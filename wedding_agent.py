
"""
Wedding Planning Multi-Agent
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


# ==========================================
# 1. TRAVEL AGENT - MCP INTERCEPTOR
# ==========================================

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

RETRYABLE_MCP_CODES = {-32603}


class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


# ==========================================
# 2. VENUE AGENT
# ==========================================

from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool
from langchain.agents import create_agent

tavily_client = TavilyClient()


@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters
      (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}


venue_agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt="""
    You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
    You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
    - Price (lowest)
    - Capacity (exact match)
    - Reviews (highest)
    You may need to make multiple searches to iteratively find the best options.
    You have a suggested limit of 12 web searches. Count every web_search call you make.
    After 12 searches, you should stop searching and summarize the best options you have
    found so far.
    """
)


# ==========================================
# 3. DJ / PLAYLIST AGENT
# ==========================================

from langchain_community.utilities import SQLDatabase

db_path = Path(
    "/Users/yisun/lca-lc-foundations/notebooks/module-2/resources/Chinook.db"
)

if not db_path.is_file():
    raise FileNotFoundError(f"Database not found: {db_path}")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")


@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"


playlist_agent = create_agent(
    model="gpt-5-nano",
    tools=[query_playlist_db],
    system_prompt="""
    You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
    Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
    If you run into errors when querying the database, try to fix them by making changes to the query.
    Do not come back empty handed, keep trying to query the db until you find a list of songs.

    This is a SQLite database. Before writing any data queries, first discover the schema.
    """
)


# ==========================================
# 4. WEDDING STATE
# ==========================================

from langchain.agents import AgentState
from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str


@tool
def update_state(
    origin: str,
    destination: str,
    guest_count: str,
    genre: str,
    runtime: ToolRuntime
) -> Command:
    """Update the state when you know all of the values: origin, destination, guest_count, genre.
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""

    return Command(update={
        "origin": origin,
        "destination": destination,
        "guest_count": guest_count,
        "genre": genre,
        "messages": [
            ToolMessage(
                "Successfully updated state",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })


# ==========================================
# 5. MAIN APPLICATION
# ==========================================

async def main():

    # --------------------------------------
    # 5.1 CONNECT TO KIWI MCP
    # --------------------------------------

    client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )

    tools = await client.get_tools()

    # --------------------------------------
    # 5.2 TRAVEL AGENT
    # --------------------------------------

    travel_agent = create_agent(
        model="gpt-5-nano",
        tools=tools,
        system_prompt="""
        You are a travel agent. Search for flights to the desired destination wedding location.
        You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
        - Price (lowest, economy class)
        - Duration (shortest)
        - Date (time of year which you believe is best for a wedding at this location)
        To make things easy, only look for one ticket, one way.
        You may need to make multiple searches to iteratively find the best options.
        You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
        If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
        Once you have found the best options, let the user know your shortlist of options.
        """
    )

    # --------------------------------------
    # 5.3 SUBAGENT TOOLS
    # --------------------------------------

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the desired destination wedding location."""
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        response = await travel_agent.ainvoke({
            "messages": [
                HumanMessage(
                    content=f"Find flights from {origin} to {destination}"
                )
            ]
        })
        return response["messages"][-1].content

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Venue agent chooses the best venue for the given location and capacity."""
        destination = runtime.state["destination"]
        capacity = runtime.state["guest_count"]
        query = f"Find wedding venues in {destination} for {capacity} guests"
        response = venue_agent.invoke({
            "messages": [
                HumanMessage(content=query)
            ]
        })
        return response["messages"][-1].content

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Playlist agent curates the perfect playlist for the given genre."""
        genre = runtime.state["genre"]
        query = f"Find {genre} tracks for wedding playlist"
        response = playlist_agent.invoke({
            "messages": [
                HumanMessage(content=query)
            ]
        })
        return response["messages"][-1].content

    # --------------------------------------
    # 5.4 COORDINATOR AGENT
    # --------------------------------------

    coordinator = create_agent(
        model="gpt-5-nano",
        tools=[
            search_flights,
            search_venues,
            suggest_playlist,
            update_state
        ],
        state_schema=WeddingState,
        checkpointer=InMemorySaver(),
        system_prompt="""
        You are a wedding coordinator.
        First find all the information you need to update the state. When you have the information, update the state.
        Once that has completed and returned, you can delegate the tasks
        to your specialists for flights, venues, and playlists.
        Once you have received their answers, coordinate the perfect wedding for me.
        """
    )

    # --------------------------------------
    # 5.5 TERMINAL CONVERSATION
    # --------------------------------------

    config = {
        "configurable": {
            "thread_id": "wedding_1"
        },
        "recursion_limit": 40
    }

    print("\nWedding Agent is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break

        if not user_input:
            continue

        response = await coordinator.ainvoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config
        )

        print("\nWedding Agent:")
        print(response["messages"][-1].content)
        print()


# ==========================================
# 6. RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    asyncio.run(main())
