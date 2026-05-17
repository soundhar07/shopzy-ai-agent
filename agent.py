"""
Australian Grocery Deal Assistant — agent.py
============================================
Derived from shopping_list_agent.ipynb.
MCP tools loaded directly from mcp_supermarket.py — the LLM reads the
server instructions + tool descriptions and builds its own payloads.
"""

import os
import uuid
import getpass
from datetime import datetime
from typing import Optional, Literal
import asyncio, json

from pydantic import BaseModel, Field, model_validator

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langchain_google_genai import ChatGoogleGenerativeAI
from trustcall import create_extractor

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from configuration import Configuration


# 0.  ENV SETUP


def _set_env(var: str) -> None:
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def setup_environment() -> None:
    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "aus_grocery_assistant"
    _set_env("GOOGLE_API_KEY")
    os.environ.setdefault("SUPERMARKET_SWEEP_API_KEY", "demo")


# 1.SCHEMAS  

class UpdateMemory(BaseModel):
    """Decision on what memory type to update."""
    update_type: Literal["user", "shopping_list", "instructions"]


class PreferredItem(BaseModel):
    """Maps a casual item name to a specific supermarket product."""
    generic_name: str = Field(description="Short name the user uses, e.g. 'milk'")
    product_name: str = Field(description="Full product name, e.g. 'Pauls Full Cream Milk 2L'")


class Profile(BaseModel):
    """Profile of the user."""
    name: Optional[str] = Field(default=None, description="The user's name")
    location: Optional[str] = Field(default=None, description="City/suburb the user shops in")
    gender: Optional[str] = Field(default=None, description="Gender identity")
    store_preference: Optional[str] = Field(default=None, description="Preferred store (Coles/Woolworths/Aldi)")
    dietary_choice: Optional[str] = Field(default="No specific dietary requirements", description="Dietary requirements")
    preferred_items: list[PreferredItem] = Field(default_factory=list, description="Known product preferences")

    @model_validator(mode="before")
    @classmethod
    def coerce_preferred_items(cls, values):
        pi = values.get("preferred_items")
        if isinstance(pi, str):
            try:
                values["preferred_items"] = json.loads(pi)
            except Exception:
                values["preferred_items"] = []
        return values


class ShoppingItem(BaseModel):
    """A single item on the active shopping list."""
    item_name: str = Field(description="Full product name, e.g. 'Pauls Full Cream Milk 2L'")
    quantity: str = Field(default="1", description="How many the user wants")
    unit_price: Optional[str] = Field(default=None, description="Price from MCP, e.g. '$2.50'")
    # was_price: Optional[str] = Field(default=None, description="Original price before discount")
    # savings: Optional[str] = Field(default=None, description="Amount saved")
    store: Optional[str] = Field(default=None, description="Store this item is available at")
    on_special: Optional[bool] = Field(default=None, description="True if currently discounted")
    product_url: Optional[str] = Field(default=None, description="Direct URL to the product page")
    status: Literal["to buy", "purchased", "removed"] = Field(default="to buy")



# 2.  MODEL + MCP TOOLS


model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

profile_extractor = create_extractor(model, tools=[Profile], tool_choice="Profile")

MCP_SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_supermarket.py")

async def get_mcp_tools():
    """Load tools directly from the MCP server — LLM reads their descriptions."""
    server_params = StdioServerParameters(command="python", args=[MCP_SERVER_SCRIPT])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await load_mcp_tools(session)



# 3.  PROMPTS  

MODEL_SYSTEM_MESSAGE = """\
You are a helpful Australian Grocery Deal Assistant.

You help users manage their shopping list and find the best deals at
Australian supermarkets (Coles, Woolworths, Aldi, Chemist Warehouse).

You have long-term memory that tracks:
1. The user's profile (name, location, store preference, dietary needs)
2. The user's shopping list (with prices, savings, store, and product URLs)
3. Instructions for managing the list

Current User Profile:
{user_profile}

Current Shopping List:
{shopping_list}

Current Instructions:
{instructions}

## Decision Rules — follow in order

### 1. Memory updates
- Personal info shared (name, location, store preference, diet) → UpdateMemory(update_type='user')
- Shopping list changes (add / remove / mark purchased) → UpdateMemory(update_type='shopping_list')
- User sets a list preference (formatting, sorting, etc.) → UpdateMemory(update_type='instructions')

### 2. Always search before adding an item to the shopping list
Before saving ANY item via UpdateMemory(update_type='shopping_list'), you MUST call
search_products first to fetch the current price, store, and deal status.
Steps:
  a. Determine whether the request is Mode A (broad/generic) or Mode B (specific brand).
     Refer to the MCP server instructions for the exact payload rules.
  b. Call search_products with the correct mode payload.
     - Use the user's store_preference to populate store_codes if available.
       (Coles → ['c'], Woolworths → ['w'], Aldi → ['a'], Chemist Warehouse → ['cw'])
  c. If search returns 0 results → retry with search_term only (no category_codes,
     no is_half_price, no is_discounted flags).
  d. Once results are returned, pick the most_discounted product from the response.
  e. Only then call UpdateMemory(update_type='shopping_list') using the price,
     store, savings, and product_url from the search result.

### 3. Presenting results to the user
- Always highlight the most_discounted item from the search response as the top pick.
- Show: product name, store, current price, was-price (if applicable), savings, and product URL.
- Use cupPrice for fair pack-size comparisons when showing multiple products.
- When displaying the shopping list, group items by store.
- Announce when the shopping list is updated. Do NOT announce profile changes.

### 4. No search needed for removals or status changes
- If the user removes an item or marks it as purchased, update memory directly
  without calling search_products.
"""

TRUSTCALL_INSTRUCTION = """\
Reflect on the following interaction.
Use the provided tools to retain any necessary memories about the user.
Use parallel tool calling to handle updates and insertions simultaneously.
System Time: {time}
"""
CREATE_INSTRUCTIONS = """\
Reflect on the following interaction.
Based on this interaction, update the instructions for managing the Shopping List.
Focus on:
- How the user prefers items to be named or described
- How the user prefers the list to be sorted or grouped
- Any store or brand preferences the user has expressed
- Any formatting preferences (e.g. include price, include URL)
Use any direct feedback from the user to refine these instructions.

Your current instructions are:
{current_instructions}
"""


# 4.  GRAPH NODES

def shopping_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Main agent node — loads memory, binds UpdateMemory + MCP tools, calls model."""
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    user_profile = None
    memories = store.search(("profile", user_id))
    if memories:
        user_profile = memories[0].value
    
    shopping_list = None
    shopping_list_memories = store.search(("shopping_list", user_id))
    if shopping_list_memories:
        shopping_list = "\n".join(f"{m.value}" for m in shopping_list_memories)
    

    instructions = ""
    memories = store.search(("instructions", user_id))
    if memories:
        instructions = memories[0].value

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile,
        shopping_list=shopping_list,
        instructions=instructions,
    )

   
    mcp_tools = asyncio.run(get_mcp_tools())
    all_tools = [UpdateMemory] + mcp_tools

    response = model.bind_tools(all_tools).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    return {"messages": [response]}


def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace = ("profile", user_id)
    existing = store.search(namespace)
    existing_memories = [(m.key, "Profile", m.value) for m in existing] if existing else None

    system_msg = SystemMessage(content=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat()))
    messages = list(merge_message_runs(messages=[system_msg] + state["messages"][:-1]))

    result = profile_extractor.invoke({
        "messages": messages,
        "existing": existing_memories,
    })
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", str(uuid.uuid4())), r.model_dump())

    return {"messages": [{"role": "tool", "content": "updated profile",
                          "tool_call_id": _last_tool_call_id(state)}]}


def update_shopping_list(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    namespace = ("shopping_list", user_id)
    existing = store.search(namespace)
    existing_memories = [(m.key, "ShoppingItem", m.value) for m in existing] if existing else None

    system_msg = SystemMessage(content=TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat()))
    messages = list(merge_message_runs(messages=[system_msg] + state["messages"][:-1]))

    extractor = create_extractor(model, tools=[ShoppingItem], tool_choice="ShoppingItem", enable_inserts=True)
    result = extractor.invoke({
        "messages": messages,
        "existing": existing_memories,
    })
    for r, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", str(uuid.uuid4())), r.model_dump())

    updated = [r.model_dump() for r in result["responses"]]
    return {"messages": [{"role": "tool", "content": f"Updated shopping list: {updated}",
                          "tool_call_id": _last_tool_call_id(state)}]}


def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    configurable = Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    
    namespace = ("instructions", user_id)
    existing = store.search(namespace)
    current = existing[0].value if existing else ""

    system_msg = SystemMessage(content=CREATE_INSTRUCTIONS.format(current_instructions=current))
    messages = list(merge_message_runs(messages=[system_msg] + state["messages"][:-1]))
    messages.append(HumanMessage(content="Please update the instructions based on the conversation"))

    new_memory = model.invoke(messages)
    key = existing[0].key if existing else str(uuid.uuid4())
    store.put(namespace, key, {"memory": new_memory.content})
    return {"messages": [{"role": "tool", "content": "updated instructions",
                          "tool_call_id": _last_tool_call_id(state)}]}


def run_mcp_tool(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Execute MCP tool calls — the model already built the correct payload."""

    last_msg = state["messages"][-1]
    results = []

    async def _invoke_all():
        server_params = StdioServerParameters(command="python", args=[MCP_SERVER_SCRIPT])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                tool_map = {t.name: t for t in tools}
                for tc in last_msg.tool_calls:
                    t = tool_map.get(tc["name"])
                    if t:
                        output = await t.ainvoke(tc["args"])
                        results.append({"role": "tool", "content": str(output),
                                        "tool_call_id": tc["id"]})

    asyncio.run(_invoke_all())
    return {"messages": results}



# 5.  ROUTING


def _last_tool_call_id(state: MessagesState) -> str:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            return msg.tool_calls[0]["id"]
    return ""


def route_message(state: MessagesState) -> Literal[
    "update_profile", "update_shopping_list", "update_instructions", "run_mcp_tool", "__end__"
]:
    last_msg = state["messages"][-1]
    if not getattr(last_msg, "tool_calls", None):
        return END

    for tc in last_msg.tool_calls:
        if tc["name"] == "UpdateMemory":
            update_type = tc["args"].get("update_type")
            if update_type == "user":            return "update_profile"
            if update_type == "shopping_list":   return "update_shopping_list"
            if update_type == "instructions":    return "update_instructions"
        else:
            return "run_mcp_tool"   # any non-UpdateMemory call goes to MCP

    return END



# 6.  GRAPH


def build_graph():
    builder = StateGraph(MessagesState,config_schema=Configuration)

    builder.add_node("shopping_mAIstro",    shopping_mAIstro)
    builder.add_node("update_profile",      update_profile)
    builder.add_node("update_shopping_list", update_shopping_list)
    builder.add_node("update_instructions", update_instructions)
    builder.add_node("run_mcp_tool",        run_mcp_tool)

    builder.add_edge(START, "shopping_mAIstro")
    builder.add_conditional_edges("shopping_mAIstro", route_message)

    # Every memory/tool node loops back to the main agent
    for node in ("update_profile", "update_shopping_list", "update_instructions", "run_mcp_tool"):
        builder.add_edge(node, "shopping_mAIstro")

    return builder.compile()
    # checkpointer=MemorySaver(), store=InMemoryStore())



setup_environment()
graph = build_graph()