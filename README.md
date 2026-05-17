# Australian Grocery Deal Assistant

An AI-powered shopping list agent that finds the best deals across Australian supermarkets (Coles, Woolworths, Aldi, Chemist Warehouse and Priceline) — built with LangGraph, Google Gemini, Trustcall, and a persistent MCP price server.

---

##  Features

- **Conversational shopping list management** — add, remove, and mark items as purchased via natural language
- **Live deal search** — searches for half-price specials and discounts in real time via the Supermarket Sweep API
- **Smart search modes** — Mode A (broad/generic) and Mode B (specific brand), with automatic fallback on 0 results
- **Long-term memory** — persists user profile, shopping list, and list-management instructions across sessions using LangGraph's `InMemoryStore`
- **Store-aware results** — filters by the user's preferred store and groups the shopping list by store
- **Persistent MCP server** — MCP tool server runs as a standalone SSE process; no subprocess restart per turn
- **Claude Desktop compatible** — connect the MCP server directly to Claude Desktop for native chat use
- **Per-turn checkpointing** — full conversation state checkpointed with `MemorySaver`

---

##  Architecture

```
┌──────────────────────────────────────────────────┐
│                  agent.py                        │
│                                                  │
│  LangGraph StateGraph (MessagesState)            │
│  ┌──────────────┐   conditional   ┌───────────┐  │
│  │shopping_     │ ─────────────►  │update_    │  │
│  │mAIstro       │                 │profile    │  │
│  │(Gemini LLM)  │ ─────────────►  │update_    │  │
│  └──────────────┘                 │shopping_  │  │
│         ▲                         │list       │  │
│         │  loop back              │update_    │  │
│         └───────────────────────  │instructions│ │
│                                   │run_mcp_   │  │
│                                   │tool       │  │
│                                   └───────────┘  │
│                                                  │
│  Memory: MemorySaver (threads) + InMemoryStore   │
│           (profile / shopping_list / instruction)│
└──────────────────────┬───────────────────────────┘
                       │ SSE  HTTP
                       ▼
┌─────────────────────────────────────────────────┐
│           mcp_supermarket.py                     │
│           FastMCP SSE Server (:8000)             │
│                                                  │
│  Tools:     search_products                      │
│  Resource:  grocery://categories                 │
│  API:       Supermarket Sweep API                │
└─────────────────────────────────────────────────┘
         ▲
         │ stdio / SSE
         │
┌────────┴────────────────────────────────────────┐
│           Claude Desktop (optional)              │
│           Chat directly with the MCP server      │
└─────────────────────────────────────────────────┘
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 3.1 Flash-Lite (`langchain-google-genai`) |
| Agent Framework | LangGraph (`StateGraph`, `MessagesState`) |
| Memory | Trustcall + LangGraph `InMemoryStore` + `MemorySaver` |
| MCP Server | FastMCP (SSE transport) |
| MCP Client | `langchain-mcp-adapters` `MultiServerMCPClient` |
| Price API | Supermarket Sweep API |
| HTTP Client | httpx |
| Schema Validation | Pydantic v2 |
| Observability | LangSmith |

---

##  Project Structure

```
grocery-agent/
├── agent.py                 # LangGraph agent — graph, nodes, routing
├── mcp_supermarket.py       # FastMCP SSE server — search_products tool
├── configuration.py         # RunnableConfig schema (user_id, thread_id)
├── requirements.txt         # Python dependencies
└── README.md
└── .env

```

---

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export GOOGLE_API_KEY=your_google_api_key
export LANGSMITH_API_KEY=your_langsmith_api_key
export SUPERMARKET_SWEEP_API_KEY=your_api_key   
```
### 3. Start the MCP server (Terminal 1)

```bash
python mcp_supermarket.py
# → FastMCP SSE server running at http://127.0.0.1:8000/sse
```

### 4. Run the agent (Terminal 2)

```bash
python agent.py
```

---

## Example Usage

```python
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "1", "user_id": "soundhar"}}

turns = [
    "My name is Soundhar. I live in Sydney and prefer Woolworths.",
    "Add milk and chips to my list.",
    "What are the best half-price deals on chips right now?",
    "Show me my full shopping list by different categories.",
    "Replace Dishwashing Liquid with Laundry Liquid.",
]

for text in turns:
    for chunk in graph.stream({"messages": [HumanMessage(content=text)]}, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
```

---

## MCP Search Modes

| Mode | When | Key Flags |
|---|---|---|
| **Mode A** — Broad/Generic | User says "milk", "chips", no brand | `search_term=""`, `category_codes=[...]`, `is_half_price=True` |
| **Mode B** — Specific Brand | User says "Pauls milk", "Cadbury Old Gold" | `search_term="Cadbury Old Gold"`, `brands=[...]`, `is_discounted=True` |
| **Fallback** | Mode A or B returns 0 results | `search_term="full cream milk"`, all flags cleared |

---

## Memory Architecture

Three namespaced stores per `user_id`:

| Namespace | Schema | Contents |
|---|---|---|
| `("profile", user_id)` | `Profile` | Name, location, store preference, dietary needs, preferred items |
| `("shopping_list", user_id)` | `ShoppingItem` | Item name, quantity, price, store, savings, URL, status |
| `("instructions", user_id)` | `dict` | LLM-generated list management preferences |

---

## 🗺️ Graph Flow

```
START
  └─► shopping_mAIstro (Gemini LLM)
            │
            ├─ UpdateMemory(user)          ──► update_profile
            ├─ UpdateMemory(shopping_list) ──► update_shopping_list
            ├─ UpdateMemory(instructions)  ──► update_instructions
            ├─ search_products (MCP)       ──► run_mcp_tool
            └─ no tool call               ──► END
                        │
                        └──────────────────► shopping_mAIstro (loop)
```

---

##  Connecting to Claude Desktop

The MCP server can be connected directly to **Claude Desktop**, letting you search for supermarket deals and manage your shopping list from inside Claude — no agent.py required.

### Step 1 — Find your Claude Desktop config file

| OS | Path |
|---|---|
| **macOS** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Windows** | `%APPDATA%\Claude\claude_desktop_config.json` |

### Step 2 — Add the MCP server entry

Open `claude_desktop_config.json` and add the following under `"mcpServers"`:

```json
{
  "mcpServers": {
    "supermarket": {
      "command": "python",
      "args": ["/absolute/path/to/mcp_supermarket.py"],
      "env": {
        "SUPERMARKET_SWEEP_API_KEY": "your_api_key"
      }
    }
  }
}
```

>  Replace `/absolute/path/to/mcp_supermarket.py` with the actual full path on your machine.  
> Example macOS: `/Users/soundhar/projects/grocery-agent/mcp_supermarket.py`

### Step 3 — Update the entry point for stdio

Claude Desktop communicates with MCP servers over **stdio**, not SSE.
Change the entry point in `mcp_supermarket.py` to:

```python
if __name__ == "__main__":
    mcp.run(transport="stdio")
```

>  Use `transport="sse"` when running with `agent.py`, and `transport="stdio"` when connecting via Claude Desktop.

### Step 4 — Restart Claude Desktop

Fully quit and reopen Claude Desktop. You should see **SupermarketPriceServer** appear in the tools panel (🔨 icon).

### Step 5 — Try it in Claude

```
Search for half-price milk deals at Woolworths
What are the best frozen meal specials right now?
Find me Cadbury Old Gold chocolate deals
```

Claude will call `search_products` directly and return live prices, savings, and product links.

---