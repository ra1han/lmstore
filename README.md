# Koala - LMStore

A semantic storage system for tools/operators that uses embeddings for similarity search and optionally LLM for final selection.

## Overview

LMStore provides a way to store tools with their semantic embeddings and retrieve the most relevant tools for a given natural language query. It combines vector similarity search with optional LLM-based selection for improved accuracy.

## Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Set the following environment variables for Azure OpenAI:

```bash
AZURE_OPENAI_ENDPOINT=<your-endpoint>
AZURE_OPENAI_KEY=<your-api-key>
AZURE_OPENAI_VERSION=<api-version>
EMBEDDING_MODEL=<embedding-deployment-name>
CHAT_MODEL=<chat-deployment-name>
```

## How LMStore Works

### Architecture

LMStore is built around three core components:

1. **Operator Storage** - Tools are stored as `Operator` objects containing:
   - `name`: Unique identifier for the tool
   - `description`: Natural language description of what the tool does
   - `embedding`: Vector representation of the tool (generated from name + description)

2. **Vector Search** - Uses cosine similarity to find semantically similar tools:
   - Query text is converted to an embedding
   - Cosine similarity is computed against all stored tool embeddings
   - Results are ranked by similarity score

3. **LLM Selection (Optional)** - Refines vector search results using an LLM:
   - Top candidates from vector search are presented to the LLM
   - LLM uses tool-calling to select the most appropriate tool
   - Provides more accurate selection for ambiguous queries

### Core API

```python
from src.lm_store import LMStore
from src.llm_adapter import chat_model, embedding_model

# Initialize with chat and embedding functions
chat = chat_model(provider="azureopenai", model="gpt-4o", ...)
embed = embedding_model(provider="azureopenai", model="text-embedding-3-small", ...)

store = LMStore(chat, embed)

# Add tools
store.add({
    "name": "create_pull_request",
    "description": "Creates a new pull request on GitHub"
})

# Vector search only
results = store.get(query="I need to merge my feature branch", limit=5)

# Vector search + LLM selection
results, llm_pick = store.get(query="I need to merge my feature branch", limit=5, llm_search=True)
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         LMStore                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ADD TOOLS                                                   │
│     ┌──────────┐     ┌───────────┐     ┌──────────────────┐    │
│     │  Tool    │ ──► │ Embedding │ ──► │ Operator Storage │    │
│     │  (dict)  │     │   Model   │     │   (in-memory)    │    │
│     └──────────┘     └───────────┘     └──────────────────────┘ │
│                                                                 │
│  2. SEARCH (llm_search=False)                                   │
│     ┌──────────┐     ┌───────────┐     ┌──────────────────┐    │
│     │  Query   │ ──► │ Embedding │ ──► │ Cosine Similarity│    │
│     │  (str)   │     │   Model   │     │   + Ranking      │    │
│     └──────────┘     └───────────┘     └──────────────────┘    │
│                                                  │              │
│                                                  ▼              │
│                                         [Top K Results]        │
│                                                                 │
│  3. SEARCH (llm_search=True)                                    │
│     ┌──────────┐     ┌───────────┐     ┌──────────────────┐    │
│     │  Query   │ ──► │  Vector   │ ──► │   LLM Selection  │    │
│     │  (str)   │     │  Search   │     │  (tool calling)  │    │
│     └──────────┘     └───────────┘     └──────────────────┘    │
│                                                  │              │
│                                                  ▼              │
│                                   [Top K Results + LLM Pick]   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Persistence

LMStore supports export/import to preserve embeddings:

```python
# Export to file (saves embeddings)
store.export("data/lmstore.db")

# Load from file (skips re-embedding)
store.load_from_file("data/lmstore.db")
```

## Evaluation Design

The evaluation framework measures the accuracy of LMStore's tool selection capabilities.

### Evaluation Data Structure

Evaluation data (`data/eval.json`) consists of query-tool pairs:

```json
[
  {
    "tool": "mcp_github_create_pull_request",
    "query": "Create a pull request from the feature/auth-fix branch into main"
  },
  {
    "tool": "mcp_stripe_list_invoices",
    "query": "List the most recent invoices for customer cus_9x82ABC"
  }
]
```

### Metrics Collected

For each query, the evaluation captures:

| Metric | Description |
|--------|-------------|
| `tool_vector_search` | Top result from pure vector search |
| `tool_llm_selection` | Tool selected by LLM from top candidates |
| `latency_vector_search` | Time for vector search (ms) |
| `latency_llm_selection` | Time for vector search + LLM (ms) |
| `tool_vector_search_match` | Whether vector search matched ground truth |
| `tool_llm_selection_match` | Whether LLM selection matched ground truth |

### Evaluation Process

```
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────────┐  │
│  │  eval.json   │     │ LMStore with │     │  For each      │  │
│  │  (queries +  │ ──► │ loaded tools │ ──► │  query:        │  │
│  │  ground truth│     │              │     │                │  │
│  └──────────────┘     └──────────────┘     │  1. Vector     │  │
│                                            │     search     │  │
│                                            │  2. LLM select │  │
│                                            │  3. Compare    │  │
│                                            └────────────────┘  │
│                                                     │          │
│                                                     ▼          │
│                                            ┌────────────────┐  │
│                                            │ evaluation_    │  │
│                                            │ report.csv     │  │
│                                            └────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Running the Evaluation

```bash
python eval/evaluate_lmstore.py
```

### Output

Results are appended to `eval/evaluation_report.csv` with:
- Per-query results for detailed analysis
- Summary statistics printed to console:
  - Total queries evaluated
  - Vector Search Accuracy (%)
  - LLM Selection Accuracy (%)
  - Average latencies for both methods

### Key Insights

The evaluation helps answer:
1. **Vector Search vs LLM Selection** - Does adding LLM selection improve accuracy?
2. **Latency Trade-offs** - What's the cost of improved accuracy?
3. **Model Comparison** - How do different embedding/chat models perform?

## Project Structure

```
koala/
├── src/
│   ├── lm_store.py       # Core LMStore implementation
│   └── llm_adapter.py    # LLM provider adapters (Azure, Ollama)
├── data/
│   ├── tools.json        # Raw tool definitions
│   ├── exported_tools.json  # Tools with pre-computed embeddings
│   └── eval.json         # Evaluation query-tool pairs
├── eval/
│   ├── evaluate_lmstore.py   # Evaluation script
│   └── evaluation_report.csv # Results
├── data-prep/
│   └── extract_tools.py  # Tool extraction utilities
└── sample.ipynb          # Interactive examples
```

## Supported Providers

| Provider | Chat | Embeddings |
|----------|------|------------|
| Azure OpenAI | ✅ | ✅ |
| Ollama | ✅ | ❌ |

## License

MIT
