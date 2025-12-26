import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import Tool

from langgraph_bigtool import create_agent

load_dotenv()

azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_api_key = os.environ["AZURE_OPENAI_KEY"]
azure_api_version = os.environ["AZURE_OPENAI_VERSION"]
chat_model = os.environ["CHAT_MODEL"]
embedding_model = os.environ["EMBEDDING_MODEL"]

# Load tools from JSON file
tools_path = Path(__file__).parent.parent / "data" / "tools.json"
with open(tools_path, "r", encoding="utf-8") as f:
    tools_config = json.load(f)


def _make_noop_tool(name: str, description: str) -> Tool:
    """Create a placeholder tool that simply returns its name when invoked."""

    def _run(*_args, **_kwargs):
        return f"Executed tool: {name}"

    return Tool.from_function(
        func=_run,
        name=name,
        description=description,
    )


all_tools = [
    _make_noop_tool(tool["name"], tool.get("description", ""))
    for tool in tools_config
]

# Create registry of tools. This is a dict mapping
# identifiers to tool instances.
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# Index tool names and descriptions in the LangGraph
# Store. Here we use a simple in-memory store.
embeddings = init_embeddings(
    f"azure_openai:{embedding_model}",
    azure_endpoint=azure_endpoint,
    azure_deployment=embedding_model,
    api_key=azure_api_key,
    openai_api_version=azure_api_version,
)

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {
            "description": f"{tool.name}: {tool.description}",
        },
    )

# Initialize agent
llm = init_chat_model(
    f"azure_openai:{chat_model}",
    azure_endpoint=azure_endpoint,
    azure_deployment=chat_model,
    api_key=azure_api_key,
    openai_api_version=azure_api_version,
)

builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)

system_instruction = (
    "You are a tool-orchestration agent. Always a tool to "
    "complete the user's request. Never respond directly. Don't try to call the tool, just select it."
)

messages = [
    {"role": "system", "content": system_instruction},
    {"role": "user", "content": "Open a PR to merge my hotfix branch into production."},
]

# Test it out
for step in agent.stream(
    {"messages": messages},
    stream_mode="updates",
):
    for _, update in step.items():
        for message in update.get("messages", []):
            message.pretty_print()