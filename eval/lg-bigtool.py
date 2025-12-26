import math
import os
import types
import uuid

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import (
    convert_positional_only_function_to_tool
)

load_dotenv()

azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_api_key = os.environ["AZURE_OPENAI_KEY"]
azure_api_version = os.environ["AZURE_OPENAI_VERSION"]
chat_model = os.environ["CHAT_MODEL"]
embedding_model = os.environ["EMBEDDING_MODEL"]

# Collect functions from `math` built-in
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(
        function, types.BuiltinFunctionType
    ):
        continue
    # This is an idiosyncrasy of the `math` library
    if tool := convert_positional_only_function_to_tool(
        function
    ):
        all_tools.append(tool)

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

query = "Use available tools to calculate arc cosine of 0.5."

# Test it out
for step in agent.stream(
    {"messages": query},
    stream_mode="updates",
):
    for _, update in step.items():
        for message in update.get("messages", []):
            message.pretty_print()