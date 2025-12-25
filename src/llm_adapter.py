from typing import Optional, Literal, Callable, Protocol, Sequence, Any
from enum import Enum
import ollama
from openai import AzureOpenAI

class LLMProvider(Enum):
    """Supported LLM providers."""
    AZURE_OPENAI = "azureopenai"
    OLLAMA = "ollama"

class ToolCandidate(Protocol):
    name: str
    description: str


ChatCallable = Callable[[Sequence[dict], Sequence[Any]], Optional[str]]
EmbeddingCallable = Callable[[str], list[float]]


def _extract_content(messages: Sequence[dict], role: str) -> str:
    """Return the most recent content for the given role."""
    for message in reversed(messages):
        if message.get("role") == role:
            return message.get("content", "")
    return ""


def _normalize_tool(tool: Any) -> tuple[str, str]:
    """Convert tool-like objects or dicts into name/description tuple."""
    if hasattr(tool, "name") and hasattr(tool, "description"):
        return getattr(tool, "name"), getattr(tool, "description")
    if isinstance(tool, dict):
        return tool.get("name", ""), tool.get("description", "")
    raise ValueError(f"Invalid tool format: {tool}")


def _azure_tools(tools: Sequence[Any]) -> list[dict]:
    return [
        {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        for name, description in (_normalize_tool(tool) for tool in tools)
    ]


def _ollama_tools(tools: Sequence[Any]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {}
            }
        }
        for name, description in (_normalize_tool(tool) for tool in tools)
    ]


class AzureChatAdapter:
    def __init__(
        self,
        model: str,
        endpoint: Optional[str],
        key: Optional[str],
        api_version: Optional[str],
    ) -> None:
        self._model = model
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
        )

    def __call__(self, messages: Sequence[dict], tools: Sequence[Any]) -> Optional[str]:
        instructions = _extract_content(messages, "system")
        query = _extract_content(messages, "user")

        response = self._client.responses.create(
            model=self._model,
            instructions=instructions,
            input=query,
            tools=_azure_tools(tools),
            tool_choice="required",
        )

        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                return item.name
        return None


class OllamaChatAdapter:
    def __init__(self, model: str) -> None:
        self._model = model

    def __call__(self, messages: Sequence[dict], tools: Sequence[Any]) -> Optional[str]:
        response = ollama.chat(
            model=self._model,
            messages=list(messages),
            tools=_ollama_tools(tools),
        )

        if response.message.tool_calls:
            tool = response.message.tool_calls[0]
            return tool.function.name
        return None


class AzureEmbeddingAdapter:
    def __init__(
        self,
        model: str,
        endpoint: Optional[str],
        key: Optional[str],
        api_version: Optional[str],
    ) -> None:
        self._model = model
        self._client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
        )

    def __call__(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding


class OllamaEmbeddingAdapter:
    def __call__(self, text: str) -> list[float]:
        raise NotImplementedError("Ollama embedding not implemented yet.")


def chat_model(
    provider: Literal["azureopenai", "ollama"],
    model: str,
    endpoint: Optional[str] = None,
    key: Optional[str] = None,
    api_version: Optional[str] = None,
) -> ChatCallable:
    try:
        normalized_provider = LLMProvider(provider)
    except ValueError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc

    if normalized_provider is LLMProvider.AZURE_OPENAI:
        return AzureChatAdapter(model, endpoint, key, api_version)
    if normalized_provider is LLMProvider.OLLAMA:
        return OllamaChatAdapter(model)

    raise ValueError(f"Unsupported provider: {provider}")


def embedding_model(
    provider: Literal["azureopenai", "ollama"],
    model: str,
    endpoint: Optional[str] = None,
    key: Optional[str] = None,
    api_version: Optional[str] = None,
) -> EmbeddingCallable:
    try:
        normalized_provider = LLMProvider(provider)
    except ValueError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc

    if normalized_provider is LLMProvider.AZURE_OPENAI:
        return AzureEmbeddingAdapter(model, endpoint, key, api_version)
    if normalized_provider is LLMProvider.OLLAMA:
        return OllamaEmbeddingAdapter()

    raise ValueError(f"Unsupported provider: {provider}")