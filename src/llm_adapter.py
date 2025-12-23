from typing import Optional, Literal
from enum import Enum
import ollama
from openai import AzureOpenAI

class LLMProvider(Enum):
    """Supported LLM providers."""
    AZURE_OPENAI = "azureopenai"
    OLLAMA = "ollama"

def chat_model(
        provider: Literal["azureopenai", "ollama"],
        model: str,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        api_version: Optional[str] = None
    ) -> callable:

    if provider == LLMProvider.AZURE_OPENAI.value:
        def azure_chat(
                messages: list[dict],
                tools: list[dict],
            ) -> str:

            azure_client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=api_version
            )

            for msg in messages:
                if msg.get("role") == "system":
                    instructions = msg.get("content", "")
                elif msg.get("role") == "user":
                    query = msg.get("content", "")
            
            tools = [
                {
                    "type": "function",
                    "name": result.name,
                    "description": result.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                for result in tools
            ]
                    
            response = azure_client.responses.create(
                model=model,
                instructions=instructions,
                input=query,
                tools=tools,
                tool_choice="required"
            )
            
            for item in response.output:
                if item.type == "function_call":
                    return item.name
            
            return None
         
        return azure_chat
    else:
        def ollama_chat(
                messages: list[dict],
                tools: list[dict],
            ) -> str:        
           
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": result.name,
                        "description": result.description,
                        "parameters": {
                        }
                    }
                }
                for result in tools
            ]

            response = ollama.chat(model=model, messages=messages, tools=tools)

            if response.message.tool_calls:
                tool = response.message.tool_calls[0]
                return tool.function.name
        
            return None
        
        return ollama_chat

def embedding_model(
        provider: Literal["azureopenai", "ollama"],
        model: str,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        api_version: Optional[str] = None
    ) -> callable:

    if provider == LLMProvider.AZURE_OPENAI.value:
        def azure_embedding(
                text: str
            ) -> str:

            azure_client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=api_version
            )
            
            response = azure_client.embeddings.create(
                model=model,
                input=text
            )

            return response.data[0].embedding
         
        return azure_embedding
    elif provider == LLMProvider.OLLAMA.value:
        def ollama_embedding(
                text: str
            ) -> str:        
           
            raise NotImplementedError("Ollama embedding not implemented yet.")
        
        return ollama_embedding