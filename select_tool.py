import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def load_tool_embeddings(filepath: str = "data/tool_embeddings.json") -> list:
    """Load tool embeddings from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for a single text using Azure OpenAI."""
    response = client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=text
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def find_top_k_similar(query_text: str, tool_embeddings: list, k: int = 3) -> list[tuple[str, str, float]]:
    """
    Find top k semantically closest tools for a given query text.
    
    Returns:
        List of tuples containing (tool_name, description, similarity_score)
    """
    # Generate embedding for the query text
    query_embedding = generate_embedding(query_text)
    
    # Calculate similarity with each tool
    similarities = []
    for tool_data in tool_embeddings:
        tool_embedding = np.array(tool_data["description_embedding"])
        similarity = cosine_similarity(query_embedding, tool_embedding)
        similarities.append((tool_data["name"], tool_data["description"], similarity))
    
    # Sort by similarity (descending) and return top k
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:k]

def create_tools_from_matches(top_matches: list[tuple[str, str, float]]) -> list[dict]:
    """
    Convert top matches into Responses API tool format with no parameters.
    """
    tools = []
    for tool_name, description, _ in top_matches:
        tool = {
            "type": "function",
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        tools.append(tool)
    return tools

def select_tool_with_openai(query_text: str, tools: list[dict]) -> dict:
    """
    Use OpenAI Responses API to select the most appropriate tool for the query.
    
    Returns:
        Dictionary containing selected tool info or None if no tool was selected
    """
    response = client.responses.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
        instructions="You are a helpful assistant. Select the most appropriate tool to help with the user's request. You must call one of the provided tools.",
        input=query_text,
        tools=tools,
        tool_choice="required"
    )
    
    # Process the response to find function calls
    for item in response.output:
        if item.type == "function_call":
            return {
                "selected_tool": item.name,
                "call_id": item.call_id,
                "response_id": response.id
            }
    
    return None

def main():
    # Load tool embeddings
    print("Loading tool embeddings...")
    tool_embeddings = load_tool_embeddings()
    print(f"Loaded {len(tool_embeddings)} tool embeddings\n")
    
    # Example query
    query_text = "List my Cloudflare accounts."
    
    print(f"Query: '{query_text}'\n")
    print("Finding top 3 semantically closest tools...\n")
    
    # Find top 3 similar tools
    top_matches = find_top_k_similar(query_text, tool_embeddings, k=3)
    
    # Print results
    print("Top 3 matches (by embedding similarity):")
    print("-" * 80)
    for i, (tool_name, description, score) in enumerate(top_matches, 1):
        print(f"{i}. {tool_name}")
        print(f"   Similarity: {score:.4f}")
        print(f"   Description: {description}")
        print()
    
    # Convert top matches to Responses API tool format
    tools = create_tools_from_matches(top_matches)
    
    # Use OpenAI to select the best tool
    print("-" * 80)
    print("Asking OpenAI to select the best tool via function call...\n")
    
    result = select_tool_with_openai(query_text, tools)
    
    if result:
        print(f"OpenAI selected tool: {result['selected_tool']}")
        print(f"Call ID: {result['call_id']}")
        print(f"Response ID: {result['response_id']}")
    else:
        print("OpenAI did not select any tool.")

if __name__ == "__main__":
    main()
