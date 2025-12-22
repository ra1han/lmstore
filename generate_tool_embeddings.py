import os
import json
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

# Load tool definitions from tools.json (array of {name, description} objects)
with open("data/tools.json", "r") as f:
    tools = json.load(f)

# Generate embeddings for each tool definition
def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using Azure OpenAI."""
    response = client.embeddings.create(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        input=texts
    )
    return [item.embedding for item in response.data]

# Extract tool names and descriptions from the array structure
tool_names = [tool["name"] for tool in tools]
tool_descriptions = [tool["description"] for tool in tools]

# Combine name + description for embedding
texts_to_embed = [f"{tool['name']}: {tool['description']}" for tool in tools]

print(f"Generating embeddings for {len(texts_to_embed)} tool definitions...")

# Generate embeddings
embeddings = generate_embeddings(texts_to_embed)

# Create a result array with tool names, descriptions, and embeddings
tool_embeddings = [
    {
        "name": name,
        "description": desc,
        "description_embedding": emb
    }
    for name, desc, emb in zip(tool_names, tool_descriptions, embeddings)
]

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")

# Save embeddings to a JSON file
output_path = "data/tool_embeddings.json"
with open(output_path, "w") as f:
    json.dump(tool_embeddings, f, indent=2)

print(f"Embeddings saved to {output_path}")
