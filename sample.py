import os
from dotenv import load_dotenv
from openai import AzureOpenAI

from src.lm_store import LMStore

if __name__ == "__main__":
    
    load_dotenv()
    
    # Initialize client
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # Create store
    store = LMStore(client)
    
    store.load_from_file('data/tool_embeddings.json')

    # # Add operators
    # operators = [
    #     {"name": "search_code", "description": "Search for code in repositories"},
    #     {"name": "list_files", "description": "List files in a directory"},
    #     {"name": "run_command", "description": "Execute a shell command"}
    # ]
    
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    # store.add(operators, embedding_model)
    # print(f"Added {store.count} operators")
    
    # Search without LLM
    query = "Do Cloudflare Workers costs depend on response sizes? I want to serve some images (map tiles) from an R2 bucket and I'm concerned about costs."    

    results = store.get(query=query, limit=3, embedding_model=embedding_model)
    print("\nSearch results:")
    for r in results:
        print(f"  {r.name}: {r.similarity_score:.4f}")
    
    # Search with LLM finalization
    chat_model = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    results, selection = store.get(
        query=query,
        limit=3,
        llm_search=True,
        chat_model=chat_model,
        embedding_model=embedding_model
    )
    
    if selection:
        print(f"\nLLM selected: {selection}")
    
    # # Export
    # json_data = store.export()
    # print(f"\nExported JSON:\n{json_data}")
