"""
LMStore Evaluation Script

Evaluates the accuracy of LMStore by comparing vector search and LLM selection
results against ground truth from eval.json.
"""

import json
import csv
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_adapter import chat_model as c_model, embedding_model as e_model
from lm_store import LMStore
from openai import AzureOpenAI


def load_eval_data(filepath: str) -> list[dict]:
    """Load evaluation data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_lmstore(
    eval_data_path: str,
    exported_tools_path: str,
    output_csv_path: str,
    embedding_model: str,
    chat_model: str
):
    """
    Evaluate LMStore accuracy using eval.json data.
    
    Args:
        eval_data_path: Path to eval.json
        exported_tools_path: Path to exported_tools.json with embeddings
        output_csv_path: Path to output CSV report
        embedding_model: Embedding model name/deployment
        chat_model: Chat model name/deployment for LLM selection
    """
    chat = c_model(    
            provider="azureopenai",
            model=os.getenv("CHAT_MODEL"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION")
            )

    embed = e_model(    
            provider="azureopenai",
            model=os.getenv("EMBEDDING_MODEL"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_VERSION")
            )
    # Initialize LMStore
    store = LMStore(chat, embed)    

    loaded_count = store.load_from_file(exported_tools_path)
    print(f"Loaded {loaded_count} tools into LMStore")
    
    # Load evaluation data
    eval_data = load_eval_data(eval_data_path)
    print(f"Loaded {len(eval_data)} evaluation queries")
    
    # Results storage
    results = []
    
    for i, item in enumerate(eval_data):
        query = item['query']
        expected_tool = item['tool']
        
        print(f"\n[{i+1}/{len(eval_data)}] Evaluating query: {query[:80]}...")
        
        # Vector search only (llm_search=False)
        start_time = time.time()
        vector_results = store.get(
            query=query,
            limit=10,
            llm_search=False
        )
        latency_vector_search = round((time.time() - start_time) * 1000, 2)  # ms
        
        # Get top vector search result
        tool_vector_search = vector_results[0].name if vector_results else ""
        
        # LLM selection (llm_search=True)
        start_time = time.time()
        llm_results, llm_selection = store.get(
            query=query,
            limit=10,
            llm_search=True,
        )
        latency_llm_selection = round((time.time() - start_time) * 1000, 2)  # ms
        
        # Get LLM selected tool
        tool_llm_selection = llm_selection if llm_selection else ""
        
        # Check matches
        tool_vector_search_match = tool_vector_search == expected_tool
        tool_llm_selection_match = tool_llm_selection == expected_tool
        
        result = {
            'query': query,
            'tool': expected_tool,
            'tool_vector_search': tool_vector_search,
            'latency_vector_search': latency_vector_search,
            'tool_llm_selection': tool_llm_selection,
            'latency_llm_selection': latency_llm_selection,
            'tool_vector_search_match': tool_vector_search_match,
            'tool_llm_selection_match': tool_llm_selection_match,
            'embedding_model': embedding_model,
            'chat_model': chat_model
        }
        results.append(result)
        
        print(f"  Expected: {expected_tool}")
        print(f"  Vector Search: {tool_vector_search} (match: {tool_vector_search_match})")
        print(f"  LLM Selection: {tool_llm_selection} (match: {tool_llm_selection_match})")
    
    # Write results to CSV
    fieldnames = [
        'query', 'tool', 'tool_vector_search', 'latency_vector_search',
        'tool_llm_selection', 'latency_llm_selection',
        'tool_vector_search_match', 'tool_llm_selection_match',
        'embedding_model', 'chat_model'
    ]
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(output_csv_path)
    
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_csv_path}")
    
    # Calculate and print summary statistics
    total = len(results)
    vector_matches = sum(1 for r in results if r['tool_vector_search_match'])
    llm_matches = sum(1 for r in results if r['tool_llm_selection_match'])
    avg_latency_vector = sum(r['latency_vector_search'] for r in results) / total
    avg_latency_llm = sum(r['latency_llm_selection'] for r in results) / total
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Vector Search Accuracy: {vector_matches}/{total} ({100*vector_matches/total:.1f}%)")
    print(f"LLM Selection Accuracy: {llm_matches}/{total} ({100*llm_matches/total:.1f}%)")
    print(f"Avg Latency (Vector Search): {avg_latency_vector:.2f} ms")
    print(f"Avg Latency (LLM Selection): {avg_latency_llm:.2f} ms")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    # Default paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    eval_data_path = os.path.join(base_dir, 'data', 'eval.json')
    exported_tools_path = os.path.join(base_dir, 'data', 'lmstore.db')
    output_csv_path = os.path.join(base_dir, 'eval', 'evaluation_report.csv')
    
    # Model configurations - update these based on your Azure OpenAI deployment
    embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
    chat_model = os.environ.get('CHAT_MODEL', 'gpt-4o-mini')
    
    print("LMStore Evaluation")
    print(f"{'='*60}")
    print(f"Eval data: {eval_data_path}")
    print(f"Tools data: {exported_tools_path}")
    print(f"Output: {output_csv_path}")
    print(f"Embedding model: {embedding_model}")
    print(f"Chat model: {chat_model}")
    print(f"{'='*60}")
    
    evaluate_lmstore(
        eval_data_path=eval_data_path,
        exported_tools_path=exported_tools_path,
        output_csv_path=output_csv_path,
        embedding_model=embedding_model,
        chat_model=chat_model
    )
