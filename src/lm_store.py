"""
LMStore Module

A semantic storage system for tools/operators that uses embeddings for
similarity search and optionally LLM for final selection.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Union
import numpy as np
from openai import AzureOpenAI, OpenAI


@dataclass
class Operator:
    """Represents an operator/tool with its embedding."""
    name: str
    description: str
    embedding: list[float]


@dataclass
class SearchResult:
    """Result from a search query."""
    name: str
    description: str
    similarity_score: float


class LMStore:
    
    def __init__(self, client: Union[AzureOpenAI, OpenAI]):

        self._client = client
        self._operators: list[Operator] = []
    
    @property
    def count(self) -> int:

        return len(self._operators)
    
    def _generate_embedding(self, text: str, model: str) -> list[float]:

        response = self._client.embeddings.create(
            model=model,
            input=text
        )

        return response.data[0].embedding
    
    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:

        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def add(
        self,
        operators: Union[dict, list[dict]],
        embedding_model: str
    ) -> int:
        """
        Add one or more operators to the store.
        
        Args:
            operators: A single operator dict {"name": str, "description": str}
                      or a list of such dicts
            embedding_model: The embedding model name/deployment to use
            
        Returns:
            Number of operators added
        """
        # Normalize to list
        if isinstance(operators, dict):
            operators = [operators]
        
        added_count = 0
        for op in operators:
            name = op.get("name")
            description = op.get("description")
            
            if not name or not description:
                raise ValueError(f"Operator must have 'name' and 'description': {op}")
            
            # Generate embedding for the description
            embedding = self._generate_embedding(name + " " + description, embedding_model)
            
            # Create and store the operator
            operator = Operator(
                name=name,
                description=description,
                embedding=embedding
            )
            self._operators.append(operator)
            added_count += 1
        
        return added_count
    
    def get(
        self,
        query: str,
        embedding_model: str,
        limit: int = 3,
        llm_search: bool = False,
        chat_model: Optional[str] = None,
    ) -> Union[list[SearchResult], tuple[list[SearchResult], Optional[str]]]:

        if not self._operators:
            if llm_search:
                return [], None
            return []
        
        if not embedding_model:
            raise ValueError("embedding_model parameter is required")
        
        if llm_search and not chat_model:
            raise ValueError("chat_model parameter is required when llm_search is True")
                
        # Generate query embedding
        query_embedding = self._generate_embedding(query, embedding_model)
        
        # Calculate similarities
        results = []
        for op in self._operators:
            similarity = self._cosine_similarity(query_embedding, op.embedding)
            results.append(SearchResult(
                name=op.name,
                description=op.description,
                similarity_score=similarity
            ))
        
        # Sort by similarity (descending) and limit
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        top_results = results[:limit]
        
        if not llm_search:
            return top_results
        
        llm_result = self._select_with_llm(query, top_results, chat_model)

        return top_results, llm_result
    
    def _select_with_llm(
        self,
        query: str,
        candidates: list[SearchResult],
        model: str
    ) -> Optional[str]:
        """
        Use LLM Responses API to select the best tool from candidates.
        
        Args:
            query: The original query
            candidates: List of candidate SearchResult objects
            model: The LLM model name/deployment
            
        Returns:
            LLMSelectionResult if a tool was selected, None otherwise
        """
        # Create tool definitions
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
            for result in candidates
        ]
        
        response = self._client.responses.create(
            model=model,
            instructions="You are a helpful assistant. Select the most appropriate tool to help with the user's request. You must call one of the provided tools.",
            input=query,
            tools=tools,
            tool_choice="required"
        )
        
        for item in response.output:
            if item.type == "function_call":
                return item.name
        
        return None
    
    def export(self, filepath: Optional[str] = None) -> Union[str, None]:
        """
        Export all operators to JSON.
        
        Args:
            filepath: Optional file path to save the JSON. 
                     If None, returns the JSON string.
            
        Returns:
            JSON string if filepath is None, otherwise None (writes to file)
        """
        data = [asdict(op) for op in self._operators]
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            return None
        
        return json_str
    
    def load(self, data: list[dict]) -> int:
        """
        Load operators from a JSON structure.
        
        Args:
            data: List of dicts with 'name', 'description', and 'embedding' fields
            
        Returns:
            Number of operators loaded
        """
        loaded_count = 0
        for item in data:
            name = item.get("name")
            description = item.get("description")
            embedding = item.get("embedding")
            
            if not name or not description or not embedding:
                raise ValueError(
                    f"Each item must have 'name', 'description', and 'embedding': {item}"
                )
            
            operator = Operator(
                name=name,
                description=description,
                embedding=embedding
            )
            self._operators.append(operator)
            loaded_count += 1
        
        return loaded_count
    
    def load_from_file(self, filepath: str) -> int:
        """
        Load operators from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Number of operators loaded
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return self.load(data)
    
    def clear(self) -> None:
        """Clear all operators from the store."""
        self._operators = []