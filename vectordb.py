#vectordb.py
from typing import List, Tuple

class SimpleVectorDB:
    def __init__(self):
        # Store documents as a list of dictionaries for clarity
        self.data = {}  # Format: {"doc_id": {"doc_embedding": List[float], "doc_text": str}}

    def add_document(self, embedding: List[float], text: str):
        """Add a single document with auto-generated ID."""
        doc_id = f"doc_{len(self.data) + 1}"
        self.data[doc_id] = {
            "doc_embedding": embedding,
            "doc_text": text
        }

    def query(self, query_embedding: List[float], n_results: int = 3) -> List[Tuple[str, float]]:
        """
        Compare query to all documents using cosine similarity.
        Returns top n_results as (id, text, similarity_score).
        """
        results = []
        
        # Calculate similarity for each document
        for doc_id, entry in self.data.items():
            similarity = self.cosine_similarity(query_embedding, entry["doc_embedding"])
            results.append((doc_id, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Pure-Python cosine similarity without numpy."""
        
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be the same length")
        #        
        
        dot_product = 0.0
        magnitude_a = 0.0
        magnitude_b = 0.0
        
        for i in range(len(vec_a)):
            dot_product += vec_a[i] * vec_b[i]
            magnitude_a += vec_a[i] ** 2
            magnitude_b += vec_b[i] ** 2
        
        magnitude_a = magnitude_a ** 0.5
        magnitude_b = magnitude_b ** 0.5
        
        if magnitude_a * magnitude_b == 0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)
    #  
#
