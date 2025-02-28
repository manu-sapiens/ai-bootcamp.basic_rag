#vectordb.py
from typing import List, Tuple, Dict, Any
import hashlib
import json
import os
import pickle

class SimpleVectorDB:
    def __init__(self, db_path="vectordb.pkl", hash_path="hash_dict.pkl"):
        # Store documents as a list of dictionaries for clarity
        self.data = {}  # Format: {"doc_id": {"doc_embedding": List[float], "doc_text": str}}
        self.hash_dict = {}  # Format: {"hash": "doc_id"}
        self.db_path = db_path
        self.hash_path = hash_path

    def compute_hash(self, text: str) -> str:
        """Compute a hash for the given text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def add_document(self, embedding: List[float], text: str) -> str:
        """
        Add a single document with auto-generated ID.
        Returns the document ID.
        If the document already exists (based on hash), returns the existing ID.
        """
        # Compute hash for the text
        doc_hash = self.compute_hash(text)
        
        # Check if this document already exists
        if doc_hash in self.hash_dict:
            return self.hash_dict[doc_hash]  # Return existing document ID
        
        # Create a new document ID
        doc_id = f"doc_{len(self.data) + 1}"
        
        # Add document to the database
        self.data[doc_id] = {
            "doc_embedding": embedding,
            "doc_text": text,
            "doc_hash": doc_hash
        }
        
        # Add hash to the hash dictionary
        self.hash_dict[doc_hash] = doc_id
        
        return doc_id

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

    def save_to_disk(self) -> None:
        """Save the database and hash dictionary to disk."""
        # Save the database
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.data, f)
        
        # Save the hash dictionary
        with open(self.hash_path, 'wb') as f:
            pickle.dump(self.hash_dict, f)
        
        print(f"Database saved to {self.db_path} and {self.hash_path}")

    def load_from_disk(self) -> bool:
        """
        Load the database and hash dictionary from disk.
        Returns True if successful, False otherwise.
        """
        # Check if both files exist
        if not (os.path.exists(self.db_path) and os.path.exists(self.hash_path)):
            return False
        
        try:
            # Load the database
            with open(self.db_path, 'rb') as f:
                self.data = pickle.load(f)
            
            # Load the hash dictionary
            with open(self.hash_path, 'rb') as f:
                self.hash_dict = pickle.load(f)
            
            print(f"Database loaded from {self.db_path} and {self.hash_path}")
            print(f"Loaded {len(self.data)} documents")
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False

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
