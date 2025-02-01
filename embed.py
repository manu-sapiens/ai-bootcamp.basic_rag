import ollama
from typing import List, Tuple

EMBED_MODEL = "all-minilm"
OLLAMA_EMBEDDING_KEY = "embeddings"

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

def embed_with_ollama(text: str) -> List[float]:
    
    ollama_response = ollama.embed(model=EMBED_MODEL, input=text)
    embedding = ollama_response.get(OLLAMA_EMBEDDING_KEY)
    if embedding is None:
        failure_message = f"No [{OLLAMA_EMBEDDING_KEY}] key found in response: {ollama_response}"
        raise ValueError(failure_message)
    #
    
    # Ensure the embedding is a flat list
    if any(isinstance(i, list) for i in embedding):
        embedding = [item for sublist in embedding for item in sublist]
    return embedding
#


# Example usage
if __name__ == "__main__":

    # Example documents about llamas
    documents = [
        "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
        "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
        "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
        "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
        "Llamas are vegetarians and have very efficient digestive systems",
        "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
    ]

    # Initialize DB
    db = SimpleVectorDB()

    # Add documents one by one
    i = 0
    for document in documents:
        # Get embedding from Ollama
        embedding = embed_with_ollama(document)
        
        # Add to DB
        db.add_document(embedding, document)
        i += 1
    #
    
    print(f"Added {len(documents)} documents to the database.")

    # Query
    query = "What animals are llamas related to?"
    print(f"QUERY: {query}")
    query_embedding = embed_with_ollama(query)
    
    # Get results
    results = db.query(query_embedding, n_results=1)
    print("\nTop result:")
    top_result = results[0]
    id = top_result[0]
    similarity = top_result[1]
    text = db.data[id]["doc_text"]
    
    print(f"ID: {id}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Text: {text}")
    
#
