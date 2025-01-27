import ollama
from typing import List, Tuple
import numpy as np

#EMBED_MODEL = "mxbai-embed-large"
EMBED_MODEL = "all-minilm"

class SimpleVectorDB:
    def __init__(self):
        self.data = []  # To store tuples of (id, embedding, document)

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str]):
        """Add embeddings and documents to the database."""
        for id_, embedding, document in zip(ids, embeddings, documents):
            embedding_array = np.array(embedding[0], dtype=np.float32)  # We know it's always a list with one item
            self.data.append((id_, embedding_array, document))

    def query(self, query_embedding: List[float], n_results: int = 1) -> List[Tuple[str, str, float]]:
        """
        Query the database for the closest embeddings.
        Returns the closest `n_results` items based on cosine similarity.
        """
        query_vector = np.array(query_embedding[0], dtype=np.float32)  # We know it's always a list with one item

        # Calculate cosine similarity between the query and all stored embeddings
        results = []
        for id_, embedding, document in self.data:
            similarity = self._cosine_similarity(query_vector, embedding)
            results.append((id_, document, similarity))

        # Sort by similarity in descending order and return the top n_results
        results = sorted(results, key=lambda x: x[2], reverse=True)
        return results[:n_results]

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate the cosine similarity between two vectors."""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


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

    # Initialize the vector database
    collection = SimpleVectorDB()

    # Store each document in the vector database
    for i, doc in enumerate(documents):
        response = ollama.embed(model=EMBED_MODEL, input=doc)
        print(f"Processing document #{i}")
        collection.add(
            ids=[str(i)],
            embeddings=response["embeddings"],
            documents=[doc]
        )

    # Example query
    prompt = "What animals are llamas related to?"
    print(f"\nQuery: {prompt}")

    # Generate embedding for the query
    response = ollama.embed(
        model=EMBED_MODEL,
        input=prompt
    )

    # Perform the query and print results
    results = collection.query(response["embeddings"], n_results=1)
    for res_id, res_doc, similarity in results:
        print(f"\nMost relevant document (ID: {res_id}, Similarity: {similarity:.4f}):")
        print(f"{res_doc}")
