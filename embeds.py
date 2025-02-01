import ollama
from typing import List

EMBED_MODEL = "all-minilm"
OLLAMA_EMBEDDING_KEY = "embeddings"

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

    # Query
    query = "What animals are llamas related to?"
    print(f"QUERY: {query}")
    query_embedding = embed_with_ollama(query)
    print(f"Embedding: {query_embedding}")
#
