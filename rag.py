from chunking import load_text, chunk_text
from vectordb import SimpleVectorDB
from embeds import embed_with_ollama
from llm import query_llm

# Example usage
if __name__ == "__main__":

    text = load_text("pg75244.txt")
    documents = chunk_text(text, num_words=256, overlap_words=128)
    print("-------------")
    print(f"Nb of chunks: {len(documents)}")
    print("-------------")
    print("Example: chunk 0:")
    print(documents[0])
    print("-------------")

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
    query = "Who killed Purcell?"
    print(f"QUERY: {query}")
    query_embedding = embed_with_ollama(query)
    
    # Get results
    results = db.query(query_embedding, n_results=1)
    print("\nTop result:")
    top_result = results[0]
    id = top_result[0]
    similarity = top_result[1]
    text = db.data[id]["doc_text"]
    
    print(f"TOP CHUNK (id: {id}, similarity:{similarity:.4f}):\n{text}\n")
    
    # Compare with LLM (for fun)
    llm_answer = query_llm(query)
    print(f"LLM: {llm_answer}")
    
#

