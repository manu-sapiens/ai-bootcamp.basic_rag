from chunking import load_text, chunk_text
from vectordb import SimpleVectorDB
from embeds import embed_with_ollama
from llm import query_llm
import time
import sys

# THIS IS NOT a WORKING RAG pipeline, but it contains all the parts necessary to build one as an
# exercise to the student.

def fancy_progress_bar(current, total, bar_length=50, prefix='Progress:', suffix='Complete', fill_char='█', empty_char='░'):
    """
    Display a fancy text-based progress bar.
    
    Args:
        current (int): Current progress value
        total (int): Total value
        bar_length (int): Length of the progress bar in characters
        prefix (str): Text to display before the progress bar
        suffix (str): Text to display after the progress bar
        fill_char (str): Character to use for filled portion of the bar
        empty_char (str): Character to use for empty portion of the bar
    """
    percent = float(current) / total
    filled_length = int(bar_length * percent)
    bar = fill_char * filled_length + empty_char * (bar_length - filled_length)
    
    # Calculate ETA
    if current > 0:
        eta = f"ETA: {(time.time() - start_time) / current * (total - current):.1f}s"
    else:
        eta = "ETA: calculating..."
    
    # Create the progress bar string
    progress_str = f"\r{prefix} |{bar}| {percent:.1%} {current}/{total} {suffix} [{eta}]"
    
    # Print the progress bar
    sys.stdout.write(progress_str)
    sys.stdout.flush()

# Example usage
if __name__ == "__main__":

    # ------- Chunking -------
    text = load_text("pg75244.txt")
    documents = chunk_text(text, num_words=256, overlap_words=128)
    print("-------------")
    print(f"Nb of chunks: {len(documents)}")
    print("-------------")
    print("Example: chunk 0:")
    print(documents[0])
    print("-------------")

    # ------- Vector DB -------
    # Initialize DB
    db = SimpleVectorDB()

    # Add documents one by one
    print("\nEmbedding documents:")
    total_docs = len(documents)
    start_time = time.time()
    
    for i, document in enumerate(documents):
        # Get embedding from Ollama
        embedding = embed_with_ollama(document)
        
        # Add document to vector DataBase
        db.add_document(embedding, document)
        
        # Update progress bar
        fancy_progress_bar(i + 1, total_docs, prefix='Embedding:', suffix='Complete')
    
    # Print a newline after the progress bar is complete
    print("\n")
    elapsed_time = time.time() - start_time
    print(f"Added {total_docs} documents to the database in {elapsed_time:.2f} seconds.")

    # ------- Query -------
    query = "Who killed Purcell?"
    print(f"QUERY: {query}")
    query_embedding = embed_with_ollama(query)
    
    # Get results
    results = db.query(query_embedding, n_results=10)
    print("\nTop result:")
    top_result = results[0]
    id = top_result[0]
    similarity = top_result[1]
    text = db.data[id]["doc_text"]
    
    print(f"TOP CHUNK (id: {id}, similarity:{similarity:.4f}):\n{text}\n")

    # ------- LLM -------    
    # Compare with LLM (for fun)
    llm_answer = query_llm(query)
    print(f"LLM: {llm_answer}")
