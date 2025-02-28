from chunking import load_text, chunk_text
from vectordb import SimpleVectorDB
from embeds import embed_with_ollama
from llm import query_llm
import time
import sys
import os

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
    # Define database file paths
    db_path = "vectordb.pkl"
    hash_path = "hash_dict.pkl"
    
    # Initialize DB
    db = SimpleVectorDB(db_path=db_path, hash_path=hash_path)
    
    # Try to load the database from disk
    db_loaded = db.load_from_disk()
    
    # If the database doesn't exist or couldn't be loaded, create a new one
    if not db_loaded:
        print("No existing database found or could not load it. Creating a new one...")
        
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
        # Add documents one by one
        print("\nEmbedding documents:")
        total_docs = len(documents)
        start_time = time.time()
        added_count = 0
        skipped_count = 0
        
        for i, document in enumerate(documents):
            # Get embedding from Ollama
            embedding = embed_with_ollama(document)
            
            # Add document to vector DataBase (returns doc_id)
            doc_id = db.add_document(embedding, document)
            
            # Check if this was a new document or an existing one
            if doc_id == f"doc_{i + 1}":
                added_count += 1
            else:
                skipped_count += 1
            
            # Update progress bar
            fancy_progress_bar(i + 1, total_docs, prefix='Embedding:', suffix='Complete')
        
        # Print a newline after the progress bar is complete
        print("\n")
        elapsed_time = time.time() - start_time
        print(f"Added {added_count} new documents to the database.")
        print(f"Skipped {skipped_count} duplicate documents.")
        print(f"Total processing time: {elapsed_time:.2f} seconds.")
        
        # Save the database to disk
        db.save_to_disk()
    else:
        print(f"Using existing database with {len(db.data)} documents.")

    # ------- Query -------
    query = "Who is darth vader the father of?" #"what is the color of the hair of Purcell?" #"Who killed Purcell?"
    print(f"QUERY: {query}")
    query_embedding = embed_with_ollama(query)
    
    # Get results
    results = db.query(query_embedding, n_results=10)
    print("\nTop result:")
    top_result = results[0]
    id = top_result[0]
    similarity = top_result[1]
    text = db.data[id]["doc_text"]
    
    texts = ""  
    i = 0
    for result in results:
        id = result[0]
        similarity = result[1]
        text = db.data[id]["doc_text"]
        texts += f"ID={id}, TEXT={text}\n"
        i += 1
    #
    prompt = f"""
    I have the following fragments of information in my database: \n 
    {texts} + \nBased on these fragments, can you answer the following question: 
    {query}?\n. If no information is relevant, please say so.
    """

    # ------- LLM -------    
    # Compare with LLM (for fun)
    llm_answer = query_llm(prompt)
    print(f"LLM: {llm_answer}")
#
