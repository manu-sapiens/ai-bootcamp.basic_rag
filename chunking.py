import os
import sys

def chunk_text(text, num_words, overlap_words):
    """
    Chunk text into overlapping chunks of fixed word size.
    Designed for in-memory use (e.g., RAG pipelines).
    
    Args:
        text (str): Input text.
        num_words (int): Words per chunk.
    overlap_words (int): Overlap as a number of words.
    
    Returns:
        list: List of chunks.
    """
    if overlap_words < 0:
        raise ValueError("Overlap must be a non-negative integer.")
    
    words = text.split()
    if not words:
        return []
    
    step = max(1, num_words - overlap_words)  # Ensure step ≥1
    
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i + num_words]
        chunks.append(" ".join(chunk))
    
    return chunks

def save_chunks(chunks, filename):
    """
    Save chunks to files for inspection. Used in standalone mode.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = "./out"
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        output_filename = os.path.join(output_dir, f"{base_name}_{idx + 1:05d}.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(chunk)

def load_text(filename):
    """
    Load text from a file with encoding fallback. Used in both modes.
    """
    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filename, "r", encoding="latin-1") as f:
                return f.read()
        except UnicodeDecodeError:
            print("Error: Unable to decode the file with UTF-8 or Latin-1 encoding.")
            sys.exit(1)

def main():
    """Standalone mode: Save chunks to files for inspection. Overlap is in words."""
    if len(sys.argv) < 4:
        print("Usage: python chunking.py <filename> <num_words> <overlap_words>")
        sys.exit(1)

    print("First argument = ", sys.argv[0])
    filename = sys.argv[1]
    num_words = int(sys.argv[2])
    overlap_words = int(sys.argv[3])

    text = load_text(filename)
    chunks = chunk_text(text, num_words, overlap_words)
    save_chunks(chunks, filename)
    print(f"Saved {len(chunks)} chunks to './out' folder.")

if __name__ == "__main__":
    # Execute only when run as a standalone script
    main()
