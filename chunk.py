import os
import sys
from math import ceil

def chunk_text(text, num_words, overlap):
    """
    Chunk the text into overlapping chunks.

    Args:
        text (str): The text to chunk.
        num_words (int): Number of words per chunk.
        overlap (float): Overlap percentage (0 to 1).

    Returns:
        list: List of text chunks.
    """
    words = text.split()
    step = max(1, int(num_words * (1 - overlap)))  # Calculate step size
    chunks = []

    for i in range(0, len(words), step):
        chunk = words[i:i + num_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))

    return chunks

def save_chunks(chunks, filename):
    """
    Save chunks to the ./out folder with numbered filenames.

    Args:
        chunks (list): List of text chunks.
        filename (str): Original filename.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = "./out"
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in enumerate(chunks):
        output_filename = os.path.join(output_dir, f"{base_name}_{idx + 1:05d}.txt")
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(chunk)

def main():
    if len(sys.argv) < 4:
        print("Usage: python chunker.py <filename> <num_words> <overlap>")
        sys.exit(1)

    filename = sys.argv[1]
    num_words = int(sys.argv[2])
    overlap = float(sys.argv[3])

    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    if overlap < 0 or overlap > 1:
        print("Error: Overlap must be a value between 0 and 1.")
        sys.exit(1)

    try:
        with open(filename, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except UnicodeDecodeError:
        try:
            with open(filename, "r", encoding="latin-1") as f:
                text = f.read()
        except UnicodeDecodeError:
            print("Error: Unable to decode the file with UTF-8 or Latin-1 encoding. Please check the file encoding.")
            sys.exit(1)

    chunks = chunk_text(text, num_words, overlap)
    save_chunks(chunks, filename)
    print(f"Chunks saved in './out' folder.")

if __name__ == "__main__":
    main()