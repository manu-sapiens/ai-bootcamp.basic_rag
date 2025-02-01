from chunk import load_text, chunk_text

text = load_text("2889.txt")
chunks = chunk_text(text, num_words=256, overlap_words=128)
print("-------------")
print(f"Nb of chunks: {len(chunks)}")
print("-------------")
print("Example: chunk 0:")
print(chunks[0])
print("-------------")
# → Pass chunks to embedding/vector DB step