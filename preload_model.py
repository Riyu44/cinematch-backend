from sentence_transformers import SentenceTransformer
print("Pre-downloading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Done.")