from sentence_transformers import SentenceTransformer

print("Downloading local retriever model for offline embeddings...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading CLIP model for image filtering...")
SentenceTransformer("clip-ViT-B-32")

print("Done.")