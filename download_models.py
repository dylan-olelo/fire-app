from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Downloading retriever model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading generator model...")
pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("All models downloaded and cached.")