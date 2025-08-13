from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("Downloading retriever model...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading generator model...")
pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

print("All models downloaded and cached.")