# query_kb.py: Offline KB query with semantic matching and conversational rephrasing

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util  # pip install sentence-transformers
from typing import Dict, List, Optional

# Add imports for caching
import pickle

# Configuration
KB_PATH = "kb/curated_kb.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, offline-capable model (~80MB)

class KBQuery:
    def __init__(self, kb_path: str = KB_PATH):
        self.kb = self.load_kb(kb_path)
        self.embedder = None  # Lazy load
        self.session_history = []
        self.intent_embeddings = self.load_or_precompute_embeddings()  # Cached
    
    def load_kb(self, path: str) -> Dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"KB file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_embedder(self):
        if self.embedder is None:
            print("Loading embedding model (this may take a moment on first run)...")
            self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self.embedder
    
    def load_or_precompute_embeddings(self) -> Dict:
        cache_path = KB_PATH + ".embeddings.pkl"
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}...")
            with open(cache_path, "rb") as f:
                return pickle.load(f) 
        
        print("Precomputing embeddings (this may take a while on first run)...")
        embedder = self.get_embedder()
        intent_embeddings = {}
        total_intents = sum(len(data.get("intents", {})) for data in self.kb.values())
        processed = 0
        
        for model, data in self.kb.items():
            intent_embeddings[model] = {}
            for intent_name, entry in data.get("intents", {}).items():
                patterns = entry.get("patterns", [])
                if patterns:
                    emb = embedder.encode(patterns, convert_to_tensor=True, show_progress_bar=False)
                    intent_embeddings[model][intent_name] = {
                        "emb": emb,
                        "entry": entry
                    }
                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed}/{total_intents} intents...")
        
        print(f"Saving embeddings cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(intent_embeddings, f)
        
        return intent_embeddings
    
    def query(self, user_query: str, vehicle_model: Optional[str] = None, top_k: int = 1) -> List[Dict]:
        """
        Queries the KB: Matches query to intents via semantic similarity, rephrases answer conversationally.
        
        - If vehicle_model is None, searches all models.
        - Returns list of top matches: {"model": str, "intent": str, "answer": str, "pages": [], "pdf": str, "rephrased": str}
        """
        full_query = " ".join(self.session_history[-3:]) + " " + user_query  # Last 3 turns for more context
        has_history = bool(self.session_history)
        
        query_emb = self.get_embedder().encode(full_query, convert_to_tensor=True)
        
        matches = []
        models_to_search = [vehicle_model] if vehicle_model else list(self.kb.keys())
        
        for model in models_to_search:
            if model not in self.intent_embeddings:
                continue
            for intent_name, data in self.intent_embeddings[model].items():
                similarities = util.pytorch_cos_sim(query_emb, data["emb"])[0]
                max_sim = similarities.max().item()
                if max_sim > 0.5:  # Threshold for relevance
                    matches.append({
                        "model": model,
                        "intent": intent_name,
                        "score": max_sim,
                        "entry": data["entry"]
                    })
        
        # Sort by score and take top_k
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:top_k]
        
        results = []
        for match in top_matches:
            entry = match["entry"]
            rephrased = self.rephrase_answer(user_query, entry["answer"], has_history=has_history)
            results.append({
                "model": match["model"],
                "intent": match["intent"],
                "answer": entry["answer"],
                "pages": entry.get("pages", []),
                "pdf": entry.get("pdf", ""),
                "rephrased": rephrased
            })
        
        # Update history after
        self.session_history.append(user_query)
        if len(self.session_history) > 5:
            self.session_history.pop(0)
        
        return results
    
    def rephrase_answer(self, query: str, raw_answer: str, has_history: bool = False) -> str:
        """Simple rephrasing to make it conversational."""
        prefix = "Building on what you asked before: " if has_history else f"Based on your question about '{query}': "
        return f"{prefix}{raw_answer}. See the relevant pages in the guide. Anything else?"

if __name__ == "__main__":
    query_engine = KBQuery()
    
    print("Welcome to the EV Guide Chat! Type 'quit' to exit.")
    vehicle_model = input("Enter vehicle model (e.g., 'Tesla Model 3') or press Enter to search all: ").strip() or None
    
    while True:
        try:
            user_query = input("\nYou: ").strip()
            if not user_query or user_query.lower() == "quit":
                print("Goodbye!")
                break
            
            results = query_engine.query(user_query, vehicle_model=vehicle_model)
            
            if results:
                top = results[0]
                print(f"\nAssistant: {top['rephrased']}")
                print(f"PDF: {top['pdf']}, Pages: {top['pages']}")
            else:
                print("\nAssistant: Sorry, I couldn't find a good match in the guide. Try rephrasing?")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
