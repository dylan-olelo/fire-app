# query_kb.py: Offline KB query with semantic matching and conversational rephrasing

import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util  # pip install sentence-transformers
from typing import Dict, List, Optional

# Add imports for caching
import pickle

# Add import for fuzzy matching
from rapidfuzz import process, fuzz  # pip install rapidfuzz

# Configuration
KB_PATH = "kb/curated_kb.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, offline-capable model (~80MB)

class KBQuery:
    def __init__(self, kb_path: str = KB_PATH):
        self.kb = self.load_kb(kb_path)
        self.embedder = None  # Lazy load
        self.session_history = []
        self.intent_embeddings = self.load_or_precompute_embeddings()  # Cached
        self.fuzzy_threshold = 90  # For keyword fallback
        self.persisted_model = None  # Store model across turns
    
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
    
    def extract_model_from_query(self, query: str) -> Optional[str]:
        # Simple regex for common patterns (expand as needed)
        import re
        match = re.search(r"(tesla|cybertruck|model [3ysx])", query, re.IGNORECASE)
        if match:
            return match.group(1).title()  # e.g., "Tesla Model 3"
        return None
    
    def query(self, user_query: str, vehicle_model: Optional[str] = None, top_k: int = 10) -> List[Dict]:
        # Extract model if not provided and first time
        if vehicle_model is None and self.persisted_model is None:
            extracted = self.extract_model_from_query(user_query)
            if extracted:
                self.persisted_model = extracted
                print(f"Detected model from query: {self.persisted_model}")
        
        # Use persisted model if available
        effective_model = vehicle_model or self.persisted_model
        
        # Prepend model to query for better matching (simple "rephrase")
        full_query = f"{effective_model} {user_query}" if effective_model else user_query
        print(f"Effective query: {full_query[:200]}...")  # Debug
        
        has_history = self.persisted_model is not None  # Only model counts as "history"
        
        query_emb = self.get_embedder().encode(full_query, convert_to_tensor=True)
        
        matches = []
        models_to_search = [vehicle_model] if vehicle_model else list(self.kb.keys())
        
        for model in models_to_search:
            if model not in self.intent_embeddings:
                continue
            for intent_name, data in self.intent_embeddings[model].items():
                similarities = util.pytorch_cos_sim(query_emb, data["emb"])[0]
                max_sim = similarities.max().item()
                # if max_sim > 1: 
                if True: # Lowered threshold for more matches
                    matches.append({
                        "model": model,
                        "intent": intent_name,
                        "score": max_sim,
                        "entry": data["entry"]
                    })
                # else:
                    # NEW: Fuzzy keyword fallback if semantic low
                    all_patterns = " ".join(data["entry"].get("patterns", []))
                    # Use token_set_ratio for stricter, token-based matching
                    fuzzy_score = process.extractOne(full_query, [all_patterns], scorer=fuzz.token_set_ratio)[1]
                    # if fuzzy_score > self.fuzzy_threshold:
                    if True:
                        matches.append({
                            "model": model,
                            "intent": intent_name,
                            "score": fuzzy_score / 100,
                            "entry": data["entry"],
                            "via_fuzzy": True
                        })
        
        # NEW: Keyword boost
        query_words = set(full_query.lower().split())
        for m in matches:
            patterns = m["entry"].get("patterns", [])
            pattern_text = " ".join(patterns).lower()
            overlap = len(query_words.intersection(pattern_text.split())) / max(1, len(query_words))
            if overlap > 0.5:  # Significant keyword match
                m["score"] += 0.2
                m["boosted"] = True

        # Sort after boost
        matches.sort(key=lambda x: x["score"], reverse=True)
        top_matches = matches[:top_k]  # Ensure defined here

        # Debug: Print top 3 candidates
        print("Top candidates:")
        for i, m in enumerate(matches[:10], 1):
            print(f"{i}. {m['intent']} (score: {m['score']:.2f}, model: {m['model']}{', boosted' if m.get('boosted') else ''}{', fuzzy' if m.get('via_fuzzy') else ''})")

        results = []
        for match in top_matches:
            entry = match["entry"]
            rephrased = self.rephrase_answer(user_query, entry["answer"], has_history=has_history, model=effective_model)
            result = {
                "model": match["model"],
                "intent": match["intent"],
                "answer": entry["answer"],
                "pages": entry.get("pages", []),
                "pdf": entry.get("pdf", ""),
                "rephrased": rephrased
            }
            if "images" in entry:
                result["images"] = entry["images"]  # Include if present
            results.append(result)
        
        return results
    
    def rephrase_answer(self, query: str, raw_answer: str, has_history: bool = False, model: str = "") -> str:
        """Simple rephrasing to make it conversational."""
        prefix = f"For {model}: " if model else ""
        return f"{prefix}{raw_answer}. See pages for details. Anything else?"

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
                if "images" in top:
                    print(f"Images: {top['images']}")
            else:
                print("\nAssistant: Sorry, I couldn't find a good match in the guide. Try rephrasing?")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
