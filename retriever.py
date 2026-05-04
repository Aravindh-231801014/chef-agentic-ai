import os
import logging
import warnings

# --- ABSOLUTE TOP LOG SILENCING ---
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import json
import numpy as np

# ---- LOAD DATASET ----
DATA_PATH = os.path.join("data", "recipes.json")
if not os.path.exists(DATA_PATH):
    # Fallback to current dir
    DATA_PATH = "recipes.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    recipes = json.load(f)

# ---- LAZY LOAD MODEL & INDEX ----
_model = None
_index = None

def get_model():
    from sentence_transformers import SentenceTransformer
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer("all-MiniLM-L6-v2", show_progress_bar=False)
        except Exception:
            print("Warning: Could not load embedding model. Using keyword fallback.")
            return None
    return _model

def build_index():
    import faiss
    global _index
    model = get_model()
    if model is None:
        return None
    # Combine title and ingredients for embedding
    corpus = [r.get("title", "") + " " + " ".join(r.get("ingredients", [])) for r in recipes]
    embeddings = model.encode(corpus)
    
    dimension = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dimension)
    _index.add(embeddings.astype('float32'))
    return _index

def get_index():
    global _index
    if _index is None:
        build_index()
    return _index

def retrieve(query, top_k=3):
    """Retrieves relevant recipes based on query."""
    model = get_model()
    
    # Keyword fallback if model or index fails
    if model is None:
        query_words = set(query.lower().split())
        scored_recipes = []
        for r in recipes:
            text = (r.get("title", "") + " " + " ".join(r.get("ingredients", []))).lower()
            score = len(query_words.intersection(set(text.split())))
            scored_recipes.append((score, r))
        
        scored_recipes.sort(key=lambda x: x[0], reverse=True)
        return [r for score, r in scored_recipes[:top_k]]

    index = get_index()
    if index is None:
        return recipes[:top_k] # Last resort fallback
    
    query_vec = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, top_k)
    
    results = [recipes[i] for i in indices[0] if i != -1]
    return results

def search(query, top_k=3):
    """Alias for retrieve to match requested handover doc."""
    return retrieve(query, top_k)