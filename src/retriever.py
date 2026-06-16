# src/retriever.py
import faiss, pickle, numpy as np
import os
from sentence_transformers import SentenceTransformer

embed_model = None
index = None
texts = None

def init_retriever():
    global embed_model, index, texts
    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    if index is None:
        index_path = "models/faiss_index.bin"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run build_index.py first.")
        index = faiss.read_index(index_path)
    if texts is None:
        meta_path = "models/chunks.pkl"
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Chunks metadata not found at {meta_path}. Please run build_index.py first.")
        with open(meta_path, "rb") as f:
            texts = pickle.load(f)

def retrieve_top_k(query: str, k: int = 5, threshold: float = 0.1):
    init_retriever()
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_emb, k)
    results = []
    for s, i in zip(scores[0], idx[0]):
        if s >= threshold:
            results.append(texts[i])
    return results
