# src/retriever.py
import faiss, pickle, numpy as np
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("models/faiss_index.bin")
with open("models/chunks.pkl", "rb") as f:
    texts = pickle.load(f)

def retrieve_top_k(query: str, k: int = 5, threshold: float = 0.3):
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q_emb, k)
    results = []
    for s, i in zip(scores[0], idx[0]):
        if s >= threshold:
            results.append(texts[i])
    return results