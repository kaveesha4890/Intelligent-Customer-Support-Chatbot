# src/embedder.py + build index
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def build_faiss_index(chunks, index_path="models/faiss_index.bin", meta_path="models/chunks.pkl"):
    texts = [c.page_content for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(texts, f)
    return index