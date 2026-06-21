import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # use cached models, no network calls

from sentence_transformers import SentenceTransformer
from src.knowledge_db import init_knowledge_db, populate_from_txt_files, retrieve_similar

embed_model = None


def init_retriever():
    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        init_knowledge_db()
        populate_from_txt_files(embed_model)  # no-op if already populated


def retrieve_top_k(query: str, k: int = 5, threshold: float = 0.45,
                   category: str = None) -> list[str]:
    init_retriever()
    query_emb = embed_model.encode(query, normalize_embeddings=True)
    return retrieve_similar(query_emb, k=k, threshold=threshold, category=category)
