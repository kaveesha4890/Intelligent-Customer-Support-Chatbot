"""
ChromaDB-backed knowledge base.
Replaces FAISS + SQLite+numpy with a proper vector database.

Benefits over the previous approach:
- HNSW index: fast approximate nearest-neighbor search that scales to millions of docs
- Built-in metadata filtering: search only within a category (e.g. fraud, billing)
- Persistent storage: survives restarts, no re-indexing needed
- Dynamic updates: add/update/delete documents without rebuilding anything
"""
import os

import chromadb
from chromadb.config import Settings

_CHROMA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chroma_db",
)
_KB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "knowledge_base",
)

_client     = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=_CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},   # use cosine similarity
        )
    return _collection


def init_knowledge_db():
    _get_collection()   # creates chroma_db/ directory and collection on first call


def populate_from_txt_files(embed_model, force: bool = False):
    """Read all .txt files from knowledge_base/ and insert into ChromaDB."""
    col = _get_collection()

    if col.count() > 0 and not force:
        return  # already populated

    if force:
        # Clear all existing documents before repopulating
        existing = col.get()
        if existing["ids"]:
            col.delete(ids=existing["ids"])

    ids, documents, embeddings, metadatas = [], [], [], []

    for category in sorted(os.listdir(_KB_DIR)):
        cat_dir = os.path.join(_KB_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()

            doc_id = f"{category}/{fname}"
            emb    = embed_model.encode(content, normalize_embeddings=True).tolist()

            ids.append(doc_id)
            documents.append(content)
            embeddings.append(emb)
            metadatas.append({"category": category, "filename": fname})

    if ids:
        col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        print(f"[knowledge_db] Imported {len(ids)} documents into ChromaDB.")


def add_document(category: str, filename: str, content: str, embed_model) -> str:
    """Add a new document. Returns its id."""
    col   = _get_collection()
    doc_id = f"{category}/{filename}"
    emb   = embed_model.encode(content, normalize_embeddings=True).tolist()
    col.add(
        ids=[doc_id],
        documents=[content],
        embeddings=[emb],
        metadatas=[{"category": category, "filename": filename}],
    )
    return doc_id


def update_document(category: str, filename: str, content: str, embed_model):
    """Update content and recompute embedding for an existing document."""
    col    = _get_collection()
    doc_id = f"{category}/{filename}"
    emb    = embed_model.encode(content, normalize_embeddings=True).tolist()
    col.update(
        ids=[doc_id],
        documents=[content],
        embeddings=[emb],
        metadatas=[{"category": category, "filename": filename}],
    )


def delete_document(category: str, filename: str):
    """Remove a document from the knowledge base."""
    col = _get_collection()
    col.delete(ids=[f"{category}/{filename}"])


def retrieve_similar(query_emb, k: int = 3, threshold: float = 0.45,
                     category: str = None) -> list[str]:
    """
    Return top-k document contents whose cosine similarity exceeds threshold.
    Optionally filter by category (e.g. 'fraud', 'billing').
    """
    col = _get_collection()
    where = {"category": category} if category else None

    results = col.query(
        query_embeddings=[query_emb.tolist()],
        n_results=min(k, col.count()),
        where=where,
        include=["documents", "distances"],
    )

    docs = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity: similarity = 1 - distance
        similarity = 1 - dist
        if similarity >= threshold:
            docs.append(doc)
    return docs


def list_documents(category: str = None) -> list[dict]:
    """Return all document metadata for inspection."""
    col = _get_collection()
    where = {"category": category} if category else None
    results = col.get(where=where, include=["metadatas"])
    return [
        {"id": id_, **meta}
        for id_, meta in zip(results["ids"], results["metadatas"])
    ]
