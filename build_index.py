from src.document_processor import load_and_chunk_documents
from src.embedder import build_faiss_index

chunks = load_and_chunk_documents()
print(f"Loaded {len(chunks)} chunks")
build_faiss_index(chunks)
print("FAISS index built and saved to models/")