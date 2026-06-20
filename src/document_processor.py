# src/document_processor.py
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(kb_path="knowledge_base"):
    docs = []
    for root, dirs, files in os.walk(kb_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": file_path}))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(docs)