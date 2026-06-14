# src/document_processor.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_documents(kb_path="knowledge_base"):
    loader = DirectoryLoader(kb_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(docs)