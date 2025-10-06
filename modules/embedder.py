import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

class Embedder:
    def __init__(self, persist_directory: str = './data/chroma_db'):
        persist = os.getenv('CHROMA_PERSIST_DIR', persist_directory)
        emb_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

        device = 'cpu'
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': False}

        self.embedding = HuggingFaceEmbeddings(
            model_name=emb_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )        
        self.persist_directory = persist

    def add_documents(self, collection_name: str, documents: List[Document]):
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding, collection_name=collection_name)
        db.add_documents(documents)
        db.persist()

    def get_retriever(self, collection_name: str, k: int = 4):
        db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding, collection_name=collection_name)
        return db.as_retriever(search_type='similarity', search_kwargs={'k': k})