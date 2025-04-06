from typing import List
import os
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'rag_system')

class GeminiEmbedder(Embeddings):
    def __init__(self, api_key, model_name="models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']

def create_vector_store(api_key, texts=None, client=None):
    """Create and initialize vector store with documents."""
    try:
        # Initialize vector store
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=GeminiEmbedder(api_key=api_key),
            persist_directory="chroma_db",
            client=client  # Pass the client if provided
        )
        
        # Add documents if provided
        if texts:
            with st.spinner('📤 Uploading documents to database...'):
                vector_store.add_documents(texts)
                st.success("✅ Documents stored successfully!")
                return vector_store
        
        return vector_store
            
    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None
   