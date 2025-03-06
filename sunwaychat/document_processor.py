import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

class DocumentProcessor:
    def __init__(self, api_key=None, model_name="gemini-2.0-flash", chunk_size=100, chunk_overlap=10):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure Gemini API
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyC0mPhzCmVmBhhfbGwQUWXTPct0WGeLiSs")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def load_and_split(self, file_path):
        """Load and split the document into chunks."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def generate_embeddings(self, chunks):
        """Generate embeddings for the document chunks using Gemini API."""
        chunk_embeddings = []
        for chunk in chunks:
            try:
                if isinstance(chunk.page_content, str):
                    # Use Gemini's embedding capability
                    result = genai.embed_content(
                        model="models/text-embedding-004",  # Gemini's embedding model
                        content=chunk.page_content,
                        task_type="retrieval_document"
                    )
                    embedding = result.get('embedding')
                    if embedding and isinstance(embedding, list) and len(embedding) > 0:
                        embedding = np.array(embedding, dtype='float32')
                        chunk_embeddings.append(embedding)
                    else:
                        print(f"Empty or invalid embedding for chunk: {chunk.page_content[:50]}...")
                else:
                    print(f"Chunk content is not a string: {type(chunk.page_content)}")
            except Exception as e:
                print(f"Error embedding chunk: {e}")
        
        # Ensure embeddings are consistently shaped
        if chunk_embeddings:
            return np.vstack(chunk_embeddings)
        else:
            print("No valid embeddings generated")
            return np.empty((0, 768), dtype='float32')  # Gemini embedding dimension is typically 768

    def save_embeddings(self, embeddings, file_path="embeddings.npy"):
        """Save embeddings to a file."""
        try:
            np.save(file_path, embeddings)
            print(f"Embeddings saved to {file_path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, file_path="embeddings.npy"):
        """Load embeddings from a file."""
        try:
            if os.path.exists(file_path):
                embeddings = np.load(file_path)
                print(f"Embeddings loaded from {file_path}")
                return embeddings
            else:
                print(f"No embeddings file found at {file_path}")
                return None
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None