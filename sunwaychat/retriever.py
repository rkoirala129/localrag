import os
import numpy as np
import google.generativeai as genai
from document_processor import DocumentProcessor
from faiss_indexer import FaissIndexer
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from langchain.llms.base import LLM
from typing import Optional, List, Any

# Custom LLM wrapper for Gemini
class GeminiLLM(LLM):
    model: genai.GenerativeModel

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self) -> str:
        return "gemini"


class FaissRetriever(BaseRetriever):
    def __init__(self, index, chunks, document_processor):
        super().__init__()  # Call super().__init__() first
        
        if index is None:
            raise ValueError("FAISS index cannot be None")
        if chunks is None:
            raise ValueError("Chunks cannot be None")
        if document_processor is None:
            raise ValueError("Document processor cannot be None")
            
        # Store the passed parameters as instance variables
        self._faiss_index = index
        self._chunks = chunks
        self._document_processor = document_processor
        
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents based on the query
        :param query: Input query string
        :return: List of relevant documents
        """
        try:
            # Generate embedding for the query
            query_embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = query_embedding_result.get('embedding')
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Convert to numpy array
            query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
            
            # Perform FAISS search
            distances, indices = self._faiss_index.search(query_embedding, k=4)
            
            # Return relevant documents
            return [Document(page_content=self._chunks[i].page_content) for i in indices[0]]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

def setup_embeddings_and_index(file_path="workload.pdf"):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "api key here")
        processor = DocumentProcessor(api_key=api_key)
        indexer = FaissIndexer(use_gpu=True)
    
        embeddings_file = "embeddings.npy"
        index_file = "faiss_index.bin"

        embeddings = processor.load_embeddings(embeddings_file)
        if embeddings is None:
            print(f"Generating new embeddings for {file_path}")
            chunks = processor.load_and_split(file_path)
            embeddings = processor.generate_embeddings(chunks)
            processor.save_embeddings(embeddings)
        else:
            print(f"Loaded existing embeddings from {embeddings_file}")
            chunks = processor.load_and_split(file_path)

        if not indexer.load_index(index_file):
            print(f"Creating new FAISS index")
            indexer.create_index(embeddings)
            indexer.save_index(index_file)
        else:
            print(f"Loaded existing FAISS index from {index_file}")

        return indexer.index, chunks, embeddings, processor
    except Exception as e:
        print(f"Error in setup_embeddings_and_index: {str(e)}")
        return None, None, None, None

