import numpy as np
import google.generativeai as genai
from langchain.docstore.document import Document
from typing import List

class CustomFaissRetriever:
    """
    A custom retriever using FAISS for document retrieval without inheriting from BaseRetriever.
    This bypasses any potential attribute handling issues from LangChain's BaseRetriever.
    """
    
    def __init__(self, index, chunks, document_processor):
        """
        Initialize the retriever with a FAISS index, document chunks, and document processor.
        
        Args:
            index: A FAISS index containing document embeddings
            chunks: List of document chunks corresponding to the embeddings
            document_processor: An instance of DocumentProcessor for embedding queries
        """
        if index is None:
            raise ValueError("FAISS index cannot be None")
        if chunks is None:
            raise ValueError("Chunks cannot be None")
        if document_processor is None:
            raise ValueError("Document processor cannot be None")
            
        self.index = index
        self.chunks = chunks
        self.document_processor = document_processor
        
    def get_relevant_documents(self, query: str, k: int = 100) -> List[Document]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: Input query string
            k: Number of documents to retrieve (default: 4)
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Generate embedding for the query
            query_embedding_result = genai.embed_content(
                model="models/text-embedding-004",
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
            distances, indices = self.index.search(query_embedding, k=k)
            
            # Return relevant documents
            documents = []
            for i in indices[0]:
                if i < len(self.chunks):  # Safety check
                    documents.append(Document(page_content=self.chunks[i].page_content))
                
            return documents
        except Exception as e:
            print(f"Error retrieving documents: {str(e)}")
            return []