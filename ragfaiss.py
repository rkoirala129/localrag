
import os
import numpy as np
import faiss
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever, Document
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Any
from langchain.schema import BaseRetriever, Document

# MODEL = "llama3:instruct"
MODEL = "gemma2:latest"

# Initialize the model and embeddings
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Load the document
loader = PyPDFLoader("constitution.pdf")
documents = loader.load()

# Split the document into chunks
chunk_size = 450
chunk_overlap = 0
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
chunks = text_splitter.split_documents(documents)

# Embed the chunks
chunk_embeddings = []

for chunk in chunks:
    try:
        # Ensure each chunk is a string
        if isinstance(chunk.page_content, str):
            embedding = embeddings.embed_query(chunk.page_content)
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                embedding = np.array(embedding, dtype='float32')
                chunk_embeddings.append(embedding)
            else:
                print(f"Empty or invalid embedding for chunk: {chunk.page_content[:50]}...")
        else:
            print(f"Chunk content is not a string: {type(chunk.page_content)}")
    except Exception as e:
        print(f"Error embedding chunk: {e}")

if chunk_embeddings:
    embeddings_array = np.vstack(chunk_embeddings)
    print(f"Embeddings array shape: {embeddings_array.shape}")
else:
    print("No embeddings generated. Check the embedding process.")
    embeddings_array = np.empty((0, 0), dtype='float32')


# Initialize FAISS index
faiss_index = None
is_gpu_index = False

if embeddings_array.size > 0:
    try:
        dim = embeddings_array.shape[1]  # Dimension of embeddings
        
        # Try GPU first
        try:
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0  # Use GPU 0
            gpu_index = faiss.GpuIndexFlatL2(res, dim, config)
            
            # Add embeddings in batches
            batch_size = 100  # Adjust this based on your GPU memory
            for i in range(0, embeddings_array.shape[0], batch_size):
                batch = embeddings_array[i:i+batch_size]
                gpu_index.add(batch)
            
            faiss_index = gpu_index
            is_gpu_index = True
            print("FAISS GPU index initialized and embeddings added.")
        except Exception as e:
            print(f"GPU indexing failed: {e}")
            print("Falling back to CPU indexing.")
            
            # CPU indexing
            cpu_index = faiss.IndexFlatL2(dim)
            cpu_index.add(embeddings_array)
            faiss_index = cpu_index
            print("FAISS CPU index initialized and embeddings added.")
        
    except Exception as e:
        print(f"Error initializing FAISS index: {e}")
else:
    print("Empty embeddings array. Cannot initialize FAISS index.")

# Define the retriever class
class FaissRetriever(BaseRetriever, BaseModel):
    index: Any = Field(description="FAISS index")
    chunks: List[Any] = Field(description="List of document chunks")
    embedding_model: Any = Field(description="Embedding model")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        print(f"FaissRetriever initialized with index: {self.index is not None}, chunks: {len(self.chunks)}, embedding_model: {self.embedding_model is not None}")

    def get_relevant_documents(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query, k=5):
        try:
            print(f"Query: {query}")
            query_embedding = self.embedding_model.embed_query(query)
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
            print(f"Query embedding shape: {query_embedding.shape}")
            distances, indices = self.index.search(query_embedding, k)
            results = [Document(page_content=self.chunks[i].page_content, metadata=self.chunks[i].metadata) for i in indices[0]]
            return results
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []

    async def aget_relevant_documents(self, query: str):
        return self.get_relevant_documents(query)
    
# Ensure faiss_index is initialized before creating retriever
if faiss_index is not None:
    try:
        retriever = FaissRetriever(index=faiss_index, chunks=chunks, embedding_model=embeddings)
        print(f"Using {'GPU' if is_gpu_index else 'CPU'} FAISS index")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Query the system
        query = "Under what circumstances do the office of prime minister be vacant?"
        result = qa_chain.invoke({"query": query})
        print("Answer:", result['result'])
        # print("\nSource Documents:")
        # for doc in result['source_documents']:
        #     print(doc.page_content[:100] + "...")  # Print first 100 characters of each source document
    except Exception as e:
        print(f"Error running QA chain: {e}")
        import traceback
        traceback.print_exc()  # This will print the full error traceback
else:
    print("Cannot run QA chain as FAISS index is not initialized.")