import os
import numpy as np
import faiss
import google.generativeai as genai
from typing import List, Optional

# Import our custom implementations instead of LangChain classes
from document_processor import DocumentProcessor
from faiss_indexer import FaissIndexer
# These would be the new files with our custom implementations
from custom_retriever import CustomFaissRetriever  
from custom_qa_chain import CustomQAChain

def setup_embeddings_and_index(file_path="workload.pdf"):
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyC0mPhzCmVmBhhfbGwQUWXTPct0WGeLiSs")
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

def main():
    # Setup embeddings and index
    faiss_index, chunks, embeddings, processor = setup_embeddings_and_index("workload.pdf")
    print("FAISS Index:", faiss_index)
    print("Number of Chunks:", len(chunks) if chunks is not None else "N/A")
    print("Embeddings Shape:", embeddings.shape if embeddings is not None else "N/A")
    print("Processor:", processor)
    if faiss_index is None or chunks is None or embeddings is None or processor is None:
        print("Failed to setup embeddings and index")
        return
    
    # Get API key for Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyC0mPhzCmVmBhhfbGwQUWXTPct0WGeLiSs")
    
    # Initialize our custom retriever
    try:
        retriever = CustomFaissRetriever(
            index=faiss_index,
            chunks=chunks,
            document_processor=processor
        )
        print("Custom retriever initialized successfully")
    except Exception as e:
        print(f"Error initializing retriever: {str(e)}")
        return
    
    # Initialize our custom QA chain
    try:
        qa_chain = CustomQAChain(
            api_key=api_key,
            retriever=retriever
        )
        print("Custom QA chain initialized successfully")
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        return
    
    # Chat loop
    print("Welcome to the chat interface! Type 'exit' to quit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            break
        try:
            answer, sources = qa_chain.query(query)
            if answer:
                print(f"Answer: {answer}")
                # Uncomment to see source documents
                # print("\nSource Documents:")
                # for doc in sources:
                #     print(doc.page_content[:100] + "...")
            else:
                print("Sorry, I couldn't find an answer.")
        except Exception as e:
            print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()




















# import os
# import numpy as np
# import faiss
# import google.generativeai as genai
# from langchain.docstore.document import Document
# from langchain_core.retrievers import BaseRetriever
# from typing import List, Optional

# from document_processor import DocumentProcessor
# from retriever import FaissRetriever, setup_embeddings_and_index
# from qa_chain import QAChain


# def embed_query(query, processor):
#     """Generate embedding for a query using the same processor."""
#     try:
#         result = genai.embed_content(
#             model="models/embedding-001",
#             content=query,
#             task_type="retrieval_query"
#         )
#         embedding = result.get('embedding')
#         if embedding and isinstance(embedding, list) and len(embedding) > 0:
#             return np.array(embedding, dtype='float32')
#         raise ValueError("Failed to generate query embedding")
#     except Exception as e:
#         print(f"Error embedding query: {e}")
#         return None

# class FaissIndexer:
#     def __init__(self, use_gpu=False):
#         self.index = None
#         self.use_gpu = use_gpu

#     def create_index(self, embeddings):
#         # Assuming embeddings is a numpy array
#         dimension = embeddings.shape[1]
#         self.index = faiss.IndexFlatL2(dimension)
#         self.index.add(embeddings)

#     def load_index(self, file_path):
#         try:
#             if os.path.exists(file_path):
#                 self.index = faiss.read_index(file_path)
#                 return True
#             return False
#         except Exception as e:
#             print(f"Error loading index: {e}")
#             return False

#     def save_index(self, file_path):
#         try:
#             faiss.write_index(self.index, file_path)
#             print(f"Index saved to {file_path}")
#         except Exception as e:
#             print(f"Error saving index: {e}")


# def main():
#     # Setup embeddings and index
#     faiss_index, chunks, embeddings, processor = setup_embeddings_and_index("workload.pdf")
#     print("FAISS Index:", faiss_index)
#     print("Number of Chunks:", len(chunks) if chunks is not None else "N/A")
#     print("Embeddings Shape:", embeddings.shape if embeddings is not None else "N/A")
#     print("Processor:", processor)
#     if faiss_index is None or chunks is None or embeddings is None or processor is None:
#         print("Failed to setup embeddings and index")
#         return
    
#     # Get API key for Gemini
#     api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyC0mPhzCmVmBhhfbGwQUWXTPct0WGeLiSs")
    
#     # Initialize retriever with explicit parameters
#     try:
#         retriever = FaissRetriever(
#             index=faiss_index,
#             chunks=chunks,
#             document_processor=processor
#         )
#         print("Retriever initialized successfully")
#     except Exception as e:
#         print(f"Error initializing retriever: {str(e)}")
#         return
    
#     # Initialize QA chain with api_key parameter
#     try:
#         qa_chain = QAChain(
#             api_key=api_key,
#             retriever=retriever
#         )
#         print("QA chain initialized successfully")
#     except Exception as e:
#         print(f"Error initializing QA chain: {str(e)}")
#         return
    
#     # Chat loop
#     print("Welcome to the chat interface! Type 'exit' to quit.")
#     while True:
#         query = input("Ask a question: ")
#         if query.lower() == "exit":
#             break
#         try:
#             answer, sources = qa_chain.query(query)
#             if answer:
#                 print(f"Answer: {answer}")
#                 # Uncomment to see source documents
#                 # print("\nSource Documents:")
#                 # for doc in sources:
#                 #     print(doc.page_content[:100] + "...")
#             else:
#                 print("Sorry, I couldn't find an answer.")
#         except Exception as e:
#             print(f"Error processing query: {str(e)}")

# if __name__ == "__main__":
#     main()