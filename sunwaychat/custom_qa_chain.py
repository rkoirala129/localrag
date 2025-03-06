import google.generativeai as genai
from typing import Optional, List, Tuple
from langchain.docstore.document import Document

class CustomQAChain:
    """
    A custom QA Chain that works with our custom retriever instead of LangChain's BaseRetriever.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash", retriever=None):
        """
        Initialize the QA Chain with Gemini LLM.

        Args:
            api_key: Gemini API key
            model_name: Gemini model name (default: "gemini-1.5-pro")
            retriever: Optional custom retriever for document retrieval
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.retriever = retriever
        
    def query(self, question: str) -> Tuple[Optional[str], List[Document]]:
        """
        Run a query through the QA chain.
        
        Args:
            question: The question to answer
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if not self.retriever:
            print("Retriever not initialized.")
            return None, []
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            if not docs:
                print("No relevant documents found.")
                return None, []
            
            # Construct prompt with retrieved context
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""Answer the following question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate answer
            response = self.model.generate_content(prompt)
            return response.text, docs
            
        except Exception as e:
            print(f"Error running query: {str(e)}")
            return None, []