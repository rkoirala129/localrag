import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List

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

# QAChain class using Gemini API
class QAChain:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro", retriever=None):
        """
        Initialize the QAChain with Gemini LLM.

        :param api_key: Gemini API key
        :param model_name: Gemini model name (default: "gemini-1.5-pro")
        :param retriever: Optional retriever for document retrieval
        """
        self.model = GeminiLLM(api_key=api_key, model_name=model_name)
        self.retriever = retriever
        self.qa_chain = None
        if retriever:
            self._initialize_chain()

    def _initialize_chain(self):
        """Initialize the RetrievalQA chain."""
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.model,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True
            )
            print("QA chain initialized.")
        except Exception as e:
            print(f"Error initializing QA chain: {e}")

    def query(self, question):
        """Run a query through the QA chain."""
        if self.qa_chain:
            try:
                result = self.qa_chain.invoke({"query": question})
                return result['result'], result['source_documents']
            except Exception as e:
                print(f"Error running query: {e}")
                return None, []
        print("QA chain not initialized.")
        return None, []


# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA

# class QAChain:
#     def __init__(self, model_name="llama3.1:8b", retriever=None):
#         self.model = Ollama(model=model_name)
#         self.retriever = retriever
#         self.qa_chain = None
#         if retriever:
#             self._initialize_chain()

#     def _initialize_chain(self):
#         """Initialize the RetrievalQA chain."""
#         try:
#             self.qa_chain = RetrievalQA.from_chain_type(
#                 llm=self.model,
#                 chain_type="stuff",
#                 retriever=self.retriever,
#                 return_source_documents=True
#             )
#             print("QA chain initialized.")
#         except Exception as e:
#             print(f"Error initializing QA chain: {e}")

#     def query(self, question):
#         """Run a query through the QA chain."""
#         if self.qa_chain:
#             try:
#                 result = self.qa_chain.invoke({"query": question})
#                 return result['result'], result['source_documents']
#             except Exception as e:
#                 print(f"Error running query: {e}")
#                 return None, []
#         print("QA chain not initialized.")
#         return None, []