import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Set Gemini API
# os.environ["GOOGLE_API_KEY"] = "AIzaSyC0mPhzCmVmBhhfbGwQUWXTPct0WGeLiSs"  
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=api_key)


def ask_question(query, faiss_dir="nepali_faiss_store", k=3):
    # Step 1: Load embeddings and vectorstore
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    vector_db = FAISS.load_local(faiss_dir, embed_model, allow_dangerous_deserialization=True)

    # Step 2: Perform similarity search
    results = vector_db.similarity_search(query, k=k)
    context = "\n\n".join([r.page_content for r in results])

    # Step 3: Build prompt
    prompt = f"""
तपाईंले तल दिइएको सन्दर्भलाई ध्यानमा राखेर प्रश्नको उत्तर दिनुहोस्। 
उत्तर नेपाली युनिकोडमा दिनुहोस् र त्यसपछि त्यसै उत्तरको अंग्रेजी अनुवाद पनि दिनुहोस्।

Please answer the following question based on the given context. 
First, respond in Nepali (Unicode), and then provide the same answer translated into English.

Context / सन्दर्भ:
{context}

Question / प्रश्न:
{query}

Answer / उत्तर (Nepali first, then English):
    """

    # Step 4: Query Gemini
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)

    # Step 5: Print answer
    print("Gemini Response:\n")
    print(response.text)

# Example usage
if __name__ == "__main__":
    ask_question("Summarize the document.")
