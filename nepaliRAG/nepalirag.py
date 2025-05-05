# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings  # NOT langchain_huggingface
# from langchain.vectorstores import FAISS
# from langchain.schema import Document
# from pdfminer.high_level import extract_text
# import easyocr
# import numpy as np
# from pdf2image import convert_from_path
# # Step 1: Extract text from Nepali PDF
# # def extract_text_from_pdf(pdf_path):
# #     doc = fitz.open(pdf_path)
# #     text = ""
# #     for page in doc:
# #         text += page.get_text()
# #     return text

# # def extract_unicode_text(pdf_path):
# #     text = extract_text(pdf_path)
# #     return text



# def extract_nepali_text_from_pdf(pdf_path):

#     try:
#          images = convert_from_path(pdf_path)
#     except Exception as e:
#         print(f"Error converting PDF to images: {e}")
#         raise
    
#     reader = easyocr.Reader(['ne'])
    
#     all_text = []
    
#     for i, image in enumerate(images):
#         print(f"Processing page {i+1}/{len(images)}")
        
#         img_np = np.array(image)
        
#         results = reader.readtext(img_np, detail=0)
        
#         page_text = ' '.join(results)
#         all_text.append(page_text)
    
#     full_text = '\n\n'.join(all_text)
#     return full_text


# # Step 2: Extract and chunk text
# extracted_text = extract_nepali_text_from_pdf("Socialclass9.pdf")

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=512,
#     chunk_overlap=50
# )
# chunks = text_splitter.split_text(extracted_text)

# # Step 3: Wrap SentenceTransformer using LangChain's HuggingFaceEmbeddings
# embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# # Step 4: Convert chunks into LangChain Document objects
# documents = [Document(page_content=chunk) for chunk in chunks]

# # Step 5: Create and save FAISS vector store
# db = FAISS.from_documents(documents, embedding_model)
# db.save_local("nepali_faiss_store")

# # Step 6: Load FAISS and perform similarity search
# db = FAISS.load_local("nepali_faiss_store", embedding_model, allow_dangerous_deserialization=True)

# query = "नेपाली समाजको वर्गीकरण के हो?"
# results = db.similarity_search(query, k=3)

# # Step 7: Print results
# for i, r in enumerate(results):
#     print(f"Result {i+1}:\n{r.page_content}\n{'-'*50}")


import os
import numpy as np
import easyocr
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import google.generativeai as genai

# Set up Gemini API
os.environ["GOOGLE_API_KEY"] = "api key here"  # Replace with your real API key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Step 1: Extract Nepali text from PDF
def extract_nepali_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        raise
    
    reader = easyocr.Reader(['ne'])
    all_text = []
    
    for i, image in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}")
        img_np = np.array(image)
        results = reader.readtext(img_np, detail=0)
        page_text = ' '.join(results)
        all_text.append(page_text)
    
    full_text = '\n\n'.join(all_text)
    return full_text

# Step 2: Extract and chunk text
extracted_text = extract_nepali_text_from_pdf("Socialclass9.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = text_splitter.split_text(extracted_text)

# Step 3: Wrap SentenceTransformer using LangChain's HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Step 4: Convert chunks into LangChain Document objects
documents = [Document(page_content=chunk) for chunk in chunks]

# Step 5: Create and save FAISS vector store
db = FAISS.from_documents(documents, embedding_model)
db.save_local("nepali_faiss_store")

# Step 6: Load FAISS and perform similarity search
db = FAISS.load_local("nepali_faiss_store", embedding_model, allow_dangerous_deserialization=True)

query = "Define the classification of Nepali society."
results = db.similarity_search(query, k=3)

# Step 7: Prepare context for Gemini
context = "\n\n".join([r.page_content for r in results])

# Step 8: Query Gemini LLM
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

model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content(prompt)

# Step 9: Output the response
print("Gemini Response:\n")
print(response.text)
