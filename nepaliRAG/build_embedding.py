# build_faiss_store.py
import os
import numpy as np
import easyocr
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

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
    
    return '\n\n'.join(all_text)

def build_faiss_store(pdf_path, faiss_dir):
    # Step 1: Extract text
    text = extract_nepali_text_from_pdf(pdf_path)

    # Step 2: Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)

    # Step 3: Convert to Documents
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Step 4: Embedding model
    embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Step 5: Create and save FAISS
    vector_db = FAISS.from_documents(documents, embed_model)
    vector_db.save_local(faiss_dir)

if __name__ == "__main__":
    build_faiss_store("Socialclass9.pdf", "nepali_faiss_store")
