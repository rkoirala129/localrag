from indexing import load_docs
from splitting import split_docs
from embed import hf_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader


pc = Pinecone(api_key="b8c4f6ed-79cc-475d-bf8d-e0b835993cf0")

directory = 'documents'
index_name = "langchain-chatbot"

documents = load_docs(directory)
docs = split_docs(documents)

index = pc.Index(index_name)
     
vectorstore = LangchainPinecone(index, hf_embedding.embed_query, "text")

# vectorstore.add_documents(docs)

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = vectorstore.similarity_search_with_score(query,k=k)
  else:
    similar_docs = vectorstore.similarity_search(query,k=k)
  return similar_docs

query = "How is Labor Law in Nepal?"
similar_docs = get_similiar_docs(query)
print(similar_docs)

     
