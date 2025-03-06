# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_huggingface import HuggingFaceEmbeddings  
model_name = "sentence-transformers/all-MiniLM-L6-v2"  
model_kwargs = {'device': 'cpu'}  
encode_kwargs = {'normalize_embeddings': False}  
hf_embedding = HuggingFaceEmbeddings(  
model_name=model_name,  
model_kwargs=model_kwargs,  
encode_kwargs=encode_kwargs  

)

# query_result = hf_embedding.embed_query("Hello world")
# len(query_result)
# print(len(query_result))
     