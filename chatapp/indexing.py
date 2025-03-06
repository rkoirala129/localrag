from langchain_community.document_loaders import DirectoryLoader

from splitting import split_docs

directory = 'documents'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)
# print(len(documents))

docs  = split_docs(documents)
# print(len(docs))
# print(docs[5].page_content)