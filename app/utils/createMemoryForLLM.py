from pydoc import text
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
DATA_PATH = 'data/unprocessed/'
#Step1: Load Raw PDF's Data
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documnets=loader.load()
    return documnets

documents=load_pdf_files(data=DATA_PATH)
# print("length of documents",len(documents))

#Step2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50) #chunk_overlap means to maintain the context of the text from the previous chunk
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(documents)
print("length of text_chunks",len(text_chunks))


#Step3: Create Vector Embedding
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

embedding_model=get_embedding_model()

#Step4: Store Vector Embeddings in FAISS
DB_FAISS_PATH='vectorstore/db_faiss'
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)