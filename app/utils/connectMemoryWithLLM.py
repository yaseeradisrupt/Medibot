import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from load_dotenv import load_dotenv
from langchain_community.vectorstores import FAISS


load_dotenv()
#Step1: Setup LLM Mistral with HuggingFace

HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    print(huggingface_repo_id)
    llm = HuggingFaceEndpoint(repo_id=huggingface_repo_id, temperature=0.5,model_kwargs={"token":HF_TOKEN,"max_length":"1024"}) #high temperature means more creative, less precise
    return llm
#step2: Connect LLM with FAISS and Create Chain


def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=["context","question"])
    return prompt

def get_chain():
    DB_FAISS_PATH='vectorstore/db_faiss'
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    custom_prompt_template="""Use the pieces of information provided in the context to answer user's qeustion. 
        If you do not know the answer, just say that you dont know, dont try to make up an answer.
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.

        """
    qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(HUGGINGFACE_REPO_ID),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 3}),#k is the number of documents to return from the database that match the query
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt(custom_prompt_template)}
        )
    return qa_chain



