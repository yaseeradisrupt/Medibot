from fastapi import FastAPI
from pydantic import BaseModel
from app.utils.connectMemoryWithLLM import get_chain
from langchain_community.vectorstores import FAISS


app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/query/")
async def query_medibot(query: Query):
    try:
        qa_chain = get_chain()
        response = qa_chain.invoke({"query": query.query})
        return {"result":"success","data": {"response":response["result"], "source_documents": response["source_documents"]}}
    except Exception as e:
        return {"result":"error","message": str(e)}