from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.utils.connectMemoryWithLLM import graph
app = FastAPI()

class Query(BaseModel):
    query: str

async def event_generator(query: str):
    async for msg, metadata in graph.astream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="custom",
    ):
        if "response" in metadata.get("tags", []):
            yield msg["content"] + "\n"

@app.post("/query/")
async def query_medibot(query: Query):
    try:
        return StreamingResponse(event_generator(query.query), media_type="text/event-stream")
    except Exception as e:
        return {"result": "error", "message": str(e)}
