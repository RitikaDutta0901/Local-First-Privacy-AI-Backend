import os

from fastapi import FastAPI, HTTPException
from langchain_community.llms import Ollama
from pydantic import BaseModel

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(title="Local-First Privacy AI Backend")

llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        response = await llm.ainvoke(request.message)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}")
    return {"response": response}
