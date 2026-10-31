import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from rag import add_chunks, search

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(title="Local-First Privacy AI Backend")

llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


class ChatRequest(BaseModel):
    message: str


class SearchRequest(BaseModel):
    query: str
    n_results: int = 3


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


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        documents = PyPDFLoader(temp_path).load()
        chunks = splitter.split_documents(documents)
        stored = add_chunks(chunks, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process PDF: {exc}")
    finally:
        os.remove(temp_path)

    return {
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "stored": stored,
        "preview": chunks[0].page_content[:200] if chunks else "",
    }


@app.post("/api/search")
def search_documents(request: SearchRequest):
    return {
        "query": request.query,
        "results": search(request.query, request.n_results),
    }
