import hashlib
import json
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from rag import add_chunks, delete_document, list_documents, search

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

app = FastAPI(title="Local-First Privacy AI Backend")

llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


class ChatRequest(BaseModel):
    message: str
    n_results: int = 3


class SearchRequest(BaseModel):
    query: str
    n_results: int = 3


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}


def build_prompt(question, context):
    if not context:
        return question

    context_text = "\n\n".join(item["content"] for item in context)
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def stream_answer(prompt, sources):
    yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"
    try:
        async for chunk in llm.astream(prompt):
            yield f"event: token\ndata: {json.dumps({'token': chunk})}\n\n"
    except Exception as exc:
        yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
    yield "event: done\ndata: [DONE]\n\n"


@app.post("/api/chat")
async def chat(request: ChatRequest):
    context = search(request.message, request.n_results)
    prompt = build_prompt(request.message, context)
    sources = [
        {
            "document_id": item["document_id"],
            "source": item["source"],
            "page": item["page"],
            "distance": item["distance"],
        }
        for item in context
    ]
    return StreamingResponse(
        stream_answer(prompt, sources),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    document_id = hashlib.sha256(contents).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        documents = PyPDFLoader(temp_path).load()
        chunks = splitter.split_documents(documents)
        stored, duplicate = add_chunks(
            chunks, document_id, file.filename, len(contents)
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process PDF: {exc}")
    finally:
        os.remove(temp_path)

    return {
        "filename": file.filename,
        "document_id": document_id,
        "pages": len(documents),
        "chunks": len(chunks),
        "stored": stored,
        "duplicate": duplicate,
        "preview": chunks[0].page_content[:200] if chunks else "",
    }


@app.get("/api/documents")
def get_documents():
    documents = list_documents()
    return {"count": len(documents), "documents": documents}


@app.delete("/api/documents/{document_id}")
def remove_document(document_id: str):
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": document_id}


@app.post("/api/search")
def search_documents(request: SearchRequest):
    return {
        "query": request.query,
        "results": search(request.query, request.n_results),
    }
