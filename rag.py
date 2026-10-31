import os
from datetime import datetime, timezone
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from fastembed import TextEmbedding

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def add_chunks(chunks, document_id, filename, size_bytes):
    if not chunks:
        return 0, False

    existing = collection.get(where={"document_id": document_id}, limit=1)
    if existing["ids"]:
        return 0, True

    uploaded_at = datetime.now(timezone.utc).isoformat()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = [vector.tolist() for vector in embedder.embed(texts)]
    ids = [uuid4().hex for _ in texts]
    metadatas = [
        {
            "document_id": document_id,
            "source": filename,
            "page": chunk.metadata.get("page", 0),
            "chunk": index,
            "size_bytes": size_bytes,
            "uploaded_at": uploaded_at,
        }
        for index, chunk in enumerate(chunks)
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(texts), False


def search(query, n_results=3):
    if collection.count() == 0:
        return []

    query_embedding = list(embedder.query_embed(query))[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )

    matches = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append(
            {
                "content": text,
                "document_id": metadata.get("document_id"),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": round(distance, 4),
            }
        )
    return matches


def list_documents():
    data = collection.get(include=["metadatas"])
    documents = {}

    for metadata in data["metadatas"]:
        document_id = metadata.get("document_id")
        if not document_id:
            continue

        document = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "filename": metadata.get("source"),
                "chunks": 0,
                "pages": set(),
                "size_bytes": metadata.get("size_bytes", 0),
                "uploaded_at": metadata.get("uploaded_at"),
            },
        )
        document["chunks"] += 1
        document["pages"].add(metadata.get("page", 0))

    for document in documents.values():
        document["pages"] = len(document["pages"])

    return list(documents.values())


def delete_document(document_id):
    existing = collection.get(where={"document_id": document_id}, limit=1)
    if not existing["ids"]:
        return False

    collection.delete(where={"document_id": document_id})
    return True
