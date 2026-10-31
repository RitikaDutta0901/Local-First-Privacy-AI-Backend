import os
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


def add_chunks(chunks, filename):
    if not chunks:
        return 0

    texts = [chunk.page_content for chunk in chunks]
    embeddings = [vector.tolist() for vector in embedder.embed(texts)]
    ids = [uuid4().hex for _ in texts]
    metadatas = [
        {"source": filename, "page": chunk.metadata.get("page", 0)}
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(texts)


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
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": round(distance, 4),
            }
        )
    return matches
