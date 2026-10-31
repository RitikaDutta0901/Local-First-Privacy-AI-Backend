import main


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_lists_document_and_detects_duplicate(client, sample_pdf):
    response = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["stored"] > 0
    assert data["duplicate"] is False

    response = client.get("/api/documents")
    documents = response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["document_id"] == data["document_id"]

    response = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
    )
    assert response.json()["duplicate"] is True
    assert response.json()["stored"] == 0


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_search_and_delete_document(client, sample_pdf):
    upload = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
    ).json()

    response = client.post(
        "/api/search",
        json={"query": "What is retrieval augmented generation?", "n_results": 2},
    )
    results = response.json()["results"]
    assert results
    assert results[0]["document_id"] == upload["document_id"]

    response = client.delete(f"/api/documents/{upload['document_id']}")
    assert response.status_code == 200
    assert client.get("/api/documents").json()["count"] == 0
    assert client.delete(f"/api/documents/{upload['document_id']}").status_code == 404


def test_chat_streams_sources_and_tokens(client, sample_pdf, monkeypatch):
    upload = client.post(
        "/api/upload",
        files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
    ).json()

    class FakeLLM:
        async def astream(self, prompt):
            assert "Context:" in prompt
            for token in ["Hello", " from", " test"]:
                yield token

    monkeypatch.setattr(main, "llm", FakeLLM())

    response = client.post("/api/chat", json={"message": "What is RAG?"})
    assert response.status_code == 200
    assert "event: sources" in response.text
    assert upload["document_id"] in response.text
    assert "event: token" in response.text
    assert '"Hello"' in response.text
    assert "event: done" in response.text
