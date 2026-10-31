import os
import tempfile
from pathlib import Path

import pytest

os.environ["CHROMA_DIR"] = tempfile.mkdtemp(prefix="chroma-test-")
os.environ["CHROMA_COLLECTION"] = "test_documents"

from fastapi.testclient import TestClient

import main
import rag


@pytest.fixture
def client():
    ids = rag.collection.get()["ids"]
    if ids:
        rag.collection.delete(ids=ids)
    return TestClient(main.app)


@pytest.fixture
def sample_pdf():
    path = Path(__file__).parent / "fixtures" / "sample.pdf"
    return path.read_bytes()
