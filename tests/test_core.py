"""Model-free regression tests using actual PDF, FAISS, BM25 and API code."""
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.errors import PdfReadError

from app import main
from app.pdf_loader import load_pdf
from app.rag import Chunk, EMBED_DIM, chunk_pages
from app.reranker import rerank
from app.store import Retrieval, VectorStore


def vectors(count):
    data = np.zeros((count, EMBED_DIM), dtype="float32")
    for i in range(count):
        data[i, 0] = 1 - i * 0.1
        data[i, 1] = np.sqrt(1 - data[i, 0] ** 2)
    return data


def test_blank_pdf_does_not_write_shared_temp_file(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    stream = BytesIO()
    writer.write(stream)
    assert load_pdf(stream.getvalue()) == []
    assert list(tmp_path.iterdir()) == []


def test_invalid_pdf_is_not_silently_accepted():
    with pytest.raises(PdfReadError):
        load_pdf(b"not a PDF")


def test_chunk_overlap_and_original_page_number():
    text = "0123456789" * 100
    chunks = chunk_pages([(text, 2), ("short second page", 5)])
    assert [c.page for c in chunks] == [2, 2, 2, 5]
    assert chunks[0].text[-50:] == chunks[1].text[:50]
    assert [c.chunk_id for c in chunks] == list(range(4))
    assert all(len(c.text) <= 500 for c in chunks)


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([("", 1), ("   ", 2)]) == []


def test_empty_store_returns_no_results():
    assert VectorStore().search(vectors(1)[0]) == []


def test_mismatched_vectors_are_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        VectorStore().add(vectors(2), [Chunk(0, "document", 1)])


def test_unmatched_keywords_do_not_pollute_dense_ranking():
    store = VectorStore()
    store.add(vectors(6), [Chunk(i, f"reference document number {i}", 1) for i in range(6)])
    query = vectors(1)[0]
    dense = store.search(query, top_k=2)
    hybrid = store.search(query, top_k=2, query_text="unrelatedxyz")
    assert [r.chunk.chunk_id for r in hybrid] == [r.chunk.chunk_id for r in dense] == [0, 1]


def test_positive_keyword_match_can_improve_ranking():
    store = VectorStore()
    texts = ["general handbook", "inspection form ZR-INSP-22", "battery charging",
             "pricing information", "office addresses", "travel guidance"]
    store.add(vectors(6), [Chunk(i, text, 1) for i, text in enumerate(texts)])
    query = vectors(1)[0]
    assert store.search(query, top_k=1)[0].chunk.chunk_id == 0
    assert store.search(query, top_k=1, query_text="ZR-INSP-22")[0].chunk.chunk_id == 1


def test_reset_clears_dense_and_keyword_indices():
    store = VectorStore()
    store.add(vectors(1), [Chunk(0, "sample text", 1)])
    store.reset()
    assert store.size() == 0
    assert store.chunks == [] and store.bm25 is None


def test_reranker_preserves_first_stage_slots(monkeypatch):
    from app import reranker
    model = SimpleNamespace(predict=lambda pairs: np.arange(len(pairs)))
    monkeypatch.setattr(reranker, "get_reranker", lambda: model)
    candidates = [Retrieval(Chunk(i, f"passage {i}", 1), 0.1) for i in range(5)]
    result = rerank("question", candidates, top_k=3)
    assert [r.chunk.chunk_id for r in result] == [0, 1, 4]


@pytest.fixture
def local_client(monkeypatch):
    monkeypatch.setattr(main, "store", VectorStore())
    monkeypatch.setattr(main, "_sessions", {})
    with TestClient(main.app) as client:
        yield client


def test_health_without_models_or_keys(local_client):
    assert local_client.get("/health").json() == {"status": "ok", "chunks_indexed": 0}


def test_query_without_document_has_clear_error(local_client):
    response = local_client.post("/query", json={"question": "hello"})
    assert response.status_code == 400
    assert "No document indexed" in response.json()["detail"]


@pytest.mark.parametrize("top_k", [0, 11])
def test_invalid_top_k_is_rejected(local_client, top_k):
    assert local_client.post("/query", json={"question": "hello", "top_k": top_k}).status_code == 422


def test_upload_rejects_non_pdf(local_client):
    response = local_client.post("/upload", files={"file": ("notes.txt", b"text", "text/plain")})
    assert response.status_code == 400


def test_query_returns_actual_source_and_history(local_client, monkeypatch):
    main.store.add(vectors(1), [Chunk(7, "The support period is 24 months.", 2)])
    monkeypatch.setattr(main, "embed_texts", lambda texts: vectors(len(texts)))
    calls = []

    def generate(*, system, messages):
        calls.append(messages)
        return "The support period is 24 months."

    monkeypatch.setattr(main, "get_llm_client", lambda: SimpleNamespace(generate=generate))
    first = local_client.post("/query", json={"question": "What is the support period?"}).json()
    assert first["sources"][0]["chunk_id"] == 7
    assert first["sources"][0]["page"] == 2
    assert "24 months" in calls[0][-1]["content"]
    second = local_client.post("/query", json={"question": "Repeat that", "session_id": first["session_id"]})
    assert second.status_code == 200
    assert second.json()["session_id"] == first["session_id"]
    assert len(calls[1]) == 3
