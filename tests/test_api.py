from fastapi.testclient import TestClient

from rag_knowledge_assistant.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ask_returns_answer_with_citations():
    body = client.get("/ask", params={"q": "what is overfitting?"}).json()
    assert not body["refused"]
    assert body["citations"]
    assert "[1]" in body["text"]
    assert body["citations"][0]["source_url"].startswith("http")


def test_ask_refuses_out_of_domain():
    body = client.get("/ask", params={"q": "banana bread cake recipe"}).json()
    assert body["refused"]
    assert body["citations"] == []


def test_stream_is_sse_with_citations_event():
    r = client.get("/stream", params={"q": "cosine similarity"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "event: citations" in r.text
    assert "event: done" in r.text


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "RAG Knowledge Assistant" in r.text
