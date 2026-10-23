"""Integration tests for FastAPI endpoints (/health, /search, /ask)."""

from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "documents_indexed" in data
        assert "embedding_model" in data
        assert "generator_model" in data


def test_search_endpoint_valid():
    with TestClient(app) as client:
        # If models are loaded into state
        payload = {
            "query": "quantum circuits and entanglement",
            "retriever_type": "hybrid",
            "top_k": 3,
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == payload["query"]
        assert data["retriever_type"] == "hybrid"
        assert len(data["results"]) <= 3
        if data["results"]:
            first = data["results"][0]
            assert "id" in first
            assert "title" in first
            assert "score" in first


def test_search_endpoint_invalid_retriever():
    with TestClient(app) as client:
        payload = {
            "query": "test query",
            "retriever_type": "invalid_type",
            "top_k": 5,
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 400


def test_ask_endpoint_valid():
    with TestClient(app) as client:
        payload = {
            "question": "What methods are used for quantum computing?",
            "retriever_type": "hybrid",
            "top_k": 2,
            "temperature": 0.0,
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "citations" in data
        assert "retrieved_docs" in data
