"""Integration tests for FastAPI endpoints (/health, /search, /ask)."""

from fastapi.testclient import TestClient

from src.api import app, state


class _DummyRetriever:
    """Mock retriever used in CI or test environments where disk indices are absent."""

    def retrieve(self, query: str, top_k: int = 5, **kwargs):
        return [
            {
                "id": "mock-paper-001",
                "title": "Quantum Circuits and Entanglement Theory",
                "abstract": "A foundational study on quantum circuit optimization.",
                "authors": "Alice Researcher, Bob Scientist",
                "categories": "quant-ph",
                "score": 0.98,
                "rank": 1,
                "retriever": "mock",
            }
        ]


class _DummyGenerator:
    """Mock generator used in CI or test environments."""

    def generate_answer(self, question: str, retrieved_docs: list, temperature: float = 0.0):
        return {
            "question": question,
            "answer": "Quantum computing utilizes superposition and entanglement [1].",
            "citations": [
                {
                    "citation_id": "[1]",
                    "paper_id": "mock-paper-001",
                    "title": "Quantum Circuits and Entanglement Theory",
                    "authors": "Alice Researcher, Bob Scientist",
                    "categories": "quant-ph",
                }
            ],
            "retrieved_docs": retrieved_docs,
        }


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
        if state.get("hybrid") is None:
            state["hybrid"] = _DummyRetriever()

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
        if state.get("hybrid") is None:
            state["hybrid"] = _DummyRetriever()
        if state.get("generator") is None:
            state["generator"] = _DummyGenerator()

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


def test_api_validation_errors():
    with TestClient(app) as client:
        # 1. Query too short (< 2 chars) -> 422
        res = client.post("/search", json={"query": "a", "top_k": 5})
        assert res.status_code == 422

        # 2. Query too long (> 500 chars) -> 422
        res = client.post("/search", json={"query": "x" * 501, "top_k": 5})
        assert res.status_code == 422

        # 3. top_k out of bounds (top_k=0) -> 422
        res = client.post("/search", json={"query": "valid query", "top_k": 0})
        assert res.status_code == 422

        # 4. top_k out of bounds (top_k=100) -> 422
        res = client.post("/search", json={"query": "valid query", "top_k": 100})
        assert res.status_code == 422

        # 5. Blank whitespace query -> 400
        res = client.post("/search", json={"query": "    ", "top_k": 5})
        assert res.status_code == 400

        # 6. Ask temperature out of bounds (temperature=1.5) -> 422
        res = client.post("/ask", json={"question": "valid question", "temperature": 1.5})
        assert res.status_code == 422

        # 7. Ask temperature negative (temperature=-0.2) -> 422
        res = client.post("/ask", json={"question": "valid question", "temperature": -0.2})
        assert res.status_code == 422


def test_ask_generation_timeout():
    import time

    class _SlowGenerator:
        def generate_answer(self, question: str, retrieved_docs: list, temperature: float = 0.0):
            time.sleep(0.3)
            return {"question": question, "answer": "Done", "citations": [], "retrieved_docs": []}

    with TestClient(app) as client:
        state["hybrid"] = _DummyRetriever()
        state["generator"] = _SlowGenerator()

        payload = {
            "question": "What is the timeout test?",
            "retriever_type": "hybrid",
            "top_k": 1,
            "timeout_seconds": 1.0,  # Pydantic minimum is ge=1.0
        }
        # Verify valid response within generous timeout
        res = client.post("/ask", json=payload)
        assert res.status_code == 200


