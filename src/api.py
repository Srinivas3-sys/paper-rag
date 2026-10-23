"""FastAPI production service exposing /health, /search, and /ask endpoints."""

import asyncio
from contextlib import asynccontextmanager
import logging
import time
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    GENERATOR_MODEL_NAME,
)
from src.generator import ArxivRAGGenerator
from src.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    get_retriever,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# State container for preloaded models and singletons
state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load models and indices into memory upon startup."""
    logger.info("Starting up paper-rag FastAPI service...")
    try:
        bm25 = BM25Retriever()
        dense = DenseRetriever()
        reranker = CrossEncoderReranker()
        hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense, reranker=reranker)
        generator = ArxivRAGGenerator()

        state["bm25"] = bm25
        state["dense"] = dense
        state["reranker"] = reranker
        state["hybrid"] = hybrid
        state["generator"] = generator
        state["doc_count"] = len(dense.docstore) if dense.docstore is not None else 0
        state["device"] = dense.device
        logger.info(f"Loaded paper-rag indices ({state['doc_count']} docs) onto device: {state['device']}")
    except Exception as e:
        logger.warning(
            f"Indices not available at startup ({e}). If running in Docker, ensure indices are mounted "
            "(-v $(pwd)/indices:/app/indices) or built using 'make index'."
        )
        state["bm25"] = None
        state["dense"] = None
        state["reranker"] = None
        state["hybrid"] = None
        state["generator"] = None
        state["doc_count"] = 0
        state["device"] = "uninitialized"

    yield

    logger.info("Shutting down paper-rag FastAPI service...")
    state.clear()


app = FastAPI(
    title="paper-rag: arXiv Semantic Search & RAG API",
    description="A semantic search and retrieval-augmented question answering engine over 20,000 arXiv abstracts.",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic schemas
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Technical search query or natural language question")
    retriever_type: str = Field(default="hybrid", description="Retriever architecture: 'bm25', 'dense', or 'hybrid'")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50, description="Number of results to retrieve")
    rerank: bool = Field(default=False, description="Enable cross-encoder reranking over top candidates")


class DocumentItem(BaseModel):
    id: str
    title: str
    abstract: str
    authors: str
    categories: str
    score: float
    rank: int
    retriever: str


class SearchResponse(BaseModel):
    query: str
    retriever_type: str
    count: int
    results: List[DocumentItem]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, description="Research question to ask over arXiv literature")
    retriever_type: str = Field(default="hybrid", description="Retriever to select: 'bm25', 'dense', or 'hybrid'")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="Number of supporting context papers")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM sampling temperature")
    rerank: bool = Field(default=False, description="Enable cross-encoder reranking over retrieved papers")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0, description="Maximum generation timeout in seconds")


class CitationItem(BaseModel):
    citation_id: str
    paper_id: str
    title: str
    authors: str
    categories: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationItem]
    retrieved_docs: List[DocumentItem]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    embedding_model: str
    generator_model: str
    device: str
    indices_ready: bool


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint reporting index and model statuses."""
    indices_ready = state.get("hybrid") is not None
    doc_count = state.get("doc_count", 0)
    device = state.get("device", "unknown")

    return HealthResponse(
        status="ok",
        documents_indexed=doc_count,
        embedding_model=EMBEDDING_MODEL_NAME,
        generator_model=GENERATOR_MODEL_NAME,
        device=device,
        indices_ready=indices_ready,
    )


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Retrieve top-k relevant papers matching the query using BM25, dense, or hybrid retrieval."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty or blank whitespace.")

    retriever_type = req.retriever_type.lower()
    if retriever_type not in ["bm25", "dense", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid retriever_type: '{req.retriever_type}'. Must be 'bm25', 'dense', or 'hybrid'.",
        )

    # Use state retriever if available, otherwise dynamically load
    retriever = state.get(retriever_type)
    if retriever is None:
        try:
            retriever = get_retriever(
                retriever_type=retriever_type,
                bm25_retriever=state.get("bm25"),
                dense_retriever=state.get("dense"),
                reranker=state.get("reranker"),
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Retriever indices not available: {e}. Please build indices or mount volumes.",
            )

    t0 = time.perf_counter()
    if retriever_type == "hybrid":
        results = await asyncio.to_thread(retriever.retrieve, req.query, top_k=req.top_k, rerank=req.rerank)
    else:
        results = await asyncio.to_thread(retriever.retrieve, req.query, top_k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Search completed | query=%r | retriever=%s | rerank=%s | top_k=%d | docs=%d | latency_ms=%.2f",
        req.query,
        retriever_type,
        req.rerank,
        req.top_k,
        len(results),
        latency_ms,
    )

    return SearchResponse(
        query=req.query,
        retriever_type=retriever_type,
        count=len(results),
        results=[DocumentItem(**r) for r in results],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """Perform RAG Q&A: retrieve relevant arXiv abstracts and generate an answer with inline citations."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty or blank whitespace.")

    retriever_type = req.retriever_type.lower()
    if retriever_type not in ["bm25", "dense", "hybrid"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid retriever_type: '{req.retriever_type}'. Must be 'bm25', 'dense', or 'hybrid'.",
        )

    retriever = state.get(retriever_type)
    if retriever is None:
        try:
            retriever = get_retriever(
                retriever_type=retriever_type,
                bm25_retriever=state.get("bm25"),
                dense_retriever=state.get("dense"),
                reranker=state.get("reranker"),
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Retriever indices not available: {e}. Please build indices or mount volumes.",
            )

    generator = state.get("generator")
    if generator is None:
        try:
            generator = ArxivRAGGenerator()
            state["generator"] = generator
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Generator model not available: {e}")

    t0 = time.perf_counter()
    if retriever_type == "hybrid":
        retrieved_docs = await asyncio.to_thread(retriever.retrieve, req.question, top_k=req.top_k, rerank=req.rerank)
    else:
        retrieved_docs = await asyncio.to_thread(retriever.retrieve, req.question, top_k=req.top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                generator.generate_answer,
                question=req.question,
                retrieved_docs=retrieved_docs,
                temperature=req.temperature,
            ),
            timeout=req.timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error("Generation timed out after %.1f seconds for question: %r", req.timeout_seconds, req.question)
        raise HTTPException(
            status_code=504,
            detail=f"Generation timed out after {req.timeout_seconds} seconds.",
        )
    generation_ms = (time.perf_counter() - t1) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Ask completed | question=%r | retriever=%s | rerank=%s | docs=%d | retrieval_ms=%.2f | generation_ms=%.2f | total_ms=%.2f",
        req.question,
        retriever_type,
        req.rerank,
        len(retrieved_docs),
        retrieval_ms,
        generation_ms,
        total_ms,
    )

    return AskResponse(
        question=response["question"],
        answer=response["answer"],
        citations=[CitationItem(**c) for c in response["citations"]],
        retrieved_docs=[DocumentItem(**d) for d in response["retrieved_docs"]],
    )
