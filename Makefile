.PHONY: help install lint typecheck test check data index evaluate api api-dev docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install     : Install pinned python dependencies"
	@echo "  make lint        : Run ruff linter and style checks"
	@echo "  make typecheck   : Run mypy static type checking on src/"
	@echo "  make test        : Run pytest test suite"
	@echo "  make check       : Run lint, typecheck, and unit tests"
	@echo "  make data        : Download and preprocess the 20k arXiv dataset sample"
	@echo "  make index       : Build BM25 and FAISS dense indices"
	@echo "  make evaluate    : Run full retrieval and generation benchmarks"
	@echo "  make api         : Run FastAPI production server (no reload)"
	@echo "  make api-dev     : Run FastAPI development server with auto-reload"
	@echo "  make docker-build: Build Docker image for paper-rag"
	@echo "  make docker-run  : Run Docker container with mounted indices volume"
	@echo "  make clean       : Remove temporary cache files"

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

typecheck:
	mypy src/ --ignore-missing-imports

test:
	pytest -v tests/

check: lint typecheck test

data:
	python3 -m src.data --sample-size 20000

index:
	python3 -m src.index

evaluate:
	python3 src/evaluate_benchmark.py
	python3 src/eval_generation.py
	python3 src/benchmark_retrieval.py

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

api-dev:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t paper-rag:latest .

docker-run:
	docker run -p 8000:8000 -v $$(pwd)/indices:/app/indices paper-rag:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
