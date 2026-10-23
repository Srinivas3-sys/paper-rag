.PHONY: help install lint test verify data index evaluate api docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  make install     : Install required python dependencies"
	@echo "  make lint        : Run ruff linter and style checks"
	@echo "  make test        : Run pytest test suite"
	@echo "  make verify      : Run automated self-correction verification script"
	@echo "  make data        : Download and preprocess the 20k arXiv dataset sample"
	@echo "  make index       : Build BM25 and FAISS dense indices"
	@echo "  make evaluate    : Evaluate BM25, Dense, Hybrid retrievers and plot results"
	@echo "  make api         : Run FastAPI development server"
	@echo "  make docker-build: Build Docker image for paper-rag"
	@echo "  make docker-run  : Run Docker container on port 8000"
	@echo "  make clean       : Remove temporary cache files"

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest -v tests/

verify:
	./verify.sh

data:
	python3 -m src.data --sample-size 20000

index:
	python3 -m src.index

evaluate:
	python3 -m src.evaluate

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t paper-rag:latest .

docker-run:
	docker run -p 8000:8000 paper-rag:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
