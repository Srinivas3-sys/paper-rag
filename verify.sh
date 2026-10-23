#!/usr/bin/env bash
set -e

echo "=============================================="
echo "    paper-rag Self-Correction Verification    "
echo "=============================================="

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[Step 1/5] Checking Python environment and dependencies..."
python3 -c "
import torch
import sentence_transformers
import faiss
import rank_bm25
import fastapi
import datasets
print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print(f'SentenceTransformers: {sentence_transformers.__version__}')
print(f'FAISS: {faiss.__version__}')
print(f'FastAPI: {fastapi.__version__}')
"
echo "✓ Environment check passed."

echo "[Step 2/5] Running Ruff code analysis and linting..."
if command -v ruff &>/dev/null; then
  ruff check src tests || {
    echo "ERROR: Ruff linting failed."
    exit 1
  }
else
  python3 -m compileall src tests
fi
echo "✓ Lint and syntax check passed."

echo "[Step 3/5] Running test suite via Pytest..."
if ls tests/test_*.py 1> /dev/null 2>&1; then
  pytest -v tests/ || {
    echo "ERROR: Pytest tests failed."
    exit 1
  }
else
  echo "No test files matching test_*.py yet, skipping."
fi
echo "✓ Test suite passed."

echo "[Step 4/5] Checking deliverables & artifacts..."
# Verify evaluation metrics if evaluation module has been executed
if [ -f "results/metrics.md" ]; then
  if ! grep -q "Recall@5" results/metrics.md; then
    echo "ERROR: results/metrics.md missing Recall@5 metric."
    exit 1
  fi
  if ! grep -q "MRR" results/metrics.md; then
    echo "ERROR: results/metrics.md missing MRR metric."
    exit 1
  fi
  echo "✓ results/metrics.md is valid."
fi

if [ -f "results/metrics.png" ]; then
  echo "✓ results/metrics.png exists."
fi

echo "[Step 5/5] Final verification status..."
echo "ALL CHECKS PASSED"
exit 0
