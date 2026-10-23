# Self-Correction Loop Log

This log tracks all verification iterations, failures, root causes, and fixes applied throughout the implementation of `paper-rag`.

| Attempt | Module / Component | Error Observed | Root Cause | Fix Applied | Status |
|---------|--------------------|----------------|------------|-------------|--------|
| 1 | Scaffolding / verify.sh | Pytest exit code 5 (no tests collected) | Pytest exits with status code 5 when 0 test files match `test_*.py` | Update verify.sh to check for `tests/test_*.py` files before invoking pytest or allow exit code 5 when no tests are defined yet | Fixed |
| 2 | Data / tests/test_data.py | Ruff F401 `pytest` imported but unused | Unused import left in test file | Removed unused `import pytest` | Fixed |
| 3 | Evaluate / src/evaluate.py | Ruff F401 and F841 in src/evaluate.py | Unused `pandas` import and unused `metrics` local variable | Removed unused import and variable | Fixed |
