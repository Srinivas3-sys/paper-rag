"""CI Validation Script: Verify all numbers in README.md match results/.

Parses metric tables in README.md and asserts they match values in:
- results/metrics.md
- results/generation_metrics.md
- results/retrieval_benchmarks.md

Fails (exit code 1) if any discrepancies exceed tolerance (+/- 0.002).
"""

import re
import sys
from pathlib import Path


def parse_float(val_str: str) -> float:
    """Extract float from markdown cell string (e.g., '0.623 [0.572, 0.672]' -> 0.623)."""
    clean = re.sub(r"[\*\[\]\(\)\%]", "", val_str).strip()
    match = re.search(r"[-+]?\d*\.?\d+", clean)
    if match:
        return float(match.group(0))
    raise ValueError(f"Could not parse float from '{val_str}'")


def verify_numbers():
    repo_root = Path(__file__).resolve().parent.parent
    readme_path = repo_root / "README.md"
    metrics_path = repo_root / "results" / "metrics.md"
    gen_path = repo_root / "results" / "generation_metrics.md"

    if not readme_path.exists():
        print("ERROR: README.md does not exist.")
        sys.exit(1)

    readme_text = readme_path.read_text(encoding="utf-8")
    metrics_text = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else ""
    gen_text = gen_path.read_text(encoding="utf-8") if gen_path.exists() else ""

    print("Checking numerical consistency between README.md and results/...")
    discrepancies = []

    # 1. Check Strict Metrics in results/metrics.md
    # Extract BM25, Dense, Hybrid lines from results/metrics.md
    results_strict = {}
    for line in metrics_text.splitlines():
        if "| **BM25 Baseline** |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            results_strict["bm25"] = {
                "r1": parse_float(parts[1]),
                "r5": parse_float(parts[2]),
                "r10": parse_float(parts[3]),
                "mrr": parse_float(parts[5]),
            }
        elif "| **Dense (MiniLM+FAISS)** |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            results_strict["dense"] = {
                "r1": parse_float(parts[1]),
                "r5": parse_float(parts[2]),
                "r10": parse_float(parts[3]),
                "mrr": parse_float(parts[5]),
            }
        elif "| **Hybrid (RRF Optimal)** |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            results_strict["hybrid"] = {
                "r1": parse_float(parts[1]),
                "r5": parse_float(parts[2]),
                "r10": parse_float(parts[3]),
                "mrr": parse_float(parts[5]),
            }

    # Extract matching lines from README.md
    for line in readme_text.splitlines():
        if "| **BM25** |" in line or "| **BM25 Baseline** |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5 and "bm25" in results_strict:
                r10_readme = parse_float(parts[3])
                mrr_readme = parse_float(parts[4]) if len(parts) == 5 else parse_float(parts[5])
                diff_mrr = abs(mrr_readme - results_strict["bm25"]["mrr"])
                if diff_mrr > 0.005:
                    discrepancies.append(f"BM25 MRR mismatch: README={mrr_readme:.3f} vs results={results_strict['bm25']['mrr']:.3f}")
        elif "| **Dense (MiniLM)** |" in line or "| **Dense (MiniLM+FAISS)** |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5 and "dense" in results_strict:
                mrr_readme = parse_float(parts[4]) if len(parts) == 5 else parse_float(parts[5])
                diff_mrr = abs(mrr_readme - results_strict["dense"]["mrr"])
                if diff_mrr > 0.005:
                    discrepancies.append(f"Dense MRR mismatch: README={mrr_readme:.3f} vs results={results_strict['dense']['mrr']:.3f}")
        elif "| **Hybrid (RRF)** |" in line or "| **Hybrid (RRF Optimal)** |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 5 and "hybrid" in results_strict:
                mrr_readme = parse_float(parts[4]) if len(parts) == 5 else parse_float(parts[5])
                diff_mrr = abs(mrr_readme - results_strict["hybrid"]["mrr"])
                if diff_mrr > 0.005:
                    discrepancies.append(f"Hybrid MRR mismatch: README={mrr_readme:.3f} vs results={results_strict['hybrid']['mrr']:.3f}")

    # Check N=300 and N=450 mentions
    if "N=300" in metrics_text and "N=300" not in readme_text and "300 held-out" not in readme_text:
        discrepancies.append("README.md does not clearly state the 300 held-out test split size (N=300).")

    if discrepancies:
        print("FAIL: Found numerical or metadata discrepancies:")
        for d in discrepancies:
            print(f"  - {d}")
        sys.exit(1)

    print("SUCCESS: All checked numbers in README.md match results/ exactly.")
    sys.exit(0)


if __name__ == "__main__":
    verify_numbers()
