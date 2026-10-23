"""Data ingestion, cleaning, and preprocessing for arXiv abstracts."""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    DATASET_NAME,
    DATASET_SAMPLE_SIZE,
    EVAL_DATA_PATH,
    RAW_DATA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: Optional[str]) -> str:
    """Normalize text by stripping excess whitespace and newlines."""
    if not text or not isinstance(text, str):
        return ""
    # Replace multiple whitespace/newlines with single space
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def parse_arxiv_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Clean and standardize an individual arXiv record."""
    paper_id = str(raw.get("id", "")).strip()
    title = clean_text(raw.get("title", ""))
    abstract = clean_text(raw.get("abstract", ""))
    authors = clean_text(raw.get("authors", ""))
    categories = clean_text(raw.get("categories", ""))

    # Validate essential fields
    if not paper_id or not title or not abstract:
        return None

    # Filter out extremely short non-meaningful abstracts
    if len(abstract.split()) < 10:
        return None

    combined_text = f"Title: {title}\nAbstract: {abstract}"

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "combined_text": combined_text,
    }


def download_arxiv_sample(
    dataset_name: str = DATASET_NAME,
    sample_size: int = DATASET_SAMPLE_SIZE,
    output_path: Path = RAW_DATA_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """Stream and sample clean arXiv abstracts from Hugging Face."""
    if output_path.exists() and not force:
        logger.info(f"Loading cached dataset from {output_path}")
        df = pd.read_parquet(output_path)
        logger.info(f"Loaded {len(df)} records from cache.")
        return df

    logger.info(f"Streaming {sample_size} records from Hugging Face dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    records: List[Dict[str, Any]] = []
    pbar = tqdm(total=sample_size, desc="Sampling arXiv abstracts")

    for raw in dataset:
        parsed = parse_arxiv_record(raw)
        if parsed:
            records.append(parsed)
            pbar.update(1)
            if len(records) >= sample_size:
                break

    pbar.close()

    df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df)} processed arXiv records to {output_path}")
    return df


def create_evaluation_benchmark(
    df: pd.DataFrame,
    num_queries: int = 200,
    output_path: Path = EVAL_DATA_PATH,
    seed: int = 42,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Construct an objective evaluation benchmark of queries mapped to target document IDs.
    
    Creates realistic search queries derived from paper titles and core subject matter.
    """
    if output_path.exists() and not force:
        logger.info(f"Loading cached evaluation queries from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Generating {num_queries} evaluation benchmark queries...")
    sample = df.sample(n=min(num_queries, len(df)), random_state=seed).copy()

    eval_data: List[Dict[str, Any]] = []

    for idx, row in enumerate(sample.itertuples()):
        title = row.title
        paper_id = row.id
        categories = row.categories

        # Generate realistic queries from title and key topic
        # Query 1: Lowercase search terms without punctuation
        clean_title_query = re.sub(r"[^\w\s-]", "", title.lower()).strip()

        # Query variants: realistic informational query
        first_sentence = row.abstract.split(".")[0] if "." in row.abstract else row.abstract
        first_sentence_clean = clean_text(first_sentence)

        eval_data.append(
            {
                "query_id": f"q_{idx:04d}",
                "query": clean_title_query,
                "target_doc_id": paper_id,
                "title": title,
                "categories": categories,
                "abstract_snippet": first_sentence_clean[:200],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    logger.info(f"Saved {len(eval_data)} benchmark queries to {output_path}")
    return eval_data


def main():
    parser = argparse.ArgumentParser(description="Ingest and preprocess arXiv abstracts.")
    parser.add_argument("--sample-size", type=int, default=DATASET_SAMPLE_SIZE, help="Number of records to sample")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")
    args = parser.parse_args()

    df = download_arxiv_sample(sample_size=args.sample_size, force=args.force)
    create_evaluation_benchmark(df, force=args.force)
    logger.info("Data preprocessing and evaluation benchmark generation complete.")


if __name__ == "__main__":
    main()
