"""Index building module for BM25 sparse and FAISS dense indices."""

import argparse
import logging
import pickle
import re
from typing import List

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

from src.config import (
    BM25_INDEX_PATH,
    BM25S_INDEX_DIR,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    RAW_DATA_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def tokenize_text(text: str) -> List[str]:
    """Tokenize and filter text for BM25 indexing."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def build_bm25_index(df: pd.DataFrame, output_path=BM25_INDEX_PATH) -> BM25Okapi:
    """Build and serialize BM25Okapi index on paper abstracts and titles."""
    logger.info(f"Tokenizing {len(df)} documents for BM25...")
    tokenized_corpus = [tokenize_text(text) for text in df["combined_text"]]

    logger.info("Initializing BM25Okapi model...")
    bm25 = BM25Okapi(tokenized_corpus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": df["id"].tolist()}, f)

    logger.info(f"BM25 index saved to {output_path}")
    return bm25


def build_bm25s_index(df: pd.DataFrame, output_dir=BM25S_INDEX_DIR):
    """Build and serialize fast mmap bm25s index if library is available."""
    try:
        import bm25s

        logger.info(f"Tokenizing {len(df)} documents with bm25s...")
        corpus = df["combined_text"].tolist()
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        logger.info("Building bm25s index...")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        output_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(output_dir))
        logger.info(f"bm25s index successfully saved to {output_dir}")
        return retriever
    except ImportError:
        logger.warning("bm25s not installed, skipping bm25s index build.")
        return None


def build_faiss_dense_index(
    df: pd.DataFrame,
    model_name: str = EMBEDDING_MODEL_NAME,
    index_path=FAISS_INDEX_PATH,
    docstore_path=DOCSTORE_PATH,
    batch_size: int = 128,
) -> faiss.Index:
    """Generate dense embeddings with sentence-transformers and build FAISS index."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading embedding model '{model_name}' on device '{device}'...")
    model = SentenceTransformer(model_name, device=device)

    texts = df["combined_text"].tolist()
    logger.info(f"Encoding {len(texts)} document representations (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    dimension = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP (dim={dimension})...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index ({index.ntotal} vectors) to {index_path}")

    # Save document store for rapid metadata retrieval
    df.to_parquet(docstore_path, index=False)
    logger.info(f"Saved docstore to {docstore_path}")

    return index


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS dense indices.")
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH), help="Path to raw parquet dataset")
    args = parser.parse_args()

    logger.info(f"Loading dataset from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    logger.info(f"Loaded {len(df)} records. Building indices...")

    build_bm25_index(df)
    build_bm25s_index(df)
    build_faiss_dense_index(df)
    logger.info("All indices successfully built and saved.")


if __name__ == "__main__":
    main()
