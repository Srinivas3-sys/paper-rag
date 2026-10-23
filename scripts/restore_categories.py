"""Restore arXiv categories to data/arxiv_20000.parquet and indices/docstore.parquet."""

import time
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from src.data import clean_text, parse_arxiv_record
from src.config import RAW_DATA_PATH, DOCSTORE_PATH

def main():
    t0 = time.time()
    print("Loading current docstore...")
    df = pd.read_parquet(RAW_DATA_PATH)
    n = len(df)
    target_ids = df["id"].tolist()
    
    print(f"Streaming first {n} records from Hugging Face...")
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    
    id_to_cats = {}
    pbar = tqdm(total=n, desc="Streaming categories")
    
    for raw in ds:
        parsed = parse_arxiv_record(raw)
        if parsed:
            pid = parsed["id"]
            cats = clean_text(raw.get("categories", ""))
            id_to_cats[pid] = cats
            pbar.update(1)
            if len(id_to_cats) >= n:
                break
    pbar.close()
    
    # Update df categories
    new_cats = [id_to_cats.get(pid, "") for pid in target_ids]
    df["categories"] = new_cats
    
    non_empty = sum(1 for c in new_cats if c.strip())
    print(f"Categories restored: {non_empty}/{n} non-empty ({non_empty/n*100:.1f}%)")
    
    # Save back
    df.to_parquet(RAW_DATA_PATH, index=False)
    print(f"Saved updated parquet to {RAW_DATA_PATH}")
    
    df.to_parquet(DOCSTORE_PATH, index=False)
    print(f"Saved updated docstore to {DOCSTORE_PATH}")
    print(f"Finished in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
