# Technical Decisions & Empirical Findings

Engineering notes on architecture choices, parameter tuning, and empirical measurements across the paper-rag system.

---

## 1. Sparse Retrieval: `bm25s` vs `rank-bm25`

- **Context**: The initial prototype used `rank-bm25`, a pure-Python inverted index that evaluates token postings sequentially in Python loops.
- **Measurement**: On the 20,000 document corpus, `rank-bm25` averaged 22.41 ms per query and serialized to a 16 MB pickle file. In contrast, `bm25s` (using memory-mapped compressed sparse column matrices) averaged 1.03 ms per query with a 12 MB index.
- **Decision**: Adopted `bm25s` as the default sparse backend with a transparent fallback to `rank-bm25` if `bm25s` is not installed or when evaluating legacy index files.
- **Outcome**: **21.8x reduction in sparse query latency** (22.4 ms -> 1.0 ms) with identical retrieved document rankings.

---

## 2. Dense Semantic Search: Vector Normalization and Sequence Truncation

- **Index Selection**: Used FAISS `IndexFlatIP` (exact inner product search). All query and document embeddings are $L_2$-normalized upon generation, making inner product algebraically equivalent to cosine similarity:
  $$\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \tilde{u} \cdot \tilde{v}$$
  On 20,000 vectors with 384 dimensions, `IndexFlatIP` search takes 2.06 ms (p50) on CPU/GPU without quantization loss or recall degradation.
- **Sequence Length Truncation**:
  - `sentence-transformers/all-MiniLM-L6-v2` has a hard sequence limit of 256 tokens.
  - Profiled all 20,000 abstracts in the corpus: **23.86% (4,771 papers) exceed 256 tokens** and have their trailing text truncated during indexing.
  - Tested `BAAI/bge-small-en-v1.5` (512 token context): Truncation drops to **0.38% (76 papers)**. However, BGE requires ~137s to encode 20k abstracts on an RTX 3050 vs ~18s for MiniLM (~7.6x longer forward pass), and single-query encoding latency increases from 3.8 ms to 9.2 ms.
  - **Decision**: Retained `all-MiniLM-L6-v2` for low latency, while documenting the 23.86% truncation rate honestly.

---

## 3. Hybrid Reciprocal Rank Fusion & Parallelization

- **RRF Parameter Tuning**: Hyperparameters $k \in \{10, 30, 60, 100\}$ and weight $\alpha \in \{0.3, 0.5, 0.7\}$ were ablated strictly on the DEV split (120 queries):
  $$\text{RRF}(d) = \frac{\alpha}{k + \text{rank}_{\text{dense}}(d)} + \frac{1 - \alpha}{k + \text{rank}_{\text{bm25}}(d)}$$
  - Default settings ($k=60, \alpha=0.6$) yielded Dev MRR = 0.5898.
  - Optimal configuration ($k=10, \alpha=0.3$) achieved Dev MRR = 0.6033. Lower $k$ sharpens the reward for rank-1 candidates, and $\alpha=0.3$ appropriately favors the stronger lexical signal on academic preprints.
- **Parallel Execution Finding**:
  - Implemented `ThreadPoolExecutor(max_workers=2)` to run dense FAISS search and BM25 search concurrently.
  - Measured latency: Sequential hybrid took 14.07 ms p50; Parallel hybrid took 15.32 ms p50.
  - **Finding**: Because `bm25s` takes only ~1 ms and FAISS takes ~2 ms, thread dispatch, Python GIL interaction, and context switching overhead exceed the parallel execution gain. Threadpool parallelism only benefits slower backends (e.g. remote vector databases or slow multi-second dense models).

---

## 4. Cross-Encoder Reranking

- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Pipeline**: Top 30 candidates retrieved by hybrid RRF are paired with the query as `(query, document_text)` and scored jointly through cross-attention.
- **Empirical Impact (Held-Out Test Set, N=222)**:
  - **Recall@1**: 0.545 -> **0.599** (+5.4% absolute)
  - **MRR**: 0.617 -> **0.656** (+0.038 absolute)
  - **nDCG@10**: 0.645 -> **0.682** (+0.037 absolute)
  - **Latency Cost**: +21.49 ms p50 on RTX 3050 GPU (30 candidate pairs). Total pipeline latency: 37.26 ms p50.
- **Decision**: Made cross-encoder reranking an optional query parameter (`rerank=true`) in the API, allowing clients to trade 21 ms of compute for a +5.4% boost in top-1 accuracy.

---

## 5. Generator Architecture & Hallucination Mitigation

- **Prompt Construction**: Placed instructions and question at the top of the prompt. Implemented token budgeting across retrieved abstracts, reserving headroom for system prompts and dropping lowest-ranked abstracts rather than slicing blindly at 512 tokens.
- **Strict Temperature Handling**: When `temperature <= 0.0`, the generator forces `do_sample=False` (deterministic greedy decoding). When `temperature > 0.0`, stochastic sampling is enabled with top-p=0.9.
- **Abstain Mechanism**: If top retrieval score is below threshold $\tau^* = 0.0160$ (tuned on DEV split for 93.8% balanced accuracy), the engine immediately returns `"not enough evidence in the indexed papers"` without calling the LLM.
- **Citation Verification**: Eliminated legacy heuristic fallbacks that automatically appended document [1] when citations were omitted. Citations are verified using `cross-encoder/nli-deberta-v3-base`.
