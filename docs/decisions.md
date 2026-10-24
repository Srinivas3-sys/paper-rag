# Engineering Decisions & Notes

First-person notes on what I tried, what worked, what failed, and the measurements behind every choice in `paper-rag`.

---

## 1. Why Dense Retrieval Lost to BM25 (And What I Checked)

When I first evaluated `all-MiniLM-L6-v2` against BM25 on a held-out test split, dense retrieval underperformed BM25 across every single regime:
- Overall Strict MRR: Dense scored 0.454 vs BM25 0.624.
- Paraphrase Strict MRR: Dense scored 0.216 vs BM25 0.222.
- Acronym Strict MRR: Dense scored 0.231 vs BM25 0.660.

Dense should not lose to BM25 on paraphrases where keyword overlap is zero. Before rewriting the pipeline or changing models, I wrote a diagnostic test suite (`scripts/diagnostics/`) to test every failure hypothesis:

1. **Was it an indexing bug or row shuffle?**
   - I ran a self-retrieval test (`scripts/diagnostics/self_retrieval.py`): I embedded 500 random documents from the corpus and queried the FAISS index with each document's own abstract and full title+abstract.
   - Result: Self-retrieval achieved **100.0% Rank-1 Recall** (500/500, mean rank 1.00).
   - I reconstructed all 20,000 vectors directly from the FAISS `IndexFlatIP` and compared them to freshly encoded embeddings (`scripts/diagnostics/index_alignment.py`): mean cosine similarity was `1.000000` with max difference `1.19e-7` (machine epsilon). The FAISS index and docstore were 100% aligned.
2. **Was it sequence truncation?**
   - `all-MiniLM-L6-v2` has a hard context window of 256 tokens.
   - I tokenized all 20,000 corpus abstracts (`scripts/diagnostics/length_truncation.py`): **23.86% (4,771 papers) exceed 256 tokens** and had their trailing technical methodology silently chopped off during embedding.
   - In comparison, 512-token models (`BAAI/bge-small-en-v1.5`) reduce truncation to 0.38% (76 papers).
3. **Was it query specificity?**
   - I inspected the generated paraphrase queries (`scripts/diagnostics/query_specificity.py`). When title words were forbidden, Flan-T5 frequently collapsed into generic prompt parroting phrases like `"What is the main problem?"` or `"What is the core problem of this paper?"`.
   - On those generic queries, the target document was ranked at position ~19,900 because thousands of papers match a query with zero domain vocabulary. Filtering out degenerate boilerplate regexes and prompting with technical methodology snippets fixed this.
4. **Was it single-target ground truth?**
   - In a 20,000-paper arXiv corpus from the same two-month period (April-May 2007), there are often dozens of preprints covering the exact same subject (e.g., graphene quantum Hall effect, AdS/CFT dipoles, dark matter halo profiles).
   - Dense retrieval retrieved highly relevant preprints in the exact same subject at rank 1 and 2, but received a strict score of 0 because ground truth only listed the single paper the query was generated from.
   - When I implemented a multi-relevant lenient metric (crediting retrieved documents sharing the target's arXiv category with dense cosine similarity $\ge 0.30$), **Dense Paraphrase MRR jumped from 0.216 to 0.526**, beating BM25's 0.398 by **+32.2%**.

---

## 2. Comparing Dense Encoders on DEV

On the 150-query DEV split, I compared three sentence embedding models (`scripts/compare_dense_encoders.py`):
1. `all-MiniLM-L6-v2` (22M params, 256 tokens): DEV MRR = 0.4273 (Paraphrase: 0.1210).
2. `BAAI/bge-small-en-v1.5` (33M params, 512 tokens): DEV MRR = 0.4964 (Paraphrase: 0.1702).
3. `intfloat/e5-small-v2` (33M params, 512 tokens, with `query: ` and `passage: ` prefixes): DEV MRR = **0.5463** (Paraphrase: **0.2064**, Acronym: 0.4325).

**What I found**:
- Moving to E5-small-v2 improved paraphrase MRR by +70.6% over MiniLM (0.2064 vs 0.1210).
- However, standalone dense (0.2064) **still did not beat standalone BM25 (0.2284)** on DEV paraphrases under strict single-target ground truth.
- On exact technical acronyms (ADMM, MIMO, QEC, GAN), BM25 dominated (DEV MRR = 0.7055 vs E5 0.4325 and MiniLM 0.2060) because sparse inverted indexes match rare character tokens exactly, whereas dense encoders blur rare acronyms into broad contextual representations.

---

## 3. Why BM25 is the Recommended Default Retriever

Across the 300 held-out test queries, the results are:
- **BM25 Alone**: MRR = **0.624** [95% CI: 0.574, 0.675], Recall@10 = 0.747, Mean Latency = **3.7 ms**.
- **Hybrid RRF ($k=10, \alpha=0.3$)**: MRR = **0.623** [95% CI: 0.572, 0.672], Recall@10 = 0.757, Mean Latency = **19.7 ms**.
- **Paired Wilcoxon Signed-Rank Test**: $W = 8721.0, p = 0.9221$, mean difference = -0.0012.

**Decision**:
BM25 alone is statistically indistinguishable from Hybrid RRF on overall MRR ($p = 0.9221$), while evaluating 5.3x faster (3.7 ms vs 19.7 ms), consuming zero GPU memory, and requiring no offline vector embeddings. Therefore, **BM25 is the recommended default retriever for throughput-sensitive serving**.

Hybrid RRF is only recommended when serving semantic natural language queries where keyword overlap is known to be low:
- On Paraphrase queries ($n=100$), Hybrid significantly outperforms BM25 (Strict MRR 0.250 vs 0.222, $W = 791.0, p = 0.0054$).
- On Acronym queries ($n=100$), BM25 significantly outperforms Hybrid (Strict MRR 0.660 vs 0.629, $W = 852.5, p = 0.0049$).

---

## 4. Why Threadpool Parallelism Failed for Fast Retrievers

In `src/retrieval.py`, I initially implemented concurrent execution using Python's `ThreadPoolExecutor(max_workers=2)`:
```python
future_dense = executor.submit(dense_retriever.retrieve, query, k)
future_bm25 = executor.submit(bm25_retriever.retrieve, query, k)
```

**Measured Latency (100 timed queries)**:
- Sequential Hybrid (Dense then BM25): **14.81 ms** (p50) / 16.52 ms (mean)
- Parallel Hybrid (`ThreadPoolExecutor(2)`): **16.15 ms** (p50) / 19.01 ms (mean)

**Why it failed**:
`bm25s` completes in ~1.0 ms and FAISS completes in ~2.5 ms. When tasks take 1-3 ms, the overhead of Python thread dispatch, OS context switches, and GIL acquisition contention costs ~1.5-2.5 ms. Parallelism only pays off when sub-tasks take >20 ms (e.g. network I/O or multi-core tensor operations).

---

## 5. Replacing `rank-bm25` with `bm25s`

- **Problem**: `rank-bm25` is a pure-Python library that computes BM25 scores sequentially in Python loops over token lists. On 20,000 papers, it averaged 22.4 ms per query and saved to a 16 MB pickle file.
- **Solution**: Migrated to `bm25s`, which builds memory-mapped Compressed Sparse Column (CSC) matrices using Scipy.
- **Result**: Query latency dropped from 22.4 ms to **1.03 ms** (**21.8x speedup**), and index size on disk dropped from 16 MB to **12 MB**, with identical top-k retrieved documents.

---

## 6. Cross-Encoder Reranking: Latency vs Quality

I integrated `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the top 30 hybrid candidates:
- **Quality Gain (Test Split, N=300)**:
  - Overall Strict MRR: 0.622 -> **0.661** (+0.039)
  - Paraphrase Strict MRR: 0.245 -> **0.355** (+0.110 gain!)
  - Recall@1: 0.547 -> **0.600** (+5.3% absolute)
- **Latency Cost**: Adds **25.05 ms** per query on an RTX 3050 GPU (and >150 ms on CPU).
- **Decision**: Made reranking an opt-in flag (`rerank=True`) in `/search` and `/ask`. It is enabled when answer accuracy is critical, but disabled by default for low-latency interactive search.

---

## 7. Generation, Abstention, and NLI Integrity

1. **Prompt Truncation Fix**:
   - The original generator concatenated context abstracts first and placed the question at the end. When combined abstracts exceeded 512 tokens, the question was silently truncated.
   - I rewrote prompt formatting to place instructions and the user question at the top, and added a token budget manager that drops lowest-ranked abstracts rather than chopping the question.
2. **Abstain Gate on Unanswerable Queries**:
   - Off-topic questions (e.g. French sourdough recipes, Australian capitals) and fictional physics papers hallucinate if forced into generation.
   - I tuned an evidence threshold ($\tau^* = 7.9662$ on BM25 scores) strictly on the DEV split ($N=20$: 12 answerable, 8 unanswerable), achieving **0.958 balanced accuracy**.
   - On the held-out TEST split ($N=40$: 28 answerable, 12 unanswerable), the abstain gate achieved a **91.7% refusal rate** [95% CI: 75.0%, 100.0%], refusing 11 of 12 unanswerable queries before invoking the LLM.
3. **NLI Citation Verification**:
   - Rather than faking citations by attaching document `[1]` automatically, the pipeline strictly parses explicit `[n]` bracket citations and verifies them against cited abstracts using `cross-encoder/nli-deberta-v3-base`.
   - On Flan-T5-base, explicit bracketed citations occurred on 2 of 28 test questions (50.0% precision). On instruction-tuned Qwen2.5-0.5B, citation precision was 100.0%.
   - In a 20-sample hand-check audit comparing DeBERTa NLI predictions to human labels, agreement was **35.0% (7/20)**. The discrepancy stems from DeBERTa expecting full declarative sentences: when small models output short noun phrases, DeBERTa flags them as `neutral` due to missing verbs, whereas a human reader recognizes valid topical grounding.
