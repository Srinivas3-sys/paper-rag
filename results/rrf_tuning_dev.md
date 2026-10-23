# RRF Hyperparameter Grid Ablation on DEV Split (N=150)

Evaluation strictly on the 150 DEV queries (50 Lexical, 50 Paraphrase, 50 Acronym).
Dense weight fixed at 1.0; BM25 weight tuned over {1.0, 1.5, 2.0, 3.0}; k tuned over {10, 30, 60}.

| Rank | Dense Encoder | k | BM25 Weight | Overall MRR | Lexical MRR | Paraphrase MRR | Acronym MRR | Overall Recall@10 |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | intfloat/e5-small-v2 | 10 | 3.0 | **0.6451** | 0.9700 | 0.2281 | 0.7372 | 0.7400 |
| 2 | intfloat/e5-small-v2 | 10 | 2.0 | **0.6363** | 0.9700 | 0.2367 | 0.7023 | 0.7467 |
| 3 | intfloat/e5-small-v2 | 30 | 3.0 | **0.6347** | 0.9700 | 0.2427 | 0.6915 | 0.7467 |
| 4 | intfloat/e5-small-v2 | 10 | 1.0 | **0.6304** | 1.0000 | 0.2640 | 0.6271 | 0.7333 |
| 5 | intfloat/e5-small-v2 | 10 | 1.5 | **0.6298** | 0.9800 | 0.2410 | 0.6684 | 0.7600 |
| 6 | intfloat/e5-small-v2 | 30 | 1.0 | **0.6230** | 1.0000 | 0.2575 | 0.6115 | 0.7400 |
| 7 | all-MiniLM-L6-v2 | 10 | 3.0 | **0.6230** | 0.9650 | 0.2403 | 0.6636 | 0.7333 |
| 8 | all-MiniLM-L6-v2 | 10 | 2.0 | **0.6229** | 0.9550 | 0.2458 | 0.6680 | 0.7467 |
| 9 | intfloat/e5-small-v2 | 30 | 2.0 | **0.6228** | 0.9700 | 0.2576 | 0.6409 | 0.7400 |
| 10 | all-MiniLM-L6-v2 | 10 | 1.5 | **0.6225** | 0.9567 | 0.2469 | 0.6640 | 0.7600 |
| 11 | intfloat/e5-small-v2 | 60 | 1.0 | **0.6219** | 1.0000 | 0.2565 | 0.6093 | 0.7400 |
| 12 | intfloat/e5-small-v2 | 30 | 1.5 | **0.6212** | 0.9900 | 0.2561 | 0.6174 | 0.7400 |
| 13 | intfloat/e5-small-v2 | 60 | 3.0 | **0.6172** | 0.9700 | 0.2506 | 0.6310 | 0.7400 |
| 14 | intfloat/e5-small-v2 | 60 | 1.5 | **0.6169** | 0.9900 | 0.2451 | 0.6157 | 0.7333 |
| 15 | all-MiniLM-L6-v2 | 30 | 3.0 | **0.6150** | 0.9650 | 0.2312 | 0.6489 | 0.7467 |
| 16 | intfloat/e5-small-v2 | 60 | 2.0 | **0.6106** | 0.9700 | 0.2447 | 0.6170 | 0.7333 |
| 17 | all-MiniLM-L6-v2 | 30 | 2.0 | **0.6069** | 0.9550 | 0.2320 | 0.6337 | 0.7400 |
| 18 | all-MiniLM-L6-v2 | 60 | 3.0 | **0.6068** | 0.9650 | 0.2261 | 0.6295 | 0.7400 |
| 19 | all-MiniLM-L6-v2 | 30 | 1.5 | **0.5903** | 0.9567 | 0.2248 | 0.5895 | 0.7400 |
| 20 | all-MiniLM-L6-v2 | 60 | 2.0 | **0.5824** | 0.9550 | 0.2338 | 0.5585 | 0.7467 |
| 21 | all-MiniLM-L6-v2 | 60 | 1.5 | **0.5795** | 0.9567 | 0.2342 | 0.5476 | 0.7467 |
| 22 | all-MiniLM-L6-v2 | 10 | 1.0 | **0.5630** | 0.9700 | 0.2193 | 0.4997 | 0.7533 |
| 23 | all-MiniLM-L6-v2 | 60 | 1.0 | **0.5517** | 0.9667 | 0.2302 | 0.4581 | 0.7533 |
| 24 | all-MiniLM-L6-v2 | 30 | 1.0 | **0.5474** | 0.9667 | 0.2177 | 0.4580 | 0.7533 |

## Selected Configuration on Dev
- **Best Dense Encoder**: `intfloat/e5-small-v2`
- **Best RRF k**: `10`
- **Best BM25 Weight**: `3.0` (Dense Weight: 1.0)
- **Dev Overall MRR**: `0.6451` (vs BM25 DEV MRR: 0.6321)
- **Dev Paraphrase MRR**: `0.2281` (vs BM25 DEV MRR: 0.2284)
