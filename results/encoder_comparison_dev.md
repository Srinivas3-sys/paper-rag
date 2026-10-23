# Dense Encoder Comparison on Dev Split (N=150)

Empirical comparison of dense embedding architectures evaluated on the DEV split across 3 regimes (50 Lexical, 50 Paraphrase, 50 Acronym).

| Dense Architecture | Parameters | Max Seq Len | Overall MRR | Lexical MRR | Paraphrase MRR | Acronym MRR | Overall Recall@10 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **all-MiniLM-L6-v2** | 22M | 256 | **0.4273** | 0.9550 | **0.1210** | 0.2060 | 0.5200 |
| **BAAI/bge-small-en-v1.5** | 33M | 512 | **0.4964** | 0.9900 | **0.1702** | 0.3291 | 0.5867 |
| **intfloat/e5-small-v2** | 33M | 512 | **0.5463** | 1.0000 | **0.2064** | 0.4325 | 0.6333 |
