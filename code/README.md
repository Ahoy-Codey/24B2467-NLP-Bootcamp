# Code — SciFact evidence retrieval & claim verification

Everything is produced by one notebook: `nlp_bootcamp_scifact.ipynb` (seed 42).

## Exact commands to reproduce (Kaggle)
1. Create a Kaggle dataset from the bootcamp zip (it must contain `data/`, `check_format.py`, `results_template.json`) and attach it to the notebook.
2. Notebook settings: **Accelerator = GPU T4 x2**, **Internet = On**, Persistence off.
3. In the configuration cell set `NAME`, `ROLL_NO` (seed is fixed to 42).
4. **Run All** (≈35–50 min). Outputs land in `/kaggle/working/submission/` and `/kaggle/working/<rollno>_nlp_bootcamp.zip`.

Locally (any CUDA GPU): `pip install torch transformers scipy scikit-learn nltk pandas sentencepiece` then
`jupyter nbconvert --to notebook --execute nlp_bootcamp_scifact.ipynb --ExecutePreprocessor.timeout=-1`
with `DATA_ROOT=/path/to/folder/containing/data` exported.

## Where each task is in the notebook
| Task | Notebook section | Files |
|---|---|---|
| 1 | "Task 1a — Explore", "Task 1 — BM25" | t1_bm25.trec, val_bm25.trec |
| 2 | "Task 2 — Dense retrieval" | t2_general.trec, t2_scientific.trec, val_general.trec |
| 3 | "Task 3a — Hybrid", "Task 3b — Cross-encoder reranking" | t3_hybrid.trec, t3_rerank.trec |
| 4 | "Task 4 — Fine-tune the bi-encoder" | t4_random.trec, t4_hard.trec |
| 5 | "Task 5a/5b/5c" | t5_verification.csv, t5_val_verification.csv, t5_val_errors.csv |

## Models (all open-source encoders, each ≤ 400M parameters; no generative LLMs)
- BM25: own scipy implementation — Okapi BM25 (own scipy implementation), k1=1.2, b=0.3, idf=log(1+(N-df+0.5)/(df+0.5)); doc = title + abstract; lowercase \w+ tokens, sklearn English stop-words removed, Porter stemming=on; grid k1 x b x stemming (72 configs) tuned on val nDCG@10
- General bi-encoder: `BAAI/bge-large-en-v1.5` (CLS pooling, L2-normalised, query instruction)
- Biomedical bi-encoder: `ncbi/MedCPT-Query-Encoder` + `ncbi/MedCPT-Article-Encoder` (CLS pooling, dot product, [title, abstract] input)
- Hybrid: weighted score interpolation of per-claim z-normalised scores over the full corpus: 0.1*BM25 + 0.9*bge-large-en-v1.5 (weights grid-searched on val, also compared against weighted RRF)
- Reranker: `ncbi/MedCPT-Cross-Encoder` on top-10 of `hybrid` (chosen among ['BAAI/bge-reranker-base', 'ncbi/MedCPT-Cross-Encoder'] × k ∈ [10, 20, 30, 50, 100] on val)
- Fine-tuning: `BAAI/bge-large-en-v1.5`, InfoNCE (τ=0.02), batch 16, lr 2e-05, 4 epochs, best-val epoch kept (random: 2, hard: 2); hard negatives = top-30 of BM25/general/MedCPT minus gold
- Verification: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` — average of abstract-level finetuned NLI and rationale-sentence fine-tuned NLI (chosen among abstract / sentence / ensemble on val F1; sentence selector `ncbi/MedCPT-Cross-Encoder` fine-tuned on gold rationale ids, val rationale recall@3 = 0.918); retriever = hybrid weighted score interpolation of per-claim z-normalised scores over the full corpus: 0.2*BM25 + 0.1*MedCPT + 0.7*bge-large-en-v1.5-ft-hard (weights grid-searched on val, also compared against weighted RRF); K=2, τ=0.42

## Validation ablation (162 claims; metrics over the 106 claims with gold evidence)
| run file | system | nDCG@10 | R@100 | MRR |
|---|---|---|---|---|
| t1_bm25.trec | bm25 | 0.868 | 0.9811 | 0.8503 |
| t2_general.trec | general | 0.9328 | 1.0 | 0.9191 |
| t2_scientific.trec | scientific | 0.8509 | 1.0 | 0.8127 |
| t3_hybrid.trec | hybrid | 0.9388 | 1.0 | 0.9276 |
| t3_rerank.trec | rerank | 0.9149 | 1.0 | 0.8906 |
| t4_random.trec | ft_random | 0.9169 | 0.9906 | 0.9007 |
| t4_hard.trec | ft_hard | 0.9303 | 1.0 | 0.9108 |

Verification (val): P=0.7946 R=0.7177 F1=0.7542. Error categories: {'entity_mismatch': 8, 'numerical': 13, 'other': 10, 'retrieval_miss': 6, 'needs_multiple_docs': 10}.

## Efficiency (T4)
| model | index_time_sec | query_latency_ms | memory_mb | index_size_mb |
|---|---|---|---|---|
| general | 81.34 | 21.82 | 2271.9 | 20.2 |
| scientific | 24.16 | 11.47 | 6520.3 | 15.2 |

General-model candidates (val nDCG@10): {'BAAI/bge-large-en-v1.5': 0.9327816176022498, 'BAAI/bge-base-en-v1.5': 0.8976479834895273}

BM25 query latency 0.39 ms; cross-encoder rerank latency 72.2 ms/claim (top-10).
`memory_mb` = peak GPU memory allocated while loading the model(s) and encoding the corpus.
