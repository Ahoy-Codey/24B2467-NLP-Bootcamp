# NLP Selection Bootcamp — SciFact report

**Name:** Nihar  |  **Roll no:** 24b2467  |  **Seed:** 42  |  all analysis on validation claims

## Q1. Two claims where BM25 beats t2_general on nDCG@10, two where t2_general beats BM25 — why?

BM25 > general: claims 673, 34. Their gold abstracts share the claim's most distinctive (highest-IDF) terms exactly ('LDL', 'causal'; 'folate', 'homocysteine'); BM25's IDF weighting rewards these exact matches (gold rank 1/1 vs 5/2 for bge), while the embedding drifts toward topically similar abstracts. General > BM25: claims 766, 541. Only 40%/50% of claim terms appear in the gold abstract — it paraphrases the claim, so BM25 ranks it 67/7 while bge matches the meaning (rank 1/1).

## Q2. Which Task 3 component gave the largest validation gain? What did it cost in latency?

Hybrid fusion gave the largest gain: 0.933 (bge-large-en-v1.5 alone) → 0.939 (+0.006). Its cost is small: ≈34 ms/claim (BM25 0.4 ms + two dense encoders; fusion itself is negligible). Cross-encoder reranking of the top-10 changed nDCG@10 by -0.024 while adding ≈72 ms/claim (≈2× the first stage), so on this data the reranker was not worth its cost.

## Q3. Did hard negatives help? Did you see overfitting, and how did you detect it?

Partly. Hard negatives beat random negatives (val nDCG@10 0.930 vs 0.917), but neither beat pretrained bge-large (0.933): it is already strong on this domain and 440 training pairs are few. I evaluated train and val nDCG@10 after every epoch: train rose to 0.986 (hard) / 0.970 (random) while val peaked at epoch 2 and then fell — clear over-fitting, detected by the widening train–val gap. I kept the best-val epoch.

## Q4. What fraction of your validation verification errors are retrieval_miss?

6 of 47 validation claims with errors are retrieval_miss (13%): no gold abstract reached the top-2 given to the verifier. Other categories: numerical 13, other 10, needs_multiple_docs 10, entity_mismatch 8. Pipeline: hybrid of BM25 + MedCPT + fine-tuned bge, then an ensemble of abstract-level and rationale-sentence DeBERTa-v3 NLI (val F1 0.754, P 0.795, R 0.718). Retrieval is no longer the bottleneck; verification is.

## Q5. What is the single change you would try next, and why?

Add contrastive training claims: perturb training claims by swapping numbers/quantities and flipping direction words (increase↔decrease, higher↔lower), paired with their gold rationales as CONTRADICT examples. Numerical errors are the largest category (13 of 47), and the training set has few minimal pairs forcing the verifier to check exact quantities and polarity. This targets that failure directly, needs no new data, and stays within the 400M model limit.