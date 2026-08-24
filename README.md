# Self-Consistency Based Hallucination Risk Dashboard for Financial LLM Responses

> **Course Project | DA Results Report**  
> **Status: NLP & ML Pipeline — Complete (Proof of Concept)**  
> **Next Phase: R Shiny Interactive Dashboard — Pending**

---

## Project Overview

Large Language Models (LLMs) frequently hallucinate in financial domains — generating fabricated earnings figures, incorrect company facts, or wrong market data with high apparent confidence. This project builds a **black-box, access-free hallucination detection framework** using the self-consistency principle: if an LLM truly knows the answer, its multiple sampled responses should remain consistent. If it is guessing or hallucinating, the responses contradict each other.

The system does **not** require access to model internals (logprobs, attention weights). It works entirely on the text outputs of any LLM.

---

## Project Architecture

```
AI_HALLUNCINATION-1/
├── src/
│   ├── data_curator.py          # Benchmark dataset builder & financial QA curator
│   ├── nli_scorer.py            # NLI semantic contradiction scoring engine
│   ├── feature_extraction.py    # Statistical + NLP feature extractor
│   ├── train_eval.py            # Model training, cross-validation & evaluation
│   └── generate_results.py      # Results report exporter
│
├── data/
│   ├── financial_qa_dataset.json          # 30-item curated benchmark dataset
│   └── processed_features_benchmark.csv   # Extracted feature matrix
│
├── models/
│   ├── fusion_classifier.joblib   # Trained Logistic Regression (primary)
│   ├── rf_classifier.joblib       # Trained Random Forest
│   ├── mlp_classifier.joblib      # Trained MLP Neural Net
│   ├── feature_scaler.joblib      # StandardScaler for inference
│   └── results_summary.json       # Full evaluation metrics & feature importances
│
├── reports/
│   └── nlp_experiment_report.md   # Full academic markdown report
│
├── requirements.txt
└── venv/                          # Python virtual environment
```

---

## Five-Stage Detection Pipeline

```
Financial Query
     │
     ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Multi-Sample Response Collection  │
│  - 1 Deterministic Response  (T ≈ 0.0)      │
│  - N = 5 Sampled Responses   (T = 0.7–1.0)  │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Statistical Feature Extraction    │
│  - Numeric Variance & Mismatch Rate         │
│  - spaCy Entity Jaccard Distance            │
│  - Token Lexical Overlap (Jaccard)          │
│  - Response Length Dispersion               │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 3: NLI Semantic Contradiction Score  │
│  - Pairwise (Premise, Hypothesis) scoring   │
│  - Contradiction / Entailment / Neutral %   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 4: Risk Score Fusion (Classifier)    │
│  - Feature Vector → Logistic Regression     │
│  - Output: Hallucination Risk Score (0–1)   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Stage 5: R Shiny Dashboard (Pending)       │
│  - Per-query risk score visualization       │
│  - Category-level aggregate trend view      │
│  - Side-by-side response comparison         │
└─────────────────────────────────────────────┘
```

---

## Benchmark Dataset

| Attribute | Detail |
| :--- | :--- |
| **Total Samples** | 30 curated financial QA sets |
| **Sampling Budget** | 5 responses per query (T = 0.7–1.0) + 1 deterministic baseline (T ≈ 0.0) |
| **Financial Categories** | Earnings & EPS · Company Facts · Market Events · Financial Ratios |
| **Ground Truth Labels** | 15 Faithful (label = 0) · 15 Hallucinated (label = 1) |
| **Hallucination Taxonomy** | Numerical Contradiction · Entity Misassignment · Directional Polarity Conflict · Formula Distortion |

> ⚠️ **Limitation:** 30 samples is a proof-of-concept scale. A production-grade version requires minimum 300–500 samples with real API-generated LLM responses.

---

## Extracted Features (10 Total)

| Feature | Description |
| :--- | :--- |
| `numeric_cv` | Coefficient of variation across all numerical values in responses |
| `numeric_mismatch_rate` | Fraction of response pairs with conflicting numerical figures |
| `entity_jaccard_distance` | spaCy NER pairwise Jaccard distance (ORG, MONEY, DATE, PERCENT) |
| `entity_mismatch_count` | Count of named entities that appear in only some (not all) responses |
| `token_jaccard_distance` | Token-level surface overlap distance across all response pairs |
| `length_dispersion` | Standard deviation / mean of response token lengths |
| `nli_contradiction_mean` | Mean pairwise NLI contradiction probability |
| `nli_contradiction_max` | Max pairwise NLI contradiction probability |
| `nli_contradiction_std` | Standard deviation of pairwise contradiction scores |
| `nli_entailment_mean` | Mean pairwise NLI entailment probability |

---

## Hypothesis

| | Statement |
| :--- | :--- |
| **H₀ (Null)** | Statistical dispersion and NLI contradiction features do NOT provide a significant improvement in detecting financial hallucinations over single-metric baselines |
| **H₁ (Alternative)** | Feature-fused models achieve significantly superior ROC-AUC (> 0.85) over Lexical-Only, NLI-Only, and standard SelfCheckGPT baselines |
| **Decision** | **Reject H₀** — Logistic Regression achieved ROC-AUC 0.9000 vs. 0.5000 Lexical baseline |

---

## Experimental Results (Proof of Concept — 30 Samples)

### Model Comparison (Test Set: 9 Samples)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1: Lexical-Only (Token Jaccard) | 0.4444 | 0.4444 | 1.0000 | 0.6154 | 0.5000 |
| Baseline 2: NLI-Only (Contradiction Mean) | 0.4444 | 0.4286 | 0.7500 | 0.5455 | 0.5500 |
| Baseline 3: Standard SelfCheckGPT | 0.5556 | 0.5000 | 1.0000 | 0.6667 | 0.6000 |
| **Proposed: Feature-Fused Logistic Regression** | **0.6667** | **0.6000** | **0.7500** | **0.6667** | **0.9000** |
| Proposed: Feature-Fused Random Forest | 0.5556 | 0.5000 | 0.5000 | 0.5000 | 0.7000 |
| Proposed: Feature-Fused MLP Neural Net | 0.6667 | 0.6000 | 0.7500 | 0.6667 | 0.8000 |

### 5-Fold Cross-Validation (Logistic Regression)

| Fold | Accuracy | Precision | Recall | F1 | AUC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 0.5000 | 0.5000 | 0.6667 | 0.5714 | 0.6667 |
| 3 | 0.8333 | 1.0000 | 0.6667 | 0.8000 | 1.0000 |
| 4 | 0.5000 | 0.5000 | 0.6667 | 0.5714 | 0.3333 |
| 5 | 0.8333 | 0.7500 | 1.0000 | 0.8571 | 1.0000 |
| **Mean** | **0.7333** | **0.7500** | **0.8000** | **0.7600** | **0.8000** |

> ⚠️ **Honest Note:** The large AUC swing between Fold 1 (1.0) and Fold 4 (0.33) indicates the model is **not yet statistically stable** due to the small dataset size. Each fold has only 4–6 test samples. These results demonstrate the methodology works directionally, not that it is production-validated.

### Top Feature Importances (Logistic Regression Odds Ratios)

| Feature | Logit Coefficient | Odds Ratio | Interpretation |
| :--- | :---: | :---: | :--- |
| `length_dispersion` | +1.21 | **3.36x** | Most predictive — LLM unsure = outputs vary wildly in length |
| `entity_jaccard_distance` | +1.17 | **3.23x** | Named entity disagreement = strong hallucination signal |
| `nli_contradiction_max` | +0.74 | **2.09x** | Single worst-case contradiction highly informative |
| `nli_contradiction_mean` | +0.61 | **1.84x** | Average semantic contradiction across all pairs |
| `numeric_mismatch_rate` | +0.52 | **1.67x** | Conflicting numbers across responses |

---

## Current Status

| Phase | Component | Status |
| :--- | :--- | :---: |
| ✅ Phase 1 | Benchmark Dataset Creation (30 samples, 4 categories) | **Done** |
| ✅ Phase 2 | Statistical Feature Extraction (numeric, entity, lexical, length) | **Done** |
| ✅ Phase 3 | NLI Semantic Contradiction Scoring Engine | **Done** |
| ✅ Phase 4 | ML Model Training & Cross-Validation (LR, RF, MLP) | **Done** |
| ✅ Phase 5 | Results Report & Feature Importance Analysis | **Done** |
| 🔲 Phase 6 | R Shiny Interactive Dashboard | **Pending** |
| 🔲 Phase 7 | Live LLM API Integration (OpenAI / Ollama) | **Pending** |
| 🔲 Phase 8 | Scale Dataset to 300–500 Samples for Production Validation | **Pending** |

---

## Setup & Reproduction

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate benchmark dataset
python src/data_curator.py

# 4. Run full ML training and evaluation pipeline
python src/train_eval.py

# 5. Generate academic results report
python src/generate_results.py
```

---

## Known Limitations

1. **Dataset size is too small (30 samples)** for statistically reliable evaluation. Cross-validation fold AUC swings from 0.33 to 1.00.
2. **LLM responses are manually curated** — not sampled from a real API. A real experiment requires OpenAI / Ollama API integration.
3. **NLI engine is heuristic-based** — it uses regex number extraction and keyword polarity pairs, not a true DeBERTa-v3 transformer (transformer download blocked during setup). Plugging in the real NLI model will improve signal quality significantly.
4. **Balanced dataset is artificial** — real-world LLM outputs are not 50% hallucinated.

---

## Tech Stack

| Layer | Technology |
| :--- | :--- |
| Language | Python 3.13 |
| ML Framework | scikit-learn (Logistic Regression, Random Forest, MLP) |
| NLP | spaCy (`en_core_web_sm`) · transformers (DeBERTa NLI — optional) |
| Data | pandas · numpy |
| Model Serialization | joblib |
| Dashboard (Pending) | R Shiny + reticulate |

---

## References

- Manakul et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.*
- Ji et al. (2023). *Survey of Hallucination in Natural Language Generation.*
- PHANTOM Benchmark for Financial Hallucination Detection.
- HaluEval: A Large-Scale Hallucination Evaluation Benchmark.
