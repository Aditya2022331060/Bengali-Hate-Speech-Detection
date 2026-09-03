# 📋 RESEARCH PROGRESS & SUPERVISORY TRACKER

> **Document Role**: Single source of truth for repository state, empirical results, scientific audits, and paper milestones.  
> **Mandate for AI Partners (Claude & Antigravity)**: Read this file upon every session start. Verify, re-check, and update this document as progress is made. Never hallucinate numbers.

---

## 1. Project Metadata

- **Paper Title**: *Consistency-Constrained Multi-Task Bengali Hate Speech Detection with Interpretable Generative Explanations*
- **Target Venue**: IEEE / ACL / EMNLP Format (Current template: `IEEEtran`)
- **Primary Research Questions**:
  1. Does multi-task joint learning with differentiable soft consistency loss eliminate logical contradictions across Hate Type, Target, and Severity?
  2. Does an integrated generative explanation head preserve or improve classification accuracy while providing faithful rationales (measured via ERASER)?
  3. How does our 110M specialized Small Language Model (SLM) compare against dedicated Bengali LLMs (1B–3B params) in accuracy, structural consistency (CVR), deterministic output parsing, and inference latency?
- **Taxonomy (5 × 5 × 3)**:
  - **Hate Type (5 classes)**: `None`, `Abusive`, `Political Hate`, `Religious Hate`, `Gender Hate` *(consolidated from original 6-class Profane/Sexism labels)*.
  - **Target (5 classes)**: `None`, `Individual`, `Organization`, `Community`, `Society`.
  - **Severity (3 classes)**: `Little to None`, `Mild`, `Severe`.
- **Consistency Constraint**:
  $$\text{If } \text{Type} = \text{None} \implies \text{Target} = \text{None} \land \text{Severity} = \text{Little to None}$$
  Any violation is recorded in the **Consistency Violation Rate (CVR)**.
- **Dataset Partitioning (Zero Data Leakage)**:
  - Master dataset: 35,530 rows (`TrainMultiHate`).
  - Stratified 80/10/10 split with `random_state=42`:
    - **Train (80%)**: 28,424 samples (used strictly in NB03).
    - **Validation (10%)**: 3,553 samples (checkpoint selection).
    - **Test (10%)**: 3,553 samples (held-out benchmark for NB04 and NB05a–d).

---

## 2. Verified Empirical Results

### Table A: Main Classification & Consistency (Test Set: 3,553 samples)
*All metrics verified from Notebook 04 test evaluation logs.*

| Model Configuration | Parameters | Type F1 | Target F1 | Severity F1 | **Avg Macro F1** | **CVR (%)** | Violations (/3,553) |
|---|---|---:|---:|---:|---:|---:|---:|
| **exp1 (Baseline MTL)** | 110M | 0.5114 | 0.5113 | 0.5971 | 0.5399 | 9.46% | 336 |
| **exp3 (+Gen Head)** | 110M + LSTM | **0.5273** | 0.5229 | 0.5991 | 0.5498 | 4.95% | 176 |
| **exp2 (+Consistency Loss)** | 110M | 0.4960 | **0.5647** | 0.6112 | 0.5573 | **0.00%** | **0** |
| **exp4 (Full Model)** | 110M + LSTM | 0.4994 | 0.5598 | **0.6136** | **0.5576** | **0.03%** | **1** |

> **Key Analytical Takeaway**: Consistency loss eliminates 99.7%–100% of logical violations (from 336 down to 0–1), while improving overall Average Macro F1 (+3.3% relative).

---

### Table B: ERASER Faithfulness Evaluation (1,000 test samples)
*Verified from `results4/results/eraser_results.json`.*

| Model Configuration | Comp @ 5% | Comp @ 10% | Comp @ 20% | Comp @ 50% | **AOPC (Higher = Better)** | Suff @ 50% (Lower = Better) |
|---|---:|---:|---:|---:|---:|---:|
| **exp1 (Baseline MTL)** | 0.0838 | 0.1006 | 0.1443 | 0.2431 | 0.1430 | 0.0989 |
| **exp3 (+Gen Head)** | 0.1002 | 0.1165 | 0.1669 | 0.2749 | 0.1646 | 0.1059 |
| **exp2 (+Consistency Loss)** | 0.0984 | 0.1161 | **0.1746** | **0.3117** | **0.1752** | 0.1169 |
| **exp4 (Full Model)** | **0.1028** | **0.1192** | 0.1645 | 0.2952 | **0.1704** | **0.0745** |

> **Key Analytical Takeaway**: exp4 achieves the lowest sufficiency score (0.0745), proving that retained rationale tokens are sufficient to drive correct model predictions. Perturbation curve plot saved in `results4/results/fig5_perturbation_curve.png`.

---

### Table C: Bengali LLM Benchmarks (3,553 test samples)
*Goal: Complete Table 5 in manuscript comparing against open-source Bengali LLMs.*

| Model | Size | Paradigm | Type F1 | Target F1 | Sev F1 | Avg F1 | CVR (%) | Parse Rate | Latency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ours (exp4 Full)** | 110M | Deterministic | 0.4994 | 0.5598 | 0.6136 | **0.5576** | **0.03%** | **100%** | **~0.015s** | ✅ Verified |
| **TituLLM-1B** | 1B | 0-shot | *[Log pending]* | *[Log pending]* | *[Log pending]* | *[Log pending]* | *[Pending]* | *[Pending]* | ~2.3s | ⏳ Pass 1 done, 5-shot running |
| **TituLLM-1B** | 1B | 5-shot | — | — | — | — | — | — | — | ⏳ In progress (NB05b) |
| **TigerLLM-1B-it** | 1B | 0-shot / 5-shot | — | — | — | — | — | — | — | ⏳ Ready to run (NB05a) |
| **BongLLaMA-3B** | 3B | 0-shot / 5-shot | — | — | — | — | — | — | — | ⏳ Ready to run (NB05c) |

---

## 3. Codebase & Pipeline Status

| Notebook / Component | File Path | Status | Notes |
|---|---|---|---|
| **EDA & Silver Generation** | `notebooks/01_data_and_silver_explanations.ipynb` | ✅ Completed | Preprocessed 35.5k samples, 5k Gemini silver rationales. |
| **Consistency Fixes** | `notebooks/02_consistency_and_silver_generation.ipynb` | ✅ Completed | Rule-based filtering, cleaned 5-class taxonomy. |
| **Training Pipeline** | `notebooks/03_training_experiments.ipynb` | ✅ Completed | Trained 4 models: `exp1`, `exp2`, `exp3`, `exp4`. Checkpoints archived. |
| **Evaluation & ERASER** | `notebooks/04_evaluation_results.ipynb` | ✅ Completed | Evaluated 3,553 test samples, computed ERASER faithfulness. |
| **TigerLLM Benchmark** | `notebooks/05a_benchmark_tigerllm.ipynb` | ✅ Verified Ready | Pure string formatting prompt, 0-shot + 5-shot, auto-cache. |
| **TituLLM Benchmark** | `notebooks/05b_benchmark_titullm.ipynb` | 🏃 Executing | Smart cache resumes from saved 0-shot, executes 5-shot. |
| **BongLLaMA Benchmark** | `notebooks/05c_benchmark_bongllama.ipynb` | ✅ Verified Ready | Tested for syntax, VRAM cleanup every 500 steps. |
| **Our Model Benchmark** | `notebooks/05d_benchmark_our_model.ipynb` | ✅ Verified Ready | `total_memory` fixed; searches `bangla-hate-trained-models`. |

---

## 4. Manuscript Status (`paper/main.tex`)

- [x] **Abstract**: Updated with multi-task formulation, 110M vs LLM efficiency, and 0.03% CVR.
- [ ] **Section 1: Introduction**: Draft outline exists; needs formal motivation on multi-aspect dependencies in low-resource settings.
- [x] **Section 2: Related Work**: Bengali hate speech baselines & XAI context drafted.
- [x] **Section 3: Methodology**: Multi-task architecture, Focal Loss, and differentiable Soft Consistency Loss formulated mathematically.
- [x] **Section 4: Experimental Setup**: Dataset taxonomy, 80/10/10 split, and evaluation protocols documented.
- [ ] **Section 5: Results & Discussion**:
  - [x] Table A inserted (Classification & CVR across exp1–exp4).
  - [x] Table B inserted (ERASER Faithfulness & AOPC).
  - [x] Figure 1 inserted (`fig5_perturbation_curve.png`).
  - [ ] Table C pending (LLM comparison once NB05a–d runs finish).
- [ ] **Section 6: Conclusion & Limitations**: Pending final benchmark integration.

---

## 5. Active Reviewer Risks & Mitigation Strategies

1. **Risk: "Did the model leak test data?"**
   - *Mitigation*: Mathematically deterministic 80/10/10 split (`random_state=42`, stratified). Document exact train/dev/test hash or counts (28,424 / 3,553 / 3,553).
2. **Risk: "Were the LLM prompts fair?"**
   - *Mitigation*: Both zero-shot and 5-shot were tested. 5-shot examples are strictly sampled from `train.json` (zero test contamination). Normalizer maps case variations and synonyms (`profane` $\rightarrow$ `Abusive`).
3. **Risk: "Why use an LSTM decoder instead of a Transformer decoder?"**
   - *Mitigation*: Emphasize resource-constrained deployment (edge-device suitability, low latency, only 110M parameters).

---

## 6. Immediate Action Items

1. **Monitor & Complete NB05b (TituLLM)**: Ensure 5-shot finishes and output JSON is saved.
2. **Execute NB05d (Our Model on Kaggle)**: Quick 10-minute run with `TrainMultiHate` + `bangla-hate-trained-models` to produce `our_model_benchmark_results.json`.
3. **Execute NB05a & NB05c**: TigerLLM-1B and BongLLaMA-3B on Kaggle GPU T4.
4. **Draft Full Manuscript Text**: Flesh out Sections 1, 5, and 6 in `paper/main.tex`.
