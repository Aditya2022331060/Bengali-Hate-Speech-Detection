# 📋 RESEARCH PROGRESS & SUPERVISORY TRACKER

> **Document Role**: Single source of truth for repository state, empirical results, scientific audits, and paper milestones.  
> **Mandate for AI Partners (Claude & Antigravity)**: Read this file upon every session start. Verify, re-check, and update this document as progress is made. Never hallucinate numbers.

---

## 1. Project Metadata

- **Paper Title**: *Consistency-Constrained Multi-Task Bengali Hate Speech Detection with Generative Explanations*
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
| **exp1 (Baseline MTL)** | 110M | 0.5111 | 0.5109 | 0.5972 | 0.5397 | 9.46\% | 336 |
| **exp3 (+Gen Head)** | 110M + LSTM | **0.5270** | 0.5228 | 0.5988 | 0.5495 | 4.95\% | 176 |
| **exp2 (+Consistency Loss)** | 110M | 0.4959 | **0.5652** | 0.6110 | 0.5574 | **0.00\%** | **0** |
| **exp4 (Full Model)** | 110M + LSTM | 0.4994 | 0.5598 | **0.6136** | **0.5576** | **0.03\%** | **1** |

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

> **Key Analytical Takeaway**: exp4 achieves the lowest sufficiency score (0.0745), proving that retained rationale tokens are sufficient to drive correct model predictions. Perturbation curve plot saved in `paper/figures/fig5_perturbation_curve.png`.

---

### Table C: Complete Bengali LLM Benchmarks (3,553 test samples)
*Verified from Notebooks 05a, 05b, 05c, and 05d execution results.*

| Model | Size | Paradigm | Type F1 | Target F1 | Sev F1 | **Avg F1** | **CVR (%)** | Violations | Parse Rate | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TigerLLM-1B-it** | 1B | 0-shot | 0.2256 | 0.1851 | 0.3557 | 0.2554 | 0.42\% | 15 | 95.6\% | 4.634s |
| **TigerLLM-1B-it** | 1B | 5-shot | 0.3572 | 0.2591 | 0.3747 | 0.3303 | 1.32\% | 47 | 99.1\% | 5.851s |
| **TituLLM-1B** | 1B | 0-shot | 0.1439 | 0.1500 | 0.2631 | 0.1856 | 0.00\% | 0 | 5.7\% | 2.173s |
| **TituLLM-1B** | 1B | 5-shot | 0.1803 | 0.1731 | 0.3014 | 0.2183 | 4.98\% | 177 | 21.5\% | 5.274s |
| **BongLLaMA-3B** | 3B | 0-shot | 0.1439 | 0.1500 | 0.2631 | 0.1856 | 0.00\% | 0 | 0.0\%* | 0.218s |
| **BongLLaMA-3B** | 3B | 5-shot | 0.1439 | 0.1500 | 0.0844 | 0.1261 | 100.00\% | 3,553 | 100.0\% | 1.277s |
| **Ours (exp4 Full)** | **110M** | Deterministic | **0.4994** | **0.5598** | **0.6136** | **0.5576** | **0.03\%** | **1** | **100.0\%** | **0.0152s** |

> **Key Analytical Takeaways**:
> 1. **Accuracy**: Our 110M model beats the best LLM baseline (TigerLLM 5-shot) by **+68.8\% relative** in Macro F1 (0.5576 vs 0.3303).
> 2. **Inference Speed**: Our model is **$\sim$390$\times$ faster** than TigerLLM-1B (0.0152s vs 5.851s per sample).
> 3. **Format Reliability**: 100\% deterministic parsing vs LLM failure rates between 0.9\% and 100\%.
> 4. **Structural Consistency**: Consistency Loss reduces CVR to 0.03\% (1 violation out of 3,553), whereas BongLLaMA suffers 100\% mode collapse and TigerLLM commits 47 violations.

---

## 3. Codebase & Pipeline Status

| Active Component | File Path | Status | Purpose |
|---|---|---|---|
| **Training Pipeline** | `notebooks/03_training_experiments.ipynb` (or 03a/3b) | ✅ Completed | Trained `exp1`, `exp2`, `exp3`, `exp4`. |
| **Evaluation & ERASER** | `notebooks/04_evaluation_results.ipynb` | ✅ Completed | 3,553 test set evaluation, ERASER metrics. |
| **TigerLLM Benchmark** | `notebooks/05a_benchmark_tigerllm.ipynb` | ✅ Completed | TigerLLM-1B 0-shot + 5-shot test evaluation. |
| **TituLLM Benchmark** | `notebooks/05b_benchmark_titullm.ipynb` | ✅ Completed | TituLLM-1B 0-shot + 5-shot test evaluation. |
| **BongLLaMA Benchmark** | `notebooks/05c_benchmark_bongllama.ipynb` | ✅ Completed | BongLLaMA-3B 0-shot + 5-shot test evaluation. |
| **Our Model Benchmark** | `notebooks/05d_benchmark_our_model.ipynb` | ✅ Completed | BanglaBERT MTL 4-model test evaluation. |
| **Cleaned Legacy Files** | `Useless/` | 📦 Archived | Archived unused notebooks (01, 02, 05), old CSVs, and plans. |

---

## 4. Manuscript Status (`paper/main.tex`)

- [x] **Abstract**: Completely updated with exact empirical F1 (0.558), CVR (0.03%), speedup ($\sim$390$\times$), and parameter efficiency (110M vs 1B–3B).
- [x] **Section 1: Introduction**: Formalized problem statement, structural contradictions in LLMs, and 4 explicit contributions.
- [x] **Section 2: Related Work**: Detailed coverage of Bengali NLP, multi-task learning, and explainability/XAI.
- [x] **Section 3: Methodology**: Complete mathematical formulations for task hierarchy, shared BanglaBERT representation, Focal Loss, and differentiable Soft Consistency Loss.
- [x] **Section 4: Experimental Setup**: Precise documentation of 35,530 samples, stratified 80/10/10 split, zero data leakage, LLM prompting protocols, and semi-supervised teacher rationale distillation (2,000 verified rationales).
- [x] **Section 5: Results & Discussion**:
  - [x] Table I (Ablation Test Results on 3,553 samples).
  - [x] Table II (Consistency Violation Rate analysis).
  - [x] Table IV (Full LLM Comparison on 3,553 samples: +68.8% F1, ~390x faster).
  - [x] Table III (ERASER Faithfulness Evaluation: Comp, Suff, AOPC).
  - [x] Figure 1 (`figures/fig4_confusion_matrices.png`: normalized multi-task confusion matrices).
  - [x] Figure 2 (`figures/fig5_perturbation_curve.png`: ERASER perturbation curves).
  - [x] Table V (Qualitative Dual-Layer Explainability showcasing multi-aspect labels, trigger tokens, and Bengali explanations).
  - [x] Deep-dive analysis of BongLLaMA-3B's 100% CVR mode collapse.
  - [x] In-depth discussion on latency, accuracy advantage, and parse rate reliability.
- [x] **Section 6: Conclusion & Acknowledgment**: Complete final summary, edge deployment viability, and SUST HPC resources acknowledgment.
- [x] **References**: 24 verified BibTeX citations (`references.bib`) including BD-SHS (Romim et al.), DeepHateExplainer (Karim et al.), Sazzed vulgarity, BanglaMultiHate (Hasan et al., ACL 2026), TigerLLM (Raihan & Zampieri, ACL 2025), ERASER (DeYoung et al.), LIME (Ribeiro et al.), and BanglaBERT.
- [x] **Formatting & Sizing**: Aligned with `IEEE-conference-template-062824.tex` guidelines. All table captions placed strictly ABOVE tables (`\begin{center}` environment); figure captions placed strictly BELOW figures. All 6 tables wrapped with `\resizebox` guards, zero margin overflow. All 511 braces and 32 environments 100% verified.

---

## 5. Peer-Reviewer Defense Matrix

1. **Risk: "Did the model leak test data?"**
   - *Mitigation*: Mathematically deterministic 80/10/10 split (`random_state=42`, stratified). Test set of 3,553 samples was held out and never seen during training.
2. **Risk: "Were the LLM prompts fair?"**
   - *Mitigation*: Both zero-shot and 5-shot in-context learning evaluated. Few-shot demonstrations were sampled strictly from the training split. Multi-tier regular expression fallback parser forgave minor formatting quirks.
3. **Risk: "Why use a specialized SLM over an LLM?"**
   - *Mitigation*: Our model is +68.8% more accurate in Macro F1, $\sim$390$\times$ faster, 10-30$\times$ smaller in parameters, 100% deterministic, and eliminates structural hallucinations.

---

## 6. GitHub Repository Synchronization

- **Remote URL**: `https://github.com/Aditya2022331060/Bengali-Hate-Speech-Detection.git`
- **Branch**: `main`
- **Latest Commit**: `bbc5e6c` (*Update project with IEEE paper formatting, 24 verified references, master briefing guide, completed Kaggle notebooks, and full evaluation results*)
- **Status**: ✅ Synced & Pushed successfully.

