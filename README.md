# 🧠 BanglaHateML — Bengali Hate Speech Detection with Consistency-Constrained Multi-Task Learning

> **Target Venue:** ICCIT 2026  
> **Task:** Multi-label Bengali hate speech classification with logically consistent label prediction and generative explanations.

---

## 📌 Project Overview

This project builds a lightweight, locally-runnable Bengali hate speech detection model that:

1. **Classifies** each comment on three dimensions simultaneously:
   - **Hate Type** — None, Abusive, Political Hate, Profane, Religious Hate, Sexism
   - **Target** — None, Individual, Organization, Community, Society
   - **Severity** — Little to None, Mild, Severe

2. **Enforces logical consistency** — A custom `ConsistencyPenaltyLoss` prevents contradictory predictions (e.g., "not hateful" but "targets an individual") using a soft differentiable penalty.

3. **Generates Bengali explanations** — An LSTM decoder trained on silver-standard explanations outputs a rationale for each prediction.

---

## 🏗️ Model Architecture

```
Input Comment (Bengali Text)
        │
   [BanglaBERT Encoder]  ← csebuetnlp/banglabert (110M params)
        │
   [CLS Token Embedding]
   ┌────┴────────────────────────────────┐
   │             │                      │
[Type Head]  [Target Head]  [Severity Head]
   │             │                      │
 6 classes    5 classes              3 classes
```

**Loss Function:**

$$\mathcal{L}_{total} = \mathcal{L}_{focal}^{type} + \mathcal{L}_{focal}^{target} + \mathcal{L}_{focal}^{severity} + \lambda \cdot \mathcal{L}_{consist}$$

The `ConsistencyPenaltyLoss` enforces:
- If `type = None` → `target` must be `None` AND `severity` must be `Little to None`
- If `severity = Severe` → `type` cannot be `None`

---

## 📊 Phase 1 Results (exp1 vs exp2)

| Metric | exp1: Baseline | exp2: + Consistency |
|--------|:--------------:|:-------------------:|
| Type F1 | 0.5237 | 0.5242 |
| Target F1 | 0.5430 | 0.5456 |
| Severity F1 | 0.6060 | 0.6071 |
| **Avg F1** | 0.5576 | **0.5589** |
| **CVR** | 1.91% | **0.00%** |

**CVR (Consistency Violation Rate):** Percentage of test predictions that are logically contradictory. Adding our `ConsistencyPenaltyLoss` drives this from 1.91% → **0.00%** while maintaining (marginally improving) F1 — this is the core paper contribution.

---

## 📂 Repository Structure

```
Bengali_hate_speech_detection/
├── notebooks/
│   ├── 01_data_preparation.ipynb       ← EDA, dataset analysis
│   ├── 02_silver_explanations.ipynb    ← Silver explanation generation (Kaggle GPU)
│   ├── 03_training_experiments.ipynb   ← Training: exp1 (baseline) + exp2 (consistency)
│   ├── 04_evaluation.ipynb             ← Detailed evaluation + ERASER faithfulness
│   └── 05_llm_benchmarks.ipynb         ← Compare vs Gemini/TigerLLM-1B
│
├── banglaHate/src/
│   ├── model.py                        ← ConsistencyConstrainedMTL architecture
│   ├── losses.py                       ← ConsistencyPenaltyLoss + FocalLoss
│   └── dataset.py                      ← BanglaHateDataset (multi-task labels + explanations)
│
├── paper/
│   └── main.tex                        ← ICCIT 2026 paper (LaTeX)
│
├── silver_explanations.json            ← 2,000 generated Bengali explanations for verification
├── verify_explanations.html            ← Human verification web tool (see below)
└── README.md                           ← This file
```

---

## 🚀 How to Run

### Step 1 — Run Training on Kaggle (Phase 1)

> **No API key needed. No local GPU required.**

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **+ New Notebook**
2. **File → Import Notebook** → Upload `notebooks/03_training_experiments.ipynb`
3. **Add Input** → Search and add `nahidgazi/trainmultihate`
4. **Accelerator → GPU T4 x2**
5. **Save Version → Save & Run All (Commit)**

The notebook will:
- Auto-detect and split the dataset (80/10/10 stratified)
- Run `exp1_baseline` (BanglaBERT + 3 heads)
- Run `exp2_consistency` (+ ConsistencyPenaltyLoss)
- Save model checkpoints and results to the Output tab (~8 hours)

**Download from Output tab:**
- `models/exp1_baseline_best.pt`
- `models/exp2_consistency_best.pt`
- `results/ablation_results.json`
- `results/test_results.json`

---

### Step 2 — Human Verification of Silver Explanations

The file `silver_explanations.json` contains **2,000 AI-generated Bengali explanations** for hate speech annotations. These are used to train the generative explanation head (exp3 + exp4).

**Each entry looks like:**
```json
{
  "original_idx": 1234,
  "comment": "এই সরকার দেশটাকে ধ্বংস করে দিচ্ছে",
  "type_of_hate": "Political Hate",
  "target_of_hate": "Organization",
  "severity_of_hate": "Severe",
  "silver_explanation": "মন্তব্যটিতে \"সরকার\" ও \"ধ্বংস\" শব্দ ব্যবহার করে রাজনৈতিক বিদ্বেষ প্রকাশ করা হয়েছে। এটি নির্দিষ্ট সংগঠনের বিরুদ্ধে তীব্র ও গুরুতর রাজনৈতিক ঘৃণামূলক বক্তব্য।",
  "source": "rule-contextual",
  "verified": false
}
```

#### How to Use `verify_explanations.html`

This is a browser-based tool for rapidly reviewing and verifying the generated explanations.

**Setup (one-time):**

1. Clone or pull this repository:
   ```bash
   git clone https://github.com/Aditya2022331060/Bengali_hate_speech_detection.git
   cd Bengali_hate_speech_detection
   ```

2. Start a local HTTP server in the repo folder:
   ```bash
   python -m http.server 8080
   ```

3. Open your browser and go to:
   ```
   http://localhost:8080/verify_explanations.html
   ```

> ⚠️ **Important:** You must use a local server (`python -m http.server`). Opening the HTML file directly via `file://` will not work because browsers block local JSON loading for security reasons.

**Verification Workflow:**

For each sample you will see:
- The original Bengali comment
- Its classification labels (Hate Type / Target / Severity)
- The AI-generated Bengali explanation (editable)

Use the keyboard shortcuts to review quickly:

| Key | Action |
|-----|--------|
| `A` | ✅ **Approve** — explanation is accurate and well-written |
| `E` | ✏️ **Save Edit** — edit the text first, then press E to save |
| `R` | ❌ **Reject** — explanation is wrong or irrelevant (excluded from training) |
| `← →` | Navigate between samples |
| `S` | 💾 Save progress to `silver_explanations_verified.json` |

**Filtering:**
Use the filter buttons at the top to focus on specific hate types or review status (Pending / Approved / Edited / Rejected).

**Saving:**
Press `S` at any time to download `silver_explanations_verified.json`. This file contains all approved + edited samples and is used in Phase 3 training.

---

## 📋 Full Pipeline

| Phase | What | Where | Status |
|-------|------|-------|--------|
| Phase 1 | Train exp1 + exp2 | Kaggle GPU | ✅ Done |
| Phase 2 | Generate + verify silver explanations | Local (this tool) | 🔄 In Progress |
| Phase 3 | Train exp3 + exp4 (generative head) | Kaggle GPU | ⏳ Pending |
| Phase 4 | LLM benchmarks + ERASER evaluation | Kaggle GPU | ⏳ Pending |
| Phase 5 | Write paper | Local | ⏳ Pending |

---

## 📦 Dataset

- **Source:** [nahidgazi/trainmultihate](https://www.kaggle.com/datasets/nahidgazi/trainmultihate)
- **Total Samples:** 35,522
- **Split:** 80% train / 10% dev / 10% test (stratified by hate type)
- **Labels:** Multi-task — (Hate Type × Target × Severity)

| Class | Count |
|-------|------:|
| None | 19,954 |
| Abusive | 8,212 |
| Political Hate | 4,227 |
| Profane | 2,331 |
| Religious Hate | 676 |
| Sexism | 122 |

---

## 📝 Citation

*Paper submitted to ICCIT 2026. Citation will be added upon acceptance.*

---

## 🔧 Requirements

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
```

**Python:** 3.10+  
**GPU:** Required for training (Kaggle T4 recommended)  
**Inference:** CPU-compatible

---

*Last Updated: August 2026*
