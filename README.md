# 🇧🇩 Consistency-Constrained Multi-Task Bengali Hate Speech Detection with Faithfulness-Evaluated Generative Explanations

> **Target Venue**: ICCIT 2026 (Primary) / IEEE Access / EMNLP 2027 BLP Workshop  
> **Authors**: Angkon Roy & Nahid Gazi (SUST CSE)

---

## 📌 Executive Summary

### What We Are Trying to Achieve
Hate speech detection is often treated as a simple classification task. However, hate speech has nuance: it has a **Type**, a **Target**, and a **Severity**. Most existing models, and even massive Frontier LLMs (like TigerLLM or Gemini), treat these properties as disjoint. This leads to **Logical Contradictions**—a model might predict that a comment has `Type = None` (no hate), but simultaneously hallucinate that its `Severity = Severe` or `Target = Individual`.

Our primary goal is to solve this by building a **Consistency-Constrained Multi-Task Learning (MTL) System**. We bind the probabilities of 3 tasks together using a novel mathematical penalty (Soft Consistency Loss). Furthermore, we want the model to be interpretable, so we train a Generative Decoder to write natural-language Bengali explanations justifying its predictions.

### Why We Are Better Than Frontier LLMs
Large Language Models (LLMs) are powerful but expensive, slow, and prone to hallucinations. By explicitly benchmarking against SOTA Bengali models (`TigerLLM-1B-it`) and Frontier models (`Gemini 1.5`), we prove that:
1. **Zero Consistency Violations**: Our model enforces $L_{consist}$, driving the Consistency Violation Rate (CVR) to 0%. Frontier LLMs lack this constraint and frequently output illegal combinations.
2. **High Efficiency**: Our model uses `BanglaBERT` (110M parameters). It is >10x smaller, faster, and cheaper to deploy than a 1B+ parameter LLM, while achieving higher or comparable F1 scores on our specific tasks.

---

## 🧠 Architecture & Methodology

### 1. Multi-Task Target Dimensions
The model predicts exactly 3 dimensions simultaneously:
1. **Hate Type** (*None, Abusive, Political Hate, Religious Hate, Gender Hate*) — 5 classes
2. **Target** (*None, Individual, Organization, Community, Society*) — 5 classes
3. **Severity** (*Little to None, Mild, Severe*) — 3 classes

### 2. The Mathematical Formulation
Our unified loss function combines Focal Loss (to handle the extreme class imbalance, e.g., 'Gender Hate' accounts for <1% of samples) and the Soft Consistency Penalty ($\mathcal{L}_{\text{consist}}$):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}^{\text{type}} + \mathcal{L}_{\text{focal}}^{\text{target}} + \mathcal{L}_{\text{focal}}^{\text{severity}} + \lambda \mathcal{L}_{\text{consist}} + \delta \mathcal{L}_{\text{gen}}$$

The soft differentiable consistency penalty enforces the rule: *If Type=None, Target MUST be None, and Severity MUST be Little to None*.
$$\mathcal{L}_{\text{consist}} = \frac{1}{N} \sum_{i=1}^{N} \left[ P_i(\text{type=None}) \cdot P_i(\text{target}\neq\text{None}) + P_i(\text{type=None}) \cdot P_i(\text{sev}\neq\text{Little}) \right]$$

---

## 🛠️ How We Are Doing It (Kaggle Pipeline)

The entire experimentation suite is built for execution on **Kaggle** (utilizing dual GPU T4s). The pipeline is divided into 5 sequential Jupyter Notebooks:

### 1. Data Preparation (`01_data_preparation.ipynb`)
- Cleans the raw JSON datasets.
- Computes Focal Loss $\alpha$ weights based on class frequencies.
- Proves 0.0% Ground Truth Consistency Violations in the annotated dataset.

### 2. Silver Explanations (`02_silver_explanations.ipynb`)
- We lack human-annotated explanations. This notebook uses the Gemini 1.5 API (with a strict structural prompt) to generate 5,000 high-quality "silver" Bengali explanations for training.
- Fallback logic to a deterministic rule-based generator if rate limits are hit.

### 3. Training & Ablations (`03_training_experiments.ipynb`)
The core script. We train 4 experimental ablations to isolate our contributions:
- `exp1_baseline`: Standard MTL without Consistency Loss (Baseline).
- `exp2_consistency`: MTL + Soft Consistency Loss (Proves CVR drops to 0%).
- `exp3_generation`: MTL + Consistency Loss + LSTM Decoder for Explanations.
- `exp4_faithfulness`: Full Model optimized for ERASER metrics.

### 4. Evaluation & Faithfulness (`04_evaluation_results.ipynb`)
- Calculates Macro F1 and Consistency Violation Rates.
- Runs **ERASER Faithfulness Metrics** (Comprehensiveness, Sufficiency, AOPC) to prove the generated explanations are actually derived from the model's reasoning, not just statistically likely text.

### 5. Frontier LLM Benchmarks (`05_frontier_llm_benchmarks.ipynb`)
- Benchmarks `TigerLLM-1B-it` and `Gemini 1.5 Flash` on zero-shot multi-task classification.
- Collects F1, Latency, and CVR to generate the "Why We Win" comparison table for the paper.

---

## 📁 Repository Directory Structure

```text
BanglaHateML/
├── notebooks/                     # Kaggle Execution Pipeline
│   ├── 01_data_preparation.ipynb
│   ├── 02_silver_explanations.ipynb
│   ├── 03_training_experiments.ipynb
│   ├── 04_evaluation_results.ipynb
│   └── 05_frontier_llm_benchmarks.ipynb
├── banglaHate/
│   └── src/                       # Local PyTorch source code (for smoke testing)
│       ├── dataset.py             # BanglaHateDataset loader
│       ├── model.py               # ConsistencyConstrainedMTL architecture
│       ├── losses.py              # FocalLoss + ConsistencyPenaltyLoss
│       └── smoke_test.py          # Automated pipeline sanity check
├── analysis_results.md            # Deep-dive research project status audit
├── train.json                     # Training dataset
└── README.md                      # This file
```

---

## 🚀 Running on Kaggle (Setup Guide)

To run these experiments yourself on Kaggle:

1. **Create a Kaggle Dataset**:
   - Go to Kaggle -> Create Dataset.
   - Upload `train.json`, `test.json`, and `val.json`.
   - Name it `bangla-hate-speech-mtl`.

2. **Setup Gemini API Key**:
   - In your Kaggle Notebook, go to **Add-ons -> Secrets**.
   - Add a new secret with Label: `GEMINI_API_KEY` and Value: `your-api-key-here`.
   - Attach the secret to the notebooks requiring API access (`02_silver_explanations` & `05_frontier_llm_benchmarks`).

3. **Upload Notebooks**:
   - Import the 5 `.ipynb` notebooks from the `notebooks/` directory into Kaggle.
   - Ensure the Accelerator is set to **GPU T4 x2**.
   - Ensure Internet is toggled **ON**.

4. **Execution Order**:
   - Run Notebook `01` -> Output dataset -> Feed to Notebook `02`.
   - Run Notebook `02` -> Output dataset with explanations -> Feed to Notebook `03`.
   - Run Notebook `03` (Training) -> Output model weights -> Feed to Notebook `04`.
   - Notebook `05` can be run independently for benchmarking.
