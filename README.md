# 🇧🇩 Consistency-Constrained Multi-Task Bengali Hate Speech Detection with Faithfulness-Evaluated Generative Explanations

> **Target Venue**: ICCIT 2026 (Primary) / IEEE Access / EMNLP 2027 BLP Workshop  
> **Authors**: Angkon Roy & Nahid Gazi (SUST CSE)

---

## 📌 Executive Summary

This repository contains the complete codebase, dataset analytics, model architectures, loss formulations, and benchmark evaluations for **multi-task Bengali hate speech classification** across 3 fine-grained dimensions:
1. **Hate Type** (*None, Abusive, Political Hate, Profane, Religious Hate, Sexism*) — 6 classes
2. **Target** (*None, Individual, Organization, Community, Society*) — 5 classes
3. **Severity** (*Little to None, Mild, Severe*) — 3 classes

---

## 🚀 Current Progress Status (Steps 1 – 4 Part 1 COMPLETED)

- [x] **Step 1: Environment & Project Setup** — PyTorch `2.5.1+cu121` + Transformers `4.44.2` active on local NVIDIA GPU (RTX 4050).
- [x] **Step 2: Exploratory Data Analysis (EDA)** — Processed **35,522 samples** (`train.json`), computed Focal Loss $\alpha$ weights, proved **0.0% Ground Truth Consistency Violations**, and generated 3 publication-ready figures (`banglaHate/figures/`).
- [x] **Step 3: Pipeline Smoke Test (`smoke_test.py`)** — **All 8 automated smoke tests passed in 12.7s** on GPU (`ConsistencyConstrainedMTL` with 110.6M params, Focal Loss + Consistency Penalty Loss).
- [x] **Step 4 (Part 1): Bengali Explanation Quality Benchmark** — Extracted 50 stratified comments, generated structured rationales, and saved benchmark reports (`explanation_50_samples.json` and `explanation_benchmark_report.md`).
- [ ] **Step 4 (Part 2)**: Bulk 5,000 Silver Explanation Generation (Scheduled next).
- [ ] **Step 5**: Training 4 BanglaBERT Ablation Models + TigerLLM / Llama Baselines.
- [ ] **Step 6**: Comprehensive Evaluation (Macro F1, CVR, ERASER Faithfulness Comp/Suff/AOPC).

---

## 📁 Repository Directory Structure

```
Bengali_hate_speech_detection/
├── Dataset/                   # Raw & processed datasets (train.json, CSVs)
├── banglaHate/
│   ├── src/                   # PyTorch source code
│   │   ├── dataset.py         # BanglaHateDataset PyTorch loader & tokenization
│   │   ├── model.py           # ConsistencyConstrainedMTL architecture
│   │   ├── losses.py          # FocalLoss + ConsistencyPenaltyLoss
│   │   ├── eda.py             # 11-step comprehensive EDA script
│   │   ├── smoke_test.py      # 8-stage automated pipeline sanity check
│   │   └── compare_explanation_generators.py # Step 4 Part 1 50-sample benchmark
│   ├── figures/               # 300 DPI publication figures
│   │   ├── fig1_class_distributions.png
│   │   ├── fig2_cooccurrence_heatmaps.png
│   │   └── fig3_text_length_distribution.png
│   ├── results/               # Experiment logs & benchmark outputs
│   │   ├── eda_report.json
│   │   ├── explanation_50_samples.json
│   │   └── explanation_benchmark_report.md
│   └── data/                  # Processed CSVs (train_primary.csv)
├── implementation_plan.md     # Implementation roadmap
├── implementation_plan2.md    # 6-week research execution calendar
├── walkthrough.md             # Complete step-by-step walkthrough & results
├── task.md                    # Task tracker
├── presentation.tex           # LaTeX Beamer presentation slides
└── README.md                  # Main documentation
```

---

## 🛠️ Quick Start & Running Scripts

### 1. Run Exploratory Data Analysis (EDA)
```bash
python banglaHate/src/eda.py
```

### 2. Run Pipeline Smoke Test (Sanity Check)
```bash
python banglaHate/src/smoke_test.py
```

### 3. Run Step 4 (Part 1) Bengali Explanation Benchmark
```bash
python banglaHate/src/compare_explanation_generators.py
```

---

## 🧮 Mathematical Loss Formulation

Our unified loss function combines Focal Loss for class imbalance, Soft Consistency Penalty ($\mathcal{L}_{\text{consist}}$) for logical constraint enforcement, and Cross-Entropy for explanation generation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}^{\text{type}} + \mathcal{L}_{\text{focal}}^{\text{target}} + \mathcal{L}_{\text{focal}}^{\text{severity}} + \lambda \mathcal{L}_{\text{consist}} + \delta \mathcal{L}_{\text{gen}}$$

Where the soft differentiable consistency penalty is defined as:
$$\mathcal{L}_{\text{consist}} = \frac{1}{N} \sum_{i=1}^{N} \left[ P_i(\text{type=None}) \cdot P_i(\text{target}\neq\text{None}) + P_i(\text{type=None}) \cdot P_i(\text{sev}\neq\text{Little}) + 2 \cdot P_i(\text{type=None}) \cdot P_i(\text{sev=Severe}) \right]$$

---

## 📄 Handover Notes for Collaborators

If you are picking up work on this codebase:
1. Read [`walkthrough.md`](file:///c:/Users/avron/Downloads/TWP/walkthrough.md) for full execution logs and figure previews.
2. Read [`task.md`](file:///c:/Users/avron/Downloads/TWP/task.md) for the active checklist.
3. Review [`presentation.tex`](file:///c:/Users/avron/Downloads/TWP/presentation.tex) for conference presentation slides.
4. Next task to execute is **Step 4 Part 2** (Bulk Silver Explanation Generation) followed by **Step 5** (Model Training).
