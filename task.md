# Task Tracker — Bengali Hate Speech MTL Research Project

## Step 1: Environment & Project Directory Setup
- [x] Python 3.11.9 verified with pandas 3.0.2, matplotlib 3.10.9, seaborn 0.13.2
- [x] Project directory structure created (`banglaHate/{data,src,models,figures,results,notebooks}`)

## Step 2: Dataset Inspection & Exploratory Data Analysis (EDA)
- [x] Primary dataset loaded: `train.json` (35,522 samples, 7 columns)
- [x] Label distributions computed for all 3 tasks
- [x] Class distribution bar chart generated (`fig1_class_distributions.png`)
- [x] Co-occurrence heatmaps generated (`fig2_cooccurrence_heatmaps.png`)
- [x] Ground truth consistency violation analysis — **0% violations (CLEAN!)**
- [x] Text length statistics computed (mean 78 chars / 14 words)
- [x] Text length distribution figure generated (`fig3_text_length_distribution.png`)
- [x] Focal Loss α weights computed for all 3 tasks
- [x] Supplementary datasets inspected (Bengali hate speech.csv, v1.0, v2.0, slang dict)
- [x] EDA summary report saved as JSON (`eda_report.json`)
- [x] Processed CSV saved (`train_primary.csv`)

## Step 3: Pipeline Smoke Test
- [x] Create `smoke_test.py` + core source files (`model.py`, `losses.py`, `dataset.py`)
- [x] PyTorch 2.5.1+cu121 + transformers 4.44.2 environment set up
- [x] Test dataset loader & tokenization — **PASSED**
- [x] Test model forward pass (`ConsistencyConstrainedMTL`, 110.6M params) — **PASSED**
- [x] Test multi-loss computation & backprop (`FocalLoss` + `ConsistencyPenaltyLoss`) — **PASSED**
- [x] Run 5-epoch overfitting sanity check (Loss: 2.5076 → 1.7729 in 12.7s) — **PASSED**
- [x] All 8 smoke tests passed on **NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB VRAM)**

## Step 4: Silver-Standard Explanation Generation
- [x] Step 4 Part 1: Sample 50 stratified comments & compare generators — **COMPLETED**
  - Evaluated `BanglaT5` vs `Structured Linguistic Anchor` across 50 samples
  - Generated [`explanation_50_samples.json`](file:///c:/Users/avron/Downloads/TWP/banglaHate/results/explanation_50_samples.json)
  - Generated [`explanation_benchmark_report.md`](file:///c:/Users/avron/Downloads/TWP/banglaHate/results/explanation_benchmark_report.md)
- [ ] Step 4 Part 2: Generate bulk 5,000 silver explanations (scheduled for later as requested)

## Step 5: Model Architecture & Loss Implementation
- [ ] Implement `model.py` (ConsistencyConstrainedMTL)
- [ ] Implement `losses.py` (FocalLoss + ConsistencyPenaltyLoss)
- [ ] Implement `dataset.py` (BanglaHateDataset)

## Step 6: Training Execution
- [ ] exp1_baseline: BanglaBERT Standard MTL
- [ ] exp2_consistency: BanglaBERT + L_consist
- [ ] exp3_generative: BanglaBERT + L_gen
- [ ] exp4_full: Full proposed model
- [ ] exp0_tiger: TigerLLM Zero-Shot Baseline
- [ ] exp0_llama: Llama-3.2-3B Few-Shot Baseline

## Step 7: Comprehensive Evaluation
- [ ] Classification Macro F1 per task
- [ ] Consistency Violation Rate (CVR)
- [ ] ERASER Faithfulness (Comp, Suff, AOPC)

## Step 8: Paper Finalization
- [ ] Results tables populated
- [ ] High-res figures created
- [ ] Paper draft completed
