# 📖 Comprehensive Guide: Metrics, Terminology, and Experiments

> **Document Purpose**: Authoritative reference explaining all mathematical terms, evaluation metrics, loss formulations, and experiment lifecycles in the research project:  
> *"Consistency-Constrained Multi-Task Bengali Hate Speech Detection with Generative Explanations"*.

---

# TABLE OF CONTENTS
1. [Core Terminology & Metric Definitions](#1-core-terminology--metric-definitions)
   - [1.1 Consistency Violation Rate (CVR)](#11-consistency-violation-rate-cvr)
   - [1.2 Macro F1 Score](#12-macro-f1-score)
   - [1.3 ERASER Faithfulness Metrics (Comprehensiveness, Sufficiency, AOPC)](#13-eraser-faithfulness-metrics)
   - [1.4 Parse Rate (%)](#14-parse-rate-)
   - [1.5 Inference Latency](#15-inference-latency)
   - [1.6 Zero-Shot vs. 5-Shot Paradigms](#16-zero-shot-vs-5-shot-paradigms)
   - [1.7 Loss Formulations (Focal Loss & Soft Consistency Loss)](#17-loss-formulations)
2. [Our Multi-Task Architecture & Ablation Study (exp1 to exp4)](#2-our-multi-task-architecture--ablation-study)
   - [Architecture Diagram](#architecture-diagram)
   - [exp1: Baseline Multi-Task Learning](#exp1-baseline-multi-task-learning)
   - [exp2: MTL + Soft Consistency Loss](#exp2-mtl--soft-consistency-loss)
   - [exp3: MTL + Generative Rationale Decoder](#exp3-mtl--generative-rationale-decoder)
   - [exp4: Full Proposed Model](#exp4-full-proposed-model)
3. [Bengali Large Language Model Baselines](#3-bengali-large-language-model-baselines)
   - [3.1 TigerLLM-1B-it](#31-tigerllm-1b-it)
   - [3.2 TituLLM-1B](#32-titullm-1b)
   - [3.3 BongLLaMA-3B-Instruct (Deep-Dive into 100% CVR Collapse)](#33-bongllama-3b-instruct-deep-dive-into-100-cvr-collapse)
4. [Master Empirical Comparison Table](#4-master-empirical-comparison-table)
5. [Reviewer Defense Guide (Key Questions & Answers)](#5-reviewer-defense-guide)

---

# 1. Core Terminology & Metric Definitions

---

### 1.1 Consistency Violation Rate (CVR)

#### What is it?
**CVR** measures the proportion of model predictions that violate the logical, hierarchical rules governing the multi-aspect hate speech taxonomy.

#### The Semantic Hierarchy:
Hate speech is not a set of unrelated labels; it has an intrinsic structural dependency:
$$\text{If a text has NO hate speech } (\text{Type} = \text{None}) \implies \text{Target MUST be } \text{None} \land \text{Severity MUST be } \text{Little to None}$$

#### Concrete Examples:
| Type Prediction | Target Prediction | Severity Prediction | Status | Explanation |
|---|---|---|---|---|
| `None` | `None` | `Little to None` | ✅ **Valid** | Non-hateful comment with no target and minimal severity. |
| `Religious Hate` | `Community` | `Severe` | ✅ **Valid** | Coherent prediction of targeted religious toxicity. |
| `None` | `Individual` | `Mild` | ❌ **Violation** | **Contradiction:** How can a non-hateful comment target an individual? |
| `None` | `None` | `Severe` | ❌ **Violation** | **Contradiction:** How can a non-hateful comment have severe threat intensity? |

#### Mathematical Formulation:
$$\text{CVR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\Big(\hat{y}_i^{\text{type}} = \text{None} \;\land\; \big(\hat{y}_i^{\text{target}} \neq \text{None} \;\lor\; \hat{y}_i^{\text{sev}} \neq \text{Little to None}\big)\Big) \times 100\%$$
Where $\mathbb{I}(\cdot)$ is the indicator function equal to 1 if the condition is true, and 0 otherwise.

#### Why standard metrics fail without CVR:
Traditional classification metrics (like Accuracy or F1) evaluate each task independently in isolation. A model could score a high F1 on Type, Target, and Severity individually, while simultaneously outputting hundreds of mutually contradictory combinations. **CVR directly exposes whether a model possesses structural understanding.**

---

### 1.2 Macro F1 Score

#### What is it?
The **Harmonic Mean** of Precision and Recall calculated across classes:
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Why Macro F1 instead of Accuracy or Micro F1?
In our dataset of 35,530 Bengali comments:
* `None`: 56.17% (19,958 samples)
* `Abusive`: 29.68% (10,547 samples)
* `Political Hate`: 11.90% (4,228 samples)
* `Religious Hate`: 1.90% (675 samples)
* `Gender Hate`: 0.34% (122 samples)

If a naive model predicted `None` for every single sample, its **Accuracy would be 56%**, but it would detect **zero** hate speech!
* **Micro F1** is dominated by the majority class (`None`).
* **Macro F1** computes the F1 independently for each of the 5 classes and takes an unweighted average:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
This guarantees that correctly identifying rare, high-risk categories like `Gender Hate` is weighted just as heavily as identifying common categories.

#### Average Macro F1:
The headline metric for overall multi-task competence across all three heads:
$$\text{Avg Macro F1} = \frac{\text{Type F1} + \text{Target F1} + \text{Severity F1}}{3}$$

---

### 1.3 ERASER Faithfulness Metrics

When our full model (`exp4`) predicts hate speech, it uses its attention weights from the `[CLS]` token to identify which tokens served as the **rationale** (the key supporting words). The **ERASER benchmark** (DeYoung et al., ACL 2020) tests whether those highlighted words are *actually faithful to the model's internal computation*.

#### 1. Comprehensiveness (Higher is Better $\uparrow$)
* **Concept**: If we erase/mask the identified rationale words from the sentence, does the model's confidence in its original prediction drop significantly?
* **Formula**:
  $$\text{Comp} = P(\hat{y} \mid X) - P(\hat{y} \mid X \setminus R)$$
  where $X$ is the input and $R$ is the rationale.
* **Interpretation**: A high score proves that the rationale contained the essential tokens driving the model's prediction.

#### 2. Sufficiency (Lower is Better $\downarrow$)
* **Concept**: If we show the model *ONLY* the rationale tokens (erasing the rest of the sentence), does the model maintain its original confidence?
* **Formula**:
  $$\text{Suff} = P(\hat{y} \mid X) - P(\hat{y} \mid R)$$
* **Interpretation**: A low score (close to 0) proves that the rationale tokens alone are sufficient to justify the decision without needing the rest of the text.

#### 3. AOPC (Area Over the Perturbation Curve $\uparrow$)
* **Concept**: We mask out varying percentages of top tokens: $k \in \{5\%, 10\%, 20\%, 50\%\}$.
* AOPC is the average Comprehensiveness score across all perturbation thresholds. A higher AOPC curve indicates consistently faithful feature attribution.

---

### 1.4 Parse Rate (%)

#### What is it?
The percentage of test samples for which a model's generated text can be successfully parsed into structured attribute values (`type`, `target`, `severity`).

#### The Problem with LLMs:
* Our 110M specialized model uses dedicated classification heads that output logits over fixed classes $\implies$ **100% Deterministic Parse Rate**.
* Generative LLMs generate free-form text tokens. If an LLM outputs conversational filler (`"Sure! Here is the JSON: ..."`), unclosed braces, or misses a key, parsing fails.
* In our tests, zero-shot BongLLaMA had a **0.0% parse rate**, and TituLLM parsed only **5.7%** of zero-shot outputs.

---

### 1.5 Inference Latency

#### What is it?
The wall-clock time in seconds required to classify a single comment:
$$\text{Latency} = \frac{\text{Total Evaluation Time (seconds)}}{\text{Total Test Samples (3,553)}}$$

#### Practical Relevance:
Social platforms like Facebook or YouTube process millions of comments per hour.
* Our 110M model: **0.0152s / sample** ($\approx 66$ items/second on a single standard T4 GPU).
* TigerLLM-1B: **5.851s / sample** ($\approx 0.17$ items/second $\implies$ **$\sim$390$\times$ slower**).

---

### 1.6 Zero-Shot vs. 5-Shot Paradigms

* **Zero-Shot**: The model receives task instructions, label options, and the input text. No input-output examples are provided. Tests pure instruction-following capability.
* **5-Shot (In-Context Learning)**: The model's prompt includes 5 diverse demonstration pairs (comment + correct JSON output) sampled strictly from the **training set** before presenting the test comment. Tests how well the model learns patterns in-context.

---

### 1.7 Loss Formulations

#### 1. Multi-Class Focal Loss
Addresses severe class imbalance by dynamically scaling down the loss for easy, well-classified examples:
$$\mathcal{L}_{\text{focal}}^t = -\frac{1}{N} \sum_{i=1}^N \alpha_{y_i^t} (1 - P(y_i^t \mid X_i))^\gamma \log P(y_i^t \mid X_i)$$
where $\gamma = 2.0$ focuses learning on hard examples (e.g., distinguishing Political Hate from Abusive).

#### 2. Differentiable Soft Consistency Loss ($\mathcal{L}_{\text{consist}}$)
Instead of applying hard rules after inference, we penalize contradictions during gradient descent:
$$\mathcal{L}_{\text{consist}} = \frac{1}{N} \sum_{i=1}^{N} P_i(\text{type}=\text{None}) \cdot \left[ \sum_{j \neq \text{None}} P_i(\text{target}=j) + \sum_{k \neq \text{Little}} P_i(\text{sev}=k) \right]$$
* If $P(\text{type}=\text{None}) \to 1$, the loss forces $P(\text{target}\neq\text{None}) \to 0$ and $P(\text{sev}\neq\text{Little}) \to 0$.
* Fully differentiable: Gradients propagate back into both the task heads and the shared BanglaBERT encoder.

---

# 2. Our Multi-Task Architecture & Ablation Study

### Architecture Diagram

```
                              Input Bengali Comment (Tokens)
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   Shared BanglaBERT Encoder (110M)    │
                        │        (csebuetnlp/banglabert)        │
                        └───────────────────┬───────────────────┘
                                            │ Contextual [CLS] Vector (d=768)
                                            ▼
                 ┌──────────────────────────┼──────────────────────────┐
                 │                          │                          │
                 ▼                          ▼                          ▼
       ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
       │    Type Head     │       │   Target Head    │       │  Severity Head   │
       │  (Linear + ReLU) │       │  (Linear + ReLU) │       │  (Linear + ReLU) │
       │    5 Classes     │       │    5 Classes     │       │    3 Classes     │
       └─────────┬────────┘       └─────────┬────────┘       └─────────┬────────┘
                 │                          │                          │
                 └──────────────────────────┼──────────────────────────┘
                                            ▼
                        ┌───────────────────────────────────────┐
                        │     Soft Consistency Penalty          │
                        │    (Eq: Type=None => Trg=0 & Sev=0)   │
                        └───────────────────┬───────────────────┘
                                            │
                                            ▼
                        ┌───────────────────────────────────────┐
                        │   LSTM Rationale Decoder (2.5M)       │
                        │  (Generates Natural Bengali Rationale)│
                        └───────────────────────────────────────┘
```

---

### exp1: Baseline Multi-Task Learning (MTL)
* **Architecture**: Shared BanglaBERT + 3 independent classification heads.
* **Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}}^{\text{type}} + \mathcal{L}_{\text{focal}}^{\text{target}} + \mathcal{L}_{\text{focal}}^{\text{sev}}$ ($\lambda=0, \delta=0$).
* **Results on 3,553 Test Set**:
  - Type F1: 0.5111 | Target F1: 0.5109 | Severity F1: 0.5972 | **Avg F1: 0.5397**
  - **Consistency Violations: 336 (9.46% CVR)**
* **Diagnosis**: Without explicit guidance, the three heads output independent predictions that contradict each other in nearly 1 out of every 10 comments.

---

### exp2: MTL + Soft Consistency Loss
* **Architecture**: Same as exp1, but with $\lambda=1.0$ for $\mathcal{L}_{\text{consist}}$.
* **Results on 3,553 Test Set**:
  - Type F1: 0.4959 | Target F1: **0.5652** | Severity F1: 0.6110 | **Avg F1: 0.5574**
  - **Consistency Violations: 0 (0.00% CVR)**
* **Diagnosis**: Soft Consistency Loss completely eliminated all 336 contradictions ($0.00\%$ CVR). Crucially, **Avg F1 increased by +3.3% relative** (from 0.5397 to 0.5574), primarily due to massive improvements in Target F1 (+0.054) and Severity F1 (+0.014). The consistency constraint acts as an inductive bias, forcing the shared encoder to learn more discriminative representations.

---

### exp3: MTL + Generative Explanation Head
* **Architecture**: exp1 + autoregressive LSTM decoder generating natural language rationales ($\delta=0.5, \lambda=0$).
* **Results on 3,553 Test Set**:
  - Type F1: **0.5270** | Target F1: 0.5228 | Severity F1: 0.5988 | **Avg F1: 0.5495**
  - **Consistency Violations: 176 (4.95% CVR)**
* **Diagnosis**: Adding the generative head forced the shared `[CLS]` embedding to encode semantic information rich enough to reconstruct explanatory text. This multi-task grounding cut violations in half (from 336 to 176) even without an explicit mathematical penalty.

---

### exp4: Full Proposed Model
* **Architecture**: BanglaBERT + 3 Classification Heads + Soft Consistency Loss + LSTM Generative Decoder ($\lambda=1.0, \delta=0.5$).
* **Results on 3,553 Test Set**:
  - Type F1: 0.4994 | Target F1: 0.5598 | Severity F1: **0.6136** | **Avg F1: 0.5576**
  - **Consistency Violations: 1 (0.03% CVR)**
  - **ERASER Sufficiency: 0.0745** (Best across all models)
  - **Latency: 0.0152s / sample**
* **Diagnosis**: The primary model of the paper. It combines the highest overall F1 (0.5576), near-zero CVR (0.03%), faithful rationale generation, and high throughput.

---

# 3. Bengali Large Language Model Baselines

---

### 3.1 TigerLLM-1B-it (Md. Nishat et al., ACL 2025)
* **Parameters**: 1 Billion (Decoder-only architecture).
* **0-Shot Performance**: Avg F1 = 0.2554, CVR = 0.42% (15 violations), Parse Rate = 95.6%, Latency = 4.63s.
* **5-Shot Performance**: Avg F1 = 0.3303, CVR = 1.32% (47 violations), Parse Rate = 99.1%, Latency = 5.85s.
* **Analysis**: Demonstrations improve instruction following (+0.075 F1 gain), but the model still suffers from 47 structural violations and runs $\sim$390$\times$ slower than our model.

---

### 3.2 TituLLM-1B (Hishab Technologies)
* **Parameters**: 1 Billion (Dedicated Bengali pretrained foundation model).
* **0-Shot Performance**: Avg F1 = 0.1856, Parse Rate = 5.7% (failed to output JSON format without examples), Latency = 2.17s.
* **5-Shot Performance**: Avg F1 = 0.2183, CVR = 4.98% (177 violations), Parse Rate = 21.5%, Latency = 5.27s.
* **Analysis**: Because TituLLM is a foundation model without extensive instruction tuning, it struggled with structural JSON constraints, resulting in low parse rates and 177 contradictions.

---

### 3.3 BongLLaMA-3B-Instruct (Deep-Dive into 100% CVR Collapse)
* **Parameters**: 3 Billion (Largest Bengali instruction LLM, based on LLaMA-3.2).
* **0-Shot Performance**: Parse Rate = 0.0% (completely failed to format output as JSON; fallback mapped to `None`, yielding 0.00% CVR and baseline 0.1856 F1).
* **5-Shot Performance**: Parse Rate = 100.0%, **CVR = 100.00% (3,553 / 3,553 violations!)**, Avg F1 = 0.1261.

#### Why did 100% CVR happen?
Mathematical analysis confirms that BongLLaMA suffered from **Catastrophic Mode Collapse (Degenerate Repetition)**:
* For **every single one of the 3,553 test instances**, the model output the exact same string:
  ```json
  {"type_of_hate": "None", "target_of_hate": "None", "severity_of_hate": "Severe"}
  ```
* **Mathematical Verification**:
  - Type F1 = **0.1439** $\iff$ matches predicting `Type: None` for 100% of samples.
  - Target F1 = **0.1500** $\iff$ matches predicting `Target: None` for 100% of samples.
  - Severity F1 = **0.0844** $\iff$ matches predicting `Severity: Severe` for 100% of samples.
* Because the model classified every comment as non-hateful (`Type: None`), but simultaneously assigned it severe intensity (`Severity: Severe`), **100% of its predictions violated the taxonomy rule**.

---

# 4. Master Empirical Comparison Table

All models evaluated on the **exact same 3,553 test samples**:

| Model | Size | Paradigm | Type F1 | Target F1 | Severity F1 | **Avg Macro F1** | **CVR (%)** | Violations | Parse Rate | Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **TigerLLM-1B-it** | 1B | 0-shot | 0.2256 | 0.1851 | 0.3557 | 0.2554 | 0.42% | 15 | 95.6% | 4.634s |
| **TigerLLM-1B-it** | 1B | 5-shot | 0.3572 | 0.2591 | 0.3747 | 0.3303 | 1.32% | 47 | 99.1% | 5.851s |
| **TituLLM-1B** | 1B | 0-shot | 0.1439 | 0.1500 | 0.2631 | 0.1856 | 0.00% | 0 | 5.7% | 2.173s |
| **TituLLM-1B** | 1B | 5-shot | 0.1803 | 0.1731 | 0.3014 | 0.2183 | 4.98% | 177 | 21.5% | 5.274s |
| **BongLLaMA-3B** | 3B | 0-shot | 0.1439 | 0.1500 | 0.2631 | 0.1856 | 0.00% | 0 | 0.0%* | 0.218s |
| **BongLLaMA-3B** | 3B | 5-shot | 0.1439 | 0.1500 | 0.0844 | 0.1261 | 100.00% | 3,553 | 100.0% | 1.277s |
| **Ours: exp1 (Baseline)** | 110M | Deterministic | 0.5111 | 0.5109 | 0.5972 | 0.5397 | 9.46% | 336 | 100.0% | **0.0141s** |
| **Ours: exp2 (+Consistency)** | 110M | Deterministic | 0.4959 | **0.5652** | 0.6110 | 0.5574 | **0.00%** | **0** | 100.0% | **0.0155s** |
| **Ours: exp3 (+Gen Head)** | 110M | Deterministic | **0.5270** | 0.5228 | 0.5988 | 0.5495 | 4.95% | 176 | 100.0% | **0.0150s** |
| **Ours: exp4 (Full Model)** | **110M** | Deterministic | 0.4994 | 0.5598 | **0.6136** | **0.5576** | **0.03%** | **1** | **100.0%** | **0.0152s** |

---

# 5. Reviewer Defense Guide

### Q1: "Why did you build an SLM instead of fine-tuning an LLM like LLaMA?"
> **Answer**: Our specialized 110M model outperforms 1B–3B Bengali LLMs by **+68.8% in Macro F1** while operating **~390× faster** (0.015s vs 5.85s). Large models require quantization, large GPU VRAM, and exhibit severe format fragility. In production moderation environments handling millions of comments, an efficient, deterministic 110M model deployable on standard edge CPUs is far more practical.

### Q2: "Did the model overfit or suffer data contamination?"
> **Answer**: No. The dataset was partitioned with a strictly deterministic 80/10/10 stratified split (`random_state=42`). The 3,553 test samples were held out and never touched during training or hyperparameter tuning. All baseline LLMs and our models were scored on the exact same 3,553 test instances.

### Q3: "Does Consistency Loss hurt performance?"
> **Answer**: No, it actually improves performance. Comparing `exp1` (0.5397 F1) to `exp2` (0.5574 F1) shows a **+3.3% relative improvement**. Enforcing logical dependencies acts as an inductive regularizer, sharpening the shared encoder's features for target and severity classification.
