# 🔬 Complete Research Execution Plan
## Consistency-Constrained Bengali Hate Speech Detection

> [!CAUTION]
> **ICCIT 2026 Deadline: August 31, 2026** — You have **~12 days**. This plan is structured for maximum speed. If the ICCIT 2026 deadline is not feasible, alternative venues are listed at the end.

---

## Phase 0: Immediate Decision — Can You Make Aug 31?

| If YES (Target ICCIT 2026) | If NO (Target Later Venue) |
|:---|:---|
| Skip Phase 3 (custom dataset) — no time | Include Phase 3 for stronger contribution |
| Use BanglaMultiHate only + silver-standard explanations | Build custom dataset as additional contribution |
| Prioritize Phases 1→2→4→5→6→7 | Follow all phases sequentially |
| Estimated work: 10-12 days intense | Estimated work: 6-8 weeks comfortable |

---

## Phase 1: Dataset Acquisition & Exploration (Day 1-2)

### 1.1 Download All Datasets

| Dataset | Size | Link | Command |
|:---|:---|:---|:---|
| **BanglaMultiHate** | ~50,746 samples | [HuggingFace](https://huggingface.co/datasets/aridhasan/BanglaMultiHate) | `load_dataset("aridhasan/BanglaMultiHate")` |
| **BanHADEX / BanHate** | ~19,203 samples | [HuggingFace](https://huggingface.co/datasets/aplycaebous/BanHate) | `load_dataset("aplycaebous/BanHate")` |
| **BD-SHS** (optional) | ~50,200 samples | [GitHub](https://github.com/naurosromim/hate-speech-dataset-for-Bengali-social-media) | Git clone |
| **Karim et al.** (optional) | ~10K samples | [GitHub](https://github.com/rezacsedu/Bengali-Hate-Speech-Dataset) | Git clone |

```python
# Step 1: Install dependencies
# pip install datasets transformers torch accelerate google-generativeai

from datasets import load_dataset

# Primary datasets
bangla_multi = load_dataset("aridhasan/BanglaMultiHate")
banhate = load_dataset("aplycaebous/BanHate")

print(f"BanglaMultiHate splits: {bangla_multi}")
print(f"BanHate splits: {banhate}")

# Inspect schema
print(f"BanglaMultiHate columns: {bangla_multi['train'].column_names}")
print(f"BanHate columns: {banhate['train'].column_names}")
```

### 1.2 Label Taxonomy Comparison (CRITICAL)

You **must** document the exact mismatch. Based on research:

| Dimension | BanglaMultiHate Labels | BanHADEX Labels | Gap |
|:---|:---|:---|:---|
| **Hate Type** | Abusive, Sexism, Religious Hate, Political Hate, Profane, **None** (6 labels) | Political, Religious, Gender, Personal Offense, Abusive/Violence, Origin, Body Shaming (7 labels) | Different category names and count |
| **Severity** | Little to None, Mild, **Severe** (3 labels) | ❌ **Not present** | BanHADEX has NO severity |
| **Target** | Individual, Organization, Community, Society, **None** (5 labels) | 7 target groups (different taxonomy) | Different granularity |
| **Explanation** | ❌ **Not present** | ✅ Human-written explanations | BanglaMultiHate has NO explanations |

> [!IMPORTANT]
> **This is your taxonomic mismatch.** The critical gaps are:
> 1. BanglaMultiHate has **severity** labels; BanHADEX does **not**
> 2. BanglaMultiHate has **no explanations**; BanHADEX does
> 3. The hate type and target taxonomies use **different label sets**
>
> Your silver-standard bridging step is justified because you need explanations conditioned on the **BanglaMultiHate 3-task taxonomy** (Type+Target+Severity), which BanHADEX simply doesn't provide.

### 1.3 Exploratory Data Analysis (EDA)

Create the following analyses and save as figures for the paper:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = bangla_multi['train'].to_pandas()

# 1. Class distribution per task
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, col in zip(axes, ['hate_type', 'target', 'severity']):
    df[col].value_counts().plot(kind='bar', ax=ax, title=f'{col} Distribution')
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=300)

# 2. Consistency violation analysis in raw data
# Count logically impossible combinations
violations = df[
    (df['hate_type'] == 'None') & 
    ((df['severity'] != 'Little to None') | (df['target'] != 'None'))
]
print(f"Ground-truth consistency violations: {len(violations)} / {len(df)} ({100*len(violations)/len(df):.2f}%)")

# 3. Co-occurrence matrix
pd.crosstab(df['hate_type'], df['severity']).to_csv('analysis/type_severity_crosstab.csv')
pd.crosstab(df['hate_type'], df['target']).to_csv('analysis/type_target_crosstab.csv')
```

### 1.4 Deliverables for Phase 1
- [ ] Both datasets downloaded and loaded
- [ ] Column schema documented in a table
- [ ] Class distribution bar charts (3 figures)
- [ ] Consistency violation count in ground truth
- [ ] Co-occurrence heatmaps (Type×Severity, Type×Target)

---

## Phase 2: Silver-Standard Explanation Generation (Day 2-4)

### 2.1 Design the Prompting Strategy

Use **Gemini 1.5 Pro** (or Gemini 2.0 Flash for speed) via the API. You need a **few-shot prompt** that:
1. Takes a Bengali comment + its ground-truth labels (Type, Target, Severity)
2. Generates a structured explanation in Bengali

### 2.2 Prompt Template

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

SYSTEM_PROMPT = """You are an expert Bengali linguist and hate speech analyst.
Given a Bengali social media comment and its classification labels, generate a concise 
explanation (2-3 sentences in Bengali) that justifies WHY the comment received those labels.

Rules:
1. The explanation must reference specific words or phrases from the comment.
2. The explanation must logically connect the comment to ALL three labels (Type, Target, Severity).
3. If the comment is labeled "None" for hate type, explain why it is NOT hateful.
4. Write in natural Bengali script."""

FEW_SHOT_EXAMPLES = """
Example 1:
Comment: "এই হিন্দুগুলোকে দেশ থেকে তাড়িয়ে দেওয়া উচিত"
Labels: Type=Religious Hate, Target=Community, Severity=Severe
Explanation: এই মন্তব্যটি ধর্মীয় ঘৃণা প্রকাশ করে কারণ এটি "হিন্দু" সম্প্রদায়কে নির্দিষ্ট করে আক্রমণ করছে। "তাড়িয়ে দেওয়া উচিত" বাক্যাংশটি সরাসরি বহিষ্কারের আহ্বান জানায়, যা তীব্রতার দিক থেকে গুরুতর (Severe) পর্যায়ে পড়ে। লক্ষ্য একটি ধর্মীয় সম্প্রদায় (Community)।

Example 2:
Comment: "আজকের আবহাওয়া খুব সুন্দর"
Labels: Type=None, Target=None, Severity=Little to None
Explanation: এই মন্তব্যটি কেবল আবহাওয়া সম্পর্কে একটি নিরপেক্ষ মতামত প্রকাশ করে। কোনো ব্যক্তি, গোষ্ঠী বা সম্প্রদায়ের বিরুদ্ধে ঘৃণা বা আক্রমণাত্মক বক্তব্য নেই, তাই এটি ঘৃণামূলক বক্তব্য নয়।

Example 3:
Comment: "এই মহিলার তো কোনো লজ্জা শরমই নেই, বেশ্যার মতো ঘুরে বেড়ায়"
Labels: Type=Sexism, Target=Individual, Severity=Severe
Explanation: মন্তব্যটি একজন নির্দিষ্ট মহিলাকে লক্ষ্য করে যৌনতাবাদী ঘৃণা প্রকাশ করে। "বেশ্যা" শব্দটি অত্যন্ত অবমাননাকর এবং যৌন-ভিত্তিক অপমান। "লজ্জা শরম নেই" বাক্যাংশটি চরিত্র হননের উদ্দেশ্যে ব্যবহৃত, যা তীব্রতায় Severe।
"""

def generate_explanation(comment, hate_type, target, severity):
    prompt = f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}

Now generate an explanation for:
Comment: "{comment}"
Labels: Type={hate_type}, Target={target}, Severity={severity}
Explanation:"""
    
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=300
        )
    )
    return response.text
```

### 2.3 Batch Processing Pipeline

```python
import json
import time
from tqdm import tqdm

def generate_silver_explanations(dataset, output_path, batch_size=10, max_samples=5000):
    """Generate silver-standard explanations for BanglaMultiHate samples."""
    results = []
    
    for i, sample in enumerate(tqdm(dataset, total=min(len(dataset), max_samples))):
        if i >= max_samples:
            break
        
        try:
            explanation = generate_explanation(
                comment=sample['text'],
                hate_type=sample['hate_type'],
                target=sample['target'],
                severity=sample['severity']
            )
            
            results.append({
                'id': i,
                'text': sample['text'],
                'hate_type': sample['hate_type'],
                'target': sample['target'],
                'severity': sample['severity'],
                'silver_explanation': explanation
            })
            
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            results.append({**sample, 'silver_explanation': None})
        
        # Rate limiting (Gemini API: ~15 RPM for free tier, ~1000 RPM for paid)
        if (i + 1) % batch_size == 0:
            time.sleep(2)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len([r for r in results if r['silver_explanation']])} explanations")
    return results

# Run generation
silver_data = generate_silver_explanations(
    dataset=bangla_multi['train'],
    output_path='data/silver_explanations.json',
    max_samples=5000  # Start with 5K, expand if time permits
)
```

### 2.4 Quality Validation of Silver Explanations

> [!WARNING]
> **You MUST do this to preempt reviewer criticism about circular reasoning.**

```python
# Human evaluation protocol:
# 1. Randomly sample 200 silver explanations
# 2. Have 2 native Bengali speakers rate each on:
#    - Correctness (1-5): Does the explanation correctly identify the hateful elements?
#    - Relevance (1-5): Does it reference the right words/phrases?
#    - Completeness (1-5): Does it address all three dimensions (Type, Target, Severity)?
# 3. Compute inter-annotator agreement (Cohen's kappa)
# 4. Report average scores in the paper

import random

sample_for_evaluation = random.sample(silver_data, min(200, len(silver_data)))
# Save for human annotators
with open('data/human_eval_sample.json', 'w', encoding='utf-8') as f:
    json.dump(sample_for_evaluation, f, ensure_ascii=False, indent=2)
```

### 2.5 Deliverables for Phase 2
- [ ] 5,000+ silver-standard explanations generated
- [ ] Human evaluation of 200-sample subset
- [ ] Inter-annotator agreement scores
- [ ] Quality statistics table for the paper

---

## Phase 3: Custom Dataset Creation (Optional — Skip if ICCIT 2026 Rush)

### 3.1 Why Build a Custom Dataset?

If you create even a small **gold-standard** dataset (300-500 samples) with ALL four annotations (Type, Target, Severity, Human Explanation), it serves as:
1. A **validation set** for silver-standard quality
2. A **contribution** in itself (first Bengali dataset with all 4 dimensions)
3. **Ground truth** for faithfulness evaluation

### 3.2 Annotation Protocol

#### Source Collection
```
Platform: YouTube (Bengali news channels, political channels)
Tool: youtube-comment-downloader (pip install youtube-comment-downloader)
Target: 500 comments (balanced across hate/non-hate)
```

#### Annotation Schema (per sample)

| Field | Type | Labels | Instructions |
|:---|:---|:---|:---|
| `text` | string | Raw comment | Preserve original spelling/code-mixing |
| `hate_type` | categorical | Abusive, Sexism, Religious Hate, Political Hate, Profane, None | Use BanglaMultiHate taxonomy |
| `target` | categorical | Individual, Organization, Community, Society, None | Use BanglaMultiHate taxonomy |
| `severity` | categorical | Little to None, Mild, Severe | Use BanglaMultiHate taxonomy |
| `explanation` | free-text | Bengali explanation (2-3 sentences) | Must reference specific words; must cover all 3 dimensions |

#### Annotation Workflow
```
Step 1: Collect 600 raw comments (over-sample to account for filtering)
Step 2: Two annotators independently label each sample
Step 3: Third annotator resolves disagreements
Step 4: Compute inter-annotator agreement (Fleiss' kappa)
Step 5: Filter to 500 clean, agreed-upon samples
Step 6: Format as JSON matching BanglaMultiHate schema + explanation field
```

#### Recommended Tool
Use **Label Studio** (free, open-source) or a simple **Google Form** for quick annotation.

### 3.3 Deliverables for Phase 3
- [ ] 500 annotated samples with all 4 fields
- [ ] Inter-annotator agreement report
- [ ] Dataset card (description, stats, license)

---

## Phase 4: Model Architecture (Day 3-6)

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    INPUT TEXT                         │
│              (Bengali comment)                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           SHARED ENCODER (BanglaBERT)               │
│         csebuetnlp/banglabert (ELECTRA)             │
│              Output: H ∈ ℝ^(seq_len × 768)          │
└──┬──────────┬──────────┬───────────┬────────────────┘
   │          │          │           │
   ▼          ▼          ▼           ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌────────────────┐
│Type  │ │Target│ │Sev.  │ │  Generative    │
│Head  │ │Head  │ │Head  │ │  Head (Decoder)│
│FC→6  │ │FC→5  │ │FC→3  │ │  LSTM / small  │
│      │ │      │ │      │ │  transformer   │
└──┬───┘ └──┬───┘ └──┬───┘ └──────┬─────────┘
   │        │        │            │
   ▼        ▼        ▼            ▼
  L_type  L_target  L_sev      L_gen
   │        │        │            │
   └────┬───┘────┬───┘            │
        │        │                │
        ▼        │                │
    L_consist    │                │
        │        │                │
        └────────┴────────────────┘
                 │
                 ▼
    L = αL_type + βL_target + γL_sev + δL_gen + λL_consist
```

### 4.2 Full PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ============================================================
# Component 1: Focal Loss (for class imbalance)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

# ============================================================
# Component 2: Consistency Penalty Loss (YOUR NOVEL CONTRIBUTION)
# ============================================================
class ConsistencyPenaltyLoss(nn.Module):
    """
    Penalizes logically contradictory predictions across tasks.
    
    Contradiction Rules (from BanglaMultiHate annotation schema):
      Rule 1: If Type=None → Target MUST be None AND Severity MUST be "Little to None"
      Rule 2: If Severity=Severe → Type MUST NOT be None
      Rule 3: If Target≠None → Type MUST NOT be None
    """
    def __init__(self, penalty_weight=1.0):
        super().__init__()
        self.penalty_weight = penalty_weight
        
        # Label indices (must match your label encoder)
        self.TYPE_NONE_IDX = 5      # 'None' in hate_type
        self.TARGET_NONE_IDX = 4    # 'None' in target
        self.SEV_NONE_IDX = 0       # 'Little to None' in severity
        self.SEV_SEVERE_IDX = 2     # 'Severe' in severity
    
    def forward(self, type_logits, target_logits, severity_logits):
        """
        Args:
            type_logits: (batch, 6) - raw logits for hate type
            target_logits: (batch, 5) - raw logits for target
            severity_logits: (batch, 3) - raw logits for severity
        Returns:
            penalty: scalar loss
        """
        # Convert to probabilities
        type_probs = F.softmax(type_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        sev_probs = F.softmax(severity_logits, dim=-1)
        
        # P(Type=None)
        p_type_none = type_probs[:, self.TYPE_NONE_IDX]
        # P(Target≠None) = 1 - P(Target=None)
        p_target_not_none = 1.0 - target_probs[:, self.TARGET_NONE_IDX]
        # P(Severity≠Little to None) = 1 - P(Severity=Little to None)
        p_sev_not_none = 1.0 - sev_probs[:, self.SEV_NONE_IDX]
        # P(Severity=Severe)
        p_sev_severe = sev_probs[:, self.SEV_SEVERE_IDX]
        
        # Rule 1: Penalize P(Type=None) × P(Target≠None)
        # If model says "no hate" but also says "there IS a target" → contradiction
        violation_1 = p_type_none * p_target_not_none
        
        # Rule 2: Penalize P(Type=None) × P(Severity≠Little)
        # If model says "no hate" but severity is not minimal → contradiction
        violation_2 = p_type_none * p_sev_not_none
        
        # Rule 3: Penalize P(Type=None) × P(Severity=Severe)
        # Extra strong penalty for the most extreme contradiction
        violation_3 = p_type_none * p_sev_severe
        
        # Total penalty: sum of soft-contradiction scores
        penalty = (violation_1 + violation_2 + 2.0 * violation_3).mean()
        
        return self.penalty_weight * penalty

# ============================================================
# Component 3: Main Model Architecture
# ============================================================
class ConsistencyConstrainedMTL(nn.Module):
    def __init__(self, 
                 encoder_name='csebuetnlp/banglabert',
                 num_type_labels=6,
                 num_target_labels=5,
                 num_severity_labels=3,
                 vocab_size=32000,
                 gen_hidden_dim=256,
                 gen_num_layers=2):
        super().__init__()
        
        # Shared encoder
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 768
        
        # Discriminative heads
        self.type_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_type_labels)
        )
        self.target_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_target_labels)
        )
        self.severity_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_severity_labels)
        )
        
        # Generative head (lightweight LSTM decoder)
        self.gen_embedding = nn.Embedding(vocab_size, gen_hidden_dim)
        self.gen_decoder = nn.LSTM(
            input_size=gen_hidden_dim,
            hidden_size=hidden_size,
            num_layers=gen_num_layers,
            batch_first=True,
            dropout=0.1
        )
        self.gen_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask, 
                explanation_ids=None, explanation_mask=None):
        # Encode input
        encoder_output = self.encoder(input_ids=input_ids, 
                                       attention_mask=attention_mask)
        pooled = encoder_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Discriminative predictions
        type_logits = self.type_head(pooled)
        target_logits = self.target_head(pooled)
        severity_logits = self.severity_head(pooled)
        
        # Generative output (only during training)
        gen_logits = None
        if explanation_ids is not None:
            gen_embeds = self.gen_embedding(explanation_ids)
            # Initialize decoder hidden state from encoder
            h0 = pooled.unsqueeze(0).repeat(2, 1, 1)  # (num_layers, batch, hidden)
            c0 = torch.zeros_like(h0)
            gen_output, _ = self.gen_decoder(gen_embeds, (h0, c0))
            gen_logits = self.gen_projection(gen_output)
        
        return type_logits, target_logits, severity_logits, gen_logits
```

### 4.3 The Full Loss Function (with formal math for the paper)

For the paper, write this equation:

$$\mathcal{L} = \alpha \mathcal{L}_{\text{type}} + \beta \mathcal{L}_{\text{target}} + \gamma \mathcal{L}_{\text{sev}} + \delta \mathcal{L}_{\text{gen}} + \lambda \mathcal{L}_{\text{consist}}$$

Where:

$$\mathcal{L}_{\text{consist}} = \frac{1}{N} \sum_{i=1}^{N} \Big[ P_i(\text{Type}{=}\text{None}) \cdot P_i(\text{Target}{\neq}\text{None}) + P_i(\text{Type}{=}\text{None}) \cdot P_i(\text{Sev}{\neq}\text{LittleToNone}) + 2 \cdot P_i(\text{Type}{=}\text{None}) \cdot P_i(\text{Sev}{=}\text{Severe}) \Big]$$

**Recommended hyperparameters:**

| Parameter | Value | Rationale |
|:---|:---|:---|
| α (type weight) | 1.0 | Primary task |
| β (target weight) | 1.0 | Equal importance |
| γ (severity weight) | 1.0 | Equal importance |
| δ (generation weight) | 0.5 | Auxiliary; don't let it dominate |
| λ (consistency weight) | 0.1 → 1.0 (warmup) | Start low, increase over epochs |

### 4.4 Deliverables for Phase 4
- [ ] Full model code (Python file)
- [ ] ConsistencyPenaltyLoss with formal math
- [ ] Architecture diagram (for paper Figure 1)
- [ ] Model parameter count documented

---

## Phase 5: Training Pipeline (Day 5-8)

### 5.1 Training Configuration

```python
# Training hyperparameters
CONFIG = {
    'encoder': 'csebuetnlp/banglabert',
    'max_seq_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,          # for encoder
    'head_learning_rate': 1e-4,     # for classification heads
    'epochs': 15,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'focal_gamma': 2.0,
    'consistency_lambda_start': 0.01,
    'consistency_lambda_end': 1.0,
    'gen_weight': 0.5,
    'gradient_accumulation_steps': 2,
    'fp16': True,
}
```

### 5.2 Training Loop Skeleton

```python
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def train_epoch(model, dataloader, optimizer, scheduler, epoch, total_epochs):
    model.train()
    
    # Warmup consistency penalty
    lambda_consist = CONFIG['consistency_lambda_start'] + \
        (CONFIG['consistency_lambda_end'] - CONFIG['consistency_lambda_start']) * \
        (epoch / total_epochs)
    
    # Loss functions
    type_criterion = FocalLoss(alpha=type_class_weights, gamma=2.0)
    target_criterion = FocalLoss(alpha=target_class_weights, gamma=2.0)
    severity_criterion = FocalLoss(alpha=severity_class_weights, gamma=2.0)
    gen_criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad token
    consistency_criterion = ConsistencyPenaltyLoss()
    
    total_loss = 0
    for batch in dataloader:
        type_logits, target_logits, sev_logits, gen_logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            explanation_ids=batch.get('explanation_ids')
        )
        
        # Individual losses
        l_type = type_criterion(type_logits, batch['type_labels'])
        l_target = target_criterion(target_logits, batch['target_labels'])
        l_sev = severity_criterion(sev_logits, batch['severity_labels'])
        l_consist = consistency_criterion(type_logits, target_logits, sev_logits)
        
        loss = l_type + l_target + l_sev + lambda_consist * l_consist
        
        # Add generation loss only for samples that have explanations
        if gen_logits is not None and 'explanation_ids' in batch:
            l_gen = gen_criterion(
                gen_logits.view(-1, gen_logits.size(-1)),
                batch['explanation_labels'].view(-1)
            )
            loss += CONFIG['gen_weight'] * l_gen
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

### 5.3 Ablation Experiments (REQUIRED for the paper)

You MUST train and evaluate these variants:

| Experiment | L_type | L_target | L_sev | L_gen | L_consist | Purpose |
|:---|:---|:---|:---|:---|:---|:---|
| **Baseline (independent heads)** | ✅ | ✅ | ✅ | ❌ | ❌ | Reproduce BLP-2025 style |
| **+ Consistency only** | ✅ | ✅ | ✅ | ❌ | ✅ | Isolate consistency contribution |
| **+ Generation only** | ✅ | ✅ | ✅ | ✅ | ❌ | Isolate explanation contribution |
| **Full model** | ✅ | ✅ | ✅ | ✅ | ✅ | Your proposed method |

### 5.4 Deliverables for Phase 5
- [ ] Training script (runnable)
- [ ] Training logs for all 4 ablation variants
- [ ] Loss curves plotted
- [ ] Best model checkpoints saved

---

## Phase 6: Evaluation (Day 7-10)

### 6.1 Classification Metrics

```python
from sklearn.metrics import classification_report, f1_score

def evaluate_classification(model, test_loader):
    model.eval()
    all_type_preds, all_type_labels = [], []
    all_target_preds, all_target_labels = [], []
    all_sev_preds, all_sev_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            type_logits, target_logits, sev_logits, _ = model(
                batch['input_ids'], batch['attention_mask'])
            
            all_type_preds.extend(type_logits.argmax(-1).cpu().tolist())
            all_type_labels.extend(batch['type_labels'].cpu().tolist())
            all_target_preds.extend(target_logits.argmax(-1).cpu().tolist())
            all_target_labels.extend(batch['target_labels'].cpu().tolist())
            all_sev_preds.extend(sev_logits.argmax(-1).cpu().tolist())
            all_sev_labels.extend(batch['severity_labels'].cpu().tolist())
    
    results = {
        'type_f1_macro': f1_score(all_type_labels, all_type_preds, average='macro'),
        'target_f1_macro': f1_score(all_target_labels, all_target_preds, average='macro'),
        'severity_f1_macro': f1_score(all_sev_labels, all_sev_preds, average='macro'),
    }
    return results
```

### 6.2 Consistency Violation Rate (YOUR KEY METRIC)

```python
def compute_consistency_violation_rate(model, test_loader):
    """Measures % of predictions that are logically contradictory."""
    model.eval()
    total = 0
    violations = 0
    
    TYPE_NONE = 5
    TARGET_NONE = 4
    SEV_LITTLE = 0
    
    with torch.no_grad():
        for batch in test_loader:
            type_logits, target_logits, sev_logits, _ = model(
                batch['input_ids'], batch['attention_mask'])
            
            type_preds = type_logits.argmax(-1)
            target_preds = target_logits.argmax(-1)
            sev_preds = sev_logits.argmax(-1)
            
            for t, tgt, s in zip(type_preds, target_preds, sev_preds):
                total += 1
                # Check: If Type=None, then Target must be None AND Severity must be Little
                if t == TYPE_NONE and (tgt != TARGET_NONE or s != SEV_LITTLE):
                    violations += 1
                # Check: If Target/Severity is not None, Type should not be None
                if t != TYPE_NONE and tgt == TARGET_NONE and s == SEV_LITTLE:
                    violations += 1  # Optional: reverse contradiction
    
    violation_rate = violations / total * 100
    return violation_rate
```

**What to report in the paper:**

| Model | Type F1 | Target F1 | Severity F1 | Consistency Violation Rate ↓ |
|:---|:---|:---|:---|:---|
| Baseline (no consistency) | — | — | — | ~X% |
| + Consistency penalty | — | — | — | ~Y% (should be much lower) |
| Full model | — | — | — | ~Z% |

### 6.3 Faithfulness Evaluation (ERASER-Style Metrics)

```python
def compute_comprehensiveness(model, test_loader, tokenizer, top_k_tokens=5):
    """
    Comprehensiveness: How much does the prediction change when we
    REMOVE the tokens that the explanation says are important?
    
    Higher = more faithful (explanation identifies truly important tokens)
    """
    model.eval()
    comp_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Step 1: Get original prediction probabilities
            type_logits, _, _, _ = model(batch['input_ids'], batch['attention_mask'])
            original_probs = F.softmax(type_logits, dim=-1)
            original_pred = original_probs.max(dim=-1)
            
            # Step 2: Identify important tokens from the generated explanation
            # (Use attention weights or explicit token overlap)
            important_tokens = get_important_tokens_from_explanation(
                batch['input_ids'], batch['generated_explanation'], tokenizer, top_k=top_k_tokens
            )
            
            # Step 3: Mask those tokens and re-run
            masked_ids = mask_tokens(batch['input_ids'], important_tokens, tokenizer.mask_token_id)
            type_logits_masked, _, _, _ = model(masked_ids, batch['attention_mask'])
            masked_probs = F.softmax(type_logits_masked, dim=-1)
            masked_pred = masked_probs.max(dim=-1)
            
            # Step 4: Comprehensiveness = drop in confidence
            comp = (original_pred.values - masked_pred.values).mean().item()
            comp_scores.append(comp)
    
    return sum(comp_scores) / len(comp_scores)


def compute_sufficiency(model, test_loader, tokenizer, top_k_tokens=5):
    """
    Sufficiency: How well can the model maintain its prediction using
    ONLY the tokens the explanation says are important?
    
    Lower = more faithful (important tokens alone are sufficient)
    """
    model.eval()
    suff_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Original prediction
            type_logits, _, _, _ = model(batch['input_ids'], batch['attention_mask'])
            original_probs = F.softmax(type_logits, dim=-1)
            original_pred = original_probs.max(dim=-1)
            
            # Keep ONLY important tokens
            sufficient_ids = keep_only_tokens(
                batch['input_ids'], 
                get_important_tokens_from_explanation(
                    batch['input_ids'], batch['generated_explanation'], 
                    tokenizer, top_k=top_k_tokens
                ),
                tokenizer.pad_token_id
            )
            type_logits_suff, _, _, _ = model(sufficient_ids, batch['attention_mask'])
            suff_probs = F.softmax(type_logits_suff, dim=-1)
            suff_pred = suff_probs.max(dim=-1)
            
            # Sufficiency = how little the prediction drops
            suff = (original_pred.values - suff_pred.values).mean().item()
            suff_scores.append(suff)
    
    return sum(suff_scores) / len(suff_scores)


def compute_aopc(model, test_loader, tokenizer, perturbation_steps=[1,2,5,10,20]):
    """
    Area Over the Perturbation Curve (AOPC):
    Average change in prediction as you progressively remove tokens.
    """
    aopc_values = []
    
    for k in perturbation_steps:
        comp_at_k = compute_comprehensiveness(model, test_loader, tokenizer, top_k_tokens=k)
        aopc_values.append(comp_at_k)
    
    # AOPC = average comprehensiveness across perturbation levels
    aopc = sum(aopc_values) / len(aopc_values)
    return aopc, list(zip(perturbation_steps, aopc_values))
```

**What to report in the paper:**

| Metric | Baseline | + Consistency | + Generation | Full Model |
|:---|:---|:---|:---|:---|
| Comprehensiveness ↑ | — | — | — | — |
| Sufficiency ↓ | — | — | — | — |
| AOPC ↑ | — | — | — | — |

### 6.4 Deliverables for Phase 6
- [ ] Classification results table (F1 per task)
- [ ] Consistency violation rate comparison
- [ ] Comprehensiveness, Sufficiency, AOPC scores
- [ ] Perturbation curve figure
- [ ] Confusion matrices (3 tasks)

---

## Phase 7: Paper Writing & Submission (Day 9-12)

### 7.1 Paper Structure (IEEE 6-page format)

| Section | Pages | Content |
|:---|:---|:---|
| Abstract | 0.3 | Update with actual numbers |
| Introduction | 0.8 | Keep current, add result highlights |
| Related Work | 0.8 | Keep current, minor edits |
| Methodology | 1.5 | Add formal L_consist equation, architecture figure |
| Experiments | 1.5 | Dataset stats, experimental setup, results tables |
| Results & Discussion | 0.8 | Analysis, ablation, faithfulness results |
| Conclusion | 0.3 | Summary + limitations |

### 7.2 Figures & Tables Needed

| Figure/Table | Description | Priority |
|:---|:---|:---|
| **Figure 1** | Architecture diagram (draw in draw.io or LaTeX tikz) | 🔴 Critical |
| **Figure 2** | Class distribution of BanglaMultiHate | 🟡 Important |
| **Figure 3** | Perturbation curve (AOPC) | 🟡 Important |
| **Table 1** | Dataset statistics | 🔴 Critical |
| **Table 2** | Main results (F1 + consistency rate) | 🔴 Critical |
| **Table 3** | Ablation study | 🔴 Critical |
| **Table 4** | Faithfulness metrics | 🔴 Critical |
| **Table 5** | Silver-standard quality (human eval) | 🟡 Important |
| **Table 6** | Qualitative examples | 🟢 Nice-to-have |

### 7.3 Key LaTeX Additions Needed in [main.tex](file:///d:/Study/Publications/banglaHate/main.tex)

```latex
% Missing sections to add:
\section{Experimental Setup}
\subsection{Dataset Statistics}
\subsection{Implementation Details}
\subsection{Baselines}

\section{Results and Discussion}
\subsection{Classification Performance}
\subsection{Consistency Analysis}
\subsection{Faithfulness Evaluation}
\subsection{Ablation Study}
\subsection{Qualitative Analysis}

\section{Limitations}

\section{Conclusion}
```

---

## 📋 Resource Requirements

| Resource | Minimum | Recommended |
|:---|:---|:---|
| **GPU** | Google Colab free (T4) | Colab Pro (A100) or Kaggle P100 |
| **Gemini API** | Free tier (15 RPM) | Paid tier for 5K+ generations |
| **Storage** | 10 GB | 50 GB |
| **Human annotators** | 2 people (for silver quality check) | 3 people (for custom dataset) |
| **Time** | 10 days (ICCIT rush) | 6-8 weeks (comfortable) |

---

## 🎯 Alternative Venues (If ICCIT 2026 Aug 31 is too tight)

| Venue | Typical Deadline | Tier |
|:---|:---|:---|
| **ECCE 2026** (CUET, Bangladesh) | ~Sep 2026 | Regional IEEE |
| **ICBSLP 2026** (Bangla Speech/Language) | ~Oct 2026 | Domain-specific |
| **BLP Workshop 2027** (co-located with ACL/EMNLP) | ~Feb 2027 | Top NLP workshop |
| **LREC-COLING 2027** | ~Oct 2026 | Tier-A international |
| **IEEE Access** (journal) | Rolling | Q1 journal |

---

## ✅ Master Checklist

### Week 1 (Day 1-6)
- [ ] Download BanglaMultiHate + BanHate datasets
- [ ] Run EDA and class distribution analysis
- [ ] Generate 5,000 silver-standard explanations via Gemini
- [ ] Validate 200 explanations (human eval)
- [ ] Implement full model architecture in PyTorch
- [ ] Implement ConsistencyPenaltyLoss
- [ ] Implement FocalLoss

### Week 2 (Day 7-12)
- [ ] Train baseline model (no consistency, no generation)
- [ ] Train ablation variants
- [ ] Train full model
- [ ] Evaluate: F1, consistency violation rate
- [ ] Evaluate: Comprehensiveness, Sufficiency, AOPC
- [ ] Create all figures and tables
- [ ] Write missing paper sections (Experiments, Results, Conclusion)
- [ ] Final paper formatting and proofreading
- [ ] Submit to ICCIT 2026 by August 31
