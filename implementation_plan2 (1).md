# 📅 Revised 6-Week Research Execution Plan
## Target: ~October 1, 2026 Submission

---

## Venue Strategy

| Venue | Deadline | Conference Date | Fit | Status |
|:---|:---|:---|:---|:---|
| **ICCIT 2026** (Primary) | Aug 31, 2026 (extended) | Dec 18-20, Cox's Bazar | ✅ Perfect | ⚡ Tight — only 12 days, but ICCIT often extends again |
| **ICCIT 2026** (if further extended) | ~Sep 15-30? | Dec 18-20 | ✅ Perfect | Past years show multiple extensions |
| **IEEE Access** (Journal — Backup) | Rolling | — | ✅ Good | Longer paper, higher impact, no deadline pressure |
| **EMNLP 2027 Workshop / BLP 2027** | ~Feb 2027 | ~Apr 2027 | ✅ Best fit | Top NLP venue, but longer wait |

> [!IMPORTANT]
> **My recommendation**: Target **ICCIT 2026** as primary (it commonly extends deadlines — watch for September extension). Use your 6 weeks to do thorough work. If ICCIT doesn't extend, submit to **IEEE Access** (journal) or **BLP Workshop 2027**. Either way, the research quality stays the same.

---

## 6-Week Calendar Overview

```
Week 1 (Aug 19-25)  ── Dataset Acquisition + EDA + Environment Setup
Week 2 (Aug 26-Sep 1) ── Silver-Standard Explanation Generation + Custom Dataset Start
Week 3 (Sep 2-8)    ── Model Architecture Implementation + Custom Dataset Annotation
Week 4 (Sep 9-15)   ── Training: Ablations + LLM Baselines (TigerLLM, Llama, Gemma)
Week 5 (Sep 16-22)  ── Evaluation (Classification + Consistency + Faithfulness)
Week 6 (Sep 23-30)  ── Paper Writing + Figures + Proofreading + Submission
```

---

## Week 1: Foundation (Aug 19-25)

### Day 1-2: Environment Setup

```bash
# Create project structure
mkdir -p banglaHate/{data,src,models,figures,results,notebooks}

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install google-generativeai
pip install scikit-learn pandas matplotlib seaborn
pip install git+https://github.com/csebuetnlp/normalizer
pip install sentencepiece  # required by BanglaT5 tokenizer
pip install label-studio  # for custom dataset annotation
```

**GPU Strategy:**

| Option | Cost | VRAM | Recommended For |
|:---|:---|:---|:---|
| Google Colab Free | Free | 15GB (T4) | EDA, small experiments |
| Google Colab Pro | $12/mo | 40GB (A100) | Training BanglaBERT MTL |
| Kaggle Notebooks | Free | 16GB (P100), 30h/week | Training (good free option) |
| Your local GPU (if any) | Free | Varies | Development & debugging |

### Day 2-3: Dataset Download & EDA

```python
# === download_datasets.py ===
from datasets import load_dataset
import pandas as pd
import json

# 1. BanglaMultiHate (~50K samples)
bangla_multi = load_dataset("aridhasan/BanglaMultiHate")
print("BanglaMultiHate:")
print(f"  Train: {len(bangla_multi['train'])}")
print(f"  Dev:   {len(bangla_multi['dev'])}")
print(f"  Test:  {len(bangla_multi['test'])}")
print(f"  Columns: {bangla_multi['train'].column_names}")

# 2. BanHate/BanHADEX (~19K samples)
banhate = load_dataset("aplycaebous/BanHate")
print("\nBanHate:")
print(f"  Splits: {list(banhate.keys())}")
print(f"  Columns: {banhate['train'].column_names}")

# Save locally as CSV for easier inspection
bangla_multi['train'].to_pandas().to_csv('data/banglamultihate_train.csv', index=False)
banhate['train'].to_pandas().to_csv('data/banhate_train.csv', index=False)
```

### Day 3-4: Comprehensive EDA

```python
# === eda.py ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

df = pd.read_csv('data/banglamultihate_train.csv')

# ──────────────────────────────────────────
# 1. Class Distributions (Figure for paper)
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Verified column names from HuggingFace dataset schema
type_col = 'type_of_hate'
target_col = 'target_of_hate'
sev_col = 'severity_of_hate'

for ax, col, title in zip(axes, [type_col, target_col, sev_col],
                           ['Hate Type', 'Target', 'Severity']):
    counts = df[col].value_counts()
    counts.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", len(counts)))
    ax.set_title(f'{title} Distribution', fontsize=14)
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved class_distribution.png")

# ──────────────────────────────────────────
# 2. Co-occurrence Heatmaps
# ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ct1 = pd.crosstab(df[type_col], df[sev_col])
sns.heatmap(ct1, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0])
axes[0].set_title('Hate Type × Severity')

ct2 = pd.crosstab(df[type_col], df[target_col])
sns.heatmap(ct2, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1])
axes[1].set_title('Hate Type × Target')

plt.tight_layout()
plt.savefig('figures/co_occurrence_heatmaps.png', dpi=300, bbox_inches='tight')
print("✅ Saved co_occurrence_heatmaps.png")

# ──────────────────────────────────────────
# 3. Consistency Violation Analysis in Ground Truth
# ──────────────────────────────────────────
# Rule: If Type=None → Target=None AND Severity=Little to None
none_type = df[df[type_col] == 'None']
violations_gt = none_type[
    (none_type[target_col] != 'None') | (none_type[sev_col] != 'Little to None')
]
print(f"\n📊 Ground Truth Consistency Analysis:")
print(f"   Total samples: {len(df)}")
print(f"   Type=None samples: {len(none_type)}")
print(f"   Violations in ground truth: {len(violations_gt)} ({100*len(violations_gt)/len(df):.2f}%)")

# ──────────────────────────────────────────
# 4. Text Length Analysis
# ──────────────────────────────────────────
text_col = 'comment'  # verified: column name is 'comment' not 'text'
df['text_length'] = df[text_col].str.len()
df['word_count'] = df[text_col].str.split().str.len()

print(f"\n📊 Text Statistics:")
print(f"   Mean length: {df['text_length'].mean():.0f} chars, {df['word_count'].mean():.0f} words")
print(f"   Median length: {df['text_length'].median():.0f} chars")
print(f"   Max length: {df['text_length'].max()} chars")
print(f"   95th percentile: {df['text_length'].quantile(0.95):.0f} chars")

# ──────────────────────────────────────────
# 5. Compare with BanHADEX schema
# ──────────────────────────────────────────
df_bh = pd.read_csv('data/banhate_train.csv')
print(f"\n📊 Schema Comparison:")
print(f"   BanglaMultiHate columns: {list(df.columns)}")
print(f"   BanHate columns: {list(df_bh.columns)}")
```

### Day 5: Taxonomy Gap Documentation

After running EDA, create this precise comparison table for the paper:

```python
# === taxonomy_comparison.py ===
# Run this AFTER inspecting both datasets
# Document the EXACT column/label differences

taxonomy_report = """
TAXONOMY ALIGNMENT REPORT
=========================

BanglaMultiHate Labels (verified from HuggingFace schema):
  - type_of_hate:     [Abusive, Sexism, Religious Hate, Political Hate, Profane, None]
  - target_of_hate:   [Individual, Organizations, Communities, Society, None]
  - severity_of_hate: [Little to None, Mild, Severe]

  Dataset columns: id, comment, category, subcategory, type_of_hate, severity_of_hate, target_of_hate
  Note: text column is 'comment' (NOT 'text'). Splits are: train, dev, test.

BanHADEX Labels:
  - hate_type: [Political, Religious, Gender, Personal Offense, Abusive/Violence, Origin, Body Shaming]
  - target:    [7 groups - check exact names from dataset]
  - severity:  NOT PRESENT
  - explanation: PRESENT (human-written Bengali text)

CRITICAL GAPS:
  1. BanHADEX has NO severity dimension → Cannot provide severity-conditioned explanations
  2. Hate type taxonomies differ (6 vs 7 categories with different names)
  3. Target taxonomies differ in granularity
  
JUSTIFICATION FOR SILVER-STANDARD BRIDGING:
  We need explanations that are conditioned on the BanglaMultiHate 3-task taxonomy
  (Type + Target + Severity). BanHADEX explanations cannot serve this purpose because:
  (a) They lack severity conditioning
  (b) They use a different categorical schema
  Therefore, LLM-generated silver-standard explanations are necessary.
"""
print(taxonomy_report)
```

### Day 6-7: Mandatory Pipeline Smoke Test (Sanity Check)

> [!IMPORTANT]
> **Senior's Advice**: Always run a 5-minute Smoke Test on 20 dummy samples before generating 5,000 explanations or running full 15-epoch fine-tuning loops. This prevents wasting hours of GPU time, API credits, or Colab crashes due to silent tensor shape mismatches, loss NaNs, or evaluation script bugs.

```python
# === smoke_test.py ===
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# 1. Create a tiny 10-sample dummy dataset
dummy_data = {
    'text': [
        "এই মন্তব্যটি খুবই খারাপ এবং ঘৃণাত্মক।",
        "আজকের দিনটি চমৎকার।",
        "ঐ বিশেষ ধর্মীয় দলকে দেশ থেকে বের করে দেওয়া উচিত।",
        "মেয়েরা কেন রাজনীতিতে নামবে, ওদের ঘরে থাকা উচিত।",
        "খেলাটি অনেক সুন্দর হয়েছে।"
    ] * 2,
    'hate_type': ['Abusive', 'None', 'Religious Hate', 'Sexism', 'None'] * 2,
    'target': ['Individual', 'None', 'Community', 'Individual', 'None'] * 2,
    'severity': ['Mild', 'Little to None', 'Severe', 'Mild', 'Little to None'] * 2,
    'silver_explanation': [
        "এই মন্তব্যটিতে গালিগালাজ রয়েছে।",
        "এটি একটি সাধারণ সুন্দর মন্তব্য।",
        "এটি ধর্মীয় সম্প্রদায়কে আক্রমণ করছে।",
        "এটি নারীর বিরুদ্ধে বৈষম্যমূলক বক্তব্য।",
        "এটি সাধারণ খেলার মন্তব্য।"
    ] * 2
}

df_dummy = pd.DataFrame(dummy_data)
df_dummy.to_csv('data/dummy_smoke_test.csv', index=False)
print("🧪 Step 1: Dummy dataset saved to data/dummy_smoke_test.csv")

# 2. Test Dataset Loader & Tokenizer
from src.dataset import BanglaHateDataset
dataset = BanglaHateDataset('data/dummy_smoke_test.csv')
loader = DataLoader(dataset, batch_size=2)
print(f"🧪 Step 2: Dataset loaded successfully! Samples: {len(dataset)}")
sample_batch = next(iter(loader))
print(f"   Input IDs shape: {sample_batch['input_ids'].shape}")

# 3. Test Model Forward Pass & Multi-Loss Calculation
from src.model import ConsistencyConstrainedMTL
from src.losses import FocalLoss, ConsistencyPenaltyLoss

model = ConsistencyConstrainedMTL(encoder_name='csebuetnlp/banglabert')
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
consist_loss_fn = ConsistencyPenaltyLoss()

type_logits, target_logits, sev_logits, gen_logits = model(
    sample_batch['input_ids'], 
    sample_batch['attention_mask']
)
print(f"🧪 Step 3: Forward pass successful!")
print(f"   Type Logits Shape: {type_logits.shape}")

loss_consist = consist_loss_fn(type_logits, target_logits, sev_logits)
print(f"   Consistency Penalty Loss: {loss_consist.item():.4f}")

# 4. Overfitting Sanity Check (5 Epochs on 10 samples)
print("🧪 Step 4: Testing 5-epoch Overfitting Sanity Check...")
for epoch in range(5):
    optimizer.zero_grad()
    t_out, tg_out, s_out, _ = model(sample_batch['input_ids'], sample_batch['attention_mask'])
    l_c = consist_loss_fn(t_out, tg_out, s_out)
    l_c.backward()
    optimizer.step()
    print(f"   Epoch {epoch+1}/5 - Loss: {l_c.item():.4f}")

print("✅ SMOKE TEST PASSED! The pipeline is ready for full dataset training.")
```

### Week 1 Deliverables Checklist
- [ ] Environment setup complete (Python, PyTorch, HuggingFace)
- [ ] Both datasets downloaded and locally saved
- [ ] EDA notebook with all visualizations
- [ ] Class distribution figure (for paper Figure 2)
- [ ] Co-occurrence heatmaps (for paper)
- [ ] Taxonomy gap report documented
- [ ] Text length statistics
- [ ] Ground truth consistency violation count
- [ ] Pipeline smoke test script (smoke_test.py) executed and passed on dummy samples

---

## Week 2: Data Preparation (Aug 26 - Sep 1)

### Day 1-3: Silver-Standard Explanation Generation

```python
# === generate_silver_explanations.py ===
import google.generativeai as genai
import json
import time
from tqdm import tqdm
import pandas as pd

genai.configure(api_key="YOUR_GEMINI_API_KEY")

SYSTEM_PROMPT = """You are an expert Bengali linguist and hate speech analyst.
Given a Bengali social media comment and its ground-truth classification labels, 
generate a concise explanation (2-3 sentences) in Bengali that justifies WHY 
the comment received those exact labels.

Rules:
1. Reference specific words or phrases from the comment.
2. Logically connect the comment to ALL three labels (Type, Target, Severity).
3. If Type=None, explain why the comment is NOT hateful.
4. Write entirely in Bengali script (বাংলা).
5. Be precise — avoid generic explanations."""

# 3 diverse few-shot examples (include hateful, non-hateful, and edge case)
FEW_SHOT = """
<example>
Comment: "এই হিন্দুগুলোকে দেশ থেকে তাড়িয়ে দেওয়া উচিত"
Labels: Type=Religious Hate, Target=Community, Severity=Severe
Explanation: এই মন্তব্যটি ধর্মীয় ঘৃণা প্রকাশ করে কারণ এটি "হিন্দু" সম্প্রদায়কে সরাসরি আক্রমণ করছে। "তাড়িয়ে দেওয়া উচিত" বাক্যাংশটি সহিংস বহিষ্কারের আহ্বান, যা তীব্রতায় Severe। লক্ষ্যবস্তু একটি ধর্মীয় সম্প্রদায় (Community)।
</example>

<example>
Comment: "আজকের আবহাওয়া খুব ভালো ছিল"
Labels: Type=None, Target=None, Severity=Little to None
Explanation: এই মন্তব্যটি আবহাওয়া সম্পর্কে একটি নিরপেক্ষ পর্যবেক্ষণ। কোনো ব্যক্তি বা গোষ্ঠীর বিরুদ্ধে আক্রমণাত্মক বা ঘৃণামূলক কোনো বক্তব্য নেই, তাই এটি ঘৃণামূলক বক্তব্য নয়।
</example>

<example>
Comment: "এই মেয়েটার কোনো লজ্জা নাই, সারাদিন ফেসবুকে ছবি দেয়"
Labels: Type=Sexism, Target=Individual, Severity=Mild
Explanation: মন্তব্যটি একজন নির্দিষ্ট মহিলাকে লক্ষ্য করে যৌনতাবাদী মনোভাব প্রকাশ করে। "লজ্জা নাই" বলে নারীর চরিত্র নিয়ে প্রশ্ন তোলা হচ্ছে, যা সামাজিক রক্ষণশীল যৌনতাবাদের প্রতিফলন। তবে সরাসরি সহিংসতার আহ্বান না থাকায় তীব্রতা Mild।
</example>
"""

model = genai.GenerativeModel('gemini-1.5-pro')

def generate_one_explanation(comment, type_of_hate, target_of_hate, severity_of_hate, max_retries=3):
    prompt = f"""{SYSTEM_PROMPT}

{FEW_SHOT}

Now generate an explanation for this new sample:
Comment: "{comment}"
Labels: Type={type_of_hate}, Target={target_of_hate}, Severity={severity_of_hate}
Explanation:"""
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                    top_p=0.9
                )
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # exponential backoff
            else:
                return None

def batch_generate(dataset_path, output_path, max_samples=5000):
    df = pd.read_csv(dataset_path)
    
    # Sample strategically: ensure all classes are represented
    # Over-sample minority classes for better explanation coverage
    sampled = df.groupby('type_of_hate', group_keys=False).apply(
        lambda x: x.sample(min(len(x), max_samples // 6), random_state=42)
    )
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    
    results = []
    for idx, row in tqdm(sampled.iterrows(), total=len(sampled)):
        explanation = generate_one_explanation(
            comment=row['comment'],
            type_of_hate=row['type_of_hate'],
            target_of_hate=row['target_of_hate'],
            severity_of_hate=row['severity_of_hate']
        )
        
        results.append({
            'original_idx': idx,
            'comment': row['comment'],
            'type_of_hate': row['type_of_hate'],
            'target_of_hate': row['target_of_hate'],
            'severity_of_hate': row['severity_of_hate'],
            'silver_explanation': explanation
        })
        
        # Save checkpoint every 100 samples
        if (len(results)) % 100 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Checkpoint: {len(results)} samples saved")
        
        # Rate limiting
        time.sleep(0.5)  # Adjust based on your API tier
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    success_count = sum(1 for r in results if r['silver_explanation'] is not None)
    print(f"\n✅ Done! {success_count}/{len(results)} explanations generated successfully")
    return results

# Run it
results = batch_generate(
    dataset_path='data/banglamultihate_train.csv',
    output_path='data/silver_explanations.json',
    max_samples=5000
)
```

> [!TIP]
> **Cost estimate for Gemini API:**
> - Gemini 1.5 Pro: ~$0.00125 per 1K input tokens + $0.005 per 1K output tokens
> - 5,000 samples × ~200 tokens avg per call ≈ $5-10 total
> - Gemini 2.0 Flash is 10x cheaper if budget is tight

### Day 4-5: Human Quality Validation

```python
# === validate_silver_quality.py ===
import json
import random

# Load generated explanations
with open('data/silver_explanations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Random sample 200 for human evaluation
successful = [d for d in data if d['silver_explanation'] is not None]
eval_sample = random.sample(successful, min(200, len(successful)))

# Create evaluation spreadsheet
eval_rows = []
for i, sample in enumerate(eval_sample):
    eval_rows.append({
        'eval_id': i + 1,
        'text': sample['text'],
        'hate_type': sample['hate_type'],
        'target': sample['target'],
        'severity': sample['severity'],
        'silver_explanation': sample['silver_explanation'],
        # Human annotator fills these:
        'correctness_1to5': '',       # Does explanation match labels?
        'relevance_1to5': '',         # Does it cite correct words?
        'completeness_1to5': '',      # Does it cover all 3 dimensions?
        'annotator_name': ''
    })

import pandas as pd
pd.DataFrame(eval_rows).to_csv('data/human_eval_form.csv', index=False)
print("✅ Saved human_eval_form.csv — distribute to 2 annotators")
```

**Annotation instructions for your 2 annotators:**
1. Read the Bengali comment
2. Read the silver-standard explanation
3. Rate on 3 scales (1=terrible, 5=perfect):
   - **Correctness**: Does the explanation accurately describe why the labels were assigned?
   - **Relevance**: Does the explanation reference the right specific words/phrases?
   - **Completeness**: Does the explanation address all three dimensions (Type, Target, Severity)?
4. Each annotator evaluates all 200 samples independently
5. Compute Cohen's Kappa for inter-annotator agreement

### Day 5-7: Start Custom Dataset Annotation

```
SOURCE COLLECTION PLAN:
━━━━━━━━━━━━━━━━━━━━━━
Platform: YouTube (Bengali news channels)
Channels to scrape:
  - Somoy TV
  - Jamuna TV  
  - Independent TV
  - Political commentary channels

Tool: youtube-comment-downloader
  pip install youtube-comment-downloader
  youtube-comment-downloader --url "VIDEO_URL" --output comments.json

Target: 600 raw comments → filter to 500 clean samples
Goal: Cover all label combinations, especially edge cases
```

**Annotation tool**: Use Google Sheets shared between your 2-3 annotators, with these columns:

| Column | Type | Description |
|:---|:---|:---|
| comment_id | int | Unique ID |
| text | string | Raw Bengali comment |
| type_of_hate | dropdown | Abusive / Sexism / Religious Hate / Political Hate / Profane / None |
| target_of_hate | dropdown | Individual / Organizations / Communities / Society / None |
| severity_of_hate | dropdown | Little to None / Mild / Severe |
| explanation | text | 2-3 sentence Bengali explanation |
| annotator | string | Annotator name |
| confidence | 1-5 | How confident are you? |

### Week 2 Deliverables
- [ ] 5,000 silver-standard explanations generated and saved
- [ ] Human evaluation form distributed to 2 annotators
- [ ] YouTube comment collection started (600 raw comments)
- [ ] Custom annotation Google Sheet set up and shared

---

## Week 3: Model Implementation (Sep 2-8)

### Day 1-2: Core Model Architecture

Implement the full model from the plan. The key files to create:

```
src/
├── model.py              # ConsistencyConstrainedMTL class
├── losses.py             # FocalLoss + ConsistencyPenaltyLoss
├── dataset.py            # Custom PyTorch Dataset class
├── train.py              # Training loop
├── evaluate.py           # Evaluation metrics
├── faithfulness.py       # ERASER metrics (Comp, Suff, AOPC)
└── utils.py              # Text normalization, label encoding
```

**Critical file — `src/dataset.py`:**

```python
# === src/dataset.py ===
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from normalizer import normalize
import json

class BanglaHateDataset(Dataset):
    """Unified dataset for classification + explanation generation."""
    
    # Label mappings (adjust after EDA confirms exact names)
    TYPE_LABELS = ['Abusive', 'Sexism', 'Religious Hate', 'Political Hate', 'Profane', 'None']
    TARGET_LABELS = ['Individual', 'Organizations', 'Communities', 'Society', 'None']
    SEVERITY_LABELS = ['Little to None', 'Mild', 'Severe']
    
    def __init__(self, data_path, tokenizer_name='csebuetnlp/banglabert',
                 max_length=256, max_explanation_length=128,
                 silver_explanations_path=None):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.max_explanation_length = max_explanation_length
        
        # Load main data
        import pandas as pd
        self.df = pd.read_csv(data_path)
        
        # Load silver explanations if available
        self.explanations = {}
        if silver_explanations_path:
            with open(silver_explanations_path, 'r', encoding='utf-8') as f:
                silver = json.load(f)
                for item in silver:
                    if item['silver_explanation']:
                        self.explanations[item['original_idx']] = item['silver_explanation']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Normalize Bengali text
        text = normalize(str(row['comment']))
        
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'type_label': self.TYPE_LABELS.index(row['type_of_hate']),
            'target_label': self.TARGET_LABELS.index(row['target_of_hate']),
            'severity_label': self.SEVERITY_LABELS.index(row['severity_of_hate']),
        }
        
        # Add explanation if available
        if idx in self.explanations:
            expl_encoding = self.tokenizer(
                normalize(self.explanations[idx]),
                max_length=self.max_explanation_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            item['explanation_ids'] = expl_encoding['input_ids'].squeeze()
            item['explanation_mask'] = expl_encoding['attention_mask'].squeeze()
            # Shift labels for autoregressive generation
            item['explanation_labels'] = item['explanation_ids'].clone()
            item['explanation_labels'][item['explanation_mask'] == 0] = -100  # ignore padding
        
        return item
```

### Day 3-4: Implement Consistency Penalty (Full Mathematical Version)

```python
# === src/losses.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyPenaltyLoss(nn.Module):
    """
    Soft constraint loss that penalizes logically contradictory 
    multi-task predictions.
    
    Based on BanglaMultiHate annotation schema:
      - If hate_type = "None" → target MUST be "None" 
                              AND severity MUST be "Little to None"
      - Contrapositive: If target ≠ "None" OR severity ≠ "Little to None" 
                       → hate_type MUST NOT be "None"
    """
    
    def __init__(self, type_none_idx=5, target_none_idx=4, 
                 sev_littletonone_idx=0, sev_severe_idx=2):
        super().__init__()
        self.type_none_idx = type_none_idx
        self.target_none_idx = target_none_idx
        self.sev_littletonone_idx = sev_littletonone_idx
        self.sev_severe_idx = sev_severe_idx
    
    def forward(self, type_logits, target_logits, severity_logits):
        # Soft probabilities
        p_type = F.softmax(type_logits, dim=-1)
        p_target = F.softmax(target_logits, dim=-1)
        p_sev = F.softmax(severity_logits, dim=-1)
        
        p_type_none = p_type[:, self.type_none_idx]
        p_type_has = 1.0 - p_type_none  # P(type IS hateful)
        p_target_none = p_target[:, self.target_none_idx]
        p_target_has = 1.0 - p_target_none
        p_sev_low = p_sev[:, self.sev_littletonone_idx]
        p_sev_has = 1.0 - p_sev_low
        p_sev_severe = p_sev[:, self.sev_severe_idx]
        
        # Violation 1: P(type=None) × P(target≠None)
        v1 = p_type_none * p_target_has
        
        # Violation 2: P(type=None) × P(severity≠Little)  
        v2 = p_type_none * p_sev_has
        
        # Violation 3: P(type=None) × P(severity=Severe) [extra penalty]
        v3 = p_type_none * p_sev_severe
        
        # Violation 4 (REVERSE): P(type≠None) × P(target=None) × P(severity=Little)
        # If something IS hateful, it should have a target and non-trivial severity
        v4 = p_type_has * p_target_none * p_sev_low
        
        # Aggregate: mean over batch
        penalty = (v1 + v2 + 2.0 * v3 + v4).mean()
        
        return penalty


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()
```

### Day 5-6: Complete Custom Dataset Annotation

- Collect remaining YouTube comments
- Complete annotations with both annotators
- Resolve disagreements with 3rd annotator
- Compute Fleiss' Kappa

### Day 7: Collect Human Evaluation Results for Silver Explanations

```python
# === analyze_human_eval.py ===
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load both annotators' evaluations
eval_a1 = pd.read_csv('data/human_eval_annotator1.csv')
eval_a2 = pd.read_csv('data/human_eval_annotator2.csv')

# Compute inter-annotator agreement
for metric in ['correctness_1to5', 'relevance_1to5', 'completeness_1to5']:
    kappa = cohen_kappa_score(eval_a1[metric], eval_a2[metric], weights='quadratic')
    avg = (eval_a1[metric].mean() + eval_a2[metric].mean()) / 2
    print(f"{metric}: Avg={avg:.2f}, Cohen's κ={kappa:.3f}")

# Overall quality summary for paper
print(f"\nSilver-Standard Quality Summary:")
print(f"  Average Correctness: {(eval_a1['correctness_1to5'].mean() + eval_a2['correctness_1to5'].mean())/2:.2f}/5")
print(f"  Average Relevance:   {(eval_a1['relevance_1to5'].mean() + eval_a2['relevance_1to5'].mean())/2:.2f}/5")
print(f"  Average Completeness:{(eval_a1['completeness_1to5'].mean() + eval_a2['completeness_1to5'].mean())/2:.2f}/5")
```

### Week 3 Deliverables
- [ ] `src/model.py` — complete model architecture
- [ ] `src/losses.py` — FocalLoss + ConsistencyPenaltyLoss
- [ ] `src/dataset.py` — data loader with explanation support
- [ ] Custom dataset: 500 annotated samples with inter-annotator agreement
- [ ] Human evaluation results for silver explanations
- [ ] Cohen's Kappa / Fleiss' Kappa computed

---

## Week 4: Training + LLM Benchmarking (Sep 9-15)

### Why Benchmark Bengali LLMs?

> [!IMPORTANT]
> Including LLM baselines is **essential** for two reasons:
> 1. **BLP-2025 winners used LLMs** — HateSense used Llama-3.1 + Gemma-2, Code_Gen used BanglaBERT + MuRIL ensembles. Reviewers will ask "why not just use an LLM?"
> 2. **Your key argument is about consistency** — If TigerLLM gets 90% F1 but has a 15% consistency violation rate, and your model gets 88% F1 with only 3% violations, **you win the argument**.

### Bengali LLM Landscape (Available Models)

| Model | Type | Size | HuggingFace ID | Why Include |
|:---|:---|:---|:---|:---|
| **BanglaBERT** | Encoder (ELECTRA) | 110M | `csebuetnlp/banglabert` | Your base + strongest monolingual encoder |
| **MuRIL** | Encoder (BERT) | 237M | `google/muril-base-cased` | BLP-2025 winner (Code_Gen) used this |
| **TigerLLM-1B-it** | Decoder LLM | 1B | `md-nishat-008/TigerLLM-1B-it` | Bengali-specific LLM (ACL 2025) |
| **Llama-3.2-3B** | Decoder LLM | 3B | `meta-llama/Llama-3.2-3B-Instruct` | Used by BLP-2025 HateSense system |
| **Gemma-2-2B** | Decoder LLM | 2B | `google/gemma-2-2b-it` | Used by BLP-2025 HateSense system |
| **BanglaLLM** | Decoder LLM | ~3B | `BanglaLLM/bangla-llama-3.2-3B-instruct` | Community Bengali Llama variant |

### Experiment Matrix (7 Experiments — 3 LLM Baselines + 4 Ablations)

#### Group A: LLM Baselines (Zero-Shot & Few-Shot — No Training Needed)

| Experiment ID | Model | Method | GPU Cost | Purpose |
|:---|:---|:---|:---|:---|
| `llm1_tiger_zeroshot` | TigerLLM-1B-it | Zero-shot prompting | Low (~1h) | Bengali-specific LLM baseline |
| `llm2_llama_fewshot` | Llama-3.2-3B-Instruct | 3-shot prompting | Low (~2h) | Strong multilingual LLM baseline |
| `llm3_gemma_fewshot` | Gemma-2-2B-it | 3-shot prompting | Low (~1h) | BLP-2025 competitor model |

#### Group B: Encoder Baselines + Your Ablations (Fine-Tuning Required)

| Experiment ID | Config | Purpose |
|:---|:---|:---|
| `exp1_baseline` | BanglaBERT + 3 MTL heads (no L_consist, no L_gen) | Encoder MTL baseline (BLP-2025 style) |
| `exp2_muril` | MuRIL + 3 MTL heads (no L_consist, no L_gen) | Multilingual encoder baseline |
| `exp3_consistency` | BanglaBERT + 3 heads + L_consist | Isolate consistency contribution |
| `exp4_generative` | BanglaBERT + 3 heads + gen head + L_gen | Isolate explanation contribution |
| `exp5_full` | BanglaBERT + 3 heads + gen head + L_consist + L_gen | **Your full proposed model** |

#### Group C: Optional (If Time Permits)

| Experiment ID | Config | Purpose |
|:---|:---|:---|
| `exp6_tiger_lora` | TigerLLM-1B + LoRA fine-tune for MTL | Best Bengali LLM with fine-tuning |
| `exp7_llama_lora` | Llama-3.2-3B + LoRA fine-tune for MTL | Best multilingual LLM with fine-tuning |

### Day 1-2: LLM Zero-Shot / Few-Shot Evaluation (Fast — No Training)

```python
# === src/evaluate_llm_baselines.py ===
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ════════════════════════════════════════════════════════════
# LLM Prompt Template for Multi-Task Hate Speech Classification
# ════════════════════════════════════════════════════════════

ZERO_SHOT_PROMPT = """You are a Bengali hate speech classifier. Given a Bengali comment, classify it into three dimensions.

Hate Type: One of [Abusive, Sexism, Religious Hate, Political Hate, Profane, None]
Target: One of [Individual, Organizations, Communities, Society, None]
Severity: One of [Little to None, Mild, Severe]

Rules:
- If Hate Type is "None", then Target MUST be "None" and Severity MUST be "Little to None".
- Respond ONLY with valid JSON.

Comment: "{comment}"

Output:"""

FEW_SHOT_PROMPT = """You are a Bengali hate speech classifier. Given a Bengali comment, classify it into three dimensions.

Hate Type: One of [Abusive, Sexism, Religious Hate, Political Hate, Profane, None]
Target: One of [Individual, Organizations, Communities, Society, None]
Severity: One of [Little to None, Mild, Severe]

Rules:
- If Hate Type is "None", then Target MUST be "None" and Severity MUST be "Little to None".

Example 1:
Comment: "এই হিন্দুগুলোকে দেশ থেকে তাড়িয়ে দেওয়া উচিত"
Output: {{"type_of_hate": "Religious Hate", "target_of_hate": "Communities", "severity_of_hate": "Severe"}}

Example 2:
Comment: "আজকের আবহাওয়া খুব সুন্দর"
Output: {{"type_of_hate": "None", "target_of_hate": "None", "severity_of_hate": "Little to None"}}

Example 3:
Comment: "এই নেতা শুধু চুরি করে, দেশটা শেষ করে দিচ্ছে"
Output: {{"type_of_hate": "Political Hate", "target_of_hate": "Individual", "severity_of_hate": "Mild"}}

Now classify:
Comment: "{comment}"
Output:"""

def evaluate_llm(model_name, test_data, prompt_template, max_samples=500):
    """Evaluate an LLM on multi-task hate speech classification."""
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = []
    
    for i, sample in enumerate(tqdm(test_data[:max_samples], desc=f"Evaluating {model_name}")):
        prompt = prompt_template.format(comment=sample['comment'])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse JSON from LLM response
        parsed = parse_llm_output(response)
        
        results.append({
            'comment': sample['comment'],
            'true_type': sample['type_of_hate'],
            'true_target': sample['target_of_hate'],
            'true_severity': sample['severity_of_hate'],
            'pred_type': parsed.get('type_of_hate', 'None'),
            'pred_target': parsed.get('target_of_hate', 'None'),
            'pred_severity': parsed.get('severity_of_hate', 'Little to None'),
            'raw_response': response
        })
    
    return results

def parse_llm_output(response_text):
    """Extract JSON from LLM response, handling malformed outputs."""
    import re
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {'type_of_hate': 'None', 'target_of_hate': 'None', 'severity_of_hate': 'Little to None'}

def compute_llm_consistency_violation_rate(results):
    """Compute % of LLM predictions that are logically contradictory."""
    total = len(results)
    violations = 0
    
    for r in results:
        pred_type = r['pred_type']
        pred_target = r['pred_target']
        pred_sev = r['pred_severity']
        
        # Rule: If Type=None → Target must be None AND Severity must be Little to None
        if pred_type == 'None' and (pred_target != 'None' or pred_sev != 'Little to None'):
            violations += 1
        # Reverse rule: If Target≠None or Severity≠Little → Type must not be None
        if pred_type != 'None' and pred_target == 'None' and pred_sev == 'Little to None':
            violations += 1
    
    return (violations / total) * 100 if total > 0 else 0

# ════════════════════════════════════════════════════════════
# Run evaluations
# ════════════════════════════════════════════════════════════
import pandas as pd
from datasets import load_dataset

test_data = load_dataset("aridhasan/BanglaMultiHate", split="test")

llm_experiments = [
    ('md-nishat-008/TigerLLM-1B-it', ZERO_SHOT_PROMPT, 'TigerLLM-1B (zero-shot)'),
    ('meta-llama/Llama-3.2-3B-Instruct', FEW_SHOT_PROMPT, 'Llama-3.2-3B (3-shot)'),
    ('google/gemma-2-2b-it', FEW_SHOT_PROMPT, 'Gemma-2-2B (3-shot)'),
]

all_llm_results = {}
for model_name, prompt, label in llm_experiments:
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"{'='*60}")
    
    results = evaluate_llm(model_name, test_data, prompt, max_samples=500)
    cvr = compute_llm_consistency_violation_rate(results)
    
    # Compute F1 scores
    from sklearn.metrics import f1_score
    type_f1 = f1_score(
        [r['true_type'] for r in results],
        [r['pred_type'] for r in results],
        average='macro', zero_division=0
    )
    target_f1 = f1_score(
        [r['true_target'] for r in results],
        [r['pred_target'] for r in results],
        average='macro', zero_division=0
    )
    sev_f1 = f1_score(
        [r['true_severity'] for r in results],
        [r['pred_severity'] for r in results],
        average='macro', zero_division=0
    )
    
    all_llm_results[label] = {
        'type_f1': type_f1, 'target_f1': target_f1, 'sev_f1': sev_f1,
        'avg_f1': (type_f1 + target_f1 + sev_f1) / 3,
        'cvr': cvr
    }
    
    print(f"  Type F1: {type_f1:.4f}")
    print(f"  Target F1: {target_f1:.4f}")
    print(f"  Severity F1: {sev_f1:.4f}")
    print(f"  Consistency Violation Rate: {cvr:.1f}%")

# Save results
with open('results/llm_baseline_results.json', 'w') as f:
    json.dump(all_llm_results, f, indent=2)
```

> [!TIP]
> **Key insight**: The LLM zero-shot/few-shot evaluation requires **no training at all** — just inference. You can run all 3 LLMs on 500 test samples in ~4-6 hours on a single T4 GPU (Colab free tier). Do this on Day 1-2 of Week 4 while your fine-tuning experiments are queued.

### Day 2-3: Optional — LoRA Fine-Tuning of TigerLLM (If Time Permits)

```python
# === src/train_lora_llm.py ===
# Only run this if you have time and GPU budget

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# LoRA config for TigerLLM-1B
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = AutoModelForCausalLM.from_pretrained(
    "md-nishat-008/TigerLLM-1B-it",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = get_peft_model(model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")

# Format training data as instruction-following
def format_for_lora(sample):
    return f"""### Instruction: Classify this Bengali comment for hate speech.
### Input: {sample['comment']}
### Output: {{"type_of_hate": "{sample['type_of_hate']}", "target_of_hate": "{sample['target_of_hate']}", "severity_of_hate": "{sample['severity_of_hate']}"}}"""

# Training with Hugging Face Trainer
training_args = TrainingArguments(
    output_dir='models/tigerllm_lora',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_strategy='epoch',
    logging_steps=50,
)
```

### Day 3-5: Fine-Tune Encoder Models (Your Ablations)

### Training Script

```python
# === src/train.py ===
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from model import ConsistencyConstrainedMTL
from losses import FocalLoss, ConsistencyPenaltyLoss
from dataset import BanglaHateDataset
import json
from tqdm import tqdm

def compute_class_weights(dataset, label_col, num_classes):
    """Compute inverse frequency weights for focal loss."""
    from collections import Counter
    counts = Counter(dataset.df[label_col].tolist())
    total = sum(counts.values())
    weights = [total / (num_classes * counts.get(label, 1)) for label in sorted(counts.keys())]
    return torch.FloatTensor(weights)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_dataset = BanglaHateDataset(
        'data/banglamultihate_train.csv',
        silver_explanations_path='data/silver_explanations.json' if config['use_gen'] else None
    )
    val_dataset = BanglaHateDataset('data/banglamultihate_dev.csv')
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Model
    model = ConsistencyConstrainedMTL(
        encoder_name=config['encoder'],
        num_type_labels=6, num_target_labels=5, num_severity_labels=3
    ).to(device)
    
    # Losses
    type_weights = compute_class_weights(train_dataset, 'type_of_hate', 6).to(device)
    target_weights = compute_class_weights(train_dataset, 'target_of_hate', 5).to(device)
    sev_weights = compute_class_weights(train_dataset, 'severity_of_hate', 3).to(device)
    
    type_loss_fn = FocalLoss(alpha=type_weights)
    target_loss_fn = FocalLoss(alpha=target_weights)
    sev_loss_fn = FocalLoss(alpha=sev_weights)
    consist_loss_fn = ConsistencyPenaltyLoss()
    gen_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    # Optimizer (differential LR)
    encoder_params = list(model.encoder.parameters())
    head_params = (list(model.type_head.parameters()) + 
                   list(model.target_head.parameters()) + 
                   list(model.severity_head.parameters()))
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': config['encoder_lr']},
        {'params': head_params, 'lr': config['head_lr']},
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * config['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_f1 = 0
    history = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_losses = {'total': 0, 'type': 0, 'target': 0, 'sev': 0, 'consist': 0, 'gen': 0}
        
        # Warmup lambda for consistency
        lambda_c = config['lambda_start'] + (config['lambda_end'] - config['lambda_start']) * (epoch / config['epochs'])
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}'):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            type_logits, target_logits, sev_logits, gen_logits = model(
                batch['input_ids'], batch['attention_mask'],
                batch.get('explanation_ids')
            )
            
            # Classification losses
            l_type = type_loss_fn(type_logits, batch['type_label'])
            l_target = target_loss_fn(target_logits, batch['target_label'])
            l_sev = sev_loss_fn(sev_logits, batch['severity_label'])
            
            loss = l_type + l_target + l_sev
            
            # Consistency penalty
            if config['use_consistency']:
                l_consist = consist_loss_fn(type_logits, target_logits, sev_logits)
                loss += lambda_c * l_consist
                epoch_losses['consist'] += l_consist.item()
            
            # Generation loss
            if config['use_gen'] and gen_logits is not None and 'explanation_labels' in batch:
                l_gen = gen_loss_fn(
                    gen_logits.view(-1, gen_logits.size(-1)),
                    batch['explanation_labels'].view(-1)
                )
                loss += config['gen_weight'] * l_gen
                epoch_losses['gen'] += l_gen.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_losses['total'] += loss.item()
        
        # Log
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
        print(f"Epoch {epoch+1}: {epoch_losses}")
        history.append(epoch_losses)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        avg_f1 = (val_metrics['type_f1'] + val_metrics['target_f1'] + val_metrics['sev_f1']) / 3
        
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            torch.save(model.state_dict(), f'models/{config["name"]}_best.pt')
            print(f"  ✅ New best model saved! Avg F1: {avg_f1:.4f}")
    
    # Save training history
    with open(f'results/{config["name"]}_history.json', 'w') as f:
        json.dump(history, f)
    
    return model, history

# ════════════════════════════════════
# Run all encoder fine-tuning experiments
# ════════════════════════════════════
experiments = [
    # Group B: Encoder baselines + ablations
    {'name': 'exp1_banglabert_baseline', 'encoder': 'csebuetnlp/banglabert',
     'use_consistency': False, 'use_gen': False,
     'batch_size': 16, 'epochs': 15, 'encoder_lr': 2e-5, 'head_lr': 1e-4,
     'lambda_start': 0, 'lambda_end': 0, 'gen_weight': 0},
    
    {'name': 'exp2_muril_baseline', 'encoder': 'google/muril-base-cased',
     'use_consistency': False, 'use_gen': False,
     'batch_size': 16, 'epochs': 15, 'encoder_lr': 2e-5, 'head_lr': 1e-4,
     'lambda_start': 0, 'lambda_end': 0, 'gen_weight': 0},
    
    {'name': 'exp3_consistency',  'encoder': 'csebuetnlp/banglabert',
     'use_consistency': True,  'use_gen': False,
     'batch_size': 16, 'epochs': 15, 'encoder_lr': 2e-5, 'head_lr': 1e-4,
     'lambda_start': 0.01, 'lambda_end': 1.0, 'gen_weight': 0},
    
    {'name': 'exp4_generative',   'encoder': 'csebuetnlp/banglat5',
     'use_consistency': False, 'use_gen': True,
     'batch_size': 16, 'epochs': 15, 'encoder_lr': 2e-5, 'head_lr': 1e-4,
     'lambda_start': 0, 'lambda_end': 0, 'gen_weight': 0.5},
    
    {'name': 'exp5_full_model',   'encoder': 'csebuetnlp/banglat5',
     'use_consistency': True,  'use_gen': True,
     'batch_size': 16, 'epochs': 15, 'encoder_lr': 2e-5, 'head_lr': 1e-4,
     'lambda_start': 0.01, 'lambda_end': 1.0, 'gen_weight': 0.5},
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"{'='*60}")
    model, history = train(exp)
```

### Week 4 Deliverables
- [ ] LLM baselines evaluated: TigerLLM-1B, Llama-3.2-3B, Gemma-2-2B (zero/few-shot)
- [ ] LLM consistency violation rates computed and logged
- [ ] MuRIL MTL baseline trained
- [ ] All 5 encoder experiments trained to completion
- [ ] Best model checkpoints saved
- [ ] Training loss curves for all experiments
- [ ] Validation metrics logged per epoch
- [ ] (Optional) TigerLLM LoRA fine-tuning completed

---

## Week 5: Evaluation (Sep 16-22)

### Run All Evaluations

```python
# === src/evaluate_all.py ===

# 1. Classification metrics (F1 per task)
# 2. Consistency violation rate (before vs after)
# 3. Faithfulness metrics for Generated Explanations (BLEU, BERTScore, Human Eval)

# See full evaluation code in the previous implementation plan.
# Key results table to produce:

RESULTS_TEMPLATE = """
Table 2: Main Classification Results
══════════════════════════════════════════════════════════════════════════
Model                          | Type F1 | Target F1 | Sev F1 | Avg F1 | CVR ↓
───────────────────────────────┼─────────┼───────────┼────────┼────────┼──────
--- LLM Baselines (No Training) ---
TigerLLM-1B (zero-shot)        |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
Llama-3.2-3B (3-shot)          |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
Gemma-2-2B (3-shot)            |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
--- Encoder Baselines (Fine-Tuned) ---
BanglaBERT MTL                 |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
MuRIL MTL                      |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
--- Our Ablations ---
BanglaBERT + Consistency       |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
BanglaBERT + Generative        |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
Full Model (Ours)              |  ?.??   |   ?.??    |  ?.??  |  ?.??  | ?.?%
══════════════════════════════════════════════════════════════════════════
CVR = Consistency Violation Rate (lower is better)

Table 3: Faithfulness of Generated Explanations (vs Silver Standard)
════════════════════════════════════════════════════════════════════════════
Model                    | BLEU-4 ↑ | ROUGE-L ↑ | BERTScore ↑ | Human Eval ↑
─────────────────────────┼──────────┼───────────┼─────────────┼─────────────
BanglaT5 + Generative    |   ?.??   |   ?.??    |    ?.??     |     ?.??
Full Model (Ours)        |   ?.??   |   ?.??    |    ?.??     |     ?.??
════════════════════════════════════════════════════════════════════════════
"""
```

### Week 5 Deliverables
- [ ] Classification results table filled
- [ ] Consistency violation rate for all 4 models
- [ ] Faithfulness metrics (Comp, Suff, AOPC)
- [ ] Confusion matrices (3 per model variant)
- [ ] Perturbation curve figure
- [ ] Statistical significance tests (if applicable)

---

## Week 6: Paper & Submission (Sep 23-30)

### Paper Sections to Add to main.tex

| Section | What to Write | Estimated Length |
|:---|:---|:---|
| §4 Experimental Setup | Dataset stats table, implementation details, baselines | 1 page |
| §5 Results & Discussion | Results tables, ablation analysis, qualitative examples | 1.5 pages |
| §6 Limitations | Silver-standard quality, language-specific constraints | 0.25 page |
| §7 Conclusion | Summary + future work | 0.25 page |
| Update Abstract | Insert actual numbers | — |
| Update §3 Methodology | Add formal L_consist equation, fix architecture description | — |

### Figures to Create

| # | Figure | Tool |
|:---|:---|:---|
| Fig 1 | Architecture diagram | draw.io or tikz |
| Fig 2 | Class distribution | matplotlib (from EDA) |
| Fig 3 | Consistency violation rate comparison | bar chart |
| Fig 4 | Perturbation curve (AOPC) | line plot |

### Final Checklist Before Submission
- [ ] Paper compiles without errors (pdflatex)
- [ ] All tables have numbers filled in
- [ ] All figures are high resolution (300 DPI)
- [ ] References are correct and complete
- [ ] Abstract updated with actual results
- [ ] IEEE formatting compliance (6 pages max)
- [ ] PDF file < 5MB
- [ ] All co-authors have reviewed

---

## 👥 Task Division (If Working as a Team of 2)

| Task | Angkon | Nahid |
|:---|:---|:---|
| **Week 1** | Dataset download, EDA | Environment setup, BanglaBERT + LLM model download |
| **Week 2** | Silver explanation generation (Gemini API) | Custom dataset: comment collection + annotation |
| **Week 3** | Model architecture coding (model.py, losses.py) | Dataset class (dataset.py) + custom dataset annotation |
| **Week 4** | LLM baselines (TigerLLM, Llama, Gemma zero/few-shot) + exp1-2 | Encoder fine-tuning experiments (exp3-5) + optional LoRA |
| **Week 5** | Classification evaluation + consistency metrics for ALL models | Faithfulness evaluation (ERASER metrics) |
| **Week 6** | Write §4-5 (Experiments, Results — include LLM comparison) | Write §6-7 (Limitations, Conclusion) + figures |

---

## ⚠️ Risk Mitigation

| Risk | Impact | Mitigation |
|:---|:---|:---|
| Gemini API rate limits | Can't generate 5K explanations | Use batch processing with checkpoints; try Gemini Flash |
| BanglaBERT training OOM on Colab | Can't train full model | Use `banglabert_small`, reduce batch_size to 8, use gradient accumulation |
| LLM inference OOM | Can't run Llama-3.2-3B | Use 4-bit quantization (`load_in_4bit=True`) or switch to TigerLLM-1B only |
| LLM outputs unparseable JSON | Can't compute F1 | `parse_llm_output()` handles malformed JSON; default to "None" labels |
| LLMs outperform your model on F1 | Reviewers question your approach | **This is expected!** Your story is about CVR, not F1. LLMs will have higher CVR. |
| Consistency penalty hurts F1 | Main contribution weakened | Tune λ carefully; even if F1 drops slightly, CVR improvement is the story |
| Silver explanations low quality | Reviewers reject contribution | Human eval validates quality; include quality table in paper |
| Custom dataset too small | Weak contribution | 300 samples with gold explanations is still valuable as validation set |
| ICCIT deadline not extended | Can't submit | Pivot to IEEE Access (journal) or BLP 2027 workshop |
