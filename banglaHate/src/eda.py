# === banglaHate/src/eda.py ===
# Comprehensive Exploratory Data Analysis for Bengali Hate Speech MTL Project
# Generates paper-ready figures (300 DPI) and dataset statistics

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from collections import Counter
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Seaborn style
sns.set_theme(style='whitegrid', font_scale=1.2)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Primary Dataset (train.json — BanglaMultiHate / BLP-2025 format)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading Primary Dataset (train.json)")
print("=" * 70)

train_path = os.path.join(DATASET_DIR, 'train.json')
with open(train_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

df = pd.DataFrame(raw_data)
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

# Save as CSV for faster reuse
csv_out = os.path.join(DATA_DIR, 'train_primary.csv')
df.to_csv(csv_out, index=False, encoding='utf-8')
print(f"\nSaved processed CSV to: {csv_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Label Distribution Statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Label Distribution Statistics")
print("=" * 70)

TYPE_ORDER = ['None', 'Abusive', 'Political Hate', 'Religious Hate', 'Gender Hate']
TARGET_ORDER = ['None', 'Individual', 'Organization', 'Community', 'Society']
SEVERITY_ORDER = ['Little to None', 'Mild', 'Severe']

for col, order, name in [('type_of_hate', TYPE_ORDER, 'Hate Type'),
                          ('target_of_hate', TARGET_ORDER, 'Target'),
                          ('severity_of_hate', SEVERITY_ORDER, 'Severity')]:
    print(f"\n--- {name} ({col}) ---")
    counts = df[col].value_counts()
    total = len(df)
    for label in order:
        c = counts.get(label, 0)
        print(f"  {label:<20s}: {c:>6d} ({c/total*100:>5.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Figure 1: Class Distribution Bar Charts (Paper Figure)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Generating Class Distribution Figure")
print("=" * 70)

# Color palettes
type_colors = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#2196F3', '#E91E63']
target_colors = ['#4CAF50', '#FF5722', '#3F51B5', '#009688', '#795548']
severity_colors = ['#4CAF50', '#FFC107', '#F44336']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Hate Type
type_counts = df['type_of_hate'].value_counts().reindex(TYPE_ORDER)
axes[0].barh(TYPE_ORDER[::-1], type_counts.values[::-1], color=type_colors[::-1], edgecolor='white', linewidth=0.5)
axes[0].set_title('Hate Type Distribution', fontsize=15, fontweight='bold')
axes[0].set_xlabel('Count', fontsize=12)
for i, (v, label) in enumerate(zip(type_counts.values[::-1], TYPE_ORDER[::-1])):
    axes[0].text(v + 100, i, f'{v:,} ({v/len(df)*100:.1f}%)', va='center', fontsize=10)

# Target
target_counts = df['target_of_hate'].value_counts().reindex(TARGET_ORDER)
axes[1].barh(TARGET_ORDER[::-1], target_counts.values[::-1], color=target_colors[::-1], edgecolor='white', linewidth=0.5)
axes[1].set_title('Target Distribution', fontsize=15, fontweight='bold')
axes[1].set_xlabel('Count', fontsize=12)
for i, (v, label) in enumerate(zip(target_counts.values[::-1], TARGET_ORDER[::-1])):
    axes[1].text(v + 100, i, f'{v:,} ({v/len(df)*100:.1f}%)', va='center', fontsize=10)

# Severity
sev_counts = df['severity_of_hate'].value_counts().reindex(SEVERITY_ORDER)
axes[2].barh(SEVERITY_ORDER[::-1], sev_counts.values[::-1], color=severity_colors[::-1], edgecolor='white', linewidth=0.5)
axes[2].set_title('Severity Distribution', fontsize=15, fontweight='bold')
axes[2].set_xlabel('Count', fontsize=12)
for i, (v, label) in enumerate(zip(sev_counts.values[::-1], SEVERITY_ORDER[::-1])):
    axes[2].text(v + 100, i, f'{v:,} ({v/len(df)*100:.1f}%)', va='center', fontsize=10)

plt.suptitle('BanglaMultiHate Dataset — Multi-Task Label Distributions (N=35,522)', fontsize=17, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'fig1_class_distributions.png')
plt.savefig(fig_path)
plt.close()
print(f"Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Figure 2: Co-occurrence Heatmaps
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Generating Co-occurrence Heatmaps")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap 1: Type × Severity
ct1 = pd.crosstab(df['type_of_hate'], df['severity_of_hate'])
ct1 = ct1.reindex(index=TYPE_ORDER, columns=SEVERITY_ORDER, fill_value=0)
sns.heatmap(ct1, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
axes[0].set_title('Hate Type × Severity', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Severity', fontsize=12)
axes[0].set_ylabel('Hate Type', fontsize=12)

# Heatmap 2: Type × Target
ct2 = pd.crosstab(df['type_of_hate'], df['target_of_hate'])
ct2 = ct2.reindex(index=TYPE_ORDER, columns=TARGET_ORDER, fill_value=0)
sns.heatmap(ct2, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1],
            linewidths=0.5, linecolor='white', cbar_kws={'label': 'Count'})
axes[1].set_title('Hate Type × Target', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_ylabel('Hate Type', fontsize=12)

plt.suptitle('Multi-Task Label Co-occurrence Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'fig2_cooccurrence_heatmaps.png')
plt.savefig(fig_path)
plt.close()
print(f"Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ground Truth Consistency Violation Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Ground Truth Consistency Violation Analysis")
print("=" * 70)

# Rule: If type=None → target MUST be None AND severity MUST be Little to None
none_type = df[df['type_of_hate'] == 'None']
v1 = none_type[none_type['target_of_hate'] != 'None']
v2 = none_type[none_type['severity_of_hate'] != 'Little to None']

# Reverse: if target != None → type should != None
has_target = df[df['target_of_hate'] != 'None']
v3 = has_target[has_target['type_of_hate'] == 'None']

total = len(df)
total_violations = len(v1) + len(v2)

print(f"Total samples: {total}")
print(f"Type=None samples: {len(none_type)}")
print(f"")
print(f"Violation 1 (Type=None but Target!=None): {len(v1)} ({len(v1)/total*100:.2f}%)")
print(f"Violation 2 (Type=None but Severity!=Little): {len(v2)} ({len(v2)/total*100:.2f}%)")
print(f"Violation 3 (Target!=None but Type=None): {len(v3)} ({len(v3)/total*100:.2f}%)")
print(f"")
print(f"Total GT violations: {total_violations} ({total_violations/total*100:.2f}%)")
print(f"Ground truth is {'CLEAN' if total_violations == 0 else 'HAS VIOLATIONS'}!")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Text Length Statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6: Text Length Statistics")
print("=" * 70)

df['char_length'] = df['comment'].str.len()
df['word_count'] = df['comment'].str.split().str.len()

print(f"Character length:")
print(f"  Mean: {df['char_length'].mean():.0f}")
print(f"  Median: {df['char_length'].median():.0f}")
print(f"  Std: {df['char_length'].std():.0f}")
print(f"  Max: {df['char_length'].max()}")
print(f"  P95: {df['char_length'].quantile(0.95):.0f}")
print(f"  P99: {df['char_length'].quantile(0.99):.0f}")
print(f"")
print(f"Word count:")
print(f"  Mean: {df['word_count'].mean():.0f}")
print(f"  Median: {df['word_count'].median():.0f}")
print(f"  Max: {df['word_count'].max()}")
print(f"  P95: {df['word_count'].quantile(0.95):.0f}")
print(f"  P99: {df['word_count'].quantile(0.99):.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Figure 3: Text Length Distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7: Generating Text Length Distribution Figure")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Character length histogram
axes[0].hist(df['char_length'].clip(upper=500), bins=60, color='#2196F3', alpha=0.85, edgecolor='white')
axes[0].axvline(df['char_length'].median(), color='#F44336', linestyle='--', linewidth=2, label=f'Median: {df["char_length"].median():.0f}')
axes[0].axvline(df['char_length'].quantile(0.95), color='#FF9800', linestyle='--', linewidth=2, label=f'P95: {df["char_length"].quantile(0.95):.0f}')
axes[0].set_title('Character Length Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Character Length (clipped at 500)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend(fontsize=10)

# Word count histogram
axes[1].hist(df['word_count'].clip(upper=80), bins=50, color='#4CAF50', alpha=0.85, edgecolor='white')
axes[1].axvline(df['word_count'].median(), color='#F44336', linestyle='--', linewidth=2, label=f'Median: {df["word_count"].median():.0f}')
axes[1].axvline(df['word_count'].quantile(0.95), color='#FF9800', linestyle='--', linewidth=2, label=f'P95: {df["word_count"].quantile(0.95):.0f}')
axes[1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Word Count (clipped at 80)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend(fontsize=10)

plt.suptitle('Comment Length Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'fig3_text_length_distribution.png')
plt.savefig(fig_path)
plt.close()
print(f"Saved: {fig_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Figure 4: Class Imbalance Ratios (for Focal Loss α weights)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 8: Computing Class Imbalance Ratios (for Focal Loss α)")
print("=" * 70)

for col, order, name in [('type_of_hate', TYPE_ORDER, 'Hate Type'),
                          ('target_of_hate', TARGET_ORDER, 'Target'),
                          ('severity_of_hate', SEVERITY_ORDER, 'Severity')]:
    counts = df[col].value_counts()
    total = len(df)
    n_classes = len(order)
    print(f"\n--- {name} Inverse-Frequency Weights (for FocalLoss alpha) ---")
    for label in order:
        c = counts.get(label, 1)
        weight = total / (n_classes * c)
        print(f"  {label:<20s}: count={c:>6d}, weight={weight:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Category Distribution (Metadata Analysis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 9: Category Distribution (Metadata)")
print("=" * 70)

cat_counts = df['category'].value_counts()
for cat, cnt in cat_counts.items():
    print(f"  {cat:<20s}: {cnt:>6d} ({cnt/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Save Complete EDA Report as JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 10: Saving EDA Summary Report")
print("=" * 70)

eda_report = {
    'dataset_name': 'BanglaMultiHate (train.json)',
    'total_samples': len(df),
    'columns': list(df.columns),
    'type_of_hate_distribution': df['type_of_hate'].value_counts().to_dict(),
    'target_of_hate_distribution': df['target_of_hate'].value_counts().to_dict(),
    'severity_of_hate_distribution': df['severity_of_hate'].value_counts().to_dict(),
    'category_distribution': df['category'].value_counts().to_dict(),
    'text_stats': {
        'char_length_mean': round(df['char_length'].mean(), 1),
        'char_length_median': round(df['char_length'].median(), 1),
        'char_length_max': int(df['char_length'].max()),
        'char_length_p95': round(df['char_length'].quantile(0.95), 1),
        'word_count_mean': round(df['word_count'].mean(), 1),
        'word_count_median': round(df['word_count'].median(), 1),
        'word_count_max': int(df['word_count'].max()),
        'word_count_p95': round(df['word_count'].quantile(0.95), 1),
    },
    'consistency_violations': {
        'total_violations': total_violations,
        'violation_rate_pct': round(total_violations / total * 100, 4),
        'gt_is_clean': total_violations == 0,
    },
    'class_imbalance': {
        'most_frequent_type': df['type_of_hate'].value_counts().index[0],
        'least_frequent_type': df['type_of_hate'].value_counts().index[-1],
        'imbalance_ratio_type': round(df['type_of_hate'].value_counts().iloc[0] / df['type_of_hate'].value_counts().iloc[-1], 1),
        'most_frequent_target': df['target_of_hate'].value_counts().index[0],
        'least_frequent_target': df['target_of_hate'].value_counts().index[-1],
        'imbalance_ratio_target': round(df['target_of_hate'].value_counts().iloc[0] / df['target_of_hate'].value_counts().iloc[-1], 1),
    }
}

report_path = os.path.join(RESULTS_DIR, 'eda_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(eda_report, f, ensure_ascii=False, indent=2)
print(f"Saved EDA report to: {report_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Inspect Supplementary Datasets
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 11: Supplementary Datasets Summary")
print("=" * 70)

# Bengali hate speech .csv
try:
    df_supp1 = pd.read_csv(os.path.join(DATASET_DIR, 'Bengali hate speech .csv'), encoding='utf-8')
    print(f"\n[Bengali hate speech .csv]")
    print(f"  Shape: {df_supp1.shape}")
    print(f"  Columns: {list(df_supp1.columns)}")
    print(f"  Schema: sentence (text), hate (0/1 binary), category (7 classes)")
    print(f"  Hate distribution: {df_supp1['hate'].value_counts().to_dict()}")
except Exception as e:
    print(f"  Error loading: {e}")

# bengali_hate_v2.0.csv
try:
    df_supp2 = pd.read_csv(os.path.join(DATASET_DIR, 'bengali_hate_v2.0.csv'), encoding='utf-8')
    print(f"\n[bengali_hate_v2.0.csv]")
    print(f"  Shape: {df_supp2.shape}")
    print(f"  Columns: {list(df_supp2.columns)}")
    print(f"  Label distribution: {df_supp2['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"  Error loading: {e}")

# bengali_hate_v1.0.csv
try:
    df_supp3 = pd.read_csv(os.path.join(DATASET_DIR, 'bengali_ hate_v1.0.csv'), encoding='utf-8', sep='\t')
    print(f"\n[bengali_hate_v1.0.csv]")
    print(f"  Shape: {df_supp3.shape}")
    print(f"  Columns: {list(df_supp3.columns)}")
    print(f"  Label distribution: {df_supp3['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"  Error loading: {e}")

# bengali_slung_abusive.txt
try:
    slang_path = os.path.join(DATASET_DIR, 'bengali_slung_abusive.txt')
    with open(slang_path, 'r', encoding='utf-8') as f:
        slang_words = [line.strip() for line in f if line.strip()]
    print(f"\n[bengali_slung_abusive.txt]")
    print(f"  Total slang/abusive words: {len(slang_words)}")
    print(f"  First 10: {slang_words[:10]}")
except Exception as e:
    print(f"  Error loading: {e}")

print("\n" + "=" * 70)
print("EDA COMPLETE!")
print("=" * 70)
print(f"\nGenerated Figures:")
print(f"  1. {os.path.join(FIG_DIR, 'fig1_class_distributions.png')}")
print(f"  2. {os.path.join(FIG_DIR, 'fig2_cooccurrence_heatmaps.png')}")
print(f"  3. {os.path.join(FIG_DIR, 'fig3_text_length_distribution.png')}")
print(f"\nSaved Data:")
print(f"  - {csv_out}")
print(f"  - {report_path}")
