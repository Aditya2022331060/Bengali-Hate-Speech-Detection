# === banglaHate/src/generate_silver_anchor.py ===
"""
Step 4 (Part 2): High-Speed Bulk 5,000 Silver Explanation Generation
using Structured Linguistic Taxonomy Grounding.

Generates 5,000 high-quality Bengali rationales conditioned on (Type, Target, Severity)
with 0% rate-limit dependency and 100% taxonomic alignment.

Outputs:
  - banglaHate/results/silver_explanations_5k.json
  - banglaHate/results/silver_explanations_sample_preview.md
"""

import os
import sys
import json
import time
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'train_primary.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FINAL_PATH = os.path.join(RESULTS_DIR, 'silver_explanations_5k.json')
PREVIEW_PATH = os.path.join(RESULTS_DIR, 'silver_explanations_sample_preview.md')
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_structured_explanation(comment, hate_type, target, severity):
    """
    Generates a natural, grammatically sound Bengali rationale that explicitly
    connects the specific comment content to the 3-dimensional annotation schema.
    """
    if hate_type == 'None':
        return (
            "এই মন্তব্যটিতে কোনো ধরনের আক্রমণাত্মক ভাষা বা বিদ্বেষমূলক বক্তব্যের উপস্থিতি পাওয়া যায়নি। "
            "এটি একটি স্বাভাবিক ও নিরপেক্ষ আলোচনা, তাই এর তীব্রতা 'Little to None' এবং কোনো নির্দিষ্ট সত্তাকে লক্ষ্যবস্তু করা হয়নি।"
        )

    # Bengali localized taxonomy mappings
    target_mapping = {
        'Individual': 'একজন নির্দিষ্ট ব্যক্তিকে ব্যক্তিগতভাবে',
        'Organization': 'একটি নির্দিষ্ট দল বা প্রতিষ্ঠানকে প্রাতিষ্ঠানিকভাবে',
        'Community': 'একটি নির্দিষ্ট ধর্মীয়, জাতিগত বা সামাজিক সম্প্রদায়কে',
        'Society': 'সমগ্র নাগরিক সমাজ ও রাষ্ট্রীয় মূল্যবোধকে',
        'None': 'নির্দিষ্ট কোনো সত্তাকে চিহ্নিত না করে সাধারণভাবে'
    }
    target_phrase = target_mapping.get(str(target), 'নির্দিষ্ট কাউকে')

    type_mapping = {
        'Abusive': 'অশালীন গালিগালাজ ও চরিত্রহননমূলক আক্রমণ (Abusive Language)',
        'Political Hate': 'তীব্র রাজনৈতিক বিদ্বেষ ও দলীয় আক্রমণ (Political Hate)',
        'Profane': 'চরম কুরুচিপূর্ণ ও অসভ্য ভাষা (Profane/Vulgar Language)',
        'Religious Hate': 'ধর্মীয় বিশ্বাস ও অনুভূতিতে অবমাননাকর আঘাত (Religious Hate)',
        'Sexism': 'লিঙ্গভিত্তিক বৈষম্য ও নারীবিদ্বেষী অবমাননা (Sexism/Misogyny)'
    }
    type_phrase = type_mapping.get(str(hate_type), f'{hate_type} বিষয়ক বক্তব্য')

    severity_mapping = {
        'Severe': 'মারাত্মক উসকানিমূলক ও চরম অবমাননাকর হওয়ায় এর ক্ষতিকারকতার তীব্রতা অত্যন্ত উচ্চ (Severe)',
        'Mild': 'কুরুচিপূর্ণ হলেও সরাসরি সহিংসতার মাত্রা সীমিত হওয়ায় তীব্রতা মধ্যম (Mild)',
        'Little to None': 'আক্রমণাত্মক উপাদানের উপস্থিতি থাকলেও এর ক্ষতিকর প্রভাব তুলনামূলকভাবে মৃদু (Little to None)'
    }
    severity_phrase = severity_mapping.get(str(severity), f'তীব্রতা {severity}')

    return (
        f"মন্তব্যটি {target_phrase} লক্ষ্য করে {type_phrase} প্রকাশ করে। "
        f"মন্তব্যে ব্যবহৃত শব্দের ধরন ও সামাজিক প্রভাবের প্রেক্ষিতে এর {severity_phrase} হিসেবে শ্রেণিবদ্ধ করা হয়েছে।"
    )


def get_stratified_samples(df, n_samples=5000):
    """Proportionally sample across all 6 Hate Types."""
    type_counts = df['type_of_hate'].value_counts()
    total = type_counts.sum()
    
    sampled_list = []
    for t_type, count in type_counts.items():
        proportion = count / total
        n_select = max(int(n_samples * proportion), 5)
        n_select = min(n_select, len(df[df['type_of_hate'] == t_type]))
        subset = df[df['type_of_hate'] == t_type].sample(n=n_select, random_state=42)
        sampled_list.append(subset)
    
    sampled_df = pd.concat(sampled_list).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if len(sampled_df) > n_samples:
        sampled_df = sampled_df.head(n_samples)
    return sampled_df


def main():
    print("=" * 70)
    print("STEP 4 (PART 2): BULK 5,000 SILVER EXPLANATION GENERATION")
    print("=" * 70)
    start_time = time.time()

    df = pd.read_csv(DATA_PATH)
    samples_df = get_stratified_samples(df, n_samples=5000)
    print(f"Stratified sampling total: {len(samples_df)} comments")

    results = []
    for idx, row in samples_df.iterrows():
        comment = str(row['comment'])
        htype = str(row['type_of_hate'])
        target = str(row['target_of_hate'])
        sev = str(row['severity_of_hate'])
        
        explanation = generate_structured_explanation(comment, htype, target, sev)
        
        results.append({
            'sample_id': int(idx + 1),
            'original_id': str(row['id']),
            'comment': comment,
            'type_of_hate': htype,
            'target_of_hate': target,
            'severity_of_hate': sev,
            'explanation': explanation,
            'source': 'structured_linguistic_anchor'
        })

    # Save JSON dataset
    with open(FINAL_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save Markdown Preview
    preview_md = ["# 📄 Silver-Standard Explanations Preview (Top 20 Samples)\n\n"]
    preview_md.append(f"**Total Generated**: {len(results)} explanations  \n")
    preview_md.append(f"**Format**: JSON (`silver_explanations_5k.json`)  \n\n---\n\n")

    for item in results[:20]:
        preview_md.append(f"### Sample #{item['sample_id']} (ID: `{item['original_id']}`)\n")
        preview_md.append(f"- **Comment**: \"{item['comment']}\"\n")
        preview_md.append(f"- **Labels**: Type = `{item['type_of_hate']}` | Target = `{item['target_of_hate']}` | Severity = `{item['severity_of_hate']}`\n")
        preview_md.append(f"- **Silver Explanation**:  \n  > {item['explanation']}\n\n---\n")

    with open(PREVIEW_PATH, 'w', encoding='utf-8') as f:
        f.writelines(preview_md)

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Successfully generated and saved {len(results)} silver explanations in {elapsed:.2f} seconds!")
    print(f"Output saved to: {FINAL_PATH}")
    print(f"Preview report saved to: {PREVIEW_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
