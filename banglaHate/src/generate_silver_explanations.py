# === banglaHate/src/generate_silver_explanations.py ===
"""
Step 4 (Part 2): Bulk Silver-Standard Bengali Explanation Generation using Gemini.

Generates structured Bengali rationales conditioned on (Type, Target, Severity) 
for training the generative explanation head of ConsistencyConstrainedMTL.

Features:
  - Incremental checkpoint saving (resumes from last checkpoint on restart)
  - Rate-limit handling with exponential backoff (free tier: 15 RPM)
  - Structured prompt template for consistent Bengali output
  - Falls back to linguistic anchor if Gemini fails after retries

Usage:
  python generate_silver_explanations.py --api_key "YOUR_KEY" [--n_samples 5000] [--batch_size 500]
"""

import os
import sys

# Force unbuffered output so we can see progress in real-time
os.environ['PYTHONUNBUFFERED'] = '1'

import json
import time
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'train_primary.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, 'silver_explanations_checkpoint.json')
FINAL_PATH = os.path.join(RESULTS_DIR, 'silver_explanations_5k.json')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Structured Linguistic Anchor (Fallback)
# ─────────────────────────────────────────────────────────────────────────────
def generate_rule_based_rationale(comment, hate_type, target, severity):
    """Fallback: Constructs a grammatically precise Bengali explanation."""
    if hate_type == 'None':
        return ("এই মন্তব্যটিতে কোনো ধরনের ঘৃণাত্মক বক্তব্য বা আক্রমণাত্মক ভাষা নেই। "
                "এটি একটি সাধারণ ও নিরপেক্ষ মন্তব্য, তাই তীব্রতা 'Little to None' এবং "
                "কোনো নির্দিষ্ট ব্যক্তি বা গোষ্ঠীকে লক্ষ্যবস্তু করা হয়নি।")

    target_bn = {
        'Individual': 'একজন নির্দিষ্ট ব্যক্তিকে',
        'Organization': 'একটি নির্দিষ্ট প্রতিষ্ঠানকে',
        'Community': 'একটি নির্দিষ্ট জনগোষ্ঠী বা সম্প্রদায়কে',
        'Society': 'সামগ্রিক সমাজকে',
        'None': 'কাউকে নির্দিষ্ট না করে'
    }.get(str(target), 'কাউকে')

    severity_bn = {
        'Severe': 'অত্যন্ত তীব্র (Severe)',
        'Mild': 'মধ্যম মাত্রার (Mild)',
        'Little to None': 'মৃদু (Little to None)'
    }.get(str(severity), str(severity))

    type_bn = {
        'Abusive': 'গালিগালাজ ও আক্রমণাত্মক আচরণ (Abusive)',
        'Political Hate': 'রাজনৈতিক বিদ্বেষ (Political Hate)',
        'Profane': 'অশালীন ও অসভ্য ভাষা (Profane)',
        'Religious Hate': 'ধর্মীয় অনুভূতিতে আঘাত ও ঘৃণা (Religious Hate)',
        'Sexism': 'লিঙ্গভিত্তিক বৈষম্য ও নারীবিদ্বেষ (Sexism)'
    }.get(str(hate_type), str(hate_type))

    return (f"মন্তব্যটি {target_bn} লক্ষ্য করে {type_bn} প্রকাশ করে। "
            f"মন্তব্যে ব্যবহৃত ভাষা ও বক্তব্যের ধরন বিবেচনা করে এর বিষাক্ততার "
            f"তীব্রতা {severity_bn} হিসেবে চিহ্নিত করা হয়েছে।")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gemini Prompt Template
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """আপনি একজন বাংলা ভাষাবিদ ও সাইবার হেট স্পিচ বিশ্লেষক। 
প্রদত্ত বাংলা কমেন্ট এবং তার লেবেল (ঘৃণার ধরন, লক্ষ্য, তীব্রতা) দেখে, 
কেন এই মন্তব্যটি এই লেবেল পেয়েছে তার একটি ২-৩ বাক্যের বাংলা ব্যাখ্যা লিখুন।

নিয়ম:
1. শুধুমাত্র বাংলায় লিখুন (কোনো ইংরেজি নয়)।
2. কমেন্টে ব্যবহৃত নির্দিষ্ট শব্দ/বাক্যাংশ উল্লেখ করুন যেগুলো এই লেবেলের কারণ।
3. ২-৩ বাক্যের মধ্যে সীমাবদ্ধ রাখুন।
4. লক্ষ্য (Target) এবং তীব্রতা (Severity) কেন এই মান পেয়েছে তা ব্যাখ্যা করুন।"""


def build_user_prompt(comment, hate_type, target, severity):
    """Build the user-facing prompt for a single sample."""
    return (f"কমেন্ট: \"{comment}\"\n"
            f"ঘৃণার ধরন: {hate_type}\n"
            f"লক্ষ্য: {target}\n"
            f"তীব্রতা: {severity}\n\n"
            f"ব্যাখ্যা:")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gemini API Call with Retry & Backoff
# ─────────────────────────────────────────────────────────────────────────────
def call_gemini_with_retry(model, comment, hate_type, target, severity, max_retries=3):
    """Call Gemini API with exponential backoff for rate limits."""
    user_prompt = build_user_prompt(comment, hate_type, target, severity)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(user_prompt)
            text = response.text.strip()
            
            # Quality check: must be >20 chars and contain Bengali characters
            if len(text) > 20 and any('\u0980' <= c <= '\u09FF' for c in text):
                return text, 'gemini'
            else:
                # Too short or not Bengali — retry
                continue
                
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'quota' in error_str.lower() or 'rate' in error_str.lower():
                # Rate limited — exponential backoff
                wait_time = (2 ** attempt) * 4 + random.uniform(0, 2)
                print(f"    [RATE LIMIT] Waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}...", flush=True)
                time.sleep(wait_time)
            else:
                print(f"    [ERROR] {error_str[:100]}", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2)
    
    # All retries failed — use fallback
    fallback = generate_rule_based_rationale(comment, hate_type, target, severity)
    return fallback, 'fallback'


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint Save/Load
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(results, checkpoint_path):
    """Save progress incrementally."""
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint_path):
    """Load existing progress if available."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stratified Sampling for 5,000 Samples
# ─────────────────────────────────────────────────────────────────────────────
def get_stratified_samples(df, n_samples=5000):
    """Sample n_samples proportionally across all 6 Hate Types."""
    type_counts = df['type_of_hate'].value_counts()
    total = type_counts.sum()
    
    sampled_list = []
    for t_type, count in type_counts.items():
        proportion = count / total
        n_select = max(int(n_samples * proportion), 5)  # at least 5 per type
        n_select = min(n_select, len(df[df['type_of_hate'] == t_type]))
        subset = df[df['type_of_hate'] == t_type].sample(n=n_select, random_state=42)
        sampled_list.append(subset)
    
    sampled_df = pd.concat(sampled_list).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Trim to exactly n_samples
    if len(sampled_df) > n_samples:
        sampled_df = sampled_df.head(n_samples)
    
    print(f"Stratified sampling: {len(sampled_df)} samples")
    for t_type, c in sampled_df['type_of_hate'].value_counts().sort_index().items():
        print(f"  - {t_type:<18s}: {c}")
    return sampled_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main Generation Loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate Silver-Standard Bengali Explanations")
    parser.add_argument("--api_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Save checkpoint every N samples")
    parser.add_argument("--model_name", type=str, default="gemini-3.6-flash", help="Gemini model to use")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 4 (PART 2): BULK SILVER EXPLANATION GENERATION")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Target samples: {args.n_samples}")
    
    # Configure Gemini
    import google.generativeai as genai
    genai.configure(api_key=args.api_key)
    
    model = genai.GenerativeModel(
        model_name=args.model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            'temperature': 0.3,
            'top_p': 0.9,
            'max_output_tokens': 200,
        }
    )
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    samples_df = get_stratified_samples(df, n_samples=args.n_samples)
    
    # Load checkpoint (resume from last progress)
    existing_results = load_checkpoint(CHECKPOINT_PATH)
    completed_ids = {r['original_id'] for r in existing_results}
    print(f"\nResuming from checkpoint: {len(existing_results)} already completed")
    
    results = list(existing_results)
    gemini_count = 0
    fallback_count = 0
    start_time = time.time()
    
    # Free tier: ~15 RPM, so we pace at ~4 seconds between calls
    REQUEST_INTERVAL = 4.2  # seconds between API calls (safe for 15 RPM)
    
    for idx, row in samples_df.iterrows():
        sample_id = str(row['id'])
        
        # Skip already completed
        if sample_id in completed_ids:
            continue
        
        comment = str(row['comment'])
        htype = str(row['type_of_hate'])
        target = str(row['target_of_hate'])
        sev = str(row['severity_of_hate'])
        
        # Generate explanation via Gemini
        explanation, source = call_gemini_with_retry(model, comment, htype, target, sev)
        
        if source == 'gemini':
            gemini_count += 1
        else:
            fallback_count += 1
        
        result_item = {
            'original_id': sample_id,
            'comment': comment,
            'type_of_hate': htype,
            'target_of_hate': target,
            'severity_of_hate': sev,
            'explanation': explanation,
            'source': source
        }
        results.append(result_item)
        completed_ids.add(sample_id)
        
        # Progress logging
        total_done = len(results)
        if total_done % 10 == 0:
            elapsed = time.time() - start_time
            rate = (total_done - len(existing_results)) / max(elapsed, 1) * 60
            eta_min = (args.n_samples - total_done) / max(rate, 0.1)
            print(f"  [{total_done}/{args.n_samples}] "
                  f"Gemini: {gemini_count} | Fallback: {fallback_count} | "
                  f"Rate: {rate:.1f}/min | ETA: {eta_min:.0f} min", flush=True)
        
        # Save checkpoint periodically
        if total_done % args.batch_size == 0:
            save_checkpoint(results, CHECKPOINT_PATH)
            print(f"  [CHECKPOINT] Saved {total_done} results", flush=True)
        
        # Rate limiting
        time.sleep(REQUEST_INTERVAL)
    
    # Final save
    save_checkpoint(results, CHECKPOINT_PATH)
    
    # Save final output
    with open(FINAL_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"GENERATION COMPLETE!")
    print(f"  Total: {len(results)} explanations")
    print(f"  Gemini-generated: {gemini_count}")
    print(f"  Fallback (rule-based): {fallback_count}")
    print(f"  Gemini success rate: {gemini_count/(gemini_count+fallback_count)*100:.1f}%")
    print(f"  Time elapsed: {elapsed_total/60:.1f} minutes")
    print(f"  Saved to: {FINAL_PATH}")
    print("=" * 70)


if __name__ == '__main__':
    main()
