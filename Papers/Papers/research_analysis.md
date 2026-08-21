# Final Verified Research Analysis
### Bengali Harmful Content — Gap Verification, Best Topic, and Backup Plan
*Thoroughly verified against online literature — August 10, 2026*

---

## ⚠️ Honest Corrections First

My thorough search revealed critical nuances that **weaken some previously claimed gaps**. You need to know this before framing your proposal:

### Correction 1: DeepHateExplainer DID report faithfulness metrics
Your uploaded doc and my earlier analysis both claimed "no faithfulness metrics exist for Bengali hate speech." **This is partially wrong.** DeepHateExplainer (Karim et al., 2020) explicitly reports **comprehensiveness and sufficiency scores** using LRP and sensitivity analysis. However — and this is key — it only does this for:
- A **single-task** (4-class hate type) setup
- **Token-level attribution** (not generative rationales)
- A **2020-era encoder** (BERT ensemble), not LLMs

**What's actually still missing:** Faithfulness evaluation for **generative free-text rationales** in a **multi-task** Bengali setup. That's a narrower but still real gap.

### Correction 2: BanglaMultiHate HAS annotation-level consistency rules
The dataset's annotation guidelines enforce logical rules at the **data labeling** level (e.g., Type=None → Severity=Little-to-None, Target=None). So "no consistency exists" is too strong.

**What's actually still missing:** No published **model** enforces these constraints at **inference time**. All BLP-2025 systems (Catalyst, HateSense, Code_Gen, CUET-NLP_Zenith) use independent softmax heads with no learned consistency constraint. The model can still predict contradictory combinations.

### Correction 3: BanHADEX does have multi-dimensional annotations
BanHADEX includes 7 fine-grained hate categories + 7 target groups + binary labels + free-text explanations. It's richer than just "single-task binary." However, it does **NOT** include the BLP-2025 style **Severity** dimension, and its categories don't exactly match BanglaMultiHate's taxonomy.

---

## The Verified Gap Landscape (What's TRULY Still Open)

After 12+ targeted searches, here's the honest picture:

| # | Claimed Gap | Verified Status | Strength for a Paper |
|:---:|:---|:---|:---:|
| **G1** | No model-level consistency constraint between multi-task heads for Bengali | ✅ **CONFIRMED OPEN** — Annotation rules exist but no model enforces them; no system measures contradiction rate | ⭐⭐⭐⭐ |
| **G2** | No faithfulness evaluation for *generative* Bengali explanations | ✅ **CONFIRMED OPEN** — DeepHateExplainer did token-level faithfulness (2020), but nobody has measured faithfulness of *free-text rationale generation* (BanHADEX-style) | ⭐⭐⭐⭐ |
| **G3** | No system combines multi-task (Type+Target+Severity) WITH generative explanation | ✅ **CONFIRMED OPEN** — BanglaMultiHate has the 3 tasks but no explanations; BanHADEX has explanations but different taxonomy & no Severity; nobody has merged them | ⭐⭐⭐⭐⭐ |
| **G4** | Dialect+code-mixed robustness never tested for explainability quality | ✅ **CONFIRMED OPEN** — BIDWESH tested classification accuracy across dialects, but never tested whether explanations degrade; no explainable system evaluated on Romanized/Banglish input | ⭐⭐⭐ |
| **G5** | No existing survey covers MTL + XAI + LLMs for Bengali hate speech together | ✅ **CONFIRMED OPEN** — The 2024 Journal of Big Data survey covers methods/datasets but predates BLP-2025, BanglaMultiHate, BanHADEX, and LoRA-tuned LLM work entirely | ⭐⭐⭐⭐ |
| ~~G6~~ | ~~No faithfulness metrics at all for Bengali~~ | ❌ **CLOSED** — DeepHateExplainer (2020) reported comprehensiveness/sufficiency | — |
| ~~G7~~ | ~~No multi-task Bengali dataset exists~~ | ❌ **CLOSED** — BanglaMultiHate (2025/2026) | — |
| ~~G8~~ | ~~No rationale-annotated Bengali hate dataset~~ | ❌ **CLOSED** — BanHADEX (2026) | — |

---

## PATH A: Original Research Contribution (Recommended)

### Best Topic

> **"Consistency-Constrained Multi-Task Bengali Hate Speech Detection with Faithfulness-Evaluated Generative Explanations"**

### Why This Works

It sits precisely at the intersection of **G1 + G2 + G3** — the three strongest verified gaps. No single existing paper addresses even two of these together.

### Precise Novelty Claims (What You Can Defend)

**Claim 1 — First model-level cross-task consistency mechanism for Bengali hate speech**
- Build a compatibility matrix from BanglaMultiHate's co-occurrence statistics
- Add a consistency loss term penalizing impossible label combinations (e.g., Type=None + Severity=Severe)
- Report **contradiction rate** as a new metric (% of predictions violating logical rules)
- Ablate: show that per-task F1 stays comparable but contradiction rate drops significantly

**Claim 2 — First faithfulness evaluation of generative Bengali hate explanations**
- Generate free-text rationales using a LoRA-tuned LLM (or a seq2seq head on BanglaBERT)
- Measure faithfulness using comprehensiveness + sufficiency + AOPC deletion curves on the generated rationales
- Report the **faithfulness-plausibility gap** (how often explanations sound good but are internally unfaithful) — this diagnostic table is genuinely novel for Bengali

**Claim 3 — First unified multi-task + generative explanation Bengali system**
- A single model that predicts Type + Target + Severity AND generates a rationale
- Neither BanglaMultiHate baselines nor BanHADEX experiments do this simultaneously

### What's NOT Novel (Be Honest About This)
- Multi-task classification itself → done (BanglaMultiHate, BLP-2025)
- Generating Bengali rationales → done (BanHADEX + LoRA experiments)
- Adversarial training, focal loss, encoder ensembles → established engineering
- Dialect-specific hate speech resources → done (BIDWESH)

### Methodology

```
Input: Bengali social media text
    ↓
┌───────────────────────────────────────────────┐
│  Shared Encoder                               │
│  Option A: BanglaBERT (or ensemble w/ MuRIL)  │
│  Option B: LoRA-tuned LLM (Llama-3.2-3B)     │
└──────────────┬────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ↓          ↓          ↓
┌────────┐┌────────┐┌──────────┐
│ Type   ││ Target ││ Severity │
│ Head   ││ Head   ││ Head     │
│(6-cls) ││(5-cls) ││(ordinal) │
└───┬────┘└───┬────┘└────┬─────┘
    │         │          │
    └─────────┼──────────┘
              ↓
    ┌─────────────────────┐
    │ Consistency Layer    │  ← YOUR NOVEL COMPONENT
    │ (compatibility      │
    │  matrix penalty)    │
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │ Rationale Generator  │  ← YOUR NOVEL COMPONENT
    │ (seq2seq or LLM     │
    │  generation head)   │
    └─────────┬───────────┘
              ↓
    ┌─────────────────────┐
    │ Faithfulness Eval    │  ← YOUR NOVEL COMPONENT
    │ (comprehensiveness,  │
    │  sufficiency, AOPC)  │
    └─────────────────────┘
```

### Loss Function
```
L = α·L_type(Focal) + β·L_target(Focal) + γ·L_severity(Ordinal-CE) 
    + δ·L_rationale(CE on explanation tokens) 
    + λ·L_consistency(penalty for invalid label combos)
```

### Datasets

| Dataset | Role |
|:---|:---|
| **BanglaMultiHate** (~51K) | Primary: Type + Target + Severity labels |
| **BanHADEX** (~19K) | Explanation supervision (rationale training) |
| **BIDWESH** (~9K) | Secondary: Dialect robustness test set |
| **BLP-2025 test set** | Benchmark comparison against leaderboard |

> [!IMPORTANT]
> **The merge problem:** BanglaMultiHate and BanHADEX use different taxonomies and are disjoint datasets. You have two options:
> 1. **Map taxonomies** — align BanHADEX's 7 categories to BanglaMultiHate's 6 types + add Severity labels via rule-based mapping or small annotation pass
> 2. **Silver-standard rationales** — use a strong LLM to generate rationales for BanglaMultiHate samples, then validate a subset with human annotators
> 
> Option 2 is more practical and itself becomes a minor methodological contribution.

### Evaluation Protocol

| Metric | What It Measures | Novelty |
|:---|:---|:---|
| Macro F1 per task | Classification quality | Standard |
| Exact-match joint accuracy | All 3 tasks correct simultaneously | Rarely reported |
| **Contradiction rate** | % of logically impossible label combos | **Novel for Bengali** |
| BLEU / ROUGE / BERTScore | Explanation text quality | Standard |
| **Comprehensiveness** | Do rationale tokens drive the prediction? | **Novel for Bengali generative XAI** |
| **Sufficiency** | Are rationale tokens enough for prediction? | **Novel for Bengali generative XAI** |
| **Faithfulness-plausibility gap** | Difference between human rating and faithfulness score | **Novel diagnostic** |

### Comparison You'd Show in Your Paper

| System | Multi-Task | Generative Explanation | Consistency | Faithfulness Eval |
|:---|:---:|:---:|:---:|:---:|
| DeepHateExplainer (2020) | ❌ | ❌ (token attribution) | N/A | ✅ (token-level only) |
| BLP-2025 Top Systems (2025) | ✅ | ❌ | ❌ | ❌ |
| BanHADEX + LoRA (2026) | Partial (no Severity) | ✅ | N/A | ❌ |
| **Your System** | ✅ | ✅ | ✅ | ✅ |

---

## PATH B: Survey / Review Paper (Backup Plan)

If the experimental contribution feels too risky (compute constraints, dataset alignment issues, timeline), here's a **strong survey topic** that fills a verified gap:

### Survey Title

> **"From Detection to Understanding: A Systematic Review of Multi-Task Learning, Explainability, and LLM Integration for Bengali Hate Speech Moderation (2020–2026)"**

### Why This Survey Is Needed

The only existing Bengali hate speech survey is from **Journal of Big Data (2024)**, which:
- Predates BanglaMultiHate, BanHADEX, BLP-2025, and all LoRA/LLM work
- Covers traditional ML + early DL only
- Does not discuss multi-task formulations, generative explainability, or faithfulness
- Does not cover dialectal or code-mixed challenges in depth

**No survey exists that covers the 2025–2026 explosion of work.** This is a genuine gap.

### Survey Structure

1. **Scope & Taxonomy** — Define the space: classification paradigms (binary → multi-class → multi-task), explainability types (post-hoc → intrinsic → generative), model families (ML → DL → Transformers → LLMs)
2. **Dataset Landscape** — Catalog all Bengali hate datasets (BD-SHS, BanglaMultiHate, BanHADEX, BIDWESH, MUTE, BanTH, BOISHOMMO) with size, task, annotation scheme, dialect coverage, and rationale availability
3. **Multi-Task Learning for Bengali Hate** — Deep analysis of BLP-2025 systems, architectures, what worked and what didn't
4. **Explainability Methods** — From DeepHateExplainer's LRP to BanHADEX's generative rationales; faithfulness vs. plausibility distinction
5. **LLM Integration** — Zero-shot vs. few-shot vs. LoRA; which models, which tasks, performance gaps
6. **Dialectal & Code-Mixed Challenges** — BIDWESH, Romanized Bengali, the transliteration bottleneck
7. **Open Problems & Research Agenda** — The consistency gap, faithfulness gap, dialect+XAI gap, multimodal frontier

### Where to Publish (Survey)
- **Expert Systems with Applications** (Elsevier) — survey-friendly, good IF
- **ACM Computing Surveys** — prestigious, survey-only venue
- **Information Fusion** (Elsevier) — if you emphasize multi-modal/multi-task fusion angle
- **Journal of Big Data** (Springer) — directly succeeds the 2024 survey

---

## My Honest Recommendation

| Factor | Path A (Original) | Path B (Survey) |
|:---|:---|:---|
| **Novelty strength** | ⭐⭐⭐⭐⭐ (verified empty intersection) | ⭐⭐⭐⭐ (survey gap is real but less "exciting") |
| **Compute needed** | Medium-High (at least 1× A100 or 2× RTX 4090 for LLM; less if encoder-only) | None |
| **Time to complete** | 4–6 months | 2–3 months |
| **Risk** | Medium (dataset alignment, faithfulness metrics implementation) | Low |
| **Publication venue** | ACL/EMNLP/NAACL workshop or main, BLP | ESWA, ACM Surveys, J Big Data |
| **For thesis** | Ideal for MS/PhD thesis | Good for MS thesis or supplementary publication |

> [!TIP]
> **Best strategy if you have time:** Do BOTH. Write the survey first (2 months), which forces you to deeply understand the landscape. Then build the system (Path A) as a follow-up paper. The survey becomes your "Related Work" section on steroids.

---

## Final Decision Needed From You

1. **Path A (original contribution)** — I'll draft the full problem statement, formal notation, and experimental plan
2. **Path B (survey)** — I'll draft the survey outline with paper counts per section and reading list  
3. **Both** — I'll create a phased plan starting with the survey

Which path do you want to commit to?
