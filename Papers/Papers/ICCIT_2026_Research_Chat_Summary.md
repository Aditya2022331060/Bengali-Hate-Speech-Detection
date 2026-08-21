# Bengali Harmful Content Research — Chat Summary
*ICCIT 2026 submission planning — compiled August 10, 2026*

---

## 1. What Was Reviewed

Two uploaded documents were checked against current literature:
- `research_analysis.md` — gap verification and topic recommendation
- `paper_sti2026.tex` — IEEE-format draft paper (consistency-constrained multi-task Bengali hate speech + faithfulness-evaluated explanations)

---

## 2. Verification of Claimed Datasets & Prior Work

| Item | Status | Notes |
|---|---|---|
| **BanglaMultiHate** | ✅ Confirmed real | First multi-task Bangla hate dataset (Type/Target/Severity). BLP-2025 split: 35,522 train / 2,512 dev / 10,200 test (~48K total). |
| **DeepHateExplainer (2020)** | ✅ Confirmed, scope correctly narrowed | Token-level LRP/sensitivity attribution, single-task (4-class), 2020 BERT ensemble. Did **not** do generative-rationale faithfulness — original claim in analysis doc was correctly walked back. |
| **BanHADEX / "BanHate"** | ⚠️ **Unresolved — action required** | Found third-party citations to "BanHADEX: Towards Explainable Hate Speech Detection in Bangla Using Human Annotated EXplanation," including a note that its explanation-evaluation metrics are generic text-similarity metrics, not faithfulness metrics — which *supports* your gap claim. However, a paper matching your exact stats (19,203 YouTube comments, 7 categories, 7 target groups) was found under the name **"BanHate"** (Raquib et al., BLP-2025 workshop), whose own abstract does **not** mention free-text explanations at all. These may be the same resource under two names, or two different resources. **Must be resolved before finalizing Gap 2/3 claims.** |
| **2024 Journal of Big Data survey (Al Maruf et al.)** | ✅ Confirmed still the standard reference | Still cited as-is by BLP-2025 papers in late 2025; no successor survey found. Supports Path B gap claim. |
| **BLP-2025 system papers** | ⚠️ Incomplete coverage | Draft only discusses Code_Gen, Catalyst, HateSense. Additional systems found in search: **Gradient Masters**, **Retriv**. The "no BLP-2025 system reports contradiction rate" claim needs verification against the full set (~15–20 papers), not a subset. |
| **X-MuTeST (AAAI 2026)** | 🆕 New finding — not in original analysis | Explainability framework for Hindi/Telugu/English hate speech using LLM reasoning + attention, evaluated with both faithfulness (Comprehensiveness, Sufficiency) and plausibility (Token-F1, IOU-F1) metrics. Not Bengali, so your gap survives, but should be cited/discussed in Related Work to preempt reviewer questions. |

---

## 3. Novelty Verdict

- **G1 (model-level cross-task consistency constraint)** — confirmed open, strong.
- **G3 (unified multi-task + generative explanation system)** — confirmed open, strong.
- **G2 (faithfulness of generative Bengali rationales)** — open, but **contingent on resolving the BanHADEX/BanHate identity question** above.

No single existing Bengali paper combines even two of these three gaps.

---

## 4. Feasibility: Kaggle + 20 Days (ICCIT Deadline)

**Compute is not the constraint.** Kaggle free tier: ~30 GPU hours/week (P100 or T4x2), 12-hour session cap. Over 20 days ≈ 80–85 total GPU hours — comfortably enough for BanglaBERT (110M params) multi-task fine-tuning, hyperparameter sweeps, and ablations (est. 15–25 GPU hours needed for the core work).

**Time and scope are the real constraints.** The full paper as drafted asks for three novel contributions + reproduction of three external SOTA baselines + a dataset merge, all in 20 days alongside writing a 6-page IEEE paper. That's not realistic without cutting scope.

### Recommended scope cut
- **Primary contribution (full rigor):** G1 — consistency-constrained multi-task learning + contradiction rate metric. Fully achievable standalone paper.
- **Secondary contribution (small, honestly-scoped pilot):** G2/G3 — faithfulness of generated rationales on a small subset (~200–500 samples), explicitly framed as preliminary, not a full benchmark.
- **Drop:** reproducing Code_Gen / Catalyst / HateSense from scratch. Cite their **published** numbers in the comparison table instead of retraining them.

### 20-Day Plan

| Days | Task |
|---|---|
| 1–2 | **Critical path**: confirm BanglaMultiHate downloads on Kaggle; resolve BanHADEX vs. BanHate identity/content question. Set up notebook + checkpointing to a Kaggle Dataset (sessions cap at 12h). |
| 3–5 | Baselines: single-task BanglaBERT per subtask, then vanilla multi-task (λ=0). Log per-task F1. |
| 6–8 | Build compatibility matrix, implement consistency loss, sweep λ ∈ {0.01, 0.1, 0.5, 1.0}, compute contradiction rate. **This is the headline result table.** |
| 9–10 | Ablations (Focal loss, ordinal severity, consistency on/off), exact-match joint accuracy. |
| 11–13 | Rationale pilot on a subset + comprehensiveness/sufficiency + qualitative examples. If dataset question blocks this, pivot time to deeper G1 error analysis instead. |
| 14–15 | Finalize tables/figures, architecture diagram, real Bengali error-analysis examples. |
| 16–18 | Write paper: fill placeholders with real numbers, trim Abstract/Intro/Contributions to match what was actually built, fit 6-page IEEE limit. |
| 19 | Internal review — every claim must be backed by a produced table. |
| 20 | Buffer: formatting, proofreading, submission portal. |

**Single biggest risk:** Day 1–2, the BanHADEX/BanHate question. If it resolves unfavorably, you need to know on day 2, not day 12.

---

## 5. Applications of the Research

- **Platform-level content moderation** — coherent (non-contradictory) multi-dimensional labels are directly usable in automated escalation pipelines.
- **Moderator-facing explainability tooling** — faithful (not just fluent) rationales let human moderators trust flags without re-reading everything, useful for a 300M-speaker language with few native-language moderation tools.
- **Regulatory/audit trails** — a documented faithfulness score for AI-generated moderation rationales is a more defensible audit artifact than an unverified explanation.
- **Template for other low-resource languages** — the consistency-loss + faithfulness-evaluation combination is transferable to other under-resourced-language hate speech efforts (Sinhala, Nepali, etc.).

---

## 6. Open Action Items

1. Pull the actual BanHADEX (or BanHate) paper/PDF and confirm: exact title, whether free-text rationales exist, dataset size/taxonomy.
2. Check all ~15–20 BLP-2025 system papers (not just 3) before claiming none report contradiction rate.
3. Add a brief Related Work mention of X-MuTeST (AAAI 2026) to preempt reviewer questions about parallel Indic-language faithfulness work.
4. Decide and commit to the scoped-down plan (G1 primary + G2/G3 pilot) before writing further.
