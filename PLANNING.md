# The Tokenizer Tax — Project Execution Plan

Oct 22, 2026 · @RRSS

## Overview

This is a 10-day sprint, Fri Oct 23 to Sun Nov 1, 2026: four people working in parallel, a cut-down experiment set that runs on free GPUs, a lightweight demo, a full report and a short paper draft. The trick is to cut breadth, not the core idea: the equal-token vs equal-content test (E3) and the fragmentation test (E4) stay, because they are what make this more than a replication.

**Deliverables by Nov 1**

| Deliverable | 10-day version | Done when |
| --- | --- | --- |
| Toolkit (`tokentax`) | One repo of scripts + notebooks that reproduce every figure | Each figure regenerates from one script |
| Demo app | Tokenizer Tax Calculator on Hugging Face Spaces (Gradio): paste text, see tokens, tax, cost, context loss; plus a results tab | Public link works |
| Project report | University format, built from the synopsis + corrected lit survey + results | Guide has it on Nov 1 |
| Research paper | 4-page short-paper draft (ACL format); polish and post to arXiv after the deadline | Draft complete |
| Demo video | 3-minute screen recording | Uploaded |

**Research questions**

1. **RQ1 – Size of the tax.** How many more tokens do Indian languages need than English across 7 tokenizers, and what does it cost?
2. **RQ2 – Context or capability?** How much of the accuracy gap closes when each language gets equal content instead of an equal token budget?
3. **RQ3 – Does fragmentation itself hurt?** With model, language and text fixed, does forcing more fragmented tokenization lower accuracy?
4. **RQ4 – Which tokenizer helps most?** Answered from E1 by comparing existing tokenizers; training your own is cut.

## Formal problem statement

The question is causal: of the accuracy gap between an Indian language and English, how much comes from lost context, how much from fragmentation itself, and how much from the model simply knowing the language less well. The tax and its costs are defined first, then the causal model, then the exact quantity the project estimates.

### Setup and definitions

- **Languages** L = {ℓ₀, ℓ₁, …, ℓₙ}, with ℓ₀ = English.
- **Parallel content**: items c (FLORES sentences or Belebele passages); x\_ℓ(c) is item c written in language ℓ. All versions of c carry the same meaning.
- **Tokenizer** τ maps a string to its canonical token sequence; δ decodes it, so δ(τ(x)) = x. A **segmentation** s of x is any token sequence that decodes to x; τ(x) is just one of many.

**Tax** (token length relative to English for the same content) and **fertility** (tokens per word):

```latex
T_\tau(\ell) = \frac{\sum_{c} |\tau(x_\ell(c))|}{\sum_{c} |\tau(x_{\ell_0}(c))|} \qquad F_\tau(\ell) = \frac{\sum_{c} |\tau(x_\ell(c))|}{\sum_{c} \mathrm{words}(x_\ell(c))}
```

**Single-byte rate** σ\_τ(ℓ): the share of tokens whose bytes are not a complete UTF-8 character, i.e. failed merges.

### Costs of the tax

With price π per token, a window of W tokens, and attention cost growing with the square of length:

```latex
\frac{\mathrm{Cost}_\tau(\ell)}{\mathrm{Cost}_\tau(\ell_0)} = T_\tau(\ell) \qquad K_\tau(\ell) = \frac{W}{T_\tau(\ell)} \qquad \frac{\mathrm{FLOPs}_{\mathrm{attn}}(\ell)}{\mathrm{FLOPs}_{\mathrm{attn}}(\ell_0)} \approx T_\tau(\ell)^2
```

K is usable context in English-equivalent tokens: at T = 8, a 128k window holds what English fits in 16k.

### Accuracy, and why it is hard to attribute

A model M answers item q (passage, question, four options, gold answer y). Accuracy depends on three things we can control: the language ℓ, the segmentation s (canonical τ or something finer), and the token budget B (input cut to its first B tokens; B = ∞ means no cut):

```latex
A_M(\ell, s, B) = \mathbb{E}_q\left[\mathbf{1}\{M(\mathrm{trunc}_B(s(x_\ell(q)))) = y_q\}\right] \qquad \Delta(\ell) = A_M(\ell_0, \tau, B) - A_M(\ell, \tau, B)
```

&#91;embedded content: Causal model of the tokenizer tax · 6 variables, 2 interventions\]

The amount of training data for a language shapes both the tokenizer (fewer merges learned, so a higher tax) and the model's competence. So a correlation between tax and accuracy across languages, as in Srivastava et al., cannot separate the paths: low-resource languages have both a high tax and low competence. With 12–14 languages the correlation also has very little statistical power. The way out is to **intervene** on one variable while holding the others fixed, which is what E3 and E4 do.

### Target: a three-way gap decomposition

For each language ℓ and model M, let s^ℓ be English re-segmented until it has as many tokens as language ℓ needs for the same passage. Then:

```latex
\underbrace{A(\ell_0, \tau, B) - A(\ell, \tau, B)}_{\Delta(\ell)} = \underbrace{A(\ell, \tau, \infty) - A(\ell, \tau, B)}_{\Delta_{\mathrm{ctx}}:\ \text{context lost}} + \underbrace{A(\ell_0, \tau, \infty) - A(\ell_0, s^{\ell}, \infty)}_{\Delta_{\mathrm{frag}}:\ \text{fragmentation}} + \underbrace{A(\ell_0, s^{\ell}, \infty) - A(\ell, \tau, \infty)}_{\Delta_{\mathrm{res}}:\ \text{competence and data}}
```

The identity is exact when English fits the budget, so that A(ℓ₀, τ, B) = A(ℓ₀, τ, ∞); B is chosen that way. Every term compares accuracies on the **same questions**, so each gets a paired test and a bootstrap CI.

The key move is Δ\_frag. Making Malayalam less fragmented is impossible without a new vocabulary, so instead make English **as fragmented as Malayalam** and measure how much a model that knows English well loses. That holds competence fixed and changes only fragmentation.

## Existing solutions and their shortcomings

Existing work either **diagnoses** the tax or **mitigates** it. No diagnostic method so far attributes the accuracy gap to its causes on a deployed model, and every mitigation needs a new tokenizer, a new model, or both.

**Diagnosing the tax**

| Approach | Representative work | What it gets right | Shortcoming |
| --- | --- | --- | --- |
| Token-length ratios on parallel text | Petrov et al. 2023; Ahia et al. 2023; Lundin et al. 2025; Tamang & Bora 2024 | Clean, cheap, same content across languages | Intrinsic only: says nothing about accuracy; price snapshots go stale |
| Mechanism analysis of the tax | Srivastava 2026 (failed merges); Mandarapu & Kunkunuru 2026 (removable vs irreducible tax) | Explains *why* token counts inflate | Still about token counts, not task accuracy |
| Cross-language correlation of fertility vs accuracy | Srivastava 2026 (Belebele) | Uses real models and a real task | Confounded by training data per language; 12–14 data points; cannot show cause |
| Train controlled models with different tokenizers | Rust et al. 2021; Ali et al. 2023; Dwivedi & Gopalan 2026 | Truly causal for the tokenizer choice | Very expensive (Ali et al. trained 24 models); small models; mostly European languages or only Hindi/Marathi; says nothing about deployed LLMs |
| Feed models non-canonical tokenizations | Zheng et al. 2025 | Causal within one model, no training needed | English-centric benchmarks; never used to explain a cross-lingual gap |

**Mitigating the tax**

| Approach | Representative work | What it gets right | Shortcoming |
| --- | --- | --- | --- |
| Larger multilingual vocabularies | o200k, Gemma (256k), XLM-R | Cuts the Indic tax a lot (about 73% in Srivastava) | Bigger embedding tables; gap shrinks but does not close |
| Indic-specific tokenizers | Sarvam-1, MUTANT-Indic / IndicSuperTokenizer, MorphTok, indictok | Lowest fertility for Indic | Need a new model or vocabulary extension + continued pretraining; lower fertility not shown to raise accuracy (Ali et al.) |
| Fairer tokenizer training | Parity-aware BPE, merge rebalancing (Thakur et al. 2025), adaptive data mixtures (Singh et al. 2025) | Targets the cause in the vocabulary | Requires retraining the tokenizer and the model |
| Byte- or morphology-level models | ByT5, MYTE, BLT (Tamang & Bora 2026 study) | Removes vocabulary bias | More compute; BLT weak on low-resource languages |
| Romanize the input | Prompting in Latin-script Hindi, Bengali, etc. | Fewer tokens with today's models | Lossy, unfamiliar to many users, model may know romanized text less well |
| Subword regularization in training | BPE-dropout (Provilkov et al. 2019) | Makes models robust to splits | A training technique, not a fix for deployed models |

**The gap in the literature**: no method tells a practitioner, for a given deployed model and language, how much of the accuracy loss is fixable by more context (a cheaper or longer-context deployment), how much by a better tokenizer, and how much only by more training data. That is what the next section's method answers.

## Proposed method: counterfactual tax decomposition

The method estimates the three terms of the decomposition above for any open model, using only inference: no training, no new tokenizer, about a GPU-hour per model. Each term comes from one controlled intervention on the same 300 Belebele questions.

### The five steps

1. **Audit (E1–E2).** Compute T, F and σ for 7 tokenizers × 13 languages on FLORES, and the cost and context consequences. This sets the per-item fragmentation targets for step 3.
2. **Context intervention (E3) → Δ\_ctx.** For each item, set the budget to the English passage's own token count in the model's tokenizer, cut the target-language passage to it, and compare with the uncut passage. Language and segmentation stay fixed; only content changes.
3. **Matched fragmentation of English (E4b, new) → Δ\_frag.** Re-segment each English passage until it has exactly as many tokens as the same passage in language ℓ (Algorithm 1). The text, the language and the model's competence stay fixed; only fragmentation changes.
4. **Residual → Δ\_res.** Whatever remains between fragmented English and the uncut Indic passage is competence and data, the part only more training data can fix.
5. **Mechanism checks (E4a).** Within each Indic language, add extra fragmentation at rates p ∈ {0.1, 0.3, 0.5}, plus one **single-byte** condition that splits every multi-byte character into raw UTF-8 bytes, reproducing the failed-merge pattern. This tests Srivastava's mechanism (failed merges, r = 0.89) causally rather than by correlation. If time allows, compare base and instruct versions of the same model, since Zheng et al. trace robustness to instruction tuning.

### Algorithms

```python
# Algorithm 1 - match_fragment: make English as fragmented as language l
def match_fragment(tok, en_ids, target_len, rng):
    s = list(en_ids)                         # canonical English tokens
    while len(s) < target_len:
        splittable = [i for i, t in enumerate(s) if len(tok.decode([t])) > 1]
        if not splittable:                   # already character-level: cap reached
            return s, True                   # (ids, capped)
        i = rng.choice(splittable)
        b = tok.token_bytes(s[i])
        cuts = [k for k in range(1, len(b)) if tok.has(b[:k]) and tok.has(b[k:])]
        k = rng.choice(cuts)                 # single bytes are always in a byte-level vocab
        s[i:i+1] = [tok.id(b[:k]), tok.id(b[k:])]
    assert tok.decode(s) == tok.decode(en_ids)  # same text, finer split
    return s, False

# target_len for item q and language l (per item, not an average):
#   round(len(tok(x_l(q))) / len(tok(x_en(q))) * len(tok(x_en(q))))  ==  len(tok(x_l(q)))
```

```python
# Algorithm 2 - decompose the gap for model M and language l on items Q
def decompose(M, tok, Q, l, rng, n_boot=1000):
    rows = []
    for q in Q:
        en, xl = tok(q.passage['en']), tok(q.passage[l])
        a_en      = correct(M, en, q)                         # A(en, canonical, full)
        a_l_full  = correct(M, xl, q)                         # A(l, canonical, full)
        a_l_cut   = correct(M, xl[:len(en)], q)               # A(l, canonical, budget B = |en|)
        en_frag, capped = match_fragment(tok, en, len(xl), rng)
        a_en_frag = correct(M, en_frag, q)                    # A(en, s^l, full)
        rows.append((a_en, a_l_full, a_l_cut, a_en_frag, capped))
    d_ctx  = mean(a_l_full  - a_l_cut)
    d_frag = mean(a_en      - a_en_frag)
    d_res  = mean(a_en_frag - a_l_full)
    return d_ctx, d_frag, d_res, bootstrap_ci(rows, n_boot), share_capped(rows)
```

`correct()` is the letter-choice scorer from the harness; `share_capped` reports how often English hit character level before matching the target, so those items can be shown separately.

### Reading the result

| If this term dominates | It means | What fixes it |
| --- | --- | --- |
| Δ\_ctx | The tax hurts by crowding content out of the window | Longer context or paying for more tokens; no retraining |
| Δ\_frag | Fine-grained input itself confuses the model | A better tokenizer (o200k-style vocab, Indic tokenizers) |
| Δ\_res | The model knows the language less well | More or better training data in that language |

This table is also the answer to RQ4, per language and per model.

### Why this has the fewest shortcomings

| Shortcoming of existing work | How this method avoids it |
| --- | --- |
| Correlation across languages is confounded by data | Every term compares one variable at a time on the same items; no cross-language correlation needed |
| Only 12–14 data points | Tests run per item (300 per language), paired, so power is far higher |
| Controlled tokenizer studies need costly retraining | Inference only, on models people actually deploy |
| Can't make Indic text less fragmented without a new vocabulary | Fragments English up to the Indic level instead |
| Non-canonical tokenization studied only in English | Used here specifically to explain a cross-lingual gap |
| Intrinsic metrics don't predict accuracy | Every number is an accuracy difference, not a token count |

### Remaining limitations (state these in the report)

- **Out-of-distribution input.** Fragmented English is a split the model never saw in training, while Indic canonical splits were seen. So Δ\_frag is best read as an **upper bound** on the pure fragmentation effect.
- **Different kind of fragment.** English fragments into characters; byte-level BPE fragments Indic into partial characters. The single-byte condition in step 5 covers that gap.
- **Cap.** English cannot be split finer than characters; when a language's tax under the model's tokenizer exceeds English's characters per token, matching is capped and those items are reported separately.
- **Truncation is an approximation of a real context limit.** Cutting a passage short mimics a full window, but positional effects at real 128k lengths are not tested.
- **Scope.** One task (reading comprehension), 2 small models, 300 items per language; findings may differ for generation tasks or larger models.

### Change to the sprint plan

Add **E4b** to Rohith's lane on Day 5 (it replaces "seeds 1–2"): English matched-fragmentation runs, 300 questions × 6 language targets × 2 models, roughly as cheap as one baseline pass. Rohit's `answer_kept` field is no longer needed for the headline result, since Δ\_ctx is measured directly.

## Critical review of the synopsis

The biggest risk is novelty: Paper 1 (Srivastava et al., 2026) already measures fertility, bytes-per-token, context erosion and the Belebele correlation for 14 Indian languages and 6 tokenizers, and explicitly leaves causality open. Objectives 1–2 alone would be a replication, so the project must win on objectives 3–4.

**What keeps it new**

- **A real intervention, not a correlation.** Equal-token vs equal-content separates *context loss* from everything else, but it cannot tell fragmentation apart from model capability. Add a within-model fragmentation experiment (E4 below): same model, same passage, same language, but tokenized into more pieces. That is the cleanest causal test available without retraining a model.
- **A context budget that actually binds.** Belebele passages are short (roughly 80–120 English words), so a single passage rarely hits a modern context limit. Use small budgets or a multi-passage setup (E3) so the equal-token condition really cuts content.
- **Control for resource level.** Use a mixed-effects regression with a resource covariate, which Paper 1 flags as the confounder.
- **Include an Indic-built tokenizer and model** (e.g. Sarvam-1) so the "what fixes it" answer is grounded.

**Fixes needed in the literature survey** (check each against the paper before the report)

| Paper | Problem | Fix |
| --- | --- | --- |
| 14 (Ahia et al.) | Link points to Petrov et al. (2305.15425), and the title does not match | The Ahia et al. 2023 paper is "Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models" (arXiv 2305.13707, EMNLP 2023); verify and correct |
| 2 (Petrov) | Write-up says "1 to 15 additional tokens"; the finding is up to 15× longer | Reword as a ratio |
| 11 (Rust) | "Fertility less than 1" is impossible; fertility is at least 1 | Re-read the paper and restate the threshold correctly |
| 3 (Tamang & Bora) | Table says BLT underperforms on low-resource; prose says PARITY does | Check and make both agree |
| 15 (SentencePiece) | Says it was trained "on raw Hindi" and proposed BPE-dropout | Neither is in that paper; BPE-dropout is Paper 16 |
| 6 (Dhar) | Typo "fi ve official languages"; there are 22 | Fix |
| All | Format mixes "Paper N:" and "3." headings | Use one citation style (ACL natbib) throughout |

**Papers to add** (search and confirm before citing): recent work on LLM robustness to non-canonical tokenizations (needed to position E4), Belebele's romanized Indic variants, and the Sarvam-1 / IndicTrans2 tokenizer reports.

## Scope decisions

Freeze scope on day 1: all 12 languages + English for the cheap tokenizer audit, but only 6 languages + English, 2 models and 300 questions per language for the GPU experiments. Everything below runs on free Colab/Kaggle GPUs.

**Languages**

| Used in | Languages (FLORES codes) |
| --- | --- |
| Tokenizer audit E1–E2 (CPU, minutes) | English plus Hindi, Marathi, Bengali, Assamese, Gujarati, Punjabi, Odia, Urdu, Tamil, Telugu, Kannada, Malayalam |
| GPU experiments E3–E4 | English, Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam |

The 6 GPU languages cover both families and the range from low tax (Hindi) to the highest (Malayalam). Kannada gives you native-speaker sanity checks.

**Tokenizers (7):** cl100k\_base and o200k\_base (`tiktoken`), Llama 3.x, Qwen2.5, Gemma 2, XLM-R, Sarvam-1 (all via Hugging Face). Llama and Gemma are gated: accept the licences tonight, approval can take hours.

**Models (2):** Qwen2.5-3B-Instruct and Gemma-2-2B-it in 4-bit on a Colab/Kaggle T4. Add Sarvam-1 only if a pilot shows the other two near chance on Indic.

**Data:** FLORES-200 devtest (1,012 sentences) for E1; Belebele, first 300 questions per language, for E3–E4; a dated CSV of API prices for E2.

**Cut for time** (list them as future work): Assamese/Gujarati/Punjabi/Odia/Urdu/Marathi in downstream tests, mBERT, the multi-passage version of E3, romanized-script test (E5), mixed-effects regression, training your own tokenizer (E7), latency timing.

## Sprint plan: 4 parallel lanes

Four lanes build independently Oct 23–28, merge into one tested `main` on Thu Oct 29, and leave Oct 30–31 for finding mistakes and Nov 1 for submission only. Lanes stay independent because every shared interface is fixed on Day 1 and every lane can work against fake data until the real data arrives.

### Sprint at a glance

| Day | Lane A – Saket: tax audit, figures, paper | Lane B – Rohit: harness, baseline, E3 | Lane C – Rohith: E4, statistics | Lane D – Sachi: demo, report |
| --- | --- | --- | --- | --- |
| D1 Fri Oct 23 | Repo scaffold, tokenizer wrapper, FLORES loader | Belebele loader, scorer, Qwen loads in 4-bit | Re-segmentation function + round-trip tests | Lit survey fixes, report template, app skeleton on fake CSVs |
| D2 Sat Oct 24 | E1 run: 7 tokenizers × 13 languages | Pilot: 50 Qs × 7 langs × 2 models; pick models | Stats helpers; E4 dry run on 20 English Qs | Calculator tab working with real tokenizers |
| D3 Sun Oct 25 | E2 cost + context; 3 figures | Full baseline: 300 Qs × 7 langs × 2 models | E4 run: model 1, English + Hindi | Report ch. 1–3; app v1 deployed |
| D4 Mon Oct 26 | Fertility–accuracy correlation; methodology text | E3 equal-token runs, model 1 | E4 run: model 1, Tamil + Malayalam | Results tab on real E1 CSV; report ch. 4 |
| D5 Tue Oct 27 | Shared figure style + E3/E4 plot scripts; paper skeleton | E3 runs, model 2 | E4 model 2 or extra seeds | Context-squeeze tab (optional); demo script |
| D6 Wed Oct 28 | Paper results section with real numbers | E3 analysis + results text | E4 analysis + results text; conclusion | App on real E3/E4 data; report ch. 7 |
| **D7 Thu Oct 29** | **Merge day: all PRs in by 11:00, `main` green by 18:00, results frozen** |  |  |  |
| D8 Fri Oct 30 | QA: checks Rohith's work | QA: checks Saket's work | QA: checks Rohit's work | QA: tests app, checks report numbers |
| D9 Sat Oct 31 | Fix round 2, paper polish | Fix round 2 | Fix round 2 | Demo video, report formatting |
| D10 Sun Nov 1 | Submit; no new work |  |  |  |

The heavy GPU runs (D3–D5) sit in lanes B and C on separate Colab/Kaggle accounts, so they never compete for the same quota.

### Day-1 contracts (agree on these by 1 pm Friday)

Parallel work only holds if nobody changes these names after Day 1. Each owner pushes a stub with the exact signature by Friday evening, even if it returns fake data, so every other lane can import it immediately.

**Shared functions**

| Module | Function | Returns | Owner | Stub by |
| --- | --- | --- | --- | --- |
| `tokentax/tokenizers.py` | `get_tokenizer(name)` | Object with `encode(text) → list[int]`, `decode(ids) → str`, `pieces(text) → list[str]` | Saket | D1 eve |
| `tokentax/tokenizers.py` | `TOKENIZERS` | List of the 7 names, e.g. `"cl100k", "o200k", "llama3", "qwen2.5", "gemma2", "xlmr", "sarvam1"` | Saket | D1 noon |
| `tokentax/data.py` | `load_flores(lang)` | List of 1,012 devtest sentences, NFC-normalized | Saket | D1 eve |
| `tokentax/data.py` | `load_belebele(lang, n=300)` | DataFrame: `qid, passage, question, opt1..opt4, gold` (gold = 1–4) | Rohit | D1 eve |
| `tokentax/eval/score.py` | `load_model(name)` and `score(model, passage_ids, question, options) → int` | Predicted option 1–4, by option log-likelihood | Rohit | D2 eve |
| `tokentax/eval/resegment.py` | `resegment(tokenizer, ids, p, seed) → list[int]` | Finer split that decodes to the same string | Rohith | D1 eve |
| `tokentax/stats.py` | `bootstrap_ci(correct)`, `mcnemar(a, b)`, `holm(pvals)` | (low, high), p-value, adjusted p-values | Rohith | D2 eve |

`score` takes passage **token ids**, not text, so E3 (truncated ids) and E4 (re-segmented ids) both plug into the same scorer.

**Result files** (one row per item; never rename a column after Day 1)

```csv
results/e1_intrinsic.csv
tokenizer,lang,sent_id,n_tokens,n_words,n_chars,n_bytes,n_single_byte

results/e2_cost.csv
tokenizer,lang,tax_vs_eng,usd_per_1k_eng_words,inr_per_1k_eng_words,pct_context_lost_128k,price_snapshot_date

results/e3_results.csv
model,lang,condition,budget_tokens,qid,passage_tokens_full,passage_tokens_used,answer_kept,pred,gold,correct

results/e4_results.csv
model,lang,p,seed,qid,n_tokens_canonical,n_tokens_used,pred,gold,correct
```

`condition` takes exactly `full`, `eqtok_1.0`, `eqtok_0.5`. Language codes are FLORES codes everywhere (`kan_Knda`, never `kn` or `Kannada`).

**Fake data on Day 1**: Saket commits a `scripts/make_fake_results.py` that writes all four CSVs with random values in the right shape. Sachi builds the app and Saket builds figure scripts against these, then swaps in real files with zero code changes.

### Lane A – Saket: tax audit, cost, figures, paper

**Mission:** every intrinsic number (E1–E2), every figure, and the paper draft. Branch `lane-a`.

**D1 Fri Oct 23**

- [ ] Repo scaffold: `pyproject.toml`, `src/tokentax/`, `scripts/`, `results/`, `tests/`, `app/`, a `README` with setup steps
- [ ] `get_tokenizer`: `tiktoken.get_encoding("cl100k_base" / "o200k_base")`; Hugging Face `AutoTokenizer.from_pretrained(...)` for the rest, always `add_special_tokens=False`
- [ ] `load_flores`: FLORES-200 devtest from the Hugging Face hub, `unicodedata.normalize("NFC", s)`
- [ ] `make_fake_results.py` (contract above); push stubs by evening

**D2 Sat Oct 24**

- [ ] `scripts/run_e1.py`: per sentence and tokenizer, count tokens, words (whitespace split), characters, UTF-8 bytes, and single-byte tokens (a token whose bytes are not a complete UTF-8 character; `tiktoken` exposes `decode_single_token_bytes`, Hugging Face byte-level tokenizers need the byte decoder)
- [ ] Tax per (tokenizer, language) = total tokens ÷ total English tokens on the same 1,012 sentences; also fertility = tokens ÷ words
- [ ] Sanity check: your cl100k and o200k taxes should be close to Srivastava et al.; note any big gap and why

**D3 Sun Oct 25**

- [ ] `configs/prices_2026-10-25.csv`: published input price per million tokens for the APIs whose tokenizers you use, with source links
- [ ] `scripts/run_e2.py`: cost per 1,000 English-equivalent words; English-equivalent words that fit in 8k and 128k windows; % of window lost
- [ ] Figures (`scripts/fig_*.py`, one PNG + one PDF each): tax heatmap (language × tokenizer), cost bar chart, context-loss chart

**D4 Mon Oct 26**

- [ ] Spearman correlation of fertility vs baseline accuracy across languages (needs Rohit's baseline CSV, due D3 night)
- [ ] Write the methodology section for E1–E2 (report ch. 5 part A)

**D5 Tue Oct 27**

- [ ] `tokentax/plotstyle.py`: one shared style (fonts, colours, language order)
- [ ] Figure scripts for E3 (grouped bars: full vs eqtok\_1.0 vs eqtok\_0.5 per language) and E4 (accuracy vs fertility lines), built on fake CSVs
- [ ] Overleaf paper skeleton: all sections, related work written

**D6 Wed Oct 28**

- [ ] Rerun all figure scripts on real CSVs
- [ ] Paper: setup + E1–E2 results with real numbers

**Done when:** `make e1 e2 figures` regenerates all CSVs and figures from scratch on a fresh Colab.

### Lane B – Rohit: harness, baseline, E3

**Mission:** the downstream accuracy pipeline and the equal-token vs equal-content experiment. Branch `lane-b`.

**D1 Fri Oct 23**

- [ ] `load_belebele`: the Belebele dataset from the Hugging Face hub; keep the first 300 questions by a fixed order so all languages share the same `qid`s
- [ ] Prompt template (English instructions, target-language content), letter-choice format: passage, question, options A–D, then `Answer:`
- [ ] Scoring by next-token probability of `  A `/`  B `/`  C `/`  D `: one forward pass per question, no generation, so it is fast and fair across languages
- [ ] Load `Qwen/Qwen2.5-3B-Instruct` in 4-bit (`bitsandbytes`) and score 10 English questions

**D2 Sat Oct 24**

- [ ] Pilot: 50 questions × 7 languages × Qwen2.5-3B and Gemma-2-2B-it
- [ ] Message the group by 8 pm: final 2 models, and any language near 25% (chance) to drop
- [ ] Finalize and push `score()` so Rohith can start E4

**D3 Sun Oct 25**

- [ ] Full baseline (`condition = full`): 300 × 7 × 2 = 4,200 items; batch by length, cache each finished item so a Colab disconnect loses nothing
- [ ] Push `e3_results.csv` (baseline rows) by night

**D4 Mon Oct 26**

- [ ] Equal-token conditions: budget = the English passage's token count in the **same model's** tokenizer; cut the language's passage ids to that budget (`eqtok_1.0`) and to half of it for every language (`eqtok_0.5`)
- [ ] `answer_kept`: true if at least half the content words (4+ characters) of the gold option still appear in the kept text; document this as an approximation
- [ ] Run model 1

**D5 Tue Oct 27**

- [ ] Run model 2; push all E3 rows

**D6 Wed Oct 28**

- [ ] Per language: context penalty and residual gap with bootstrap CIs; McNemar for `full` vs `eqtok_1.0`
- [ ] Write E3 results paragraph + table

**Done when:** E3 CSV has every (model, language, condition, qid) row and `make e3` rebuilds it from the cache.

### Lane C – Rohith: E4 and statistics

**Mission:** the fragmentation intervention and every statistical test. Branch `lane-c`.

**D1 Fri Oct 23**

- [ ] `resegment()` for byte-level BPE (Qwen): for each token, with probability *p*, split it into two pieces that are both in the vocabulary (pick a valid split point at random; byte-level vocabularies always contain single bytes, so a split always exists)
- [ ] Test: for 1,000 FLORES sentences in each of 7 languages and *p* ∈ {0.1, 0.3, 0.5}, `decode(resegment(ids)) == decode(ids)`; this test must never fail

**D2 Sat Oct 24**

- [ ] `stats.py`: bootstrap CI (1,000 resamples over `qid`), exact McNemar and Holm via `statsmodels`
- [ ] E4 dry run: 20 English questions with Rohit's `score()`; check accuracy at *p* = 0 matches his baseline exactly

**D3 Sun Oct 25**

- [ ] E4 run, Qwen: English + Hindi, *p* ∈ {0.1, 0.3, 0.5}, seed 0 (*p* = 0 rows come from the baseline)

**D4 Mon Oct 26**

- [ ] E4 run, Qwen: Tamil + Malayalam

**D5 Tue Oct 27**

- [ ] If Gemma is model 2: extend `resegment` to SentencePiece pieces (split the piece string at a character boundary where both halves exist, else use byte-fallback tokens) and run; in all cases also run E4b: English matched fragmentation (Algorithm 1), 300 Qs × 6 language targets

**D6 Wed Oct 28**

- [ ] Per language: accuracy vs mean fertility ratio, slope with bootstrap CI; compare English slope with Indic slopes
- [ ] Write E4 results, limitations (out-of-distribution caveat), conclusion

**Done when:** round-trip test passes in CI and `make e4` rebuilds every row.

### Lane D – Sachi: demo app and report

**Mission:** the public demo, the video, and assembling the report. Branch `lane-d`.

**D1 Fri Oct 23**

- [ ] Apply every lit survey fix from the review table
- [ ] Report file in the university template, all chapter headings in place
- [ ] `app/app.py`: Gradio Blocks with two tabs, reading only from `results/*.csv` (fake ones for now)

**D2 Sat Oct 24**

- [ ] Calculator tab: language dropdown + FLORES sentence picker (so the English translation is known) or free text; for each tokenizer show token count, colour-coded pieces (`gr.HighlightedText`), tax vs English, and cost from the price CSV

**D3 Sun Oct 25**

- [ ] Report chapters 1–3 (intro, literature survey, gaps and objectives)
- [ ] Deploy v1 to a Hugging Face Space; add `HF_TOKEN` as a Space secret so gated tokenizers load

**D4 Mon Oct 26**

- [ ] Results tab: tax heatmap and cost table from the real E1–E2 CSVs (Plotly)
- [ ] Report chapter 4: system design, with an architecture diagram (repo → CSVs → figures, app, report)

**D5 Tue Oct 27**

- [ ] Optional context-squeeze tab; otherwise polish
- [ ] Write the 3-minute demo script

**D6 Wed Oct 28**

- [ ] Results tab shows E3 and E4 charts from real CSVs; test on a phone
- [ ] Report chapter 7 (demo) with screenshots

**Done when:** the Space runs from `main` with real CSVs and loads in under 30 seconds.

### Handoffs and Git workflow

Only seven handoffs cross lanes; if one slips, the receiving lane keeps working on fake data and nobody waits.

| By | From | To | What | If late |
| --- | --- | --- | --- | --- |
| D1 eve | Saket | All | Repo scaffold, `get_tokenizer`, `load_flores`, fake CSVs | Others code against the contract signatures |
| D1 eve | Rohit | Rohith | `load_belebele` | Rohith tests `resegment` on FLORES only |
| D2 eve | Rohit | Rohith | `score()` + chosen models | Rohith moves E4 English run to D3 morning |
| D2 eve | Rohith | Rohit | `stats.py` | Rohit computes plain accuracies first |
| D3 night | Rohit | Saket, Rohith, Sachi | Baseline rows in `e3_results.csv` | Correlation and E4 *p* = 0 rows move to D4 |
| D3 night | Saket | Sachi | Real `e1_intrinsic.csv`, `e2_cost.csv` | App stays on fake data one more day |
| D5 night | Rohit, Rohith | Saket, Sachi | Final E3 and E4 CSVs | Analysis and figures shift to D6 morning |

**Git rules**

- `main` is protected; each lane works on its own branch (`lane-a` … `lane-d`).
- Rebase on `main` every morning; small PRs (shared modules: as soon as they work, reviewed by whoever imports them).
- Lanes own separate files, so conflicts are rare: Saket `tokenizers.py`, `data.py` (FLORES part), `metrics.py`, `cost.py`, `fig_*`; Rohit `eval/score.py`, `eval/budgets.py`, `data.py` (Belebele part); Rohith `eval/resegment.py`, `stats.py`; Sachi `app/`, `report/`.
- Big result files: commit CSVs (small); keep model caches on each person's Drive, not in Git.
- A GitHub Action runs `pytest` on every PR (round-trip test, metric tests, CSV-schema test).

**Daily rhythm**: 15-minute call at 9 pm. Each person says: done today, blocked by, handoff status. Update the checkboxes in this doc right after.

### Merge day – Thu Oct 29

By 6 pm, `main` holds every lane's final code, results, figures and app, all rebuilt from scratch once, and is tagged `v1.0-rc`. After that, only fixes go in.

| Time | Step | Who |
| --- | --- | --- |
| by 11:00 | Each lane opens its final PR: code, CSVs, figures, its report sections | All |
| 11:00–13:00 | Cross-review PRs: Saket reviews lane C, Rohith reviews lane B, Rohit reviews lane A, Saket also reviews lane D's app | All |
| 13:00–14:00 | Merge in order: A → B → C → D (shared modules first, app last) | Saket merges |
| 14:00–16:00 | Clean rebuild on a fresh Colab: clone `main`, install, `make e1 e2 figures`, rebuild E3/E4 CSVs from caches, `pytest` | Rohit + Rohith |
| 14:00–16:00 | Redeploy the Space from `main`; click through every tab | Sachi |
| 16:00–18:00 | Paste final numbers and figures into the report; one combined file | Sachi assembles, all write |
| 18:00 | Tag `v1.0-rc`; **results frozen**; send the guide the report draft + app link | Saket |

**Merge-day exit checklist**

- [ ] All four PRs merged, `pytest` green
- [ ] Fresh-clone rebuild produces identical CSVs (diff them against the committed ones)
- [ ] Every figure in the report regenerated today
- [ ] App on Hugging Face runs from `main`
- [ ] Report has every chapter, even if rough
- [ ] Tag `v1.0-rc` pushed

### QA days – Fri Oct 30 and Sat Oct 31

Nobody checks their own work: each person audits another lane on Day 8, everyone fixes on Day 9. Every bug becomes a GitHub issue, fixed by a PR to `main`, and the affected script is rerun so CSVs and figures stay in sync.

**Who checks whom (Day 8)**

| Checker | Checks | Focus |
| --- | --- | --- |
| Saket | Rohit's E3 | Gold-label indexing, budget computed in the right tokenizer, `answer_kept` logic |
| Rohit | Rohith's E4 + stats | Round-trip, *p* = 0 equals baseline, CI and McNemar code |
| Rohith | Saket's E1–E2 | Tax formula, single-byte count, price units, context-loss maths |
| Sachi | The app + every number in the report | Each figure and number traced to a CSV |

**Sanity tests that catch real mistakes**

- [ ] English tax = exactly 1.0 for every tokenizer
- [ ] English rows: `eqtok_1.0` accuracy equals `full` accuracy (the budget is English's own length)
- [ ] E4 at *p* = 0 equals the baseline item for item; `n_tokens_used` ≥ `n_tokens_canonical` always
- [ ] Gold labels: Belebele answers are 1–4; check 5 items by hand that `gold` points to the right option
- [ ] Option-shuffle check: shuffle options on 100 English items; accuracy should stay close (catches letter-position bias)
- [ ] No-passage check: score 100 items with the passage removed; accuracy should drop toward chance (if not, the model answers from the question alone)
- [ ] Kannada by hand: tokenize 5 sentences in a separate notebook and match the CSV counts
- [ ] Rerun one random (model, language) cell from scratch without cache and match the CSV
- [ ] App: every language, empty input, a 5,000-character input, phone screen, cold start after the Space sleeps
- [ ] Report: figure numbers and captions, table units, every citation in the reference list, lit survey summaries paraphrased in your own words

**Day 9 – Sat Oct 31**

- [ ] Morning: close every Day 8 issue; rerun affected scripts; regenerate figures
- [ ] Sachi records the 3-minute demo video from the frozen app
- [ ] Afternoon: two full read-throughs of the report by people who did not write those chapters
- [ ] Apply guide's feedback if it arrived
- [ ] Tag `v1.0`; repo README lists how to reproduce every figure

### Submission day – Sun Nov 1

- [ ] Final PDF of the report, checked page by page
- [ ] Links working from a logged-out browser: repo, app, video
- [ ] Signatures and team details on the cover page
- [ ] Submit before noon; no new features or reruns today
- [ ] Paper draft goes to arXiv in the following week after one more polish

## Experimental design

Four experiments fit in 10 days: E1–E2 measure the tax (CPU, a few hours), E3 and E4 test causes (GPU, roughly 2–4 hours each on a T4 with option log-likelihood scoring). Run order matters: E1 and the baseline first, because every later number is compared to them.

| ID | Question | What you change | Main number | Runs |
| --- | --- | --- | --- | --- |
| E1 | How big is the tax? | 7 tokenizers × 13 languages on FLORES | Token ratio vs English, fertility, bytes/token | CPU |
| E2 | What does it cost? | E1 tokens × price sheet; context windows | INR per 1k words, % of window lost | CPU |
| E3 | Context or capability? | Equal-content vs equal-token budget | Share of the gap that closes | 2 models × 7 langs × 3 conditions × 300 Qs |
| E4 | Does fragmentation itself hurt? | Same text split into more tokens | Accuracy drop per unit of fertility | 1–2 models × 4 langs × 4 rates × 300 Qs |

### E3 – Equal-token vs equal-content (single passage)

For each Belebele question, the English passage's token count *E* sets the budget.

- **Equal-content:** the full passage in every language (the baseline run).
- **Equal-token 1.0×:** every language's passage is cut to *E* tokens, so Indic languages lose the end of the passage.
- **Equal-token 0.5×:** cut to *E*/2 for every language, English included, which shows how fast accuracy falls when content is lost.
- Record per item whether the cut removed the sentence containing the answer (Belebele marks which passage sentences are relevant only implicitly, so approximate this by checking whether the gold option's key words still appear; note the approximation).

Read the result as two gaps:

```latex
\text{context penalty} = \text{Acc}_{\text{equal-content}} - \text{Acc}_{\text{equal-token}} \qquad \text{residual gap} = \text{Acc}_{\text{English}} - \text{Acc}_{\text{equal-content}}
```

The context penalty is what the tax costs through lost content alone; the residual gap is capability plus any direct fragmentation effect, which E4 probes.

### E4 – Fragmentation intervention

Same model, same passage, more pieces.

1. Tokenize the passage normally, then re-split it by randomly undoing merges at rate *p* ∈ {0.1, 0.3, 0.5} (BPE-dropout applied at inference). Every variant must decode back to the identical string; the Day 1 test enforces this.
2. Record induced fertility and accuracy at each *p*.
3. Plot accuracy against fertility for English, Hindi, Tamil, Malayalam. Compare slopes.
4. If English loses accuracy as fast as Indic when fragmented, fragmentation hurts any language and the Indic gap is partly a fragmentation gap. If not, the gap is more about training data.

Caveat for the report: models never saw non-canonical splits in training, so part of any drop is unfamiliarity. Keep *p* small and say so.

### Statistics (kept simple)

- 1,000-sample bootstrap 95% CIs for every accuracy and tax number.
- McNemar test for paired conditions on the same questions, Holm correction across languages.
- One Spearman correlation of fertility vs accuracy across languages, with the caveat that resource level confounds it (the reason E3–E4 exist).

## Demo app and repository

The product is the "Tokenizer Tax Calculator": a free Hugging Face Space with two tabs (Context squeeze only if Day 6 has spare time), backed by the same `tokentax` package that produced the paper's numbers.

**App tabs**

| Tab | What the user does | What they see |
| --- | --- | --- |
| Calculator | Pastes text or picks a FLORES sentence, picks a language | Token count per tokenizer, colour-coded token boundaries, tax vs the English translation, cost in INR for 1M words, how much of a 128k window it uses |
| Results | Browses findings | Interactive tax heatmap, cost table, E3 budget curves, E4 fragmentation slopes, filterable by language and model |
| Context squeeze | Picks a Belebele question and a token budget | Which passages fit for English vs the chosen language, side by side, and the model's recorded answer |

Keep the app light: tokenizers only, no LLM inference at runtime (load recorded results from CSV), so it runs on the free CPU tier. Use `gr.HighlightedText` (Gradio) or a small HTML renderer to show token splits — showing a Malayalam word split into a dozen byte pieces is the most memorable moment in the demo.

**Demo script (3–4 minutes)**

1. One English sentence vs its Malayalam translation in the calculator: tokens and cost side by side.
2. Switch tokenizer cl100k → o200k → Sarvam: the tax shrinks.
3. Context squeeze: at the same budget, English keeps the answer passage and Odia loses it.
4. Results tab: one chart for each of RQ2 and RQ3, one sentence each.

**Repository layout**

```markdown
tokentax/
├── configs/          scope.yaml, prices_2026-xx.csv, prompts/
├── src/tokentax/
│   ├── data.py       FLORES, Belebele, IndicCorp loaders + NFC normalization
│   ├── tokenizers.py one wrapper over tiktoken / HF / SentencePiece
│   ├── metrics.py    fertility, tax, bytes/token, single-byte rate
│   ├── cost.py       price sheet → INR/USD, context erosion
│   ├── eval/         belebele.py, budgets.py (E3), resegment.py (E4)
│   └── stats.py      bootstrap, McNemar, mixed-effects
├── scripts/          run_e1.py … run_e7.py (each writes results/*.csv)
├── notebooks/        exploration only, never the source of a final number
├── results/          CSVs + figures (versioned)
├── app/              Gradio/Streamlit Space
├── paper/            LaTeX (ACL template) — synced to Overleaf
├── tests/            decode-roundtrip, metric unit tests
└── Makefile          make e1 … make all, make app
```

**Engineering rules**: fixed random seeds, results keyed by (experiment, model, language, condition) so nothing is computed twice, a unit test that every E4 re-segmentation decodes to the original string, and the price sheet dated in its filename.

## Report and research paper

Write the report first (it is due Nov 1), and cut the 4-page paper out of it; post the paper to arXiv after one more polish.

**Report chapters** (adapt to RV University's template; owner in brackets)

1. Introduction and problem statement (Sachi, Day 3)
2. Literature survey: corrected table + summaries (Sachi, Day 3)
3. Gap analysis and objectives (Sachi, Day 3)
4. System design: toolkit and app architecture (Sachi, Day 6)
5. Methodology: E1–E4 (Saket, Day 5)
6. Results and discussion (Rohit + Rohith, Day 8)
7. Demo with screenshots (Sachi, Day 8)
8. Conclusion, limitations, future work: the cut items (Rohith, Day 8)
9. References and appendix tables

**Paper (ACL format, 4 pages + references)**

| Section | Content | Length |
| --- | --- | --- |
| Introduction | The tax, the open causal question, 3 contributions | 0.5 page |
| Related work | Petrov, Ahia, Dhar, Srivastava; Rust, Ali | 0.5 page |
| Setup | Languages, tokenizers, models, metrics | 0.5 page |
| Results | E1–E2 heatmap + cost table; E3 gap split; E4 slopes | 2 pages |
| Limitations + conclusion | 2 small models, 300 items, OOD caveat | 0.5 page |

Later submission targets (check current calls first): multilingual and low-resource NLP workshops at ACL/EMNLP venues, Indian-language workshops such as WILDRE, and ICON.

**Rule:** every number in the report comes from a CSV in `results/`, never retyped from memory; whoever writes a results paragraph checks it against the CSV on Day 9.

## Timeline, team and risks

The sprint runs Fri Oct 23 to Sun Nov 1, 2026. All four work in parallel from Day 1, and results freeze at the end of Day 7 so the last three days are protected for QA, fixes and polish.

&#91;embedded content: 16-week timeline · 9 phases, 2 checkpoints\]

Days 1–3 build and baseline in parallel; Days 4–5 run E3 and E4 side by side, one per GPU account.

**Team split**

| Member | Owns | Writes in the report |
| --- | --- | --- |
| Rohit Ajit Bharadwaj | Belebele harness, baseline, E3 | Results (E3) |
| Rohith P Hegde | E4 intervention, statistics | Results (E4), conclusion |
| S Sachithananthan | Demo app, video, report assembly | Chapters 1–4, demo chapter |
| Saket M | E1–E2 audit, figures, paper draft | Methodology, results (E1–E2) |

**Risks and fallbacks**

| Risk | Fallback |
| --- | --- |
| Gemma or Llama licence not approved by Day 2 | Use Qwen2.5 + Sarvam-1 (not gated); keep the gated tokenizers in E1 only if approved |
| Models near chance on some languages | Drop those languages from E3–E4, keep them in E1, say why |
| Colab GPU quota runs out | Each member uses their own Colab + Kaggle account; cut to 200 questions; cache every result |
| E4 shows no clear effect | Still a finding: report it with CIs; the E3 split carries the paper |
| Writing slips | Report chapters 1–5 are written during Days 3–5, not after |
| Someone falls sick | Every script lives in the repo with a README line, so anyone can rerun it |

**Guide check-ins:** Day 1 (scope), Day 6 (demo link + first results), Day 9 (full report draft).
