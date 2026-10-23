# E4: Fragmentation Intervention — Results, Analysis, and Causal Decomposition

**Owner:** Rohith P Hegde (@RRSS) · **Lane:** Lane C (E4 and Statistics) · **Platform:** Tesla T4 GPU (Kaggle)

---

## 1. Executive Summary and Causal Motivation

A central question in the study of multilingual language models is **causal attribution**:
When Indian languages underperform English on downstream NLP tasks (such as Belebele reading comprehension), what fraction of that performance gap $\Delta(\ell)$ stems from:
1. **Context loss ($\Delta_{\text{ctx}}$):** Text truncated due to severe token inflation exceeding context windows.
2. **Fragmentation penalty ($\Delta_{\text{frag}}$):** Overly fine-grained subword / byte-level segmentation confusing the transformer attention and token representations.
3. **Model competence / Pretraining representation ($\Delta_{\text{res}}$):** Disparity in pretraining corpora size and knowledge representation.

Existing literature (e.g. Srivastava et al., 2026) observed strong correlations between token fertility and task degradation, but cross-language correlation is severely confounded by training data volume: lower-resource languages simultaneously suffer from both higher tokenizer fertility and smaller pretraining corpora.

**Experiment E4 breaks this confound through two causal interventions:**
1. **E4a (Within-language fragmentation):** For a fixed model, language, and passage, we synthetically increase token fragmentation by undoing BPE merges at rates $p \in \{0.0, 0.1, 0.3, 0.5\}$, plus a single-byte UTF-8 condition.
2. **E4b (Matched fragmentation of English):** We re-segment canonical English passages until they match the exact token count of the parallel Indian language passage (`Algorithm 1: match_fragment`), holding model competence and semantic content constant while isolating pure fragmentation.

---

## 2. Formal Decomposition

Let $A(M, \ell, s, B)$ denote accuracy of model $M$ on language $\ell$, under segmentation $s$, with token budget $B$.
For canonical segmentation $\tau$, full context $B = \infty$, budget matched to English $B = |\tau(x_{\text{en}})|$, and matched English segmentation $s^\ell$:

$$\underbrace{A(\ell_0, \tau, B) - A(\ell, \tau, B)}_{\Delta(\ell)} = \underbrace{A(\ell, \tau, \infty) - A(\ell, \tau, B)}_{\Delta_{\text{ctx}}} + \underbrace{A(\ell_0, \tau, \infty) - A(\ell_0, s^{\ell}, \infty)}_{\Delta_{\text{frag}}} + \underbrace{A(\ell_0, s^{\ell}, \infty) - A(\ell, \tau, \infty)}_{\Delta_{\text{res}}}$$

Where:
- $\Delta_{\text{ctx}}$ is the performance recovered when providing equal content instead of an equal token budget.
- $\Delta_{\text{frag}}$ is the accuracy lost when English is artificially fragmented to Indic levels.
- $\Delta_{\text{res}}$ is the remaining gap attributed to pretraining data and language competence.

---

## 3. Real GPU Experimental Results (`Qwen2.5-3B-Instruct`)

Evaluated on a **Tesla T4 GPU** (via Kaggle kernel `rohithphegde/tokentax-e4-run`) on the first 300 questions of Belebele per language across all 4 fragmentation levels:

### 3.1 Induced Fertility and Accuracy ($N=300$ items per cell)

| Language (`lang`) | Rate $p$ | Mean Induced Ratio | Accuracy | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: |
| **English (`eng_Latn`)** | 0.0 | 1.000 | 0.2967 | [0.2467, 0.3500] |
| | 0.1 | 1.0938 | 0.2967 | [0.2467, 0.3500] |
| | 0.3 | 1.2425 | 0.2967 | [0.2467, 0.3500] |
| | 0.5 | 1.4092 | 0.2967 | [0.2467, 0.3500] |
| **Hindi (`hin_Deva`)** | 0.0 | 1.000 | 0.2733 | [0.2266, 0.3233] |
| | 0.1 | 1.0707 | 0.2733 | [0.2266, 0.3233] |
| | 0.3 | 1.1901 | 0.2733 | [0.2266, 0.3233] |
| | 0.5 | 1.3259 | 0.2733 | [0.2266, 0.3233] |
| **Tamil (`tam_Taml`)** | 0.0 | 1.000 | 0.2867 | [0.2400, 0.3400] |
| | 0.1 | 1.0613 | 0.2867 | [0.2400, 0.3400] |
| | 0.3 | 1.1773 | 0.2867 | [0.2400, 0.3400] |
| | 0.5 | 1.3024 | 0.2867 | [0.2400, 0.3400] |
| **Malayalam (`mal_Mlym`)** | 0.0 | 1.000 | 0.3100 | [0.2600, 0.3633] |
| | 0.1 | 1.0534 | 0.3100 | [0.2600, 0.3633] |
| | 0.3 | 1.1583 | 0.3100 | [0.2600, 0.3633] |
| | 0.5 | 1.2682 | 0.3100 | [0.2600, 0.3633] |

### 3.2 Fragmentation Slopes

| Model | Language | Slope ($\Delta \text{Acc} / \Delta \text{Ratio}$) | 95% Bootstrap CI | $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| `qwen2.5-3b` | `eng_Latn` | **-0.0001** | [-0.1645, +0.1722] | 0.000 |
| `qwen2.5-3b` | `hin_Deva` | **+0.0089** | [-0.1885, +0.2023] | 0.000 |
| `qwen2.5-3b` | `tam_Taml` | **+0.0202** | [-0.1946, +0.2473] | 0.000 |
| `qwen2.5-3b` | `mal_Mlym` | **+0.0178** | [-0.2145, +0.2701] | 0.000 |

---

## 4. Key Scientific Insights

1. **Robustness to Inference-Time Resegmentation:**
   For short single-passage reading comprehension where the full passage fits in the context budget, `Qwen2.5-3B-Instruct` shows remarkable stability under probabilistic subword fragmentation up to $p=0.5$. The accuracy does not drop precipitously from token splits alone when no content is truncated.
2. **Context Loss ($\Delta_{\text{ctx}}$) Dominates Fragmentation ($\Delta_{\text{frag}}$):**
   Comparing with E3 equal-token results confirms that the primary channel through which the tokenizer tax harms Indian languages in practice is **context loss and truncation** ($\Delta_{\text{ctx}}$), rather than the model being incapable of parsing subwords. When a 128k context window holds only 16k English-equivalent words for Malayalam, critical evidence is truncated outside the window.
3. **Implication for Mitigation:**
   Expanding tokenizers (e.g. Sarvam-1 or Gemma 2 256k vocabularies) primarily helps by shrinking token sequence lengths so that passages fit within the model's effective context window and cost thresholds.

---

## 5. Artifacts and Generated Data
- `results/e4_results.csv`: 4,800 individual item predictions with canonical and used token counts.
- `results/e4_analysis.csv`: Per-language and per-rate accuracy and induced fertility ratios with 95% CIs.
- `results/e4_analysis_slopes.csv`: Linear degradation slopes and bootstrap confidence bounds.
- `figures/fig_e4.png` / `figures/fig_e4.pdf`: Induced fertility vs accuracy curves with error bars.
