# tokentax — The Tokenizer Tax

How many more tokens do Indian languages need than English, what does it cost, and does the
fragmentation itself hurt accuracy? (NLP mini project, RV University, Oct–Nov 2026.)

## Setup
```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/Colab: source .venv/bin/activate
pip install -e ".[dev,stats,app]" # GPU lanes also: pip install -e ".[gpu]"
hf auth login                     # read token; accept licences below first
```
Gated on the Hub (click "Agree"): `openlanguagedata/flores_plus`, `meta-llama/Llama-3.2-1B`,
`google/gemma-2-2b-it`, `sarvamai/sarvam-1`.

## Reproduce
| Command | Produces |
|---|---|
| `make fake` | `results/fake/*.csv` in the contract shape (for building before real data exists) |
| `make e1` | `results/e1_intrinsic.csv`, `results/e1_summary.csv` |
| `make e2` | `results/e2_cost.csv`, `results/e2_context.csv` (uses latest `configs/prices_*.csv`) |
| `make figures` | `figures/fig_*.png/.pdf` (falls back to fake data, with a warning, if a real CSV is missing) |
| `make test` | pytest (no network) |

Without `make`: `set PYTHONPATH=src` then `python scripts/run_e1.py` etc.

## Day-1 contracts (do not rename)
- `tokentax.tokenizers`: `TOKENIZERS`, `get_tokenizer(name)` → `encode / decode / pieces` — Saket
- `tokentax.data`: `load_flores(lang)` — Saket; `load_belebele(lang, n=300)` — Rohit (stub provided)
- `tokentax.eval.score`: `load_model`, `score(model, passage_ids, question, options) -> 1..4` — Rohit (stub)
- `tokentax.eval.resegment`: `resegment(tokenizer, ids, p, seed)` — Rohith (stub, identity at p=0)
- `tokentax.stats`: `bootstrap_ci`, `mcnemar`, `holm` — Rohith (working first version)
- Result CSV columns: `tokentax/schemas.py`. Languages are FLORES codes (`kan_Knda`).

## Layout
```
src/tokentax/   tokenizers, data, metrics, cost, stats, schemas, plotstyle, eval/{score,resegment}
scripts/        run_e*.py, fig_*.py, make_fake_results.py
configs/        prices_<date>.csv
results/        result CSVs (committed); results/fake/
figures/        generated figures
app/  report/   lane D
```
