PY ?= python
export PYTHONPATH := src

.PHONY: fake e1 e2 figures test
fake:
	$(PY) scripts/make_fake_results.py
e1:
	$(PY) scripts/run_e1.py
e2:
	$(PY) scripts/run_e2.py
figures:
	$(PY) scripts/fig_tax_heatmap.py
	$(PY) scripts/fig_cost.py
	$(PY) scripts/fig_context_loss.py
	$(PY) scripts/fig_e3.py
	$(PY) scripts/fig_e4.py
	$(PY) scripts/fig_decomposition.py
corr:
	$(PY) scripts/corr_fertility_accuracy.py
check-srivastava:
	$(PY) scripts/compare_srivastava.py
test:
	$(PY) -m pytest -q -m "not network"
