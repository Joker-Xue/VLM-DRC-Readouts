PY ?= python
RUN := experiments/results/corrected_debate_20260922_v3_n2000

.PHONY: reproduce data test metrics figures check clean

reproduce: data test metrics figures check

data: $(RUN)/per_region.json

$(RUN)/per_region.json: $(RUN)/per_region.json.gz
	gunzip -k -f $<

test:
	$(PY) -m pytest -q experiments/tests experiments/agents/test_protocol.py

metrics: data
	mkdir -p outputs
	$(PY) -m experiments.eval.audited_paper_metrics --out outputs/recomputed-metrics.json --bootstrap 3000
	$(PY) -m experiments.eval.eval_corrected_debate --run $(RUN) --out outputs/corrected-debate-evaluation.json --bootstrap 3000

figures: metrics
	PYTHONPATH=. $(PY) figures/plot_paper_figures.py
	$(PY) figures/plot_corrected_debate.py

check:
	$(PY) scripts/check_reproduction.py

clean:
	rm -rf outputs figures/out $(RUN)/per_region.json
