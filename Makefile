## StockSense Makefile
## Targets: data, clean, features, eval, explain, recommend, test, api, app, all
## Run `make all` from a clean clone to reproduce the entire pipeline end-to-end.

PYTHON = python
SRC    = src

.PHONY: all data clean features eval explain recommend test api app

## 1. Build the 5 NovaMart raw CSVs with simulated inventory identities & injected traps
data:
	$(PYTHON) $(SRC)/build_novamart.py

## 2. Clean data, impute missing values, audit integrity, generate master table & EDA figures
clean:
	$(PYTHON) $(SRC)/clean.py
	$(PYTHON) $(SRC)/master.py

## 3. Engineer leak-safe features, purge gaps, and build chronological partitions
features:
	$(PYTHON) $(SRC)/features.py

## 4. Execute time-series CV, benchmark 6 models, test significance, output model_registry.json
eval:
	$(PYTHON) $(SRC)/model_eval.py

## 5. Generate per-row TreeSHAP attributions and manager language explanations
explain:
	$(PYTHON) $(SRC)/explain.py

## 6. Score test period, compute dynamic safety stock, export recommendations and dashboard data
recommend:
	$(PYTHON) $(SRC)/recommend.py
	$(PYTHON) $(SRC)/dashboard_data.py

## 7. Run complete pytest test suite (30 unit & integrity tests)
test:
	pytest tests/ -v

## 8. Run end-to-end reproducible pipeline
all: data clean features eval explain recommend test

## 9. Launch FastAPI decision-support service (Port 8000)
api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

## 10. Launch Gradio decision-support frontend (Port 7860)
app:
	$(PYTHON) dashboard/gradio_app.py
