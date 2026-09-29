## StockSense Makefile
## Targets: data, clean, features, train, api, app, extracts, test, all
## Run `make all` from the repo root to reproduce everything from scratch.

PYTHON = python
SRC    = src

.PHONY: all data clean features train api app extracts test

## Build the 5 NovaMart CSVs from data/external/sales_data.csv
data:
	$(PYTHON) $(SRC)/build_novamart.py

## Run cleaning, build master.csv, generate data quality report
clean:
	$(PYTHON) $(SRC)/clean.py

## Engineer features and build target columns
features:
	$(PYTHON) $(SRC)/features.py

## Train regression and classification models
train:
	$(PYTHON) $(SRC)/train_regression.py
	$(PYTHON) $(SRC)/train_classification.py

## Produce recommendation tables and Tableau extracts
extracts:
	$(PYTHON) $(SRC)/recommend.py

## Start FastAPI backend (runs in foreground; use & or a separate shell)
api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

## Start Gradio frontend (expects API running on port 8000)
app:
	$(PYTHON) dashboard/gradio_app.py

## Run all pytest tests
test:
	pytest tests/ -v

## Run full pipeline end-to-end
all: data clean features train extracts test
