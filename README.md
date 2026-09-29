# StockSense 🛒

> **IntelliData 2026 Data Science Hackathon** — Sri Eshwar College of Engineering  
> Team: _[TEAM NAME]_

A decision-support system that forecasts next-7-day demand, predicts stock-out
risk, and recommends reorder quantities for every Store × Product in the NovaMart
supermarket chain.

---

## ⚠️ Data Caveat

The source dataset (`atomicd/retail-store-inventory-and-demand-forecasting` on
Kaggle) is **synthetic**. The five NovaMart CSVs are derived from it; stock-outs
are additionally **simulated** as described in `docs/ASSUMPTIONS.md`. Every number
in the reports comes directly from saved code output — nothing is typed by hand.

---

## Architecture

```
data/external/sales_data.csv
        │
        ▼
src/build_novamart.py  ──►  data/raw/  (transactions, products, stores,
        │                               inventory, external_factors)
        ▼
src/clean.py           ──►  data/processed/master.csv
        │
        ▼
src/features.py        ──►  data/processed/features.parquet
        │
        ├──► src/train_regression.py    ──►  models/regression_*.joblib
        └──► src/train_classification.py ─►  models/classification_*.joblib
                                                    │
                                         src/recommend.py
                                                    │
                              ┌─────────────────────┴──────────────────────┐
                              │                                             │
                    src/api/main.py (FastAPI)             dashboard/gradio_app.py
                         port 8000                               port 7860
```

---

## Quick Start

```bash
# 1. Clone and place the Kaggle CSV
git clone <repo-url>
cd STOCKSENSE_TEAM_NAME
# copy sales_data.csv → data/external/sales_data.csv

# 2. Create environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 3. Run full pipeline
make all

# 4. Start API + app (two separate terminals)
make api
make app
```

---

## Swapping in Real Data

To use the organiser's real CSVs instead of the synthetic ones:

1. Place them in `data/raw/` with the exact field names listed in
   `docs/ASSUMPTIONS.md`.
2. Run `make clean features train extracts` (skip `make data`).
3. The rest of the pipeline is data-agnostic.

---

## Project Structure

```
data/external/      Kaggle source (git-ignored)
data/raw/           NovaMart CSVs (generated, git-ignored)
data/processed/     master.csv, features.parquet (generated, git-ignored)
notebooks/          01_dq_eda.ipynb  02_features_models.ipynb  03_recommendation.ipynb
src/                Python scripts (build_novamart, clean, features, train_*, recommend, explain, api/)
models/             Saved joblib models (generated, git-ignored)
dashboard/          gradio_app.py + tableau_extracts/
reports/            data_quality_report.md, model_comparison.md, figures/, metrics/
docs/               PROJECT_BRIEF.md, ASSUMPTIONS.md, PITCH_OUTLINE.md
tests/              pytest suite
```

---

## Deliverables Checklist

- [ ] `data/processed/master.csv` — cleaned master table
- [ ] `notebooks/01_dq_eda.ipynb` — EDA & statistics
- [ ] `notebooks/02_features_models.ipynb` — features & models
- [ ] `notebooks/03_recommendation.ipynb` — recommendation layer
- [ ] `reports/data_quality_report.md`
- [ ] `reports/model_comparison.md`
- [ ] `models/` — saved joblib models
- [ ] `src/api/main.py` — FastAPI backend
- [ ] `dashboard/gradio_app.py` — Gradio prototype
- [ ] `dashboard/tableau_extracts/` — Tableau CSVs + DASHBOARD_SPEC.md
- [ ] `docs/PITCH_OUTLINE.md`
- [ ] `pytest` passing (leakage test + risk-tier test)

---

## License

MIT
