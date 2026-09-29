<USER_REQUEST>
# StockSense: Lean Antigravity Prompt Pack (PS scope only)

## How to use

1. Create an empty repo folder, open it in Antigravity.
2. Put `sales_data.csv` (Kaggle: `atomicd/retail-store-inventory-and-demand-forecasting`) in `data/external/`.
3. Save the **MASTER BRIEF** as `docs/PROJECT_BRIEF.md` (and in the workspace rules file if your setup has one).
4. Paste the **PHASE PROMPTS** one at a time (P0 to P6). Check the acceptance line, commit, move on.

---

# MASTER BRIEF

## Mission

Build **StockSense** for the IntelliData 2026 Data Science Hackathon (Sri Eshwar College of Engineering). NovaMart Retail is a multi-city supermarket chain. Build a decision-support system that, for every Store x Product:

1. Forecasts **next-7-day demand** (regression).
2. Predicts **stock-out probability** and classifies risk High / Medium / Low (classification).
3. Recommends a **reorder quantity** with a plain-English reason.
4. Presents it in a manager-ready prototype.

A prediction without an action is incomplete. Three students will be questioned on every module, so write readable code with short comments at decision points.

Judging: Business Understanding 10, Data Cleaning and Quality 15, EDA and Statistics 15, Feature Engineering 15, ML Modelling and Validation 20, Explainability and Recommendation 10, Visualization/Prototype 10, Pitch 5.

## Rules

1. **ML only, permitted algorithms only:** Linear Regression, Decision Tree, SVM, XGBoost, KNN, Naive Bayes, Random Forest. No deep learning. No LightGBM, CatBoost, Prophet, ARIMA, and **no logistic regression**.
   - Regression: baseline, Linear Regression, Decision Tree, Random Forest, XGBoost (optionally KNN).
   - Classification: Decision Tree, Random Forest, XGBoost (optionally KNN, GaussianNB).
2. **No leakage.** Features at date t use only information available at t. Lags via `shift(>=1)`; rolling stats on shifted series. Known-in-advance columns (weekend, holiday/festival, planned promo) are allowed; list them in `docs/ASSUMPTIONS.md`.
3. **Time-aware splits only.** Chronological train/validation/test. Never shuffle. Leave a 7-day gap between splits, because 7-day targets overlap. Fit imputers and scalers on train only.
4. **Reproducible.** Seed 42, `requirements.txt`, one command to rerun. Large or generated files are git-ignored and rebuilt by scripts.
5. **Honest.** The source data is synthetic and stock-outs are simulated. Say so in the README. Every number in reports comes from saved code output, never typed by hand.
6. **Autonomous.** Don't ask questions. Make sensible choices and log them in `docs/ASSUMPTIONS.md`. Commit after every phase.
7. **Stack.** Python 3.11, pandas, numpy, scikit-learn, xgboost, scipy, statsmodels, matplotlib, seaborn, plotly, fastapi, uvicorn, pydantic, gradio, joblib, pyarrow, pytest, holidays.

## Repository layout (follow the PS structure)

```
STOCKSENSE_TEAM_NAME/
  data/external/  data/raw/  data/processed/
  notebooks/   (01_dq_eda.ipynb, 02_features_models.ipynb, 03_recommendation.ipynb)
  src/         (build_novamart.py, clean.py, features.py, train_regression.py,
                train_classification.py, recommend.py, explain.py, api/)
  models/
  dashboard/   (gradio_app.py, tableau_extracts/)
  reports/     (data_quality_report.md, model_comparison.md, figures/, metrics/)
  docs/        (PROJECT_BRIEF.md, ASSUMPTIONS.md)
  tests/
  README.md  requirements.txt  Makefile  .gitignore
```

## A. Turn the Kaggle file into the 5 NovaMart CSVs (`build_novamart.py`)

First **inspect** `data/external/sales_data.csv` (columns, dtypes, ranges, stores, products, date range) and adapt to what is really there. Likely columns: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Promotion, Competitor Pricing, Seasonality, Epidemic, Demand. Verify; don't assume.

Generate the PS files: `transactions.csv`, `products.csv`, `stores.csv`, `inventory.csv`, `external_factors.csv`, with exactly the PS field names.

1. **Demand.** Treat the dataset's `Demand` (or `Units Sold`) as true daily demand per date x store x product.
2. **Stores.** Map to S01..S0N with Coimbatore, Chennai, Madurai, Salem; add `store_type` (Supermarket/Hypermarket/Express), `floor_area_sqft`, `avg_daily_customers`, `region`.
3. **Products.** Map each product to a NovaMart category (Groceries, Beverages, Dairy, Snacks, Personal Care, Household, Frozen) with `brand`, `mrp`, `cost_price`, `shelf_life_days` (Dairy short, Personal Care long), `supplier_id`. IDs like `P101`.
4. **Inventory (stock-outs).** The Kaggle data has few stock-outs, so simulate stock movement: each product-store gets a `reorder_lvl` and `lead_days` (1 to 4); daily `sold = min(demand, on_hand)`, `closing = opening + received - sold`; reorder when `closing < reorder_lvl`, arriving after `lead_days`. Tune so about 6 to 12% of rows have a stock-out. Print the achieved rate.
5. **Transactions.** Split daily sales into transaction rows: quantity 1 to 6, `selling_price`, `discount_pct`, `promotion_flag`, `customer_id`, `payment_mode` (UPI/Card/Cash), `hour` (peaks late morning and evening). If the row count gets too large (over about 2 million), use larger quantities per transaction.
6. **External factors.** Per date x city: `temp_c`, `rain_mm` (from Weather Condition), `holiday`, `festival`, `weekend`, `local_event`. Use the `holidays` package for Tamil Nadu holidays plus a hand-made list of festivals (Pongal, Tamil New Year, Ganesh Chaturthi, Deepavali, etc.).
7. **Inject the six PS traps** and save what was injected in `data/raw/_traps_manifest.json`: blank temperatures (~3%), duplicate transactions (~0.5%), category spelling variants (Beverages / beverage / BEVERAGES), impossible quantities (~0.2%), inventory arithmetic mismatches (~1.5%), and a few new products with only 5 days of history.

Deterministic (fixed seed). Print a build summary.

## B. Round 1: cleaning, Data Quality Report, master table, EDA, statistics

- Audit data types, missing values, duplicates, invalid values. Standardise categories, dates, IDs.
- Remove **true** duplicates only. Flag negative or zero quantities and treat them (document the rule).
- Check the inventory identity `closing == opening + received - sold`; flag and reconcile mismatches.
- Impute temperature (per city, time-aware) and justify. For sparse-history products, add a `history_days` flag and explain the fallback (use category-level averages).
- Aggregate transactions to daily Store x Product and merge product, store, inventory, external tables. **Master table grain: one row = one date x one store x one product.** Save to `data/processed/master.csv`.
- **Data Quality Report** (`reports/data_quality_report.md`, generated by code): issue, count, action taken, justification.
- **EDA** (each chart answers a business question, with a short written insight):
  - Which categories generate the most revenue? (share + Pareto)
  - Which stores are growing or declining? (weekly trend)
  - Do promotions increase units sold? (promo vs non-promo)
  - How does weekend demand differ?
  - Which products are volatile? (coefficient of variation)
  - Which stores repeatedly stock out? (heatmap by store and category)
  - KPIs: Revenue, Units Sold, Stock-out Rate, Inventory Turnover, Days of Inventory, Promotion Lift, Estimated Lost Sales.
- **Statistics** (at least 3; state business question, H0, H1, test, p-value, business interpretation):
  1. Do promotions significantly increase sales? (normality check, then t-test or Mann-Whitney U)
  2. Does mean demand differ across store types? (ANOVA or Kruskal-Wallis)
  3. Is stock-out frequency associated with promotion status? (chi-square)

## C. Round 2: features and models

**Features** (all leak-safe): time (day_of_week, weekend_flag, month, week_no, festival_flag); lag (lag_1, lag_7, lag_14 demand); rolling (rolling_mean_7, rolling_mean_14, rolling_std_7); inventory (days_of_inventory, inventory_to_demand_ratio, reorder_gap); price/promo (discount_pct, price_change, promotion_flag); store/product (store_type, category, brand, shelf_life, lead_time); external (temperature, rain, holiday, local_event).

**Targets**
- `next_7_day_demand` = total units demanded over t+1..t+7.
- `stockout_flag` = 1 if a stock-out occurs in t+1..t+7. **Operational definition:** a day counts as a stock-out when closing stock is 0 (or sales were capped by available stock). Document it clearly. `stockout_probability` is the model's predicted probability.

**Model 1, demand forecast.** Baseline first (previous 7-day mean), then Linear Regression, Decision Tree, Random Forest, XGBoost. Time-based train/validation/test split. Metrics: MAE, RMSE, MAPE (non-zero targets only, and say so), R². Compare all models vs the baseline in a table and explain which metric matters most for the business (under-forecasting causes stock-outs, so also check the bias direction). Include an actual-vs-predicted plot and residual plots.

**Model 2, stock-out risk.** Decision Tree, Random Forest, XGBoost (optionally KNN/GaussianNB). Handle class imbalance (class weights / `scale_pos_weight`) and compare against no handling. Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix. Choose the decision threshold on the validation set with recall in mind. Then **check probability calibration** (calibration plot), since the risk tiers depend on the probability values.

**Model selection.** `reports/model_comparison.md`: one table of all models x all metrics, the chosen model per task, and a justification. Save models with joblib.

**Explainability (mandatory).** Global: feature importance and permutation importance (on the test set). Per row: XGBoost `pred_contribs` (or the tree model's contributions), converted to manager-friendly sentences with a % share, for example "Promotion active +31%, Weekend approaching +22%, Recent sales growth +19%". Use simple templates only, no free-text generation.

## D. Round 3: recommendation, API, Gradio, Tableau

**Business logic (from the PS)**
- Recommended stock = forecast demand + safety stock (safety stock = a simple documented rule, e.g. z x rolling_std_7 x sqrt(lead_days)).
- Reorder quantity = `max(0, recommended_stock - current_stock - incoming_stock)`.
- Risk: High if probability >= 0.70, Medium if 0.40 <= p < 0.70, Low if p < 0.40.

**Manager table** (sorted by risk): Store, Product, Current Stock, 7-Day Forecast, Stock-out Prob., Risk, Recommended Order.

**Recommendation card** per Store x Product: predicted 7-day demand, current and incoming stock, stock-out probability and risk, recommended additional replenishment, "Why?" (top drivers), and a one-line MANAGER ACTION (e.g. "Raise replenishment order today").

**Optional What-If** (only if time allows): change discount, supplier delay (extra lead days), or festival demand and re-score.

**FastAPI backend** (pydantic schemas, models loaded at startup, tested with pytest and `TestClient`):
- `GET /health`
- `GET /stores`, `GET /products`
- `POST /predict` (store_id, product_id, as_of_date) returning forecast, stock-out probability, risk
- `POST /recommend` returning the full recommendation card
- `GET /recommendations` (filters: store, category, risk)
- `POST /explain` returning top drivers in manager language
- `GET /metrics` returning saved model metrics

**Gradio frontend** (calls the FastAPI service). Tabs: Manager Action Centre (filterable risk table with reorder quantities, CSV download), Recommendation Card (pick store + product, show the card and a driver chart), Model Performance (metrics table, confusion matrix, residual plots), and optionally What-If. Red/amber/green risk colouring.

**Tableau extracts** (`dashboard/tableau_extracts/`): tidy CSVs (`fact_daily.csv` with actuals and predictions, `fact_recommendations.csv`, `fact_feature_importance.csv`, `kpi_summary.csv`, `dim_store.csv`, `dim_product.csv`) and `DASHBOARD_SPEC.md` describing each sheet. Required sections: Executive Summary (revenue, growth, stock-out rate, inventory value, products at risk), Demand Intelligence (actual vs forecast, category trend, store trend, forecast error), Inventory Risk (High/Medium/Low table or heatmap), Manager Action Centre (reorder quantity + reason), Model Performance (metrics, confusion matrix, residuals), Explainability (top features).

## E. Definition of done

- One command runs data build, cleaning, features, training, and extracts from a clean clone.
- `pytest` passes, including a leakage test and a test that risk tiers respect the thresholds.
- README covers setup and usage, architecture (Mermaid or ASCII), data caveats, and how to swap in the organizers' real CSVs.
- Every PS deliverable exists: cleaned master CSV, notebooks, Data Quality Report, model comparison with justification, saved models, dashboard, README, and a pitch outline.

---

# PHASE PROMPTS (paste one at a time)

## P0: Scaffold

> Read `docs/PROJECT_BRIEF.md` fully. Make a short plan, then execute only Phase 0: create the repo layout, `requirements.txt`, `.gitignore`, a `Makefile` (targets: data, clean, features, train, api, app, extracts, test, all), an empty `docs/ASSUMPTIONS.md`, and a README skeleton. Inspect `data/external/sales_data.csv` and record the schema findings in `docs/ASSUMPTIONS.md`. Commit.

## P1: Build the NovaMart CSVs

> Implement section A in `src/build_novamart.py`. Follow the spec exactly, including the inventory simulation and the six traps with `_traps_manifest.json`. Print the build summary. Acceptance: five CSVs in `data/raw/` match the PS field names, stock-out rate is 6 to 12%, and a rerun with the same seed gives identical files. Commit.

## P2: Round 1

> Implement section B: cleaning code in `src/clean.py`, the master table, `reports/data_quality_report.md` generated by code, all EDA charts with written insights, and the three statistical tests, in `notebooks/01_dq_eda.ipynb`. Acceptance: master table has no duplicate date-store-product keys, the inventory-identity result is reported, and detected trap counts are compared with `_traps_manifest.json`. Commit.

## P3: Features and demand forecast

> Implement the feature groups and both targets in `src/features.py`, add a pytest proving no feature uses future information, then train and compare the regression models per section C (baseline first, time-based split, all four metrics, plots). Write the regression part of `reports/model_comparison.md`. Acceptance: leakage test passes and all metrics come from saved JSON. Commit.

## P4: Stock-out model and explainability

> Implement the classification task per section C: the stock-out definition, models, imbalance handling comparison, threshold choice, all six metrics, calibration plot, feature and permutation importance, and the per-row manager-language explanations. Update `reports/model_comparison.md`. Acceptance: every High-risk row can produce a sentence explanation. Commit.

## P5: Recommendation layer

> Implement the business logic in section D (`src/recommend.py`): safety stock, reorder quantity, risk tiers, manager table, and recommendation cards. Produce `notebooks/03_recommendation.ipynb` and the Tableau extracts with `DASHBOARD_SPEC.md`. Acceptance: risk tiers respect the thresholds and the manager table matches the PS format. Commit.

## P6: API, Gradio and delivery

> Build the FastAPI service and Gradio app per section D, with tests. Then finish section E: README, `docs/PITCH_OUTLINE.md` (problem, what we found, what we predict, model performance, why it can be trusted, which products/stores need action now, business impact), and a final checklist of PS deliverables. Acceptance: API and app start cleanly and every endpoint and tab works. Commit.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-29T14:00:12+05:30.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to GPT-OSS 120B (Medium). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>