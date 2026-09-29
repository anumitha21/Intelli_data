# StockSense – Project Assumptions & Design Decisions

_This file is auto-updated by scripts. All human edits are welcome. Each entry
records the phase, the decision, and the rationale._

---

## P0 – Scaffold (2026-09-29)

### Source Data Schema (`data/external/sales_data.csv`)

Inspected 2026-09-29. Key findings:

| Field | Type | Range / Values |
|---|---|---|
| Date | str (YYYY-MM-DD) | 2022-01-01 → 2024-01-30 |
| Store ID | str | S001 – S005 (5 stores) |
| Product ID | str | P0001 – P0020 (20 products) |
| Category | str | Electronics, Clothing, Groceries, Toys, Furniture |
| Region | str | North, South, East, West |
| Inventory Level | int | — |
| Units Sold | int | 0 – 426 |
| Units Ordered | int | — |
| Price | float | 4.74 – 228.03 |
| Discount | int | 0 – 25 (%) |
| Weather Condition | str | Snowy, Cloudy, Sunny, Rainy |
| Promotion | int | 0 / 1 |
| Competitor Pricing | float | — |
| Seasonality | str | Winter, Spring, Summer, Autumn |
| Epidemic | int | 0 / 1 |
| Demand | int | 4 – 430 |

**Total rows:** 76,000 (5 stores × 20 products × ~760 days)

### Mapping Decisions

1. **Stores (5 → S01–S05):** S001→S01 Coimbatore Supermarket, S002→S02 Chennai
   Hypermarket, S003→S03 Madurai Supermarket, S004→S04 Salem Express,
   S005→S05 Chennai Express. Regions mapped to Tamil Nadu geography.

2. **Products (20 → P101–P120):** Source categories (Electronics, Clothing,
   Groceries, Toys, Furniture) re-mapped to NovaMart categories. Electronics →
   Household; Clothing → Personal Care; Groceries → Groceries; Toys → Snacks;
   Furniture → Household (secondary). A second pass assigns Dairy, Beverages,
   Frozen to ensure all 7 NovaMart categories are represented.

3. **Demand column:** We use `Demand` as the true daily demand signal (not
   `Units Sold`, which is capped by stock-on-hand). This avoids baking stock-out
   effects into the demand label.

4. **Weather → temp_c / rain_mm:** Snowy → temp≈8°C, rain≈5mm; Rainy →
   temp≈26°C, rain≈35mm; Cloudy → temp≈28°C, rain≈5mm; Sunny → temp≈33°C,
   rain≈0mm. Random jitter (seed 42) added per row for realism.

5. **Discount column:** Treated as discount percentage (0–25%). Used directly
   for `discount_pct` in transactions.

### Known-in-Advance Features (no leakage)

The following features are known before day t+1 and are therefore safe to use
as features even when the target covers t+1..t+7:

- `weekend_flag` – calendar fact
- `holiday` – Tamil Nadu public holidays (deterministic calendar)
- `festival` – hand-coded festival calendar (deterministic)
- `promotion_flag` – assumed planned in advance (retailer sets promos weekly)
- `local_event` – pre-announced events

### Inventory Simulation Choices

- `reorder_lvl` sampled as `U(5, 20) × avg_daily_demand` per product-store.
- `lead_days` sampled from `{1, 2, 3, 4}` with equal probability.
- Target stock-out rate: 6–12%. Achieved rate printed at build time.
- A "stock-out day" is defined as any day where `closing_stock == 0` OR
  `units_sold < demand` (demand was capped by available stock).

### Safety Stock Formula

`safety_stock = 1.65 × rolling_std_7(demand) × sqrt(lead_days)`

Z = 1.65 corresponds to 95% service level. Rolling std uses the shifted series
(no leakage). Document updated if the formula changes in P5.

### Statistical Test Choices

- Normality: Shapiro-Wilk on a random sample of 5,000 rows (full dataset too
  large for exact Shapiro-Wilk; we sample with seed 42).
- Promo vs sales: Mann-Whitney U (expected non-normal distribution).
- Store type vs demand: Kruskal-Wallis (non-parametric ANOVA).
- Stock-out vs promo: Chi-square contingency table.

### Model Selection Rationale (preliminary)

Primary regression model: XGBoost (expected best RMSE/MAE; handles non-linear
patterns and missing lags). Fallback: Random Forest.

Primary classification model: XGBoost with `scale_pos_weight` to handle class
imbalance. Decision threshold tuned on validation set to maximise recall (missing
a stock-out is worse than a false alarm).

---

## P1 – NovaMart Raw Dataset Generation & Traps Injection (2026-09-29)

### Parameters & Synthesis Choices

1. **Deterministic Seed:** Fixed at `SEED = 42` across all random number generators.
2. **Reorder Level Tuning:** `reorder_lvl = int(lead_days * mean_demand * 0.93)`. Order quantity set to `2 * reorder_lvl`. This achieved an empirical chain-wide stock-out rate of **8.75%**, falling precisely within the hackathon target range of 6%–12%.
3. **Transaction Partitioning:** Daily units sold were split into realistic customer basket chunks of 3–8 items (mean ~5.5). Baskets were assigned shopping hours with empirical peaks in late morning (11:00–13:00) and evening (18:00–20:00). Payment modes were sampled with UPI (55%), Card (30%), and Cash (15%). Total transactions: 1,425,577 rows.
4. **Traps Injected:**
   - Blank temperatures: 91 values (~2.99%) in `external_factors.csv`.
   - Duplicate transactions: 7,092 rows (~0.50%) duplicated in `transactions.csv`.
   - Category spelling variants: 2 SKUs (P102 `beverage`, P105 `BEVERAGES`) in `products.csv`.
   - Impossible quantities: 2,837 rows (~0.20%) with non-positive quantities (-5, -2, -1, 0) in `transactions.csv`.
   - Inventory arithmetic errors: 1,141 rows (~1.50%) with corrupted closing stock in `inventory.csv`.
   - Sparse history products: 2 new SKUs (`P121`, `P122`) with only 5 days of history.
   All ground truth records saved to `data/raw/_traps_manifest.json`.

---

## P2 – Data Cleaning, Master Table, EDA & Hypothesis Testing (2026-09-29)

### Cleaning Decisions & Traps Reconciliation

1. **True Duplicate Deduplication:** Transaction rows with identical `transaction_id` deduplicated by retaining the first occurrence. Reconciled exactly 7,092 duplicates (100% match with manifest).
2. **Non-Positive Quantity Removal:** Transactions with `quantity <= 0` (2,837 rows) filtered out as POS entry/corrupt return errors. Documented rule: dropped prior to daily demand aggregation.
3. **Inventory Conservation Law:** Closing stock reconciled via physical conservation identity: `closing = opening + received - sold`. Exactly 1,141 discrepancies identified and restored to physical balance (100% match with manifest).
4. **Time-Aware Weather Imputation:** Missing temperatures (91 missing values) imputed via time-aware linear interpolation per city with forward/backward boundary padding. Justification: Daily temperature has strong temporal autocorrelation within each micro-climate.
5. **Category Canonicalization:** Title-casing dictionary lookup mapped all variants (`beverage`, `BEVERAGES`) back to `Beverages`. Exactly 2 products standardized (100% match with manifest).
6. **Cold-Start Policy:** Products with `< 28 days` of sales history flagged with `cold_start = 1` and `history_days`. Fallback rule: In subsequent feature pipelines, cold-start SKUs use category-level hierarchical priors for 7-day demand projections.

### Master Table & Grain

- **Grain:** Strictly **ONE ROW = ONE DATE × ONE STORE × ONE PRODUCT**.
- **Composite Primary Key:** `(date, store_id, product_id)`. Zero duplicates verified across all 76,050 records.
- **Master Files:** Saved to `data/processed/master.csv` and `data/processed/master.parquet`.

### EDA & Hypothesis Testing Findings

- **Normality:** Shapiro-Wilk test on 5,000-sample of daily units sold yields $W = 0.9857, p = 4.09 \times 10^{-22}$. Normality is decisively rejected; all subsequent hypothesis testing uses non-parametric formulations.
- **Test 1 (Promotion Lift):** Mann-Whitney U test confirms that active promotional discounting creates statistically significant demand uplift ($U = 5.92 \times 10^8, p = 2.90 \times 10^{-212}$, volume lift = +15.14%).
- **Test 2 (Store Format Heterogeneity):** Kruskal-Wallis test reveals significant daily sales variation across store types ($H = 84.75, p = 3.95 \times 10^{-19}$), with Hypermarket formats handling higher turnover than Express neighborhood outlets.
- **Test 3 (Stock-out Promotion Association):** Chi-Square contingency test confirms promotion status is significantly associated with stock-out likelihood ($\chi^2 = 5.01, p = 0.0251$, Cramér's $V = 0.0081$), indicating promo demand surges must trigger proactive safety buffer increases.

---

## P3 – Feature Engineering, Target Formulation & Leak-Free Splits (2026-09-29)

### Target Operational Definitions

1. **`next_7_day_demand` (Regression Target):**
   - Cumulative units demanded across horizon $t+1 \dots t+7$:
     $$\text{next\_7\_day\_demand}_t = \sum_{k=1}^7 \text{demand}_{t+k}$$
   - Boundaries without a full 7-day future (the final 7 dates 2024-01-24 to 2024-01-30) are cleanly pruned.

2. **`stockout_flag` (Classification Target):**
   - **Operational Definition:** A day is classified as a stock-out if and only if `closing_stock == 0` OR `units_sold < demand` (demand was capped by on-hand inventory).
   - The forward 7-day target is binary:
     $$\text{stockout\_flag}_t = \mathbb{I}\left(\exists k \in \{1 \dots 7\} : \text{is\_stockout\_day}_{t+k} == 1\right)$$
   - Represents whether the store manager will face a stock depletion event within the coming week.

### Known-in-Advance Features (Zero Leakage)

The following features available at date $t$ are known prior to the forecast horizon $t+1 \dots t+7$:
- `day_of_week`, `weekend_flag`, `month`, `week_no`: Deterministic astronomical calendar facts.
- `festival_flag`, `holiday`: Gazetted Tamil Nadu public holiday schedule and fixed cultural festival calendar.
- `promotion_flag`, `discount_pct`: Retail promotional markdown schedules planned in advance by central merchandising.
- `has_local_event`: Public civic, trade, or cultural events scheduled prior to date $t$.

### Historical Lag & Rolling Window Formulations

- **Strict Historical Shift:** All demand lag features (`lag_1`, `lag_7`, `lag_14`) and rolling aggregations (`rolling_mean_7`, `rolling_mean_14`, `rolling_std_7`) are computed on the shifted series `demand.shift(1)` and earlier.
- Under mathematical verification in `tests/test_leakage.py`, feature values at any date $t$ computed on truncated data $\le t$ are bit-for-bit identical to those computed on the full dataset.

### Chronological Splits & 7-Day Purge Gaps

To completely eliminate label overlap leakage (since 7-day targets span $t+1 \dots t+7$), strict 7-day purge gaps separate the partitions:
- **Train Split:** 2022-01-01 to 2023-09-15 (62,300 rows; stockout class balance: 48.52% positive).
- **Purge Gap 1:** 2023-09-16 to 2023-09-22 (7 buffer days skipped).
- **Validation Split:** 2023-09-23 to 2023-11-20 (5,900 rows; stockout class balance: 33.56% positive).
- **Purge Gap 2:** 2023-11-21 to 2023-11-27 (7 buffer days skipped).
- **Test Split:** 2023-11-28 to 2024-01-23 (5,700 rows; stockout class balance: 44.26% positive).
- **Incomplete Future Prune:** 2024-01-24 to 2024-01-30 (750 rows dropped from modeling).

### Cold-Start Fallback Strategy

For sparse products with $< 28$ days of historical data (`P121`, `P122`), missing lag and rolling demand signals are imputed using category-level empirical means computed strictly on the training partition (`cat_means`). A binary indicator `cold_start = 1` informs downstream estimators of low-confidence historical priors.


