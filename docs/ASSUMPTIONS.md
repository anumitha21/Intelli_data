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

_Further entries added by P1–P6 scripts._
