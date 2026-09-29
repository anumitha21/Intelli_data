# StockSense Tableau Dashboard Technical Specification

> **Project:** StockSense — NovaMart Retail Decision Support Prototype  
> **Target Audience:** Regional Inventory Directors, Category Merchandisers, Store Operations Managers  
> **Source Directory:** `dashboard/tableau_extracts/`

---

## 1. Schema & Data Model

```
        dim_store (1) ────┐
                          ▼
                    fact_daily (N) ◄──── dim_product (1)
                          ▲
                          │
                fact_recommendations (1:1 latest snapshot)
```

- **`dim_store.csv`**: Store hierarchy, geographic coordinates, square footage, store formats.
- **`dim_product.csv`**: SKU taxonomy, category, brand, unit cost, MRP, shelf life.
- **`fact_daily.csv`**: Grain: `date × store_id × product_id`. Actual demand, sales turnover, stock positions, and model forecasts.
- **`fact_recommendations.csv`**: Replenishment queue, risk classifications, revenue at risk, dynamic safety buffer.
- **`fact_feature_importance.csv`**: Global TreeSHAP gain weights.
- **`kpi_summary.csv`**: Chain-wide operational benchmark KPIs.

---

## 2. Dashboard Sheets Specification

### Sheet 1: Executive Summary
- **Purpose:** High-level executive pulse across chain performance, stock efficiency, and revenue health.
- **KPI Cards:** Total Chain Revenue (₹1.09B), Total Units Sold (7.61M), Chain Stock-out Rate (8.75%), Annual Inventory Turnover (161x), Total Revenue at Risk (₹26.2M).
- **Visualizations:**
  1. *Monthly Revenue Trajectory:* Dual-axis line and bar chart showing sales expansion.
  2. *Category Pareto Revenue Contribution:* Pareto curve showing 80% revenue concentration across Groceries, Beverages, and Household items.
- **Filters:** Region, City, Store Format, Calendar Quarter.

### Sheet 2: Demand Intelligence
- **Purpose:** Diagnostic exploration of customer demand velocity and actual-vs-forecast alignment.
- **Visualizations:**
  1. *Actual vs 7-Day Predicted Demand:* Multi-line time series by Store and Category.
  2. *Promotion Lift Waterfall:* Compares baseline daily units sold vs active promotion lift (+15.1%).
  3. *Weekend Surge Heatmap:* Day-of-week demand index across store formats.
- **Filters:** Category, Brand, Promotion Status.

### Sheet 3: Inventory Risk Matrix
- **Purpose:** Rapid triage of operational stock-out probabilities across network nodes.
- **Visualizations:**
  1. *Risk Tiers Distribution:* Donut chart of High (p >= 0.70), Medium (0.40 <= p < 0.70), Low (p < 0.40).
  2. *Store × Category Risk Heatmap:* Color gradient (Red = High Risk, Green = Safe) identifying persistent stock depletion in Express format stores.
  3. *Days of Inventory Runway vs Lead Time:* Scatter plot identifying vulnerable SKUs whose stock runway is less than supplier lead days.
- **Filters:** Risk Tier, Store ID, Shelf Life Group.

### Sheet 4: Manager Action Centre (Interactive Queue)
- **Purpose:** Primary operational decision screen for Store Managers raising replenishment orders.
- **Layout:** Tidy interactive grid sorted by Risk Tier (High first) and Revenue at Risk.
- **Columns:** Store ID, Product Name, Category, Current Stock, 7-Day Forecast, Stock-out Prob (%), Risk Badge (Red/Amber/Green), Recommended Order Qty, Safety Stock Buffer.
- **Interactive Action:** Clicking any SKU triggers the **Recommendation Card Pop-up** displaying the plain-English "Why?" driver explanation and Manager Directive.
- **Filters:** Risk Tier, Store ID, Reorder Qty > 0 checkbox.

### Sheet 5: Model Performance & Diagnostics
- **Purpose:** Governance and statistical confidence validation for technical reviewers.
- **Visualizations:**
  1. *Probability Calibration Deciles:* Reliability diagram comparing mean predicted probability against observed empirical stock-outs.
  2. *Classification Confusion Matrix:* Precision (50.5%), Recall (70.6%), ROC-AUC (0.6165).
  3. *Demand Forecast Residual Distribution:* Histogram of actual vs predicted residuals.

### Sheet 6: Explainability & Driver Attribution
- **Purpose:** Transparent AI audit of the top decision drivers influencing stock-out probability.
- **Visualizations:**
  1. *Global TreeSHAP Feature Gain:* Bar chart of top predictive drivers (lead time, days of inventory, reorder gap, seasonality).
  2. *Scenario Simulator (What-If):* Parameter slider adjusting markdown (+10%) or delivery delay (+2 days) showing real-time reorder adjustments.
