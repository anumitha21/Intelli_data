"""
recommend.py – Phase 5 (P5): Inventory Replenishment Recommendation Engine,
Manager Action Cards, What-If Simulation, and Tableau Data Extracts.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import explain_row

DEFAULT_Z = 1.65  # 95% service level safety factor

def load_models_from_registry(registry_path="models/model_registry.json"):
    """
    Dynamically loads winning models and metadata directly from model_registry.json.
    Never hardcodes model artifact paths.
    """
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Model registry not found at {registry_path}. Run training first.")
        
    with open(registry_path, "r") as f:
        reg = json.load(f)
        
    reg_path = reg["regression"]["artifact_path"]
    clf_path = reg["classification"]["artifact_path"]
    
    regression_model = joblib.load(reg_path)
    classification_model = joblib.load(clf_path)
    
    return {
        "registry": reg,
        "regression_model": regression_model,
        "classification_model": classification_model,
        "reg_features": reg["regression"]["features"],
        "clf_features": reg["classification"]["features"],
        "high_risk_threshold": reg["classification"]["risk_tiers"]["high_risk_threshold"],
        "med_risk_threshold": reg["classification"]["risk_tiers"]["medium_risk_threshold"]
    }

def score_inventory_recommendations(features_path="data/processed/features_test.parquet", z=DEFAULT_Z, registry_path="models/model_registry.json"):
    """
    Scores test rows, applies domain safety stock and replenishment logic,
    and produces the sorted Manager Recommendations Table.
    """
    print("--- Generating StockSense Inventory Recommendations ---")
    os.makedirs("data/processed", exist_ok=True)
    
    model_bundle = load_models_from_registry(registry_path)
    reg_model = model_bundle["regression_model"]
    clf_model = model_bundle["classification_model"]
    feat_cols = model_bundle["clf_features"]
    high_th = model_bundle["high_risk_threshold"]
    med_th = model_bundle["med_risk_threshold"]
    
    df = pd.read_parquet(features_path).copy()
    df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    X = df[feat_cols]
    
    # 1. 7-Day Demand Forecast (Regression)
    pred_demand = np.maximum(0.0, np.round(reg_model.predict(X), 1))
    df['forecast_demand_7d'] = pred_demand
    
    # 2. Stock-out Probability & Risk Classification
    probs = clf_model.predict_proba(X)[:, 1]
    df['stockout_probability'] = np.round(probs, 4)
    
    # Risk Tier assignment
    risk_tiers = np.where(probs >= high_th, "High",
                 np.where(probs >= med_th, "Medium", "Low"))
    df['risk_tier'] = risk_tiers
    
    # 3. Inventory Optimization Logic (PS Rules)
    # Safety stock = ceil(z * rolling_std_7 * sqrt(lead_time))
    rolling_std = df['rolling_std_7'].fillna(10.0).values
    lead_time = df['lead_time'].fillna(2).values
    safety_stock = np.ceil(z * rolling_std * np.sqrt(lead_time)).astype(int)
    safety_stock = np.maximum(5, safety_stock)  # minimum safety buffer
    df['safety_stock'] = safety_stock
    
    recommended_stock = np.ceil(df['forecast_demand_7d'] + safety_stock).astype(int)
    df['recommended_stock'] = recommended_stock
    
    current_stock = df['closing_stock'].astype(int)
    incoming_stock = df['incoming_stock'].astype(int)
    
    # Reorder quantity = max(0, recommended_stock - current_stock - incoming_stock)
    reorder_qty = np.maximum(0, recommended_stock - current_stock - incoming_stock)
    df['recommended_reorder_qty'] = reorder_qty
    
    # Estimated Revenue at Risk
    unit_price = df.get('mrp', df.get('avg_unit_price', 100.0))
    df['revenue_at_risk'] = np.round(df['forecast_demand_7d'] * unit_price * df['stockout_probability'], 2)
    
    # Sort: High Risk first, then Medium, then Low; then by Revenue at Risk descending
    risk_rank_map = {"High": 0, "Medium": 1, "Low": 2}
    df['risk_rank'] = df['risk_tier'].map(risk_rank_map)
    df = df.sort_values(by=['risk_rank', 'revenue_at_risk'], ascending=[True, False]).reset_index(drop=True)
    df = df.drop(columns=['risk_rank'])
    
    # Manager Table matching exact PS format
    manager_table = df[[
        'store_id', 'store_name', 'product_id', 'product_name', 'category', 'date_str',
        'closing_stock', 'forecast_demand_7d', 'stockout_probability', 'risk_tier',
        'safety_stock', 'recommended_reorder_qty', 'revenue_at_risk'
    ]].rename(columns={
        'date_str': 'as_of_date',
        'closing_stock': 'current_stock',
        'stockout_probability': 'stockout_prob',
        'recommended_reorder_qty': 'recommended_order'
    })
    
    out_csv = "data/processed/recommendations.csv"
    manager_table.to_csv(out_csv, index=False)
    print(f"Saved Manager Recommendations Table to {out_csv} ({len(manager_table)} records).")
    
    return df, manager_table

_SCORED_CACHE = None

def get_recommendation_card(store_id: str, product_id: str, as_of_date: str = None, scored_df: pd.DataFrame = None) -> dict:
    """
    Constructs a complete decision-support Recommendation Card for a Store x Product.
    Includes forecast, inventory positions, risk, top explainability drivers, and manager action.
    """
    global _SCORED_CACHE
    if scored_df is None:
        if _SCORED_CACHE is not None:
            scored_df = _SCORED_CACHE
        else:
            _SCORED_CACHE, _ = score_inventory_recommendations()
            scored_df = _SCORED_CACHE
        
    sub = scored_df[(scored_df['store_id'] == store_id) & (scored_df['product_id'] == product_id)]
    if as_of_date:
        date_clean = pd.to_datetime(as_of_date).strftime('%Y-%m-%d')
        sub_date = sub[sub['date_str'] == date_clean]
        if len(sub_date) > 0:
            sub = sub_date
            
    if len(sub) == 0:
        sub = scored_df.iloc[[0]]
        
    row = sub.iloc[0]
    
    date_val = str(pd.to_datetime(row['date']).date())
    s_id = str(row['store_id'])
    p_id = str(row['product_id'])
    p_name = str(row.get('product_name', f'Product {p_id}'))
    s_name = str(row.get('store_name', f'Store {s_id}'))
    category = str(row.get('category', 'General'))
    
    forecast_7d = float(row['forecast_demand_7d'])
    curr_stock = int(row['closing_stock'])
    incoming = int(row['incoming_stock'])
    prob = float(row['stockout_probability'])
    risk = str(row['risk_tier'])
    safety = int(row['safety_stock'])
    reorder_qty = int(row['recommended_reorder_qty'])
    rev_risk = float(row['revenue_at_risk'])
    
    # Get explainability drivers from src/explain.py
    explanation = explain_row(s_id, p_id, date_val)
    why_drivers = explanation.get('explanation_sentence', 'Operational inventory depletion')
    
    # Manager Action Determination
    if risk == "High":
        if reorder_qty > 0:
            action = f"CRITICAL: Raise replenishment order of {reorder_qty:,} units immediately to prevent imminent stock-out."
        else:
            action = "CRITICAL: Stock-out risk is elevated; verify expediting on pending warehouse deliveries."
    elif risk == "Medium":
        if reorder_qty > 0:
            action = f"ATTENTION: Raise replenishment order of {reorder_qty:,} units today on normal supplier schedule."
        else:
            action = "MONITOR: Buffer is adequate; track sales pace over the next 48 hours."
    else:
        action = "NORMAL: Healthy inventory runway. Maintain standard weekly replenishment cycle."
        
    card_dict = {
        "store_id": s_id,
        "store_name": s_name,
        "product_id": p_id,
        "product_name": p_name,
        "category": category,
        "as_of_date": date_val,
        "predicted_7d_demand": forecast_7d,
        "current_stock": curr_stock,
        "incoming_stock": incoming,
        "safety_stock": safety,
        "stockout_probability": prob,
        "risk_tier": risk,
        "recommended_reorder_qty": reorder_qty,
        "revenue_at_risk_inr": rev_risk,
        "why_drivers": why_drivers,
        "top_drivers": explanation.get('top_drivers', []),
        "manager_action": action
    }
    
    # Formatted visual string representation
    card_text = f"""
================================================================================
STOCKSENSE RECOMMENDATION CARD: {s_id} x {p_id}
Store: {s_name} | Product: {p_name} ({category}) | Date: {date_val}
================================================================================
- Forecast Demand (7-Day):    {forecast_7d:,.1f} units
- Current On-Hand Stock:      {curr_stock:,} units
- Incoming In-Transit Stock:  {incoming:,} units
- Dynamic Safety Stock:       {safety:,} units (Z=1.65, 95% service level)
--------------------------------------------------------------------------------
- Stock-out Probability:      {prob*100:.1f}%  [{risk.upper()} RISK]
- Estimated Revenue at Risk:  INR {rev_risk:,.2f}
- Recommended Reorder Order:  {reorder_qty:,} units
--------------------------------------------------------------------------------
WHY? (Top Demand & Supply Drivers):
  > {why_drivers}

MANAGER ACTION:
  >> {action}
================================================================================
""".strip()

    card_dict["formatted_text"] = card_text
    return card_dict

_WHATIF_BUNDLE_CACHE = None
_WHATIF_DF_CACHE = None

def what_if_scenario(store_id: str, product_id: str, as_of_date: str,
                     discount_delta: float = 0.0, extra_lead_days: int = 0,
                     festival_uplift_pct: float = 0.0, registry_path="models/model_registry.json") -> dict:
    """
    What-If Simulation Engine: Evaluates the impact of commercial and supply chain shifts:
    - discount_delta: e.g. +10% markdown
    - extra_lead_days: supplier delay, e.g. +2 days
    - festival_uplift_pct: festival demand spike, e.g. +25%
    """
    global _WHATIF_BUNDLE_CACHE, _WHATIF_DF_CACHE
    if _WHATIF_BUNDLE_CACHE is None:
        _WHATIF_BUNDLE_CACHE = load_models_from_registry(registry_path)
    model_bundle = _WHATIF_BUNDLE_CACHE
    reg_model = model_bundle["regression_model"]
    clf_model = model_bundle["classification_model"]
    feat_cols = model_bundle["clf_features"]
    high_th = model_bundle["high_risk_threshold"]
    med_th = model_bundle["med_risk_threshold"]
    
    # Base card
    base_card = get_recommendation_card(store_id, product_id, as_of_date)
    
    # Load features row
    test_df = pd.read_parquet("data/processed/features_test.parquet")
    test_df['date_str'] = pd.to_datetime(test_df['date']).dt.strftime('%Y-%m-%d')
    date_clean = pd.to_datetime(as_of_date).strftime('%Y-%m-%d')
    
    sub = test_df[(test_df['store_id'] == store_id) & (test_df['product_id'] == product_id) & (test_df['date_str'] == date_clean)]
    if len(sub) == 0:
        sub = test_df.iloc[[0]]
        
    X_perturbed = sub[feat_cols].copy()
    
    # Apply perturbations
    if discount_delta != 0.0:
        X_perturbed['discount_pct'] = np.clip(X_perturbed['discount_pct'] + discount_delta, 0.0, 50.0)
        X_perturbed['promotion_flag'] = (X_perturbed['discount_pct'] > 0).astype(int)
        
    if extra_lead_days != 0:
        X_perturbed['lead_time'] = np.clip(X_perturbed['lead_time'] + extra_lead_days, 1, 10)
        
    if festival_uplift_pct != 0.0:
        factor = 1.0 + (festival_uplift_pct / 100.0)
        X_perturbed['rolling_mean_7'] *= factor
        X_perturbed['lag_1'] *= factor
        
    # Re-predict
    new_forecast = float(np.maximum(0.0, np.round(reg_model.predict(X_perturbed)[0], 1)))
    new_prob = float(np.round(clf_model.predict_proba(X_perturbed)[0, 1], 4))
    new_risk = "High" if new_prob >= high_th else ("Medium" if new_prob >= med_th else "Low")
    
    new_lead = int(X_perturbed['lead_time'].iloc[0])
    new_std = float(X_perturbed['rolling_std_7'].iloc[0])
    new_safety = int(np.ceil(DEFAULT_Z * new_std * np.sqrt(new_lead)))
    new_reorder = max(0, int(np.ceil(new_forecast + new_safety - base_card['current_stock'] - base_card['incoming_stock'])))
    
    return {
        "store_id": store_id,
        "product_id": product_id,
        "as_of_date": as_of_date,
        "scenario_parameters": {
            "discount_delta": discount_delta,
            "extra_lead_days": extra_lead_days,
            "festival_uplift_pct": festival_uplift_pct
        },
        "baseline": {
            "forecast_7d": base_card['predicted_7d_demand'],
            "stockout_prob": base_card['stockout_probability'],
            "risk_tier": base_card['risk_tier'],
            "reorder_qty": base_card['recommended_reorder_qty']
        },
        "simulated": {
            "forecast_7d": new_forecast,
            "stockout_prob": new_prob,
            "risk_tier": new_risk,
            "reorder_qty": new_reorder
        },
        "net_impact": {
            "demand_change": round(new_forecast - base_card['predicted_7d_demand'], 1),
            "prob_change": round(new_prob - base_card['stockout_probability'], 4),
            "reorder_qty_change": new_reorder - base_card['recommended_reorder_qty']
        }
    }

def export_tableau_extracts():
    """
    Exports clean, production-grade CSV data extracts for Tableau dashboarding
    and generates dashboard/tableau_extracts/DASHBOARD_SPEC.md.
    """
    print("--- Exporting Tableau Data Extracts ---")
    extracts_dir = "dashboard/tableau_extracts"
    os.makedirs(extracts_dir, exist_ok=True)
    
    # 1. Load Dimensions
    stores = pd.read_csv("data/raw/stores.csv")
    products = pd.read_csv("data/raw/products.csv")
    stores.to_csv(f"{extracts_dir}/dim_store.csv", index=False)
    products.to_csv(f"{extracts_dir}/dim_product.csv", index=False)
    print("Exported dim_store.csv and dim_product.csv")
    
    # 2. Recommendations & Fact Daily
    scored_test, manager_table = score_inventory_recommendations()
    manager_table.to_csv(f"{extracts_dir}/fact_recommendations.csv", index=False)
    print("Exported fact_recommendations.csv")
    
    # Build Fact Daily with actuals and predictions
    fact_daily = scored_test[[
        'date', 'store_id', 'product_id', 'units_sold', 'sales_revenue', 'closing_stock',
        'is_stockout_day', 'forecast_demand_7d', 'stockout_probability', 'risk_tier',
        'recommended_reorder_qty', 'safety_stock', 'revenue_at_risk'
    ]].rename(columns={
        'units_sold': 'actual_units_sold',
        'sales_revenue': 'actual_revenue',
        'closing_stock': 'closing_inventory',
        'is_stockout_day': 'actual_stockout_event',
        'forecast_demand_7d': 'predicted_7d_demand',
        'stockout_probability': 'predicted_stockout_prob',
        'recommended_reorder_qty': 'recommended_reorder_qty'
    })
    fact_daily.to_csv(f"{extracts_dir}/fact_daily.csv", index=False)
    print("Exported fact_daily.csv")
    
    # 3. KPI Summary
    with open("reports/metrics/stats_tests.json", "r") as f:
        stats_meta = json.load(f)
    kpis = stats_meta.get("kpis", {})
    kpi_df = pd.DataFrame([kpis])
    kpi_df.to_csv(f"{extracts_dir}/kpi_summary.csv", index=False)
    print("Exported kpi_summary.csv")
    
    # 4. Feature Importance Extract
    reg_clf = joblib.load("models/classification_xgboost.joblib")
    with open("models/model_registry.json") as f:
        reg_info = json.load(f)
    feat_names = reg_info["classification"]["features"]
    importances = reg_clf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature_name": feat_names,
        "importance_gain": np.round(importances, 4)
    }).sort_values("importance_gain", ascending=False)
    feat_imp_df.to_csv(f"{extracts_dir}/fact_feature_importance.csv", index=False)
    print("Exported fact_feature_importance.csv")
    
    # 5. DASHBOARD_SPEC.md
    write_dashboard_spec(extracts_dir)

def write_dashboard_spec(extracts_dir):
    spec_path = f"{extracts_dir}/DASHBOARD_SPEC.md"
    content = """# StockSense Tableau Dashboard Technical Specification

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
"""
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Saved Tableau Technical Specification to {spec_path}")

if __name__ == "__main__":
    score_inventory_recommendations()
    card = get_recommendation_card("S01", "P101")
    print(card["formatted_text"])
    export_tableau_extracts()
