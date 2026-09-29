"""
explain.py – Model Explainability & Per-Row Manager Language Explanations
Computes per-row feature contributions using XGBoost pred_contribs (TreeSHAP),
calculates signed % share of total absolute contribution, and translates into
manager-ready structured sentences using deterministic phrase templates.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------------------
# Template Map: Feature names & contextual state to manager language
# -------------------------------------------------------------
def get_manager_phrase(feature_name: str, value: float, context_row: dict = None) -> str:
    """
    Translates a feature and its value into a concise, plain-English manager phrase.
    Deterministic template mapping without free-text hallucinations.
    """
    context_row = context_row or {}
    
    if feature_name == "promotion_flag":
        return "Promotion active" if value > 0 else "Regular pricing (No promo)"
    elif feature_name == "discount_pct":
        return "High promotional markdown" if value >= 10 else "Low discount level"
    elif feature_name == "festival_flag":
        return "Festival period surge" if value > 0 else "Standard non-festival calendar"
    elif feature_name == "weekend_flag":
        return "Weekend shopping surge" if value > 0 else "Weekday shopping baseline"
    elif feature_name == "holiday":
        return "Public holiday demand surge" if value > 0 else "Standard business day"
    elif feature_name == "has_local_event":
        return "Local commercial event / fair" if value > 0 else "Standard local traffic"
    elif feature_name == "reorder_gap":
        return "Stock below reorder level" if value > 0 else "Stock above reorder threshold"
    elif feature_name == "days_of_inventory":
        return "Critically low days of inventory" if value < 3.0 else "Adequate stock runway"
    elif feature_name == "inventory_to_demand_ratio":
        return "Inventory-to-demand ratio depleted" if value < 1.0 else "Healthy stock-to-demand ratio"
    elif feature_name == "incoming_stock":
        return "Warehouse shipment received" if value > 0 else "No incoming shipment"
    elif feature_name == "lead_time":
        return "Long supplier lead time" if value >= 3 else "Fast supplier replenishment"
    elif feature_name == "shelf_life":
        return "Perishable short shelf life" if value <= 7 else "Extended shelf life product"
    elif feature_name == "cold_start":
        return "New product launch (Cold start)" if value > 0 else "Established sales history"
    elif feature_name == "rolling_mean_7":
        rm14 = context_row.get('rolling_mean_14', value)
        return "Recent sales velocity growth" if value >= rm14 else "Softening sales velocity"
    elif feature_name == "rolling_mean_14":
        return "Elevated bi-weekly sales pace" if value > 100 else "Stable baseline sales"
    elif feature_name == "rolling_std_7":
        return "High demand volatility" if value > 25 else "Predictable demand pattern"
    elif feature_name == "lag_1":
        return "Strong previous day sales spike" if value > 100 else "Moderate yesterday sales"
    elif feature_name == "lag_7":
        return "High prior week sales anchor" if value > 100 else "Steady weekly pace"
    elif feature_name == "lag_14":
        return "Historical fortnightly sales momentum"
    elif feature_name == "temperature":
        return "Elevated summer temperatures" if value > 32 else "Moderate regional temperature"
    elif feature_name == "rain":
        return "Heavy rainfall disruption" if value > 10 else "Dry weather conditions"
    elif feature_name == "day_of_week":
        return "Peak late-week demand" if value >= 4 else "Mid-week cadence"
    elif feature_name == "month":
        return f"Month {int(value)} seasonal cycle"
    elif feature_name == "week_no":
        return f"Calendar week {int(value)} seasonality"
    elif feature_name == "price_change":
        return "Retail price adjustment" if abs(value) > 0.05 else "Standard catalog price"
    else:
        return feature_name.replace("_", " ").title()

# -------------------------------------------------------------
# Core Explainer
# -------------------------------------------------------------
class StockSenseExplainer:
    def __init__(self, model_path="models/classification_xgboost.joblib", registry_path="models/model_registry.json"):
        self.clf = joblib.load(model_path)
        self.booster = self.clf.get_booster()
        
        with open(registry_path, "r") as f:
            reg = json.load(f)
            
        self.feature_cols = reg["classification"]["features"]
        self.high_risk_thresh = reg["classification"]["risk_tiers"]["high_risk_threshold"]
        self.med_risk_thresh = reg["classification"]["risk_tiers"]["medium_risk_threshold"]
        self.decision_thresh = reg["classification"]["decision_threshold"]

    def explain_row_from_features(self, row_series: pd.Series, top_k=5) -> dict:
        """Explains a single feature row pandas Series."""
        X_df = pd.DataFrame([row_series[self.feature_cols]])
        prob = float(self.clf.predict_proba(X_df)[0, 1])
        
        # Risk classification
        if prob >= self.high_risk_thresh:
            risk = "High"
        elif prob >= self.med_risk_thresh:
            risk = "Medium"
        else:
            risk = "Low"
            
        # Compute TreeSHAP contributions
        dmat = xgb.DMatrix(X_df)
        contribs = self.booster.predict(dmat, pred_contribs=True)[0]
        feat_contribs = contribs[:-1] # exclude bias term
        
        # Calculate absolute sum and % share
        abs_contribs = np.abs(feat_contribs)
        total_abs = float(np.sum(abs_contribs))
        if total_abs == 0.0:
            total_abs = 1e-6
            
        drivers = []
        context_dict = row_series.to_dict()
        
        for i, f_name in enumerate(self.feature_cols):
            raw_c = float(feat_contribs[i])
            share_pct = round((abs(raw_c) / total_abs) * 100.0, 1)
            val = float(row_series[f_name])
            phrase = get_manager_phrase(f_name, val, context_dict)
            sign = "+" if raw_c >= 0 else "-"
            
            drivers.append({
                "feature": f_name,
                "phrase": phrase,
                "value": val,
                "contribution": round(raw_c, 4),
                "share_pct": share_pct,
                "sign": sign,
                "token": f"{phrase} {sign}{int(round(share_pct))}%"
            })
            
        # Sort top K by absolute contribution
        drivers.sort(key=lambda d: abs(d["contribution"]), reverse=True)
        top_drivers = drivers[:top_k]
        
        # Format sentence: "Promotion active +31%, Weekend approaching +22%, Recent sales growth +19%"
        sentence = ", ".join([d["token"] for d in top_drivers])
        
        return {
            "date": str(pd.to_datetime(row_series.get('date', '2024-01-01')).date()),
            "store_id": str(row_series.get('store_id', 'S01')),
            "product_id": str(row_series.get('product_id', 'P101')),
            "stockout_probability": round(prob, 4),
            "risk_tier": risk,
            "top_drivers": top_drivers,
            "explanation_sentence": sentence
        }

def explain_row(store_id: str, product_id: str, date: str, dataset_path="data/processed/features_test.parquet", top_k=5) -> dict:
    """
    API function: Returns the top drivers and manager-friendly sentence explanation
    for a given (store_id, product_id, date).
    """
    explainer = StockSenseExplainer()
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    df['date_str'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    date_clean = pd.to_datetime(date).strftime('%Y-%m-%d')
    
    sub = df[(df['store_id'] == store_id) & (df['product_id'] == product_id) & (df['date_str'] == date_clean)]
    if len(sub) == 0:
        # Fallback to closest matching row
        sub = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)]
        if len(sub) == 0:
            sub = df.head(1)
            
    row = sub.iloc[0]
    return explainer.explain_row_from_features(row, top_k=top_k)

def generate_sample_explanations():
    """Generates 5 sample explanations for High-Risk store x product rows and writes markdown report."""
    print("Generating 5 sample High-Risk explanations...")
    os.makedirs("reports/figures", exist_ok=True)
    explainer = StockSenseExplainer()
    
    test_df = pd.read_parquet("data/processed/features_test.parquet")
    
    # Find rows with High predicted risk
    high_risk_explanations = []
    for _, row in test_df.iterrows():
        exp = explainer.explain_row_from_features(row)
        if exp["risk_tier"] == "High":
            high_risk_explanations.append(exp)
        if len(high_risk_explanations) >= 5:
            break
            
    # If fewer than 5 high risk, fill with highest available probabilities
    if len(high_risk_explanations) < 5:
        sorted_rows = []
        for _, row in test_df.head(50).iterrows():
            sorted_rows.append(explainer.explain_row_from_features(row))
        sorted_rows.sort(key=lambda x: x['stockout_probability'], reverse=True)
        high_risk_explanations = sorted_rows[:5]
        
    md_lines = [
        "# NovaMart Retail: Manager Language Stock-out Explanations",
        "",
        "> **Methodology:** TreeSHAP per-row contributions (`xgb.pred_contribs`), normalized by total absolute feature attribution and mapped to deterministic domain phrases.",
        "",
        "---",
        ""
    ]
    
    for i, exp in enumerate(high_risk_explanations, 1):
        md_lines.extend([
            f"### Sample Explanation {i}: Store {exp['store_id']} × Product {exp['product_id']} ({exp['date']})",
            f"- **Predicted Stock-out Probability:** `{exp['stockout_probability'] * 100:.1f}%`",
            f"- **Assigned Risk Tier:** `{exp['risk_tier']}`",
            f"- **Manager Plain-English Explanation:**",
            f"  > **\"{exp['explanation_sentence']}\"**",
            "",
            "| Rank | Feature | Manager Phrase | Contribution (Log-Odds) | Attribution Share (%) | Direction |",
            "|---|---|---|---|---|---|"
        ])
        for rank, d in enumerate(exp['top_drivers'], 1):
            md_lines.append(f"| #{rank} | `{d['feature']}` | {d['phrase']} | `{d['contribution']:+.4f}` | **{d['share_pct']:.1f}%** | `{d['sign']}` |")
        md_lines.append("\n---\n")
        
    out_file = "reports/figures/sample_explanations.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Sample explanations saved to {out_file}.")

if __name__ == "__main__":
    generate_sample_explanations()
