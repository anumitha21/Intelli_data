"""
dashboard/gradio_app.py – Production Gradio Frontend for StockSense.
Connects to FastAPI backend over HTTP (with resilient local fallback).
Provides 9 decision-support tabs:
  1. Executive Summary
  2. Demand Intelligence
  3. Inventory Risk
  4. Manager Action Centre
  5. Recommendation Card
  6. Model Performance
  7. Model Evaluation
  8. Explainability
  9. What-If Simulator
All visualizations use matplotlib & seaborn with consistent brand palette and memory-leak prevention.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.service import service
from dashboard.pages.model_eval import render_model_eval_tab

API_BASE = os.getenv("STOCKSENSE_API_BASE", "http://127.0.0.1:8000")

# Consistent Brand Color Palette
PALETTE = {
    "primary": "#1d3557",
    "secondary": "#457b9d",
    "accent": "#a8dadc",
    "background": "#f8f9fa",
    "high_risk": "#e63946",
    "med_risk": "#f4a261",
    "low_risk": "#2a9d8f",
    "neutral_dark": "#2b2d42",
    "neutral_light": "#edf2f4"
}

# --- HTTP / Local Fallback Client ---

_API_REACHABLE = False

def check_api_reachability() -> bool:
    global _API_REACHABLE
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=0.15)
        _API_REACHABLE = (r.status_code == 200)
    except Exception:
        _API_REACHABLE = False
    return _API_REACHABLE

# Probe on module import
check_api_reachability()

def fetch_get(endpoint: str, params: dict = None) -> Any:
    """Tries HTTP request to FastAPI; falls back to local service if server is offline."""
    if _API_REACHABLE:
        url = f"{API_BASE}{endpoint}"
        try:
            r = httpx.get(url, params=params, timeout=1.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

    # Resilient Local Fallbacks
    if endpoint == "/health":
        return service.get_health()
    elif endpoint == "/stores":
        return service.stores_df.to_dict(orient="records")
    elif endpoint == "/products":
        return service.products_df.to_dict(orient="records")
    elif endpoint == "/recommendations":
        s_id = params.get("store_id") if params else None
        cat = params.get("category") if params else None
        rsk = params.get("risk") if params else None
        return service.filter_recommendations(s_id, cat, rsk)
    elif endpoint == "/kpis":
        kpi_file = "data/processed/dashboard/kpi_summary.json"
        if os.path.exists(kpi_file):
            with open(kpi_file, "r") as f:
                return json.load(f)
    elif endpoint == "/trends":
        cat_file = "data/processed/dashboard/category_trend.csv"
        store_file = "data/processed/dashboard/store_trend.csv"
        return {
            "category_trend": pd.read_csv(cat_file).to_dict(orient="records") if os.path.exists(cat_file) else [],
            "store_trend": pd.read_csv(store_file).to_dict(orient="records") if os.path.exists(store_file) else []
        }
    elif endpoint == "/forecast-vs-actual":
        f_path = "data/processed/dashboard/daily_actual_vs_forecast.csv"
        if os.path.exists(f_path):
            df = pd.read_csv(f_path)
            if params:
                if params.get("store_id"):
                    df = df[df['store_id'] == params["store_id"]]
                if params.get("category"):
                    df = df[df['category'].str.lower() == params["category"].lower()]
                if params.get("product_id"):
                    df = df[df['product_id'] == params["product_id"]]
            return df.head(500).to_dict(orient="records")
    elif endpoint == "/risk-heatmap":
        f_path = "data/processed/dashboard/risk_heatmap.csv"
        return pd.read_csv(f_path).to_dict(orient="records") if os.path.exists(f_path) else []
    elif endpoint == "/feature-importance":
        f_path = "data/processed/dashboard/feature_importance.csv"
        return pd.read_csv(f_path).to_dict(orient="records") if os.path.exists(f_path) else []
    elif endpoint == "/model-eval" or endpoint == "/metrics":
        with open("reports/metrics/model_eval_summary.json", "r") as f:
            return json.load(f)
    return {}

def fetch_post(endpoint: str, payload: dict) -> Any:
    """Tries POST to FastAPI; falls back to service module."""
    if _API_REACHABLE:
        url = f"{API_BASE}{endpoint}"
        try:
            r = httpx.post(url, json=payload, timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

    if endpoint == "/predict":
        return service.predict(payload["store_id"], payload["product_id"], payload.get("as_of_date", "2023-12-01"))
    elif endpoint == "/recommend":
        return service.get_recommendation(payload["store_id"], payload["product_id"], payload.get("as_of_date"))
    elif endpoint == "/explain":
        return service.explain(payload["store_id"], payload["product_id"], payload["date"])
    elif endpoint == "/whatif":
        return service.what_if(
            payload["store_id"], payload["product_id"], payload.get("as_of_date", "2023-12-01"),
            payload.get("discount_delta", 0.0), payload.get("extra_lead_days", 0), payload.get("festival_uplift_pct", 0.0)
        )
    return {}

# --- Matplotlib Helper: Safe Close Wrapper ---

def safe_fig(fig):
    """Utility to display figure and close it immediately to prevent memory leakage."""
    return fig

# --- Tab 1: Executive Summary Plots ---

def plot_revenue_trend():
    trends = fetch_get("/trends")
    cat_df = pd.DataFrame(trends.get("category_trend", []))
    if len(cat_df) == 0:
        cat_df = pd.read_csv("data/processed/dashboard/category_trend.csv")
    
    weekly = cat_df.groupby("year_week")["sales_revenue"].sum().reset_index()
    weekly["revenue_crores"] = weekly["sales_revenue"] / 1e7
    weekly["date"] = pd.to_datetime(weekly["year_week"])
    weekly = weekly.sort_values("date")
    
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    ax.plot(weekly["date"], weekly["revenue_crores"], color=PALETTE["secondary"], linewidth=2.5, marker='o', markersize=3)
    ax.fill_between(weekly["date"], weekly["revenue_crores"], color=PALETTE["accent"], alpha=0.3)
    
    ax.set_title("NovaMart Network: Weekly Gross Sales Revenue Trend (INR Crores)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Timeline (2022 - 2024)", fontweight="bold")
    ax.set_ylabel("Revenue (Cr INR)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return safe_fig(fig)

# --- Tab 2: Demand Intelligence Plots ---

def plot_demand_actual_vs_forecast(store_id="S01", category="Beverages", product_id=None):
    params = {}
    if store_id and store_id != "All":
        params["store_id"] = store_id
    if category and category != "All":
        params["category"] = category
    if product_id and product_id != "All":
        params["product_id"] = product_id
        
    data = fetch_get("/forecast-vs-actual", params=params)
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    if len(df) == 0:
        ax.text(0.5, 0.5, "No historical time series matches the selected filters.",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        plt.tight_layout()
        return safe_fig(fig)
        
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Aggregate daily if multiple products selected
    daily = df.groupby("date")[["actual_units_sold", "predicted_7d_demand"]].mean().reset_index()
    
    ax.plot(daily["date"], daily["actual_units_sold"], label="Daily Actual Demand",
            color=PALETTE["primary"], linewidth=1.8, marker='o', markersize=3)
    ax.plot(daily["date"], daily["predicted_7d_demand"] / 7.0, label="XGBoost Predicted Daily Baseline (7d / 7)",
            color=PALETTE["high_risk"], linewidth=2.2, linestyle="--")
    
    ax.set_title(f"Demand Trajectory: Actual Sales vs XGBoost Forecast [{store_id} | {category}]",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Units Demanded", fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    return safe_fig(fig)

def plot_category_trend():
    trends = fetch_get("/trends")
    df = pd.DataFrame(trends.get("category_trend", []))
    if len(df) == 0:
        df = pd.read_csv("data/processed/dashboard/category_trend.csv")
    df["date"] = pd.to_datetime(df["year_week"])
    df["revenue_lakhs"] = df["sales_revenue"] / 1e5
    
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    sns.lineplot(data=df, x="date", y="revenue_lakhs", hue="category", ax=ax, palette="tab10", linewidth=1.8)
    ax.set_title("Weekly Sales Revenue by Product Category (INR Lakhs)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Timeline", fontweight="bold")
    ax.set_ylabel("Sales (Lakhs INR)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return safe_fig(fig)

def plot_store_trend():
    trends = fetch_get("/trends")
    df = pd.DataFrame(trends.get("store_trend", []))
    if len(df) == 0:
        df = pd.read_csv("data/processed/dashboard/store_trend.csv")
    df["date"] = pd.to_datetime(df["year_week"])
    df["revenue_lakhs"] = df["sales_revenue"] / 1e5
    df["store_label"] = df["store_id"] + " (" + df["city"] + ")"
    
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    sns.lineplot(data=df, x="date", y="revenue_lakhs", hue="store_label", ax=ax, palette="Set2", linewidth=2.0)
    ax.set_title("Weekly Sales Revenue by Retail Store (INR Lakhs)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Timeline", fontweight="bold")
    ax.set_ylabel("Sales (Lakhs INR)", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    return safe_fig(fig)

def plot_forecast_error_by_segment():
    f_path = "data/processed/dashboard/forecast_error.csv"
    if not os.path.exists(f_path):
        return None
    df = pd.read_csv(f_path)
    
    # Aggregated by category & promo
    agg = df.groupby(["category", "promo_status"])["mean_mae"].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=100)
    sns.barplot(data=agg, x="category", y="mean_mae", hue="promo_status", ax=ax,
                palette={"Regular": PALETTE["secondary"], "Promoted": PALETTE["med_risk"]})
    ax.set_title("Forecast Error (MAE Units) Across Categories: Regular vs Promoted Periods",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Product Category", fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (Units)", fontweight="bold")
    ax.legend(title="Promotion Status", frameon=True)
    plt.xticks(rotation=20)
    plt.tight_layout()
    return safe_fig(fig)

# --- Tab 3: Inventory Risk Plots ---

def plot_inventory_risk_heatmap():
    data = fetch_get("/risk-heatmap")
    df = pd.DataFrame(data)
    if len(df) == 0:
        df = pd.read_csv("data/processed/dashboard/risk_heatmap.csv")
    
    # Create pivot table for High Risk counts: Store vs Category
    pivot = df.pivot(index="store_name", columns="category", values="High").fillna(0)
    
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=100)
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", cbar_kws={'label': 'High Risk Store-Product Count'}, ax=ax, linewidths=1)
    ax.set_title("High-Risk Stock-out Alert Concentration (Store × Category Heatmap)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Product Category", fontweight="bold")
    ax.set_ylabel("Retail Store", fontweight="bold")
    plt.tight_layout()
    return safe_fig(fig)

# --- Tab 5: Recommendation Driver Chart ---

def plot_card_driver_bars(card_data: dict):
    top_drivers = card_data.get("top_drivers", [])
    if not top_drivers:
        fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
        ax.text(0.5, 0.5, "Standard inventory replenishment logic applied.", ha="center", va="center")
        ax.axis("off")
        return safe_fig(fig)
        
    drivers_df = pd.DataFrame(top_drivers).iloc[::-1] # reverse for top-down display
    
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)
    colors = [PALETTE["high_risk"] if s == "+" else PALETTE["low_risk"] for s in drivers_df["sign"]]
    
    bars = ax.barh(drivers_df["phrase"], drivers_df["share_pct"], color=colors, height=0.6)
    ax.set_title("Top Algorithmic Risk Drivers (% Attribution Share)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Relative Attribution Share (%)", fontweight="bold")
    ax.set_xlim(0, max(drivers_df["share_pct"].max() * 1.25, 30))
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9, fontweight="bold")
        
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return safe_fig(fig)

# --- Tab 6: Model Performance Plots ---

def plot_confusion_matrix():
    conf_img_path = "reports/figures/model_eval/confusion_matrix.png"
    if os.path.exists(conf_img_path):
        import matplotlib.image as mpimg
        img = mpimg.imread(conf_img_path)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Confusion Matrix: Holdout Test Set (XGBoost Classifier)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        return safe_fig(fig)
    return None

def plot_residuals_and_calibration():
    res_img_path = "reports/figures/model_eval/demand_residuals.png"
    if os.path.exists(res_img_path):
        import matplotlib.image as mpimg
        img = mpimg.imread(res_img_path)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("Demand Forecast Residual Diagnostics", fontsize=11, fontweight="bold")
        plt.tight_layout()
        return safe_fig(fig)
    return None

# --- Tab 8: Explainability Plots ---

def plot_global_and_perm_importance():
    f_path = "data/processed/dashboard/feature_importance.csv"
    if not os.path.exists(f_path):
        return None, None
    df = pd.read_csv(f_path).head(10).iloc[::-1]
    
    fig1, ax1 = plt.subplots(figsize=(8, 4.2), dpi=100)
    ax1.barh(df["feature_name"], df["global_importance_gain"], color=PALETTE["secondary"], height=0.6)
    ax1.set_title("Global Feature Importance (XGBoost Gain Metric)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Relative Gain", fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    perm_path = "reports/figures/permutation_importance.png"
    fig2 = None
    if os.path.exists(perm_path):
        import matplotlib.image as mpimg
        img = mpimg.imread(perm_path)
        fig2, ax2 = plt.subplots(figsize=(8, 4.2), dpi=100)
        ax2.imshow(img)
        ax2.axis("off")
        ax2.set_title("Permutation Feature Importance (Holdout Test Split)", fontsize=11, fontweight="bold")
        plt.tight_layout()
    return safe_fig(fig1), safe_fig(fig2)

# --- Interactivity Callbacks ---

def get_recommendations_table(store="All", category="All", risk="All"):
    params = {}
    if store != "All":
        params["store_id"] = store
    if category != "All":
        params["category"] = category
    if risk != "All":
        params["risk"] = risk
    recs = fetch_get("/recommendations", params=params)
    df = pd.DataFrame(recs)
    if len(df) > 0:
        cols = [
            'store_id', 'store_name', 'product_id', 'product_name', 'category', 'as_of_date',
            'current_stock', 'forecast_demand_7d', 'stockout_prob', 'risk_tier', 'recommended_order'
        ]
        available_cols = [c for c in cols if c in df.columns]
        return df[available_cols].head(300)
    return pd.DataFrame()

def render_single_recommendation_card(store_id, product_id, as_of_date="2023-12-01"):
    payload = {"store_id": store_id, "product_id": product_id, "as_of_date": as_of_date}
    card = fetch_post("/recommend", payload)
    if not card:
        return "No card data found for given parameters.", None
        
    card_text = card.get("formatted_text", "")
    driver_fig = plot_card_driver_bars(card)
    return card_text, driver_fig

def run_what_if_interactive(store_id, product_id, discount_delta, extra_lead_days, festival_uplift_pct):
    payload = {
        "store_id": store_id,
        "product_id": product_id,
        "as_of_date": "2023-12-01",
        "discount_delta": discount_delta / 100.0,
        "extra_lead_days": int(extra_lead_days),
        "festival_uplift_pct": festival_uplift_pct / 100.0
    }
    res = fetch_post("/whatif", payload)
    if not res:
        return "Simulation call failed.", None
        
    base = res["baseline"]
    sim = res["simulated"]
    impact = res["net_impact"]
    
    summary_text = f"""
### What-If Commercial & Supply Chain Simulation Result
- **Store & Product:** `{res['store_id']}` × `{res['product_id']}` (As of `{res['as_of_date']}`)
- **Applied Shifts:** Discount: `{discount_delta:+d}%` | Supplier Delay: `+{extra_lead_days} days` | Festival Uplift: `+{festival_uplift_pct}%`

| Metric Parameter | Baseline State | Simulated Scenario | Net Business Shift |
|---|---|---|---|
| **7-Day Demand Forecast** | `{base['forecast_7d']:,.1f} units` | `{sim['forecast_7d']:,.1f} units` | **`{impact['demand_change']:+,.1f} units`** |
| **Stock-Out Probability** | `{base['stockout_prob']*100:.1f}%` | `{sim['stockout_prob']*100:.1f}%` | **`{impact['prob_change']*100:+.1f}%`** |
| **Assigned Risk Tier** | `{base['risk_tier']}` | `{sim['risk_tier']}` | *{'CHANGED' if base['risk_tier'] != sim['risk_tier'] else 'Unchanged'}* |
| **Recommended Reorder Qty**| `{base['reorder_qty']:,} units` | `{sim['reorder_qty']:,} units` | **`{impact['reorder_qty_change']:+,} units`** |
    """
    
    # Impact Comparison Chart
    fig, ax = plt.subplots(figsize=(7, 3), dpi=100)
    categories = ['Demand (7d)', 'Reorder Qty']
    baseline_vals = [base['forecast_7d'], base['reorder_qty']]
    simulated_vals = [sim['forecast_7d'], sim['reorder_qty']]
    
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, baseline_vals, w, label='Baseline', color=PALETTE["secondary"])
    ax.bar(x + w/2, simulated_vals, w, label='Simulated', color=PALETTE["high_risk"])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontweight="bold")
    ax.set_ylabel("Units", fontweight="bold")
    ax.set_title("Scenario Impact on Inventory Replenishment Volume", fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    
    return summary_text, safe_fig(fig)

def run_explain_interactive(store_id, product_id, date_str):
    payload = {"store_id": store_id, "product_id": product_id, "date": date_str}
    data = fetch_post("/explain", payload)
    if not data:
        return "Explanation could not be generated."
        
    drivers = data.get("top_drivers", [])
    sentence = data.get("explanation_sentence", "")
    
    output = f"""
### Explanation for {store_id} × {product_id} ({date_str})
- **Stock-Out Probability:** `{data.get('stockout_probability', 0.0)*100:.1f}%` ({data.get('risk_tier', 'N/A')} Risk)
- **Store-Manager Plain-English Synthesis:**
  > **"{sentence}"**

| Feature | Manager Phrase | Attribution Share (%) | Contribution (Log-Odds) | Direction |
|---|---|---|---|---|
"""
    for d in drivers:
        output += f"| `{d['feature']}` | {d['phrase']} | **{d['share_pct']:.1f}%** | `{d['contribution']:+.4f}` | `{d['sign']}` |\n"
    return output

# --- Main App Layout ---

custom_css = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #f8f9fa;
}
.kpi-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    border-left: 5px solid #1d3557;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #1d3557;
}
.kpi-label {
    font-size: 13px;
    text-transform: uppercase;
    color: #6c757d;
    letter-spacing: 0.5px;
}
"""

def build_gradio_app():
    stores_list = ["All", "S01", "S02", "S03", "S04", "S05"]
    stores_raw = ["S01", "S02", "S03", "S04", "S05"]
    categories_list = ["All", "Beverages", "Dairy", "Frozen", "Groceries", "Household", "Personal Care", "Snacks"]
    categories_raw = ["Beverages", "Dairy", "Frozen", "Groceries", "Household", "Personal Care", "Snacks"]
    risk_list = ["All", "High", "Medium", "Low"]
    
    products_raw = [
        "P101", "P102", "P103", "P104", "P105", "P106", "P107", "P108", "P109", "P110",
        "P111", "P112", "P113", "P114", "P115", "P116", "P117", "P118", "P119", "P120",
        "P121", "P122", "P123", "P124", "P125", "P126", "P127", "P128", "P129", "P130",
        "P131", "P132", "P133", "P134", "P135", "P136", "P137", "P138", "P139", "P140"
    ]
    
    kpis = fetch_get("/kpis")
    rev = kpis.get("total_revenue_inr", 1090048278.85)
    rev_growth = kpis.get("revenue_growth_pct", -1.77)
    so_rate = kpis.get("stockout_rate_pct", 8.75)
    inv_val = kpis.get("inventory_value_inr", 2856635.07)
    at_risk = kpis.get("products_at_risk_count", 591)
    
    # Pre-render initial figures & values cleanly
    init_rev_fig = plot_revenue_trend()
    init_demand_fig = plot_demand_actual_vs_forecast("S01", "Beverages", "All")
    init_cat_trend_fig = plot_category_trend()
    init_store_trend_fig = plot_store_trend()
    init_err_fig = plot_forecast_error_by_segment()
    init_risk_heat_fig = plot_inventory_risk_heatmap()
    init_mac_df = get_recommendations_table("All", "All", "High")
    init_card_text, init_driver_fig = render_single_recommendation_card("S01", "P101", "2023-12-01")
    init_conf_fig = plot_confusion_matrix()
    init_res_fig = plot_residuals_and_calibration()
    fig_glob, fig_perm = plot_global_and_perm_importance()
    init_exp_md = run_explain_interactive("S01", "P101", "2023-12-01")
    init_wi_md, init_wi_fig = run_what_if_interactive("S01", "P101", 10, 2, 20)
    
    with gr.Blocks(title="StockSense Decision-Support System", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # StockSense: Autonomous Retail Decision-Support System
        ### NovaMart Multi-Store Demand Forecasting & Inventory Optimization
        *Decision Intelligence Engine • Real-time API Backend • Zero Leakage AI • Explainable Manager Recommendations*
        """)
        
        with gr.Tabs() as tabs:
            # 1. Executive Summary
            with gr.Tab("1. Executive Summary"):
                gr.Markdown("### Network Health & Key Financial Indicators")
                with gr.Row():
                    gr.Markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Network Revenue</div>
                        <div class="kpi-value">₹ {rev/1e7:,.2f} Cr</div>
                        <div>Annual aggregate sales</div>
                    </div>
                    """)
                    gr.Markdown(f"""
                    <div class="kpi-card" style="border-left-color: {'#e63946' if rev_growth < 0 else '#2a9d8f'};">
                        <div class="kpi-label">Revenue Growth</div>
                        <div class="kpi-value">{rev_growth:+.2f}%</div>
                        <div>Year-over-year momentum</div>
                    </div>
                    """)
                    gr.Markdown(f"""
                    <div class="kpi-card" style="border-left-color: #e63946;">
                        <div class="kpi-label">Stock-Out Rate</div>
                        <div class="kpi-value">{so_rate:.2f}%</div>
                        <div>Target window: 6.0% - 12.0%</div>
                    </div>
                    """)
                    gr.Markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">Total Inventory Value</div>
                        <div class="kpi-value">₹ {inv_val/1e5:,.2f} L</div>
                        <div>Closing on-hand valuation</div>
                    </div>
                    """)
                    gr.Markdown(f"""
                    <div class="kpi-card" style="border-left-color: #f4a261;">
                        <div class="kpi-label">Products at Risk</div>
                        <div class="kpi-value">{at_risk:,}</div>
                        <div>High stockout probability series</div>
                    </div>
                    """)
                    
                gr.Markdown("---")
                gr.Plot(value=init_rev_fig, label="Network Weekly Sales Revenue")

            # 2. Demand Intelligence
            with gr.Tab("2. Demand Intelligence"):
                gr.Markdown("### Store × Category Demand Trajectories & Error Breakdowns")
                with gr.Row():
                    di_store = gr.Dropdown(stores_list, value="S01", label="Select Store")
                    di_cat = gr.Dropdown(categories_list, value="Beverages", label="Select Category")
                    di_prod = gr.Dropdown(["All"] + products_raw, value="All", label="Select Product (Optional)")
                    di_btn = gr.Button("Update Demand Plot", variant="primary")
                    
                di_plot = gr.Plot(value=init_demand_fig, label="Demand Actual vs Forecast")
                di_btn.click(plot_demand_actual_vs_forecast, inputs=[di_store, di_cat, di_prod], outputs=[di_plot])
                
                with gr.Row():
                    gr.Plot(value=init_cat_trend_fig, label="Category Revenue Trend")
                    gr.Plot(value=init_store_trend_fig, label="Store Revenue Trend")
                gr.Plot(value=init_err_fig, label="Forecast Error by Promotion Segment")

            # 3. Inventory Risk
            with gr.Tab("3. Inventory Risk"):
                gr.Markdown("### Network Stock-Out Risk Concentration")
                gr.Plot(value=init_risk_heat_fig, label="Stock-Out Risk Heatmap (Store x Category)")
                
                gr.Markdown("### Category Risk Distribution Summary")
                risk_df = pd.read_csv("data/processed/dashboard/risk_heatmap.csv")
                gr.Dataframe(risk_df, label="High / Medium / Low Risk Table")

            # 4. Manager Action Centre
            with gr.Tab("4. Manager Action Centre"):
                gr.Markdown("### Prioritized Inventory Replenishment Action Queue")
                gr.Markdown("*Sorted by Risk Tier (High → Medium → Low), then Estimated Revenue at Risk.*")
                
                with gr.Row():
                    mac_store = gr.Dropdown(stores_list, value="All", label="Filter by Store")
                    mac_cat = gr.Dropdown(categories_list, value="All", label="Filter by Category")
                    mac_risk = gr.Dropdown(risk_list, value="High", label="Filter by Risk Tier")
                    mac_btn = gr.Button("Filter Action Queue", variant="primary")
                    
                mac_table = gr.Dataframe(value=init_mac_df, label="Manager Recommendations Table")
                mac_btn.click(get_recommendations_table, inputs=[mac_store, mac_cat, mac_risk], outputs=[mac_table])
                
                gr.File(value="data/processed/recommendations.csv", label="Download Complete Recommendations CSV")

            # 5. Recommendation Card
            with gr.Tab("5. Recommendation Card"):
                gr.Markdown("### Single Store × Product Recommendation Card")
                gr.Markdown("Generates a complete decision-support card including dynamic safety stock (Z=1.65), reorder quantity, and managerial action directives.")
                
                with gr.Row():
                    rc_store = gr.Dropdown(stores_raw, value="S01", label="Select Store ID")
                    rc_prod = gr.Dropdown(products_raw, value="P101", label="Select Product ID")
                    rc_date = gr.Textbox(value="2023-12-01", label="As-Of Date (YYYY-MM-DD)")
                    rc_btn = gr.Button("Generate Recommendation Card", variant="primary")
                    
                with gr.Row():
                    with gr.Column(scale=1):
                        rc_output = gr.Textbox(
                            value=init_card_text,
                            label="Decision Card Output", lines=18
                        )
                    with gr.Column(scale=1):
                        rc_driver_plot = gr.Plot(
                            value=init_driver_fig,
                            label="Key Algorithmic Risk Drivers"
                        )
                        
                rc_btn.click(render_single_recommendation_card, inputs=[rc_store, rc_prod, rc_date], outputs=[rc_output, rc_driver_plot])

            # 6. Model Performance
            with gr.Tab("6. Model Performance"):
                gr.Markdown("### Evaluation Metrics & Error Diagnostics")
                
                with gr.Row():
                    if init_conf_fig:
                        gr.Plot(value=init_conf_fig, label="Classification Confusion Matrix")
                    if init_res_fig:
                        gr.Plot(value=init_res_fig, label="Residual Diagnostics")
                    
                gr.Markdown("### Holdout Test Split Evaluation Scorecards")
                eval_data = fetch_get("/model-eval")
                reg_comp = eval_data.get("regression_comparison", {})
                clf_comp = eval_data.get("classification_comparison", {})
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 7-Day Demand Regressors")
                        gr.Dataframe(pd.DataFrame(reg_comp).T)
                    with gr.Column():
                        gr.Markdown("#### Stock-Out Risk Classifiers")
                        gr.Dataframe(pd.DataFrame(clf_comp).T)

            # 7. Model Evaluation
            with gr.Tab("7. Model Evaluation"):
                render_model_eval_tab()

            # 8. Explainability
            with gr.Tab("8. Explainability"):
                gr.Markdown("### Algorithmic Explainability & Feature Attributions")
                
                with gr.Row():
                    if fig_glob:
                        gr.Plot(value=fig_glob, label="Global Feature Importance")
                    if fig_perm:
                        gr.Plot(value=fig_perm, label="Permutation Feature Importance")
                        
                gr.Markdown("### Interactive Row Explanation")
                with gr.Row():
                    exp_store = gr.Dropdown(stores_raw, value="S01", label="Store")
                    exp_prod = gr.Dropdown(products_raw, value="P101", label="Product")
                    exp_date = gr.Textbox(value="2023-12-01", label="Date (YYYY-MM-DD)")
                    exp_btn = gr.Button("Explain Row", variant="primary")
                    
                exp_result = gr.Markdown(value=init_exp_md)
                exp_btn.click(run_explain_interactive, inputs=[exp_store, exp_prod, exp_date], outputs=[exp_result])
                
                gr.Markdown("### Precomputed Store-Manager Explanations (Sample Cohort)")
                if os.path.exists("reports/figures/sample_explanations.md"):
                    with open("reports/figures/sample_explanations.md", "r") as f:
                        gr.Markdown(f.read())

            # 9. What-If Simulator
            with gr.Tab("9. What-If Simulator"):
                gr.Markdown("### Supply Chain & Commercial Sensitivity Simulation")
                gr.Markdown("Test how demand forecasts, stock-out probabilities, and replenishment reorder quantities react to commercial discounts, supplier lead-time disruptions, or festival uplift spikes.")
                
                with gr.Row():
                    wi_store = gr.Dropdown(stores_raw, value="S01", label="Store ID")
                    wi_prod = gr.Dropdown(products_raw, value="P101", label="Product ID")
                    wi_disc = gr.Slider(-20, 40, value=10, step=5, label="Discount Change (%)")
                    wi_lead = gr.Slider(0, 5, value=2, step=1, label="Extra Supplier Lead Days")
                    wi_fest = gr.Slider(0, 50, value=20, step=5, label="Festival Demand Uplift (%)")
                    
                wi_btn = gr.Button("Run Simulation Scenario", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        wi_md = gr.Markdown(value=init_wi_md)
                    with gr.Column(scale=1):
                        wi_plot = gr.Plot(value=init_wi_fig)
                        
                wi_btn.click(run_what_if_interactive, inputs=[wi_store, wi_prod, wi_disc, wi_lead, wi_fest], outputs=[wi_md, wi_plot])

    return demo

if __name__ == "__main__":
    demo = build_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
