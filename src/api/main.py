"""
main.py – Production FastAPI Service for StockSense.
Implements all decision-support endpoints, health checks, model evaluation data,
diagnostic trends, and recommendation cards.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.schemas import (
    HealthResponse, StoreItem, ProductItem, PredictRequest, PredictResponse,
    RecommendRequest, RecommendCardResponse, ExplainRequest, ExplainResponse,
    WhatIfRequest, WhatIfResponse
)
from src.api.service import service

app = FastAPI(
    title="StockSense Decision-Support API",
    version="1.0.0",
    description="Autonomous decision-support backend for NovaMart retail inventory replenishment."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint: returns status and current active model names from registry."""
    return service.get_health()

@app.get("/stores", response_model=List[StoreItem])
def get_stores():
    """Returns available store entities for UI dropdowns."""
    return service.stores_df.to_dict(orient="records")

@app.get("/products", response_model=List[ProductItem])
def get_products():
    """Returns available product catalogue for UI dropdowns."""
    return service.products_df.to_dict(orient="records")

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    """Predicts next-7-day demand and stock-out probability for a Store x Product."""
    try:
        return service.predict(req.store_id, req.product_id, req.as_of_date)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecommendCardResponse)
def recommend_endpoint(req: RecommendRequest):
    """Generates the full Store Manager Recommendation Card with plain-English action."""
    try:
        return service.get_recommendation(req.store_id, req.product_id, req.as_of_date)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendations")
def list_recommendations(
    store_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    risk: Optional[str] = Query(None)
):
    """Returns the priority manager action queue with optional filters."""
    return service.filter_recommendations(store_id, category, risk)

@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(req: ExplainRequest):
    """Returns top 5 feature attribution drivers in manager language."""
    try:
        return service.explain(req.store_id, req.product_id, req.date)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    """Returns saved model performance metrics and split information."""
    metrics_path = "reports/metrics/model_eval_summary.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return service.registry

@app.get("/model-eval")
def get_model_eval():
    """Returns complete evaluation data for the Model Evaluation Gradio page."""
    with open("reports/metrics/model_eval_summary.json", "r") as f:
        return json.load(f)

@app.get("/kpis")
def get_kpis():
    """Returns executive KPI summary for executive summary dashboard cards."""
    kpi_file = "data/processed/dashboard/kpi_summary.json"
    if os.path.exists(kpi_file):
        with open(kpi_file, "r") as f:
            return json.load(f)
    with open("reports/metrics/stats_tests.json", "r") as f:
        data = json.load(f)
    return data.get("kpis", {})

@app.get("/trends")
def get_trends():
    """Returns weekly sales trends by category and store."""
    cat_file = "data/processed/dashboard/category_trend.csv"
    store_file = "data/processed/dashboard/store_trend.csv"
    import pandas as pd
    cat_df = pd.read_csv(cat_file) if os.path.exists(cat_file) else pd.DataFrame()
    store_df = pd.read_csv(store_file) if os.path.exists(store_file) else pd.DataFrame()
    return {
        "category_trend": cat_df.to_dict(orient="records"),
        "store_trend": store_df.to_dict(orient="records")
    }

@app.get("/forecast-vs-actual")
def get_forecast_vs_actual(
    store_id: Optional[str] = None,
    category: Optional[str] = None,
    product_id: Optional[str] = None
):
    """Returns time-series actuals vs predicted demand."""
    f_path = "data/processed/dashboard/daily_actual_vs_forecast.csv"
    import pandas as pd
    if not os.path.exists(f_path):
        return []
    df = pd.read_csv(f_path)
    if store_id:
        df = df[df['store_id'] == store_id]
    if category:
        df = df[df['category'].str.lower() == category.lower()]
    if product_id:
        df = df[df['product_id'] == product_id]
    return df.head(500).to_dict(orient="records")

@app.get("/risk-heatmap")
def get_risk_heatmap():
    """Returns store x category risk count distribution."""
    f_path = "data/processed/dashboard/risk_heatmap.csv"
    import pandas as pd
    if os.path.exists(f_path):
        return pd.read_csv(f_path).to_dict(orient="records")
    return []

@app.get("/feature-importance")
def get_feature_importance():
    """Returns global and permutation feature importances."""
    f_path = "data/processed/dashboard/feature_importance.csv"
    import pandas as pd
    if os.path.exists(f_path):
        return pd.read_csv(f_path).to_dict(orient="records")
    return []

@app.post("/whatif", response_model=WhatIfResponse)
def whatif_endpoint(req: WhatIfRequest):
    """Simulates commercial and supply chain shifts (discount, lead time delay, festival spike)."""
    try:
        return service.what_if(
            req.store_id, req.product_id, req.as_of_date,
            req.discount_delta, req.extra_lead_days, req.festival_uplift_pct
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
