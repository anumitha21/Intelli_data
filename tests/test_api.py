"""
tests/test_api.py – Comprehensive unit tests for FastAPI decision-support service.
Exercises all 14 endpoints including error handling, dynamic registry reloading,
and parameter filtering using fastapi.testclient.TestClient.
"""

import os
import json
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "regression_model" in data
    assert "classification_model" in data
    assert data["decision_threshold"] > 0
    assert data["high_risk_threshold"] == 0.70
    assert data["medium_risk_threshold"] == 0.40

def test_stores_and_products_endpoints():
    r_stores = client.get("/stores")
    assert r_stores.status_code == 200
    stores = r_stores.json()
    assert len(stores) > 0
    assert "store_id" in stores[0]
    assert "city" in stores[0]

    r_prods = client.get("/products")
    assert r_prods.status_code == 200
    prods = r_prods.json()
    assert len(prods) > 0
    assert "product_id" in prods[0]
    assert "product_name" in prods[0]

def test_predict_endpoint_valid_and_invalid():
    # Valid prediction
    payload = {
        "store_id": "S01",
        "product_id": "P101",
        "as_of_date": "2023-12-01"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["store_id"] == "S01"
    assert data["product_id"] == "P101"
    assert data["forecast_demand_7d"] >= 0.0
    assert 0.0 <= data["stockout_probability"] <= 1.0
    assert data["risk_tier"] in ["High", "Medium", "Low"]

    # Invalid store_id
    r_bad_store = client.post("/predict", json={"store_id": "S999", "product_id": "P101"})
    assert r_bad_store.status_code == 404
    assert "Invalid store_id" in r_bad_store.json()["detail"]

    # Invalid product_id
    r_bad_prod = client.post("/predict", json={"store_id": "S01", "product_id": "P9999"})
    assert r_bad_prod.status_code == 404
    assert "Invalid product_id" in r_bad_prod.json()["detail"]

def test_recommend_endpoint():
    payload = {
        "store_id": "S01",
        "product_id": "P101",
        "as_of_date": "2023-12-01"
    }
    r = client.post("/recommend", json=payload)
    assert r.status_code == 200
    card = r.json()
    assert card["store_id"] == "S01"
    assert card["product_id"] == "P101"
    assert "manager_action" in card
    assert card["recommended_reorder_qty"] >= 0
    assert len(card["top_drivers"]) > 0

    # Bad request
    r_bad = client.post("/recommend", json={"store_id": "S999", "product_id": "P101"})
    assert r_bad.status_code == 404

def test_recommendations_list_and_filtering():
    r_all = client.get("/recommendations")
    assert r_all.status_code == 200
    all_recs = r_all.json()
    assert len(all_recs) > 0

    # Filter by risk
    r_high = client.get("/recommendations?risk=High")
    assert r_high.status_code == 200
    high_recs = r_high.json()
    for row in high_recs:
        assert row["risk_tier"] == "High"

    # Filter by store
    r_store = client.get("/recommendations?store_id=S01")
    assert r_store.status_code == 200
    store_recs = r_store.json()
    for row in store_recs:
        assert row["store_id"] == "S01"

def test_explain_endpoint():
    payload = {
        "store_id": "S01",
        "product_id": "P101",
        "date": "2023-12-01"
    }
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["store_id"] == "S01"
    assert "top_drivers" in data
    assert len(data["top_drivers"]) <= 5
    assert "explanation_sentence" in data

    # Bad input
    r_bad = client.post("/explain", json={"store_id": "S999", "product_id": "P101", "date": "2023-12-01"})
    assert r_bad.status_code == 404

def test_kpis_and_trends_endpoints():
    r_kpis = client.get("/kpis")
    assert r_kpis.status_code == 200
    kpis = r_kpis.json()
    assert "total_revenue_inr" in kpis
    assert "stockout_rate_pct" in kpis
    assert "products_at_risk_count" in kpis

    r_trends = client.get("/trends")
    assert r_trends.status_code == 200
    trends = r_trends.json()
    assert "category_trend" in trends
    assert "store_trend" in trends

def test_forecast_vs_actual_endpoint():
    r = client.get("/forecast-vs-actual?store_id=S01")
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list)
    if len(items) > 0:
        assert items[0]["store_id"] == "S01"

def test_risk_heatmap_and_feature_importance():
    r_heat = client.get("/risk-heatmap")
    assert r_heat.status_code == 200
    assert isinstance(r_heat.json(), list)

    r_imp = client.get("/feature-importance")
    assert r_imp.status_code == 200
    assert isinstance(r_imp.json(), list)

def test_model_eval_and_metrics_endpoints():
    r_eval = client.get("/model-eval")
    assert r_eval.status_code == 200
    eval_data = r_eval.json()
    assert "regression_comparison" in eval_data
    assert "classification_comparison" in eval_data

    r_metrics = client.get("/metrics")
    assert r_metrics.status_code == 200

def test_whatif_endpoint():
    payload = {
        "store_id": "S01",
        "product_id": "P101",
        "as_of_date": "2023-12-01",
        "discount_delta": 0.10,
        "extra_lead_days": 1,
        "festival_uplift_pct": 0.15
    }
    r = client.post("/whatif", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "baseline" in data
    assert "simulated" in data
    assert "net_impact" in data
    assert "reorder_qty_change" in data["net_impact"]

def test_dynamic_registry_swapping_in_api():
    """Confirms that changing the registry dynamically changes /health and model loading."""
    registry_file = "models/model_registry.json"
    with open(registry_file, "r") as f:
        orig = json.load(f)

    try:
        mod = json.loads(json.dumps(orig))
        mod["regression"]["winner_model_name"] = "Linear Regression (Swapped)"
        with open(registry_file, "w") as f:
            json.dump(mod, f, indent=2)

        # Health endpoint should reload and reflect swapped name
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["regression_model"] == "Linear Regression (Swapped)"
    finally:
        with open(registry_file, "w") as f:
            json.dump(orig, f, indent=2)

        # Re-check restoration
        r_restored = client.get("/health")
        assert r_restored.json()["regression_model"] == orig["regression"]["winner_model_name"]
