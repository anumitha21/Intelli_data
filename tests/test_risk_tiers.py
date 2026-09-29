"""
test_risk_tiers.py – Unit tests for Phase 5 recommendations, risk tier thresholds,
non-negative reorder quantities, dynamic model registry loading, and Tableau extracts.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from src.recommend import (
    load_models_from_registry,
    score_inventory_recommendations,
    get_recommendation_card,
    what_if_scenario
)

@pytest.fixture(scope="module")
def recommendations_data():
    rec_path = "data/processed/recommendations.csv"
    if not os.path.exists(rec_path):
        _, manager_table = score_inventory_recommendations()
        return manager_table
    return pd.read_csv(rec_path)

def test_risk_tiers_strict_threshold_compliance(recommendations_data):
    """
    Verify that EVERY row strictly respects the risk tier thresholds:
    - High: p >= 0.70
    - Medium: 0.40 <= p < 0.70
    - Low: p < 0.40
    """
    df = recommendations_data
    
    high_mask = df['risk_tier'] == 'High'
    med_mask = df['risk_tier'] == 'Medium'
    low_mask = df['risk_tier'] == 'Low'
    
    # Assert all rows are covered
    assert (high_mask | med_mask | low_mask).all(), "Found invalid risk tier labels"
    
    # High risk validation
    assert (df.loc[high_mask, 'stockout_prob'] >= 0.70).all(), "High risk tier contains p < 0.70"
    
    # Medium risk validation
    assert (df.loc[med_mask, 'stockout_prob'] >= 0.40).all(), "Medium risk tier contains p < 0.40"
    assert (df.loc[med_mask, 'stockout_prob'] < 0.70).all(), "Medium risk tier contains p >= 0.70"
    
    # Low risk validation
    assert (df.loc[low_mask, 'stockout_prob'] < 0.40).all(), "Low risk tier contains p >= 0.40"

def test_reorder_quantity_never_negative(recommendations_data):
    """Verify that recommended reorder quantity is strictly non-negative (>= 0)."""
    df = recommendations_data
    reorder_vals = df['recommended_order'].values
    assert (reorder_vals >= 0).all(), f"Found negative reorder quantities! Min: {reorder_vals.min()}"
    assert np.issubdtype(reorder_vals.dtype, np.integer), "Reorder quantity must be integer units"

def test_dynamic_registry_model_swapping(tmp_path):
    """
    Verify that swapping the model path in models/model_registry.json
    changes the model loaded by recommend.py (no hardcoded model names).
    """
    registry_path = "models/model_registry.json"
    with open(registry_path, "r") as f:
        reg = json.load(f)
        
    # Create temporary swapped registry pointing to random forest
    swapped_reg = reg.copy()
    swapped_reg["classification"]["artifact_path"] = "models/classification_random_forest.joblib"
    
    tmp_reg_file = str(tmp_path / "model_registry_swapped.json")
    with open(tmp_reg_file, "w") as f:
        json.dump(swapped_reg, f)
        
    # Load using standard registry
    standard_bundle = load_models_from_registry(registry_path)
    # Load using swapped registry
    swapped_bundle = load_models_from_registry(tmp_reg_file)
    
    standard_clf_type = type(standard_bundle["classification_model"]).__name__
    swapped_clf_type = type(swapped_bundle["classification_model"]).__name__
    
    assert standard_clf_type != swapped_clf_type or standard_bundle["classification_model"] is not swapped_bundle["classification_model"]
    assert "RandomForest" in swapped_clf_type or "Forest" in str(swapped_clf_type)
    assert "XGB" in standard_clf_type or "xgb" in str(standard_clf_type).lower()

def test_manager_table_columns_and_format(recommendations_data):
    """Verify manager table has the exact PS columns."""
    expected_cols = [
        'store_id', 'store_name', 'product_id', 'product_name', 'category', 'as_of_date',
        'current_stock', 'forecast_demand_7d', 'stockout_prob', 'risk_tier',
        'safety_stock', 'recommended_order', 'revenue_at_risk'
    ]
    for col in expected_cols:
        assert col in recommendations_data.columns, f"Missing required manager table column: {col}"

def test_tableau_extracts_exist_and_load():
    """Verify all 6 Tableau CSV extracts and DASHBOARD_SPEC.md exist and load cleanly."""
    extracts_dir = "dashboard/tableau_extracts"
    expected_files = [
        "fact_daily.csv", "fact_recommendations.csv", "fact_feature_importance.csv",
        "kpi_summary.csv", "dim_store.csv", "dim_product.csv", "DASHBOARD_SPEC.md"
    ]
    for filename in expected_files:
        p = os.path.join(extracts_dir, filename)
        assert os.path.exists(p), f"Missing Tableau extract artifact: {p}"
        if filename.endswith(".csv"):
            df = pd.read_csv(p)
            assert len(df) > 0, f"Tableau CSV {filename} is empty"

def test_recommendation_card_and_what_if():
    """Verify recommendation card dictionary and What-If scenario output."""
    card = get_recommendation_card("S01", "P101")
    assert isinstance(card, dict)
    assert "manager_action" in card
    assert "why_drivers" in card
    assert card["recommended_reorder_qty"] >= 0
    
    sim = what_if_scenario("S01", "P101", card["as_of_date"], discount_delta=10.0, extra_lead_days=2)
    assert isinstance(sim, dict)
    assert "baseline" in sim
    assert "simulated" in sim
    assert "net_impact" in sim
