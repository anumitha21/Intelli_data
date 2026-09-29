"""
test_explain.py – Unit tests for explainability module and template mapping.
"""

import os
import pytest
import pandas as pd
from src.explain import get_manager_phrase, StockSenseExplainer, explain_row

def test_template_map_different_inputs_give_different_phrases():
    """Verify that distinct feature values produce distinct manager phrases."""
    # Promotion active vs inactive
    p_active = get_manager_phrase("promotion_flag", 1.0)
    p_inactive = get_manager_phrase("promotion_flag", 0.0)
    assert p_active != p_inactive
    assert "Promotion active" in p_active
    assert "Regular pricing" in p_inactive
    
    # Days of inventory low vs high
    doi_low = get_manager_phrase("days_of_inventory", 1.2)
    doi_high = get_manager_phrase("days_of_inventory", 15.0)
    assert doi_low != doi_high
    assert "low" in doi_low.lower()
    
    # Lead time long vs short
    lead_long = get_manager_phrase("lead_time", 4.0)
    lead_short = get_manager_phrase("lead_time", 1.0)
    assert lead_long != lead_short
    assert "Long" in lead_long
    assert "Fast" in lead_short

def test_explain_row_structure_and_types():
    """Verify explain_row returns all required keys, exactly 5 drivers, and non-empty sentence."""
    exp = explain_row("S01", "P101", "2023-12-01")
    assert isinstance(exp, dict)
    assert "top_drivers" in exp
    assert "explanation_sentence" in exp
    assert "risk_tier" in exp
    assert "stockout_probability" in exp
    
    drivers = exp["top_drivers"]
    assert len(drivers) == 5, f"Expected 5 drivers, got {len(drivers)}"
    
    for d in drivers:
        assert "feature" in d
        assert "phrase" in d
        assert "contribution" in d
        assert "share_pct" in d
        assert d["share_pct"] >= 0.0
        assert d["sign"] in ["+", "-"]
        
    sentence = exp["explanation_sentence"]
    assert isinstance(sentence, str)
    assert len(sentence) > 10
    assert "%" in sentence

def test_every_high_risk_row_produces_explanation():
    """Verify that every row predicted as High risk produces a valid explanation sentence."""
    explainer = StockSenseExplainer()
    test_df = pd.read_parquet("data/processed/features_test.parquet").head(100)
    
    high_risk_count = 0
    for _, row in test_df.iterrows():
        res = explainer.explain_row_from_features(row)
        if res["risk_tier"] == "High":
            high_risk_count += 1
            assert len(res["explanation_sentence"]) > 0
            assert len(res["top_drivers"]) == 5
            
    assert high_risk_count > 0, "No High risk rows found in sample"
