"""
test_master.py – Tests for Phase 2 master table integrity and uniqueness.
"""

import os
import pytest
import pandas as pd

@pytest.fixture(scope="module")
def master_data():
    csv_path = "data/processed/master.csv"
    parquet_path = "data/processed/master.parquet"
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        pytest.skip("Master dataset not found. Run make clean first.")

def test_master_exists(master_data):
    """Verify master table exists and has rows."""
    assert len(master_data) > 0, "Master table is empty"
    assert len(master_data) == 76050, f"Expected 76,050 rows, got {len(master_data)}"

def test_master_primary_key_uniqueness(master_data):
    """Verify composite primary key (date, store_id, product_id) has zero duplicates."""
    dups = master_data.duplicated(subset=['date', 'store_id', 'product_id']).sum()
    assert dups == 0, f"Found {dups} duplicate primary keys in master table"

def test_inventory_identity_conservation(master_data):
    """Verify that after reconciliation, closing == opening + received - sold holds for 100% of rows."""
    expected_closing = master_data['opening_stock'] + master_data['received_stock'] - master_data['units_sold']
    mismatches = (master_data['closing_stock'] != expected_closing).sum()
    assert mismatches == 0, f"Found {mismatches} inventory arithmetic mismatches in cleaned master"

def test_data_quality_report_exists():
    """Verify that reports/data_quality_report.md exists and is non-empty."""
    report_path = "reports/data_quality_report.md"
    assert os.path.exists(report_path), "Data Quality Report does not exist"
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert len(content) > 500, "Data Quality Report is suspiciously short"
    assert "_traps_manifest.json" in content or "Manifest vs Detected" in content

def test_all_eda_figures_exist():
    """Verify all 6 EDA figures exist in reports/figures/."""
    expected_figures = [
        "reports/figures/eda_1_category_pareto.png",
        "reports/figures/eda_2_store_weekly_trend.png",
        "reports/figures/eda_3_promotion_impact.png",
        "reports/figures/eda_4_weekend_demand.png",
        "reports/figures/eda_5_demand_volatility.png",
        "reports/figures/eda_6_stockout_heatmap.png",
    ]
    for fig in expected_figures:
        assert os.path.exists(fig), f"Missing EDA figure: {fig}"

def test_stats_tests_json_valid():
    """Verify reports/metrics/stats_tests.json exists with all 3 hypothesis tests."""
    metrics_path = "reports/metrics/stats_tests.json"
    assert os.path.exists(metrics_path), "stats_tests.json does not exist"
    import json
    with open(metrics_path, "r") as f:
        data = json.load(f)
    assert "kpis" in data
    assert "statistical_tests" in data
    assert len(data["statistical_tests"]) == 3
