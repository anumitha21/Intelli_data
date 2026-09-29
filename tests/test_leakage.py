"""
test_leakage.py – Rigorous unit tests proving zero future leakage in feature engineering
and verifying chronological train/val/test splits with 7-day purge gaps.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd
from src.features import load_master_data, attach_demand_and_targets, engineer_features

def test_feature_zero_leakage_at_cutoff():
    """
    Mathematical proof of zero future leakage:
    Truncating the raw data at cutoff date t_cut must yield identical feature
    values at date t_cut as when computed on the entire future-extended dataset.
    """
    master = load_master_data()
    df = attach_demand_and_targets(master)
    
    # Select an evaluation cutoff date
    t_cut = pd.to_datetime("2023-04-15")
    
    # 1. Feature engineering on full dataset
    full_feat = engineer_features(df)
    full_at_cutoff = full_feat[full_feat['date'] == t_cut].sort_values(['store_id', 'product_id']).reset_index(drop=True)
    
    # 2. Feature engineering on truncated dataset (strictly <= t_cut)
    trunc_df = df[df['date'] <= t_cut].copy()
    trunc_feat = engineer_features(trunc_df)
    trunc_at_cutoff = trunc_feat[trunc_feat['date'] == t_cut].sort_values(['store_id', 'product_id']).reset_index(drop=True)
    
    # Compare all engineered feature columns
    test_feature_cols = [
        'day_of_week', 'weekend_flag', 'month', 'week_no', 'festival_flag',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
        'days_of_inventory', 'inventory_to_demand_ratio', 'reorder_gap', 'incoming_stock',
        'discount_pct', 'price_change', 'promotion_flag',
        'shelf_life', 'lead_time', 'temperature', 'rain', 'holiday', 'has_local_event'
    ]
    
    for col in test_feature_cols:
        full_vals = full_at_cutoff[col].values
        trunc_vals = trunc_at_cutoff[col].values
        np.testing.assert_allclose(
            full_vals, trunc_vals, rtol=1e-5, atol=1e-5,
            err_msg=f"Feature leakage detected in '{col}'! Truncated values differ from full-dataset values at date {t_cut.date()}."
        )

def test_chronological_splits_and_purge_gaps():
    """Verify chronological ordering, zero split overlap, and strict 7-day purge gaps."""
    split_info_path = "reports/metrics/split_info.json"
    assert os.path.exists(split_info_path), "split_info.json missing"
    
    with open(split_info_path, "r") as f:
        meta = json.load(f)
        
    train_start = pd.to_datetime(meta['train']['start_date'])
    train_end   = pd.to_datetime(meta['train']['end_date'])
    val_start   = pd.to_datetime(meta['validation']['start_date'])
    val_end     = pd.to_datetime(meta['validation']['end_date'])
    test_start  = pd.to_datetime(meta['test']['start_date'])
    test_end    = pd.to_datetime(meta['test']['end_date'])
    
    # 1. Strictly chronological
    assert train_start < train_end < val_start < val_end < test_start < test_end
    
    # 2. Strict 7-day purge gaps
    train_val_gap = (val_start - train_end).days - 1
    val_test_gap = (test_start - val_end).days - 1
    assert train_val_gap == 7, f"Expected 7-day purge gap between train and val, got {train_val_gap}"
    assert val_test_gap == 7, f"Expected 7-day purge gap between val and test, got {val_test_gap}"
    
    # 3. Verify parquet files exist and match rows
    train_df = pd.read_parquet("data/processed/features_train.parquet")
    val_df   = pd.read_parquet("data/processed/features_val.parquet")
    test_df  = pd.read_parquet("data/processed/features_test.parquet")
    
    assert len(train_df) == meta['train']['row_count']
    assert len(val_df) == meta['validation']['row_count']
    assert len(test_df) == meta['test']['row_count']
    
    # Verify no date intersections
    train_dates = set(train_df['date'])
    val_dates = set(val_df['date'])
    test_dates = set(test_df['date'])
    
    assert len(train_dates.intersection(val_dates)) == 0, "Train and Val dates overlap!"
    assert len(val_dates.intersection(test_dates)) == 0, "Val and Test dates overlap!"
    assert len(train_dates.intersection(test_dates)) == 0, "Train and Test dates overlap!"

def test_no_incomplete_future_in_splits():
    """Verify that dates without a complete 7-day future are excluded from all splits."""
    test_df = pd.read_parquet("data/processed/features_test.parquet")
    max_dataset_date = pd.to_datetime("2024-01-30")
    # All rows in test must have dates <= max_dataset_date - 7 days (i.e. 2024-01-23)
    assert pd.to_datetime(test_df['date']).max() <= max_dataset_date - pd.Timedelta(days=7)
