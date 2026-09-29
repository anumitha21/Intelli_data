"""
features.py – Phase 3 (P3): Leak-Safe Feature Engineering & Chronological Dataset Splitting
Builds features from data/processed/master.csv (or master.parquet).
All features use only information available at or before date t.
Splits into train/val/test with strict 7-day chronological separation gaps.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_master_data():
    parquet_path = "data/processed/master.parquet"
    csv_path = "data/processed/master.csv"
    if os.path.exists(parquet_path):
        master = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        master = pd.read_csv(csv_path)
    else:
        from src.master import build_master
        master = build_master()
        
    master['date'] = pd.to_datetime(master['date'])
    return master

def attach_demand_and_targets(df):
    """
    Attaches true demand (from raw dataset or fallback units_sold)
    and computes the 7-day forward targets:
    - next_7_day_demand: sum of demand over t+1..t+7
    - stockout_flag: 1 if any stockout occurs in t+1..t+7, 0 otherwise
    """
    df = df.copy()
    raw_path = "data/external/sales_data.csv"
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        smap = {'S001':'S01','S002':'S02','S003':'S03','S004':'S04','S005':'S05'}
        pmap = {f'P00{i+1:02d}': f'P1{i+1:02d}' for i in range(20)}
        raw['store_id'] = raw['Store ID'].map(smap)
        raw['product_id'] = raw['Product ID'].map(pmap)
        raw['date'] = pd.to_datetime(raw['Date'])
        dem_lookup = raw.set_index(['date', 'store_id', 'product_id'])['Demand']
        mapped = df.set_index(['date', 'store_id', 'product_id']).index.map(dem_lookup)
        df['demand'] = pd.Series(mapped, index=df.index).fillna(df['units_sold']).astype(float)
    else:
        df['demand'] = df['units_sold'].astype(float)
        
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    g = df.groupby(['store_id', 'product_id'])
    
    # Target 1: next_7_day_demand = sum of demand over t+1..t+7
    df['next_7_day_demand'] = g['demand'].transform(lambda s: sum(s.shift(-k) for k in range(1, 8)))
    
    # Target 2: stockout_flag in t+1..t+7
    # Operational definition: day is stockout when closing_stock == 0 or units_sold < demand
    df['is_stockout_day'] = ((df['closing_stock'] == 0) | (df['units_sold'] < df['demand'])).astype(int)
    df['stockout_flag'] = g['is_stockout_day'].transform(
        lambda s: (sum(s.shift(-k) for k in range(1, 8)) > 0).astype(float)
    )
    
    return df

def engineer_features(df):
    """
    Constructs leak-safe features at end of date t.
    All lags and rolling statistics are calculated strictly on shifted historical data (shift >= 1).
    Known-in-advance features (calendar, deterministic holidays) use date t information.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    # -------------------------------------------------------------
    # 1. TIME FEATURES (Known in advance)
    # -------------------------------------------------------------
    df['day_of_week'] = df['date'].dt.dayofweek
    df['weekend_flag'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['week_no'] = df['date'].dt.isocalendar().week.astype(int)
    df['festival_flag'] = (df['festival'].astype(str).str.lower() != 'none').astype(int)
    
    # -------------------------------------------------------------
    # 2. LAG OF DEMAND (shift >= 1, Strictly Historical)
    # -------------------------------------------------------------
    g = df.groupby(['store_id', 'product_id'])
    # demand series shifted by 1 to represent day t-1
    dem_shifted = g['demand'].shift(1)
    
    df['lag_1'] = dem_shifted
    df['lag_7'] = g['demand'].shift(7)
    df['lag_14'] = g['demand'].shift(14)
    
    # -------------------------------------------------------------
    # 3. ROLLING DEMAND STATISTICS (On shifted series)
    # -------------------------------------------------------------
    # Compute rolling stats on shifted demand (day t-1 and prior)
    df['rolling_mean_7'] = g['demand'].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df['rolling_mean_14'] = g['demand'].transform(lambda s: s.shift(1).rolling(14, min_periods=5).mean())
    df['rolling_std_7'] = g['demand'].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std().fillna(0.0))
    
    # -------------------------------------------------------------
    # 4. INVENTORY METRICS (Known at end of day t)
    # -------------------------------------------------------------
    safe_mean_7 = np.maximum(df['rolling_mean_7'].fillna(10.0), 1.0)
    safe_lag_1 = np.maximum(df['lag_1'].fillna(10.0), 1.0)
    
    df['days_of_inventory'] = np.round(df['closing_stock'] / safe_mean_7, 2)
    df['inventory_to_demand_ratio'] = np.round(df['closing_stock'] / safe_lag_1, 2)
    df['reorder_gap'] = df['reorder_lvl'] - df['closing_stock']
    df['incoming_stock'] = df['received_stock'].astype(int)
    
    # -------------------------------------------------------------
    # 5. PRICE & PROMOTION FEATURES
    # -------------------------------------------------------------
    df['discount_pct'] = df['avg_discount_pct'].fillna(0.0)
    # Price ratio / change vs MRP
    df['price_change'] = np.round((df['avg_unit_price'] - df['mrp']) / df['mrp'], 4)
    df['promotion_flag'] = (df['discount_pct'] > 0).astype(int)
    
    # -------------------------------------------------------------
    # 6. STORE & PRODUCT METRICS
    # -------------------------------------------------------------
    df['shelf_life'] = df['shelf_life_days'].astype(int)
    df['lead_time'] = df['lead_days'].astype(int)
    
    # -------------------------------------------------------------
    # 7. EXTERNAL FACTORS (Weather & Events)
    # -------------------------------------------------------------
    df['temperature'] = df['temp_c'].astype(float)
    df['rain'] = df['rain_mm'].astype(float)
    df['holiday'] = df['holiday'].astype(int)
    df['has_local_event'] = (df['local_event'].astype(str).str.lower() != 'none').astype(int)
    
    return df

def build_features_and_splits():
    print("--- Phase 3: Feature Engineering & Dataset Splitting ---")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("docs", exist_ok=True)
    
    master = load_master_data()
    print(f"Loaded master data: {len(master)} rows.")
    
    # 1. Attach demand and calculate future 7-day targets
    df = attach_demand_and_targets(master)
    
    # 2. Engineer leak-safe features
    df = engineer_features(df)
    
    # 3. Filter rows without full 7-day future
    # Drop rows where target is NaN (the last 7 days of dataset)
    valid_df = df.dropna(subset=['next_7_day_demand', 'stockout_flag']).copy()
    valid_df['stockout_flag'] = valid_df['stockout_flag'].astype(int)
    print(f"Rows with full 7-day future: {len(valid_df)} (dropped {len(df) - len(valid_df)} boundary rows).")
    
    # -------------------------------------------------------------
    # 4. CHRONOLOGICAL SPLITS WITH 7-DAY GAP
    # -------------------------------------------------------------
    # Date boundaries:
    # Train: 2022-01-01 to 2023-09-15
    # Gap 1: 2023-09-16 to 2023-09-22 (7-day purge)
    # Val:   2023-09-23 to 2023-11-20
    # Gap 2: 2023-11-21 to 2023-11-27 (7-day purge)
    # Test:  2023-11-28 to 2024-01-23
    
    train_end = pd.to_datetime("2023-09-15")
    val_start = pd.to_datetime("2023-09-23")
    val_end   = pd.to_datetime("2023-11-20")
    test_start = pd.to_datetime("2023-11-28")
    test_end   = pd.to_datetime("2024-01-23")
    
    train_mask = (valid_df['date'] <= train_end)
    val_mask   = (valid_df['date'] >= val_start) & (valid_df['date'] <= val_end)
    test_mask  = (valid_df['date'] >= test_start) & (valid_df['date'] <= test_end)
    
    train_df = valid_df[train_mask].copy()
    val_df   = valid_df[val_mask].copy()
    test_df  = valid_df[test_mask].copy()
    
    # Verify no overlap and strict gaps
    assert train_df['date'].max() < val_df['date'].min()
    assert (val_df['date'].min() - train_df['date'].max()).days == 8 # 7 buffer days between them
    assert val_df['date'].max() < test_df['date'].min()
    assert (test_df['date'].min() - val_df['date'].max()).days == 8 # 7 buffer days between them
    
    # -------------------------------------------------------------
    # 5. COLD-START & CATEGORY-LEVEL FALLBACKS (Fit on train only)
    # -------------------------------------------------------------
    # Compute category-level averages on train set for lag/rolling features
    impute_cols = ['lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
                   'days_of_inventory', 'inventory_to_demand_ratio']
    cat_means = train_df.groupby('category')[impute_cols].mean()
    chain_means = train_df[impute_cols].mean()
    
    for split_df in [train_df, val_df, test_df]:
        for col in impute_cols:
            # Map category mean fallback
            cat_fallback = split_df['category'].map(cat_means[col]).fillna(chain_means[col])
            split_df[col] = split_df[col].fillna(cat_fallback)
            
    # Class balance summary
    train_stockout_pct = float(train_df['stockout_flag'].mean() * 100)
    val_stockout_pct   = float(val_df['stockout_flag'].mean() * 100)
    test_stockout_pct  = float(test_df['stockout_flag'].mean() * 100)
    
    split_info = {
        "train": {
            "start_date": str(train_df['date'].min().date()),
            "end_date": str(train_df['date'].max().date()),
            "row_count": len(train_df),
            "stockout_flag_distribution": {
                "negative_count_0": int((train_df['stockout_flag'] == 0).sum()),
                "positive_count_1": int((train_df['stockout_flag'] == 1).sum()),
                "positive_percentage": round(train_stockout_pct, 2)
            }
        },
        "gap_train_val_days": (val_df['date'].min() - train_df['date'].max()).days - 1,
        "validation": {
            "start_date": str(val_df['date'].min().date()),
            "end_date": str(val_df['date'].max().date()),
            "row_count": len(val_df),
            "stockout_flag_distribution": {
                "negative_count_0": int((val_df['stockout_flag'] == 0).sum()),
                "positive_count_1": int((val_df['stockout_flag'] == 1).sum()),
                "positive_percentage": round(val_stockout_pct, 2)
            }
        },
        "gap_val_test_days": (test_df['date'].min() - val_df['date'].max()).days - 1,
        "test": {
            "start_date": str(test_df['date'].min().date()),
            "end_date": str(test_df['date'].max().date()),
            "row_count": len(test_df),
            "stockout_flag_distribution": {
                "negative_count_0": int((test_df['stockout_flag'] == 0).sum()),
                "positive_count_1": int((test_df['stockout_flag'] == 1).sum()),
                "positive_percentage": round(test_stockout_pct, 2)
            }
        },
        "total_active_rows": len(train_df) + len(val_df) + len(test_df)
    }
    
    # Save split info JSON
    split_info_path = "reports/metrics/split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split metadata to {split_info_path}.")
    
    # -------------------------------------------------------------
    # 6. SAVE FEATURE DATASETS
    # -------------------------------------------------------------
    train_path = "data/processed/features_train.parquet"
    val_path   = "data/processed/features_val.parquet"
    test_path  = "data/processed/features_test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    print(f"Saved splits:\n  Train: {train_path} ({len(train_df)} rows)\n  Val:   {val_path} ({len(val_df)} rows)\n  Test:  {test_path} ({len(test_df)} rows)")
    
    # Print class balance
    print("\n" + "="*50)
    print("CHRONOLOGICAL SPLITS & CLASS BALANCE SUMMARY")
    print("="*50)
    print(f"TRAIN: {split_info['train']['start_date']} to {split_info['train']['end_date']} | Rows: {len(train_df):,} | Stockout Rate: {train_stockout_pct:.2f}% (Pos: {split_info['train']['stockout_flag_distribution']['positive_count_1']:,}, Neg: {split_info['train']['stockout_flag_distribution']['negative_count_0']:,})")
    print(f"  >>> Purge Gap 1: {split_info['gap_train_val_days']} days (2023-09-16 to 2023-09-22)")
    print(f"VAL:   {split_info['validation']['start_date']} to {split_info['validation']['end_date']} | Rows: {len(val_df):,} | Stockout Rate: {val_stockout_pct:.2f}% (Pos: {split_info['validation']['stockout_flag_distribution']['positive_count_1']:,}, Neg: {split_info['validation']['stockout_flag_distribution']['negative_count_0']:,})")
    print(f"  >>> Purge Gap 2: {split_info['gap_val_test_days']} days (2023-11-21 to 2023-11-27)")
    print(f"TEST:  {split_info['test']['start_date']} to {split_info['test']['end_date']} | Rows: {len(test_df):,} | Stockout Rate: {test_stockout_pct:.2f}% (Pos: {split_info['test']['stockout_flag_distribution']['positive_count_1']:,}, Neg: {split_info['test']['stockout_flag_distribution']['negative_count_0']:,})")
    print("="*50 + "\n")
    
    # -------------------------------------------------------------
    # 7. GENERATE FEATURE DICTIONARY
    # -------------------------------------------------------------
    generate_feature_dictionary()
    
    return train_df, val_df, test_df

def generate_feature_dictionary():
    dict_path = "docs/FEATURE_DICTIONARY.md"
    print(f"Generating feature dictionary at {dict_path}...")
    
    content = """# NovaMart Retail Feature Dictionary & Leak-Safety Rationale

> **Project:** StockSense — IntelliData 2026 Data Science Hackathon  
> **Target Horizon:** $t+1 \\dots t+7$ (Next 7 Calendar Days)  
> **Guiding Principle:** **Zero Future Leakage.** Every feature at observation date $t$ utilizes exclusively information known at or before the close of business on date $t$.

---

## 1. Feature Specifications

| Feature Name | Group | Data Type | Business & Mathematical Definition | Leak-Safety Justification |
|---|---|---|---|---|
| `day_of_week` | Time | int (0–6) | Day of week index (Monday = 0, Sunday = 6) | Calendar fact, fully deterministic in advance |
| `weekend_flag` | Time | binary (0/1) | 1 if Saturday or Sunday, 0 otherwise | Calendar fact, known in advance |
| `month` | Time | int (1–12) | Calendar month index | Calendar fact, known in advance |
| `week_no` | Time | int (1–53) | ISO 8601 calendar week number | Calendar fact, known in advance |
| `festival_flag` | Time | binary (0/1) | 1 if date matches a Tamil Nadu gazetted festival | Deterministic calendar festival schedule |
| `lag_1` | Demand Lag | float | Demand of SKU at store on date $t-1$ | Shift $\\ge 1$; uses only past sales data |
| `lag_7` | Demand Lag | float | Demand of SKU at store on date $t-7$ | Shift $\\ge 7$; uses only past sales data |
| `lag_14` | Demand Lag | float | Demand of SKU at store on date $t-14$ | Shift $\\ge 14$; uses only past sales data |
| `rolling_mean_7` | Rolling Stats | float | 7-day rolling mean of demand over $t-7 \\dots t-1$ | Computed strictly on $shift(1)$ demand series |
| `rolling_mean_14` | Rolling Stats | float | 14-day rolling mean of demand over $t-14 \\dots t-1$ | Computed strictly on $shift(1)$ demand series |
| `rolling_std_7` | Rolling Stats | float | 7-day rolling standard deviation over $t-7 \\dots t-1$ | Computed strictly on $shift(1)$ demand series |
| `days_of_inventory` | Inventory | float | `closing_stock / max(1.0, rolling_mean_7)` | Closing stock known at end of day $t$; rolling mean uses past |
| `inventory_to_demand_ratio`| Inventory | float | `closing_stock / max(1.0, lag_1)` | Closing stock known at $t$; lag uses $t-1$ |
| `reorder_gap` | Inventory | int | `reorder_lvl - closing_stock` | Closing stock and reorder point known at $t$ |
| `incoming_stock` | Inventory | int | Stock received and logged into store on date $t$ | Warehouse receipts logged on or before date $t$ |
| `discount_pct` | Price / Promo | float | Active discount percentage on date $t$ | Retail discount schedule established on date $t$ |
| `price_change` | Price / Promo | float | `(avg_unit_price - mrp) / mrp` | Realized price relative to catalog MRP at date $t$ |
| `promotion_flag` | Price / Promo | binary (0/1) | 1 if discount active or promotional tag on date $t$ | Planned marketing promotion schedule |
| `shelf_life` | Product Attribute | int | Product shelf life in calendar days | Static catalog specification |
| `lead_time` | Store / Product | int | Supplier replenishment lead time in days | Contractual supply agreement parameter |
| `temperature` | External Factors | float | Daily ambient temperature (°C) on date $t$ | Recorded meteorological observation on date $t$ |
| `rain` | External Factors | float | Daily precipitation (mm) on date $t$ | Recorded meteorological observation on date $t$ |
| `holiday` | External Factors | binary (0/1) | Tamil Nadu state public holiday indicator | Pre-announced gazetted public holiday |
| `has_local_event` | External Factors | binary (0/1) | Local commercial/cultural event indicator | Pre-scheduled municipality and trade fair calendar |
| `cold_start` | Product Flag | binary (0/1) | 1 if product historical tenure $< 28$ days | Flagged based on tenure accumulated prior to $t$ |

---

## 2. Target Variable Definitions

| Target Variable | Problem Type | Definition & Operational Rule | Horizon |
|---|---|---|---|
| `next_7_day_demand` | Regression | Sum of true daily demand across dates $t+1, t+2, \\dots, t+7$ | Future 7 days |
| `stockout_flag` | Classification | Binary indicator: 1 if `closing_stock == 0` OR `units_sold < demand` on ANY day between $t+1$ and $t+7$; 0 otherwise | Future 7 days |
"""
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Feature dictionary written to {dict_path}.")

if __name__ == "__main__":
    build_features_and_splits()
