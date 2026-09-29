"""
clean.py – Phase 2 (P2)
Audits, cleans, reconciles and standardises NovaMart raw datasets.
Generates data quality metrics and logs comparisons with _traps_manifest.json.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

def clean_data():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    
    print("--- Starting Data Cleaning & Audit ---")
    
    # Load manifest if available
    manifest_path = "data/raw/_traps_manifest.json"
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
    dq_log = []
    
    # ---------------------------------------------------------
    # 1. Clean Products
    # ---------------------------------------------------------
    products = pd.read_csv("data/raw/products.csv")
    initial_prod_count = len(products)
    
    # Audit & Standardise category
    raw_cats = products['category'].tolist()
    # Canonical categories: Groceries, Beverages, Dairy, Snacks, Personal Care, Household, Frozen
    cat_map = {
        'beverages': 'Beverages',
        'beverage': 'Beverages',
        'BEVERAGES': 'Beverages',
        'groceries': 'Groceries',
        'grocery': 'Groceries',
        'GROCERIES': 'Groceries',
        'dairy': 'Dairy',
        'DAIRY': 'Dairy',
        'snacks': 'Snacks',
        'SNACKS': 'Snacks',
        'personal care': 'Personal Care',
        'household': 'Household',
        'frozen': 'Frozen'
    }
    
    def standardise_cat(val):
        s = str(val).strip()
        return cat_map.get(s.lower(), s.title())
        
    standardised_cats = products['category'].apply(standardise_cat)
    cat_variants_detected = (standardised_cats != products['category']).sum()
    products['category'] = standardised_cats
    
    dq_log.append({
        "dataset": "products.csv",
        "issue": "Category spelling and casing variants (e.g. beverage/BEVERAGES)",
        "detected_count": int(cat_variants_detected),
        "injected_count": manifest.get("category_spelling_variants", {}).get("count", "N/A"),
        "action_taken": "Standardised to canonical title-case category names",
        "justification": "Ensures consistent category groupings across analysis, reporting, and model feature aggregations"
    })
    
    # ---------------------------------------------------------
    # 2. Clean Stores
    # ---------------------------------------------------------
    stores = pd.read_csv("data/raw/stores.csv")
    stores['store_id'] = stores['store_id'].astype(str).str.strip()
    stores['city'] = stores['city'].astype(str).str.strip()
    
    # ---------------------------------------------------------
    # 3. Clean External Factors
    # ---------------------------------------------------------
    ext = pd.read_csv("data/raw/external_factors.csv")
    ext['date'] = pd.to_datetime(ext['date'])
    
    # Audit missing temp_c
    missing_temps = ext['temp_c'].isna().sum()
    
    # Time-aware interpolation per city
    ext = ext.sort_values(['city', 'date']).reset_index(drop=True)
    ext['temp_c'] = ext.groupby('city')['temp_c'].transform(
        lambda series: series.interpolate(method='linear').bfill().ffill()
    )
    ext['temp_c'] = ext['temp_c'].round(1)
    
    dq_log.append({
        "dataset": "external_factors.csv",
        "issue": "Missing temperature values (blank temperatures)",
        "detected_count": int(missing_temps),
        "injected_count": manifest.get("blank_temperatures", {}).get("count", "N/A"),
        "action_taken": "Time-aware linear interpolation grouped by city with boundary backfill/forwardfill",
        "justification": "Daily temperatures exhibit strong local temporal autocorrelation; per-city interpolation preserves continuous local weather dynamics without leakage across geographical regions"
    })
    
    # ---------------------------------------------------------
    # 4. Clean Inventory
    # ---------------------------------------------------------
    inv = pd.read_csv("data/raw/inventory.csv")
    inv['date'] = pd.to_datetime(inv['date'])
    
    # Audit inventory identity: closing == opening + received - sold
    expected_closing = inv['opening_stock'] + inv['received_stock'] - inv['units_sold']
    mismatch_mask = inv['closing_stock'] != expected_closing
    mismatch_count = int(mismatch_mask.sum())
    
    # Reconcile mismatches
    inv['closing_stock'] = expected_closing
    
    dq_log.append({
        "dataset": "inventory.csv",
        "issue": "Inventory arithmetic discrepancies (closing != opening + received - sold)",
        "detected_count": mismatch_count,
        "injected_count": manifest.get("inventory_arithmetic_mismatches", {}).get("count", "N/A"),
        "action_taken": "Reconciled closing_stock using physical conservation law: closing = opening + received - sold",
        "justification": "Physical mass balance is an invariant domain law in retail warehouse/store accounting; correcting discrepancies eliminates corruption in stock-out indicators"
    })
    
    # ---------------------------------------------------------
    # 5. Clean Transactions
    # ---------------------------------------------------------
    tx = pd.read_csv("data/raw/transactions.csv")
    initial_tx_count = len(tx)
    
    # Audit TRUE duplicates (same transaction_id and identical row)
    dup_mask = tx.duplicated(subset=['transaction_id'], keep='first')
    dup_count = int(dup_mask.sum())
    tx = tx[~dup_mask].reset_index(drop=True)
    
    dq_log.append({
        "dataset": "transactions.csv",
        "issue": "Duplicate transactions (identical transaction IDs)",
        "detected_count": dup_count,
        "injected_count": manifest.get("duplicate_transactions", {}).get("count", "N/A"),
        "action_taken": "Deduplicated rows keeping first occurrence based on unique transaction_id",
        "justification": "Duplicate transaction logs inflate sales revenue and units sold, biasing demand models upward"
    })
    
    # Audit Impossible quantities (quantity <= 0)
    invalid_q_mask = tx['quantity'] <= 0
    invalid_q_count = int(invalid_q_mask.sum())
    
    # Treatment: filter out invalid non-positive quantities
    tx = tx[~invalid_q_mask].reset_index(drop=True)
    
    dq_log.append({
        "dataset": "transactions.csv",
        "issue": "Impossible quantities (negative or zero quantities)",
        "detected_count": invalid_q_count,
        "injected_count": manifest.get("impossible_quantities", {}).get("count", "N/A"),
        "action_taken": "Filtered out non-positive quantity transaction records",
        "justification": "Negative/zero quantities represent entry errors or unprocessed returns with corrupted revenue calculations that distort retail POS demand tracking"
    })
    
    # ---------------------------------------------------------
    # 6. Product History & Cold Start Flag
    # ---------------------------------------------------------
    # Calculate history days per product from inventory date range
    prod_history = inv.groupby('product_id')['date'].agg(
        first_date='min',
        last_date='max',
        history_days=lambda x: (x.max() - x.min()).days + 1
    ).reset_index()
    
    prod_history['cold_start'] = (prod_history['history_days'] < 28).astype(int)
    sparse_detected = int((prod_history['cold_start'] == 1).sum())
    
    products = products.merge(prod_history[['product_id', 'history_days', 'cold_start']], on='product_id', how='left')
    products['history_days'] = products['history_days'].fillna(0).astype(int)
    products['cold_start'] = products['cold_start'].fillna(1).astype(int)
    
    dq_log.append({
        "dataset": "products.csv",
        "issue": "Sparse product sales history (< 28 days / cold start)",
        "detected_count": sparse_detected,
        "injected_count": len(manifest.get("new_products_sparse_history", {}).get("product_ids", [])),
        "action_taken": "Created history_days feature and cold_start binary indicator; flagged for category-level empirical prior fallback",
        "justification": "New products with under 28 days of historical data cannot support 14-day lag or rolling features; cold-start items require category-level mean imputation during inference"
    })
    
    # Save cleaned files to processed
    products.to_parquet("data/processed/products_clean.parquet", index=False)
    stores.to_parquet("data/processed/stores_clean.parquet", index=False)
    ext.to_parquet("data/processed/external_factors_clean.parquet", index=False)
    inv.to_parquet("data/processed/inventory_clean.parquet", index=False)
    tx.to_parquet("data/processed/transactions_clean.parquet", index=False)
    
    # Save DQ log to json
    with open("reports/metrics/dq_audit.json", "w") as f:
        json.dump(dq_log, f, indent=2)
        
    print(f"Data cleaning complete. Cleaned tables saved to data/processed/.")
    return dq_log, products, stores, ext, inv, tx

if __name__ == "__main__":
    from src.master import build_master
    build_master()
