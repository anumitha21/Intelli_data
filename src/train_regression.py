"""
train_regression.py – Trains and evaluates demand forecasting regression models.
Permitted algorithms: Baseline, Linear Regression, Decision Tree, Random Forest, XGBoost.
Metrics: MAE, RMSE, MAPE, R2.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def mape_score(y_true, y_pred):
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def train_regression():
    print("--- Training Demand Regression Models ---")
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)
    
    train_df = pd.read_parquet("data/processed/features_train.parquet")
    val_df   = pd.read_parquet("data/processed/features_val.parquet")
    test_df  = pd.read_parquet("data/processed/features_test.parquet")
    
    feature_cols = [
        'day_of_week', 'weekend_flag', 'month', 'week_no', 'festival_flag',
        'lag_1', 'lag_7', 'lag_14',
        'rolling_mean_7', 'rolling_mean_14', 'rolling_std_7',
        'days_of_inventory', 'inventory_to_demand_ratio', 'reorder_gap', 'incoming_stock',
        'discount_pct', 'price_change', 'promotion_flag',
        'shelf_life', 'lead_time', 'temperature', 'rain', 'holiday', 'has_local_event',
        'cold_start'
    ]
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df['next_7_day_demand'].copy()
    
    X_val = val_df[feature_cols].copy()
    y_val = val_df['next_7_day_demand'].copy()
    
    X_test = test_df[feature_cols].copy()
    y_test = test_df['next_7_day_demand'].copy()
    
    # 1. Baseline Model (7 * rolling_mean_7)
    base_pred_test = X_test['rolling_mean_7'] * 7.0
    
    models = {
        "Baseline (7-Day Mean)": None,
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        "XGBoost Regressor": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        if model is None:
            pred_val = X_val['rolling_mean_7'] * 7.0
            pred_test = base_pred_test
        else:
            print(f"Fitting {name}...")
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            fitted_models[name] = model
            
        mae = float(mean_absolute_error(y_test, pred_test))
        rmse = float(root_mean_squared_error(y_test, pred_test))
        mape = float(mape_score(y_test.values, pred_test))
        r2 = float(r2_score(y_test, pred_test))
        
        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE_pct": round(mape, 2),
            "R2": round(r2, 4)
        }
        print(f"  {name} -> MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")
        
    # Save winning regression model (XGBoost)
    winner_name = "XGBoost Regressor"
    winner_model = fitted_models[winner_name]
    winner_path = "models/regression_xgboost.joblib"
    joblib.dump(winner_model, winner_path)
    joblib.dump(fitted_models["Random Forest"], "models/regression_random_forest.joblib")
    print(f"Saved winning regression model to {winner_path}")
    
    with open("reports/metrics/regression_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results, winner_model, feature_cols

if __name__ == "__main__":
    train_regression()
