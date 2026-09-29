"""
service.py – Business logic and model orchestration service for FastAPI endpoints.
Loads models and data once at module initialization.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend import load_models_from_registry, get_recommendation_card, what_if_scenario, score_inventory_recommendations
from src.explain import explain_row

class StockSenseService:
    def __init__(self):
        self.registry_path = "models/model_registry.json"
        self.reload_models()
        self.load_metadata()
        
    def reload_models(self):
        """Loads models dynamically from registry."""
        bundle = load_models_from_registry(self.registry_path)
        self.reg_model = bundle["regression_model"]
        self.clf_model = bundle["classification_model"]
        self.reg_features = bundle["reg_features"]
        self.clf_features = bundle["clf_features"]
        self.high_risk_th = bundle["high_risk_threshold"]
        self.med_risk_th = bundle["med_risk_threshold"]
        self.registry = bundle["registry"]
        
    def load_metadata(self):
        self.stores_df = pd.read_csv("data/raw/stores.csv")
        self.products_df = pd.read_csv("data/raw/products.csv")
        self.test_features = pd.read_parquet("data/processed/features_test.parquet")
        self.test_features['date_str'] = pd.to_datetime(self.test_features['date']).dt.strftime('%Y-%m-%d')
        
        # Load precomputed recommendation table
        rec_path = "data/processed/recommendations.csv"
        if not os.path.exists(rec_path):
            _, self.recommendations_df = score_inventory_recommendations()
        else:
            self.recommendations_df = pd.read_csv(rec_path)
            
    def get_health(self) -> dict:
        self.reload_models() # check current registry
        return {
            "status": "healthy",
            "regression_model": self.registry["regression"]["winner_model_name"],
            "classification_model": self.registry["classification"]["winner_model_name"],
            "decision_threshold": self.registry["classification"]["decision_threshold"],
            "high_risk_threshold": self.high_risk_th,
            "medium_risk_threshold": self.med_risk_th
        }

    def predict(self, store_id: str, product_id: str, as_of_date: str) -> dict:
        self.validate_store_product(store_id, product_id)
        date_clean = pd.to_datetime(as_of_date).strftime('%Y-%m-%d')
        
        sub = self.test_features[
            (self.test_features['store_id'] == store_id) &
            (self.test_features['product_id'] == product_id) &
            (self.test_features['date_str'] == date_clean)
        ]
        if len(sub) == 0:
            # Fallback to closest matching historical row
            sub = self.test_features[
                (self.test_features['store_id'] == store_id) &
                (self.test_features['product_id'] == product_id)
            ]
            if len(sub) == 0:
                raise ValueError(f"No records found for Store {store_id} and Product {product_id}")
                
        row_feat = sub.iloc[[0]][self.clf_features]
        forecast = float(np.maximum(0.0, np.round(self.reg_model.predict(row_feat)[0], 1)))
        prob = float(np.round(self.clf_model.predict_proba(row_feat)[0, 1], 4))
        risk = "High" if prob >= self.high_risk_th else ("Medium" if prob >= self.med_risk_th else "Low")
        
        return {
            "store_id": store_id,
            "product_id": product_id,
            "as_of_date": date_clean,
            "forecast_demand_7d": forecast,
            "stockout_probability": prob,
            "risk_tier": risk
        }

    def validate_store_product(self, store_id: str, product_id: str):
        valid_stores = set(self.stores_df['store_id'])
        valid_prods = set(self.products_df['product_id'])
        if store_id not in valid_stores:
            raise KeyError(f"Invalid store_id '{store_id}'. Valid stores: {sorted(list(valid_stores))}")
        if product_id not in valid_prods:
            raise KeyError(f"Invalid product_id '{product_id}'. Valid products: {sorted(list(valid_prods))}")

    def get_recommendation(self, store_id: str, product_id: str, as_of_date: str = None) -> dict:
        self.validate_store_product(store_id, product_id)
        return get_recommendation_card(store_id, product_id, as_of_date)

    def filter_recommendations(self, store_id: str = None, category: str = None, risk: str = None) -> list:
        df = self.recommendations_df.copy()
        if store_id:
            df = df[df['store_id'] == store_id]
        if category:
            df = df[df['category'].str.lower() == category.lower()]
        if risk:
            df = df[df['risk_tier'].str.lower() == risk.lower()]
        return df.to_dict(orient="records")

    def explain(self, store_id: str, product_id: str, date: str) -> dict:
        self.validate_store_product(store_id, product_id)
        return explain_row(store_id, product_id, date)

    def what_if(self, store_id: str, product_id: str, as_of_date: str,
                discount_delta: float, extra_lead_days: int, festival_uplift_pct: float) -> dict:
        self.validate_store_product(store_id, product_id)
        return what_if_scenario(store_id, product_id, as_of_date, discount_delta, extra_lead_days, festival_uplift_pct)

# Singleton service instance
service = StockSenseService()
