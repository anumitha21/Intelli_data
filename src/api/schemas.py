"""
schemas.py – Pydantic Data Contracts for the StockSense FastAPI Service.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = "healthy"
    regression_model: str
    classification_model: str
    decision_threshold: float
    high_risk_threshold: float
    medium_risk_threshold: float

class StoreItem(BaseModel):
    store_id: str
    store_name: str
    city: str
    store_type: str
    region: str

class ProductItem(BaseModel):
    product_id: str
    product_name: str
    category: str
    brand: str
    mrp: float
    cost_price: float
    shelf_life_days: int

class PredictRequest(BaseModel):
    store_id: str = Field(..., example="S01")
    product_id: str = Field(..., example="P101")
    as_of_date: Optional[str] = Field("2023-12-01", example="2023-12-01")

class PredictResponse(BaseModel):
    store_id: str
    product_id: str
    as_of_date: str
    forecast_demand_7d: float
    stockout_probability: float
    risk_tier: str

class DriverItem(BaseModel):
    feature: str
    phrase: str
    contribution: float
    share_pct: float
    sign: str

class RecommendRequest(BaseModel):
    store_id: str = Field(..., example="S01")
    product_id: str = Field(..., example="P101")
    as_of_date: Optional[str] = Field(None, example="2023-12-01")

class RecommendCardResponse(BaseModel):
    store_id: str
    store_name: str
    product_id: str
    product_name: str
    category: str
    as_of_date: str
    predicted_7d_demand: float
    current_stock: int
    incoming_stock: int
    safety_stock: int
    stockout_probability: float
    risk_tier: str
    recommended_reorder_qty: int
    revenue_at_risk_inr: float
    why_drivers: str
    top_drivers: Optional[List[Dict[str, Any]]] = None
    manager_action: str
    formatted_text: str

class ExplainRequest(BaseModel):
    store_id: str = Field(..., example="S01")
    product_id: str = Field(..., example="P101")
    date: str = Field(..., example="2023-12-01")

class ExplainResponse(BaseModel):
    store_id: str
    product_id: str
    date: str
    stockout_probability: float
    risk_tier: str
    top_drivers: List[DriverItem]
    explanation_sentence: str

class WhatIfRequest(BaseModel):
    store_id: str = Field(..., example="S01")
    product_id: str = Field(..., example="P101")
    as_of_date: str = Field(..., example="2023-12-01")
    discount_delta: float = Field(0.0, example=10.0)
    extra_lead_days: int = Field(0, example=2)
    festival_uplift_pct: float = Field(0.0, example=15.0)

class WhatIfResponse(BaseModel):
    store_id: str
    product_id: str
    as_of_date: str
    baseline: Dict[str, Any]
    simulated: Dict[str, Any]
    net_impact: Dict[str, Any]
