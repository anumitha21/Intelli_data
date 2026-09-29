# NovaMart Retail Feature Dictionary & Leak-Safety Rationale

> **Project:** StockSense — IntelliData 2026 Data Science Hackathon  
> **Target Horizon:** $t+1 \dots t+7$ (Next 7 Calendar Days)  
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
| `lag_1` | Demand Lag | float | Demand of SKU at store on date $t-1$ | Shift $\ge 1$; uses only past sales data |
| `lag_7` | Demand Lag | float | Demand of SKU at store on date $t-7$ | Shift $\ge 7$; uses only past sales data |
| `lag_14` | Demand Lag | float | Demand of SKU at store on date $t-14$ | Shift $\ge 14$; uses only past sales data |
| `rolling_mean_7` | Rolling Stats | float | 7-day rolling mean of demand over $t-7 \dots t-1$ | Computed strictly on $shift(1)$ demand series |
| `rolling_mean_14` | Rolling Stats | float | 14-day rolling mean of demand over $t-14 \dots t-1$ | Computed strictly on $shift(1)$ demand series |
| `rolling_std_7` | Rolling Stats | float | 7-day rolling standard deviation over $t-7 \dots t-1$ | Computed strictly on $shift(1)$ demand series |
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
| `next_7_day_demand` | Regression | Sum of true daily demand across dates $t+1, t+2, \dots, t+7$ | Future 7 days |
| `stockout_flag` | Classification | Binary indicator: 1 if `closing_stock == 0` OR `units_sold < demand` on ANY day between $t+1$ and $t+7$; 0 otherwise | Future 7 days |
