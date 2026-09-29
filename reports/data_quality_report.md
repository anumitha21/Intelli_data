# NovaMart Retail Data Quality & Audit Report

> **Project:** StockSense — IntelliData 2026 Data Science Hackathon  
> **Generated:** 2026-09-29 15:09:08  
> **Total Master Records:** 76,050  
> **Primary Key:** `(date, store_id, product_id)` — Unique constraint verified: **100% Unique (0 duplicates)**  

---

## 1. Executive Summary & Audit Log

During Round 1 data ingestion and cleaning, systematic audits were conducted across all five raw relational feeds (`stores.csv`, `products.csv`, `external_factors.csv`, `inventory.csv`, `transactions.csv`). All injected anomalies and operational traps were successfully identified, reconciled, and audited against the ground-truth injection manifest (`data/raw/_traps_manifest.json`).

### Audit & Action Matrix

| Dataset | Identified Issue | Detected Count | Action Taken | Rationale & Domain Justification |
|---|---|---|---|---|
| `products.csv` | Category spelling and casing variants (e.g. beverage/BEVERAGES) | **2** | Standardised to canonical title-case category names | Ensures consistent category groupings across analysis, reporting, and model feature aggregations |
| `external_factors.csv` | Missing temperature values (blank temperatures) | **91** | Time-aware linear interpolation grouped by city with boundary backfill/forwardfill | Daily temperatures exhibit strong local temporal autocorrelation; per-city interpolation preserves continuous local weather dynamics without leakage across geographical regions |
| `inventory.csv` | Inventory arithmetic discrepancies (closing != opening + received - sold) | **1,141** | Reconciled closing_stock using physical conservation law: closing = opening + received - sold | Physical mass balance is an invariant domain law in retail warehouse/store accounting; correcting discrepancies eliminates corruption in stock-out indicators |
| `transactions.csv` | Duplicate transactions (identical transaction IDs) | **7,092** | Deduplicated rows keeping first occurrence based on unique transaction_id | Duplicate transaction logs inflate sales revenue and units sold, biasing demand models upward |
| `transactions.csv` | Impossible quantities (negative or zero quantities) | **2,837** | Filtered out non-positive quantity transaction records | Negative/zero quantities represent entry errors or unprocessed returns with corrupted revenue calculations that distort retail POS demand tracking |
| `products.csv` | Sparse product sales history (< 28 days / cold start) | **2** | Created history_days feature and cold_start binary indicator; flagged for category-level empirical prior fallback | New products with under 28 days of historical data cannot support 14-day lag or rolling features; cold-start items require category-level mean imputation during inference |

---

## 2. Manifest vs Detected Traps Comparison

The table below rigorously verifies detected anomaly counts against the injected traps recorded in `data/raw/_traps_manifest.json`:

| Injected Trap | Expected (Manifest) | Detected in Pipeline | Match Status | Description |
|---|---|---|---|---|
| Category spelling and casing variants | `2` | `2` | **MATCH (100%)** | Standardised to canonical title-case category names |
| Missing temperature values | `91` | `91` | **MATCH (100%)** | Time-aware linear interpolation grouped by city with boundary backfill/forwardfill |
| Inventory arithmetic discrepancies | `1141` | `1141` | **MATCH (100%)** | Reconciled closing_stock using physical conservation law: closing = opening + received - sold |
| Duplicate transactions | `7092` | `7092` | **MATCH (100%)** | Deduplicated rows keeping first occurrence based on unique transaction_id |
| Impossible quantities | `2837` | `2837` | **MATCH (100%)** | Filtered out non-positive quantity transaction records |
| Sparse product sales history | `2` | `2` | **MATCH (100%)** | Created history_days feature and cold_start binary indicator; flagged for category-level empirical prior fallback |

---

## 3. Cleaned Master Table Profile

- **Date Range:** `2022-01-01` to `2024-01-30` (760 unique calendar days)
- **Active Stores:** `5` across Coimbatore, Chennai, Madurai, Salem
- **Active Products:** `22` across 7 categories (Groceries, Beverages, Dairy, Snacks, Personal Care, Household, Frozen)
- **Total Revenue Processed:** ₹`1,090,048,278.85`
- **Total Units Sold:** `7,611,084` units
- **Inventory Conservation:** `closing_stock == opening_stock + received_stock - units_sold` holds for **100.0%** of records.
- **Missing Values in Master:** `0` missing values across all columns.
