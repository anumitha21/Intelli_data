"""
build_novamart.py – Phase 1 (P1)
Converts data/external/sales_data.csv into 5 NovaMart CSVs in data/raw/:
1. stores.csv
2. products.csv
3. external_factors.csv
4. inventory.csv
5. transactions.csv
Plus data/raw/_traps_manifest.json with all injected anomalies recorded.
"""

import json
import os
import numpy as np
import pandas as pd

SEED = 42

def run():
    np.random.seed(SEED)
    os.makedirs("data/raw", exist_ok=True)
    
    print("Loading external sales data...")
    raw_df = pd.read_csv("data/external/sales_data.csv")
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])
    raw_df = raw_df.sort_values(by=['Date', 'Store ID', 'Product ID']).reset_index(drop=True)
    
    # -------------------------------------------------------------
    # 1. STORES
    # -------------------------------------------------------------
    store_mapping = {
        'S001': {'store_id': 'S01', 'store_name': 'NovaMart Coimbatore Supermarket', 'city': 'Coimbatore', 'store_type': 'Supermarket', 'floor_area_sqft': 25000, 'avg_daily_customers': 1800, 'region': 'West'},
        'S002': {'store_id': 'S02', 'store_name': 'NovaMart Chennai Hypermarket', 'city': 'Chennai', 'store_type': 'Hypermarket', 'floor_area_sqft': 52000, 'avg_daily_customers': 4200, 'region': 'North'},
        'S003': {'store_id': 'S03', 'store_name': 'NovaMart Madurai Supermarket', 'city': 'Madurai', 'store_type': 'Supermarket', 'floor_area_sqft': 22000, 'avg_daily_customers': 1600, 'region': 'South'},
        'S004': {'store_id': 'S04', 'store_name': 'NovaMart Salem Express', 'city': 'Salem', 'store_type': 'Express', 'floor_area_sqft': 8500, 'avg_daily_customers': 750, 'region': 'Central'},
        'S005': {'store_id': 'S05', 'store_name': 'NovaMart Chennai Express', 'city': 'Chennai', 'store_type': 'Express', 'floor_area_sqft': 9500, 'avg_daily_customers': 900, 'region': 'North'},
    }
    stores_df = pd.DataFrame(list(store_mapping.values()))
    stores_df.to_csv("data/raw/stores.csv", index=False)
    print(f"Saved data/raw/stores.csv ({len(stores_df)} stores)")

    # -------------------------------------------------------------
    # 2. PRODUCTS
    # -------------------------------------------------------------
    product_specs = [
        ('P101', 'Aachi Ponni Boiled Rice 5kg', 'Groceries', 'Aachi', 320.0, 240.0, 180, 'SUP01'),
        ('P102', 'Tata Tea Gold 500g', 'Beverages', 'Tata Tea', 290.0, 225.0, 365, 'SUP02'),
        ('P103', 'Heritage Toned Milk 1L', 'Dairy', 'Heritage', 58.0, 48.0, 4, 'SUP03'),
        ('P104', 'Aavin Standardised Milk 500ml', 'Dairy', 'Aavin', 27.0, 22.0, 3, 'SUP03'),
        ('P105', 'Bru Instant Coffee 200g', 'Beverages', 'Bru', 340.0, 265.0, 365, 'SUP02'),
        ('P106', 'Britannia Good Day Biscuits 200g', 'Snacks', 'Britannia', 45.0, 34.0, 180, 'SUP04'),
        ('P107', 'Haldirams Mixture 400g', 'Snacks', 'Haldirams', 120.0, 92.0, 120, 'SUP04'),
        ('P108', 'Amul Salted Butter 500g', 'Dairy', 'Amul', 275.0, 230.0, 180, 'SUP03'),
        ('P109', 'Hamam Neem Soap 100g 3-Pack', 'Personal Care', 'Hamam', 145.0, 108.0, 730, 'SUP05'),
        ('P110', 'Dettol Original Handwash 750ml', 'Personal Care', 'Dettol', 165.0, 120.0, 730, 'SUP05'),
        ('P111', 'Vim Dishwash Gel 750ml', 'Household', 'Vim', 155.0, 115.0, 730, 'SUP06'),
        ('P112', 'Surf Excel Easy Wash 1kg', 'Household', 'Surf Excel', 140.0, 110.0, 730, 'SUP06'),
        ('P113', 'McCain French Fries 450g', 'Frozen', 'McCain', 135.0, 98.0, 180, 'SUP07'),
        ('P114', 'ITC Master Chef Veg Burger Patty 360g', 'Frozen', 'ITC', 150.0, 112.0, 180, 'SUP07'),
        ('P115', 'Sunfeast Dark Fantasy Choco Fills 300g', 'Snacks', 'Sunfeast', 110.0, 82.0, 180, 'SUP04'),
        ('P116', 'Fortune Sunlite Sunflower Oil 1L', 'Groceries', 'Fortune', 165.0, 132.0, 270, 'SUP01'),
        ('P117', 'Tata Salt Iodized 1kg', 'Groceries', 'Tata', 28.0, 21.0, 730, 'SUP01'),
        ('P118', 'Colgate Total Toothpaste 150g', 'Personal Care', 'Colgate', 130.0, 95.0, 730, 'SUP05'),
        ('P119', 'Comfort After Wash Fabric Conditioner 860ml', 'Household', 'Comfort', 235.0, 180.0, 730, 'SUP06'),
        ('P120', 'Real Mixed Fruit Juice 1L', 'Beverages', 'Real', 125.0, 95.0, 180, 'SUP02'),
    ]
    
    new_product_specs = [
        ('P121', 'MTR Masala Idli Mix 500g', 'Groceries', 'MTR', 95.0, 72.0, 180, 'SUP01'),
        ('P122', 'Amul Masti Spiced Buttermilk 200ml', 'Dairy', 'Amul', 15.0, 11.0, 15, 'SUP03'),
    ]

    prod_mapping = {f"P00{i+1:02d}": product_specs[i][0] for i in range(20)}
    
    products_list = []
    for spec in product_specs + new_product_specs:
        products_list.append({
            'product_id': spec[0],
            'product_name': spec[1],
            'category': spec[2],
            'brand': spec[3],
            'mrp': spec[4],
            'cost_price': spec[5],
            'shelf_life_days': spec[6],
            'supplier_id': spec[7]
        })
    products_df = pd.DataFrame(products_list)
    
    # PS Trap 3: Category spelling variants
    trap_category_variants = {
        'P102': 'beverage',
        'P105': 'BEVERAGES',
    }
    for pid, var in trap_category_variants.items():
        products_df.loc[products_df['product_id'] == pid, 'category'] = var
    
    products_df.to_csv("data/raw/products.csv", index=False)
    print(f"Saved data/raw/products.csv ({len(products_df)} products)")

    # -------------------------------------------------------------
    # 3. EXTERNAL FACTORS
    # -------------------------------------------------------------
    cities = ['Coimbatore', 'Chennai', 'Madurai', 'Salem']
    dates = pd.date_range(raw_df['Date'].min(), raw_df['Date'].max(), freq='D')
    
    tn_festivals_holidays = {
        '2022-01-01': ('New Year Day', 1),
        '2022-01-14': ('Pongal', 1),
        '2022-01-15': ('Mattu Pongal / Thiruvalluvar Day', 1),
        '2022-01-16': ('Uzhavar Thirunal', 1),
        '2022-01-26': ('Republic Day', 1),
        '2022-04-14': ('Tamil New Year / Dr Ambedkar Jayanti', 1),
        '2022-04-15': ('Good Friday', 1),
        '2022-05-01': ('May Day', 1),
        '2022-05-03': ('Ramzan (Id-ul-Fitr)', 1),
        '2022-07-10': ('Bakrid', 1),
        '2022-08-09': ('Muharram', 1),
        '2022-08-15': ('Independence Day', 1),
        '2022-08-31': ('Vinayagar Chaturthi', 1),
        '2022-10-02': ('Gandhi Jayanti', 1),
        '2022-10-04': ('Ayutha Pooja', 1),
        '2022-10-05': ('Vijaya Dasami', 1),
        '2022-10-24': ('Deepavali', 1),
        '2022-12-25': ('Christmas', 1),
        '2023-01-01': ('New Year Day', 1),
        '2023-01-15': ('Pongal', 1),
        '2023-01-16': ('Mattu Pongal / Thiruvalluvar Day', 1),
        '2023-01-17': ('Uzhavar Thirunal', 1),
        '2023-01-26': ('Republic Day', 1),
        '2023-04-07': ('Good Friday', 1),
        '2023-04-14': ('Tamil New Year / Dr Ambedkar Jayanti', 1),
        '2023-04-22': ('Ramzan (Id-ul-Fitr)', 1),
        '2023-05-01': ('May Day', 1),
        '2023-06-29': ('Bakrid', 1),
        '2023-07-29': ('Muharram', 1),
        '2023-08-15': ('Independence Day', 1),
        '2023-09-19': ('Vinayagar Chaturthi', 1),
        '2023-10-02': ('Gandhi Jayanti', 1),
        '2023-10-23': ('Ayutha Pooja', 1),
        '2023-10-24': ('Vijaya Dasami', 1),
        '2023-11-12': ('Deepavali', 1),
        '2023-12-25': ('Christmas', 1),
        '2024-01-01': ('New Year Day', 1),
        '2024-01-15': ('Pongal', 1),
        '2024-01-16': ('Mattu Pongal', 1),
        '2024-01-17': ('Uzhavar Thirunal', 1),
        '2024-01-26': ('Republic Day', 1),
    }

    date_weather = raw_df.groupby('Date')['Weather Condition'].agg(lambda x: x.mode().iloc[0]).to_dict()
    ext_rows = []
    base_temps = {'Chennai': 31.0, 'Coimbatore': 27.5, 'Madurai': 32.5, 'Salem': 29.5}
    
    for d in dates:
        d_str = d.strftime('%Y-%m-%d')
        w_cond = date_weather.get(d, 'Sunny')
        month = d.month
        is_weekend = 1 if d.weekday() >= 5 else 0
        fest_info = tn_festivals_holidays.get(d_str, ('None', 0))
        
        for c in cities:
            b_temp = base_temps[c]
            seasonal_offset = 3.5 * np.sin((month - 2) * np.pi / 6.0)
            if w_cond == 'Rainy':
                cond_temp_offset = -4.0
                rain = float(np.round(np.random.uniform(15.0, 48.0), 1))
            elif w_cond == 'Cloudy':
                cond_temp_offset = -1.5
                rain = float(np.round(np.random.uniform(1.0, 8.0), 1))
            elif w_cond == 'Snowy':
                cond_temp_offset = -7.0
                rain = float(np.round(np.random.uniform(5.0, 18.0), 1))
            else:
                cond_temp_offset = 2.0
                rain = 0.0
            
            temp = float(np.round(b_temp + seasonal_offset + cond_temp_offset + np.random.uniform(-1.0, 1.0), 1))
            local_event = 'None'
            if np.random.rand() < 0.03:
                local_event = f"{c} Retail Expo / Cultural Fair"
            
            ext_rows.append({
                'date': d_str,
                'city': c,
                'temp_c': temp,
                'rain_mm': rain,
                'holiday': fest_info[1],
                'festival': fest_info[0],
                'weekend': is_weekend,
                'local_event': local_event
            })
            
    ext_df = pd.DataFrame(ext_rows)
    
    # PS Trap 1: Blank temperatures (~3%)
    n_ext = len(ext_df)
    n_blank_temp = int(np.round(0.03 * n_ext))
    blank_temp_indices = sorted(np.random.choice(n_ext, size=n_blank_temp, replace=False).tolist())
    ext_df.loc[blank_temp_indices, 'temp_c'] = np.nan
    ext_df.to_csv("data/raw/external_factors.csv", index=False)
    print(f"Saved data/raw/external_factors.csv ({len(ext_df)} rows, {n_blank_temp} blank temps injected: {n_blank_temp/n_ext*100:.2f}%)")

    # -------------------------------------------------------------
    # 4. INVENTORY SIMULATION (Target 6-12% stockouts)
    # -------------------------------------------------------------
    raw_df['store_id'] = raw_df['Store ID'].map(lambda x: store_mapping[x]['store_id'])
    raw_df['product_id'] = raw_df['Product ID'].map(prod_mapping)
    raw_df['date_str'] = raw_df['Date'].dt.strftime('%Y-%m-%d')
    
    date_strs = [d.strftime('%Y-%m-%d') for d in dates]
    stores_list = sorted(list(store_mapping.values()), key=lambda x: x['store_id'])
    prod_ids = [p[0] for p in product_specs]
    
    inventory_records = []
    stockout_count = 0
    total_inventory_rows = 0
    
    for s_info in stores_list:
        sid = s_info['store_id']
        for pid in prod_ids:
            sub = raw_df[(raw_df['store_id'] == sid) & (raw_df['product_id'] == pid)].sort_values('Date')
            demand_map = dict(zip(sub['date_str'], sub['Demand']))
            mean_dem = sub['Demand'].mean()
            
            lead_days = (hash(f"{sid}_{pid}") + SEED) % 4 + 1
            # factor = 0.93 with order_qty = 2 * reorder_lvl gives ~8.5% stockout rate (within 6-12%)
            reorder_lvl = int(np.round(lead_days * mean_dem * 0.93))
            order_qty = 2 * reorder_lvl
            
            current_stock = reorder_lvl
            pending_orders = []
            
            for t_idx, d_str in enumerate(date_strs):
                opening = current_stock
                received = sum(qty for arr_idx, qty in pending_orders if arr_idx == t_idx)
                pending_orders = [(arr_idx, qty) for arr_idx, qty in pending_orders if arr_idx > t_idx]
                
                on_hand = opening + received
                dem = int(demand_map.get(d_str, int(mean_dem)))
                sold = min(dem, on_hand)
                closing = on_hand - sold
                
                is_stockout = (closing == 0) or (sold < dem)
                if is_stockout:
                    stockout_count += 1
                total_inventory_rows += 1
                
                if closing < reorder_lvl and len(pending_orders) == 0:
                    pending_orders.append((t_idx + lead_days, order_qty))
                        
                current_stock = closing
                
                inventory_records.append({
                    'date': d_str,
                    'store_id': sid,
                    'product_id': pid,
                    'opening_stock': opening,
                    'received_stock': received,
                    'units_sold': sold,
                    'closing_stock': closing,
                    'reorder_lvl': reorder_lvl,
                    'lead_days': lead_days,
                })

    # PS Trap 6: New products with only 5 days of history
    last_5_dates = date_strs[-5:]
    sparse_pids = ['P121', 'P122']
    for s_info in stores_list:
        sid = s_info['store_id']
        for pid in sparse_pids:
            lead_days = 2
            reorder_lvl = 60
            current_stock = 120
            for d_str in last_5_dates:
                opening = current_stock
                received = 0
                dem = int(np.random.randint(25, 60))
                sold = min(dem, opening)
                closing = opening - sold
                if (closing == 0) or (sold < dem):
                    stockout_count += 1
                total_inventory_rows += 1
                current_stock = closing
                inventory_records.append({
                    'date': d_str,
                    'store_id': sid,
                    'product_id': pid,
                    'opening_stock': opening,
                    'received_stock': received,
                    'units_sold': sold,
                    'closing_stock': closing,
                    'reorder_lvl': reorder_lvl,
                    'lead_days': lead_days,
                })

    inv_df = pd.DataFrame(inventory_records)
    achieved_stockout_rate = stockout_count / total_inventory_rows
    print(f"Inventory simulation completed: {len(inv_df)} rows. Stock-out count = {stockout_count}, Rate = {achieved_stockout_rate*100:.2f}% (Target: 6-12%)")
    assert 0.06 <= achieved_stockout_rate <= 0.12, f"Stockout rate {achieved_stockout_rate} outside 6-12%!"

    # PS Trap 5: Inventory arithmetic mismatches (~1.5%)
    n_inv = len(inv_df)
    n_mismatch = int(np.round(0.015 * n_inv))
    mismatch_indices = sorted(np.random.choice(n_inv, size=n_mismatch, replace=False).tolist())
    for idx in mismatch_indices:
        diff = int(np.random.choice([-15, -10, -5, 5, 10, 15]))
        inv_df.loc[idx, 'closing_stock'] += diff
        
    inv_df.to_csv("data/raw/inventory.csv", index=False)
    print(f"Saved data/raw/inventory.csv ({len(inv_df)} rows, {n_mismatch} arithmetic mismatches injected: {n_mismatch/n_inv*100:.2f}%)")

    # -------------------------------------------------------------
    # 5. TRANSACTIONS (Fast Vectorized Basket Generation)
    # -------------------------------------------------------------
    # We partition daily units_sold into realistic baskets (quantities 1-6)
    # To ensure high performance, we generate baskets efficiently in batches.
    print("Generating transactions data...")
    price_map = {row['product_id']: row['mrp'] for _, row in products_df.iterrows()}
    disc_lookup = raw_df.groupby(['date_str', 'store_id', 'product_id'])['Discount'].mean().to_dict()
    promo_lookup = raw_df.groupby(['date_str', 'store_id', 'product_id'])['Promotion'].max().to_dict()
    
    hours_dist = np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    hours_weights = np.array([0.02, 0.04, 0.08, 0.14, 0.14, 0.10, 0.05, 0.05, 0.06, 0.08, 0.12, 0.08, 0.03, 0.01])
    hours_weights /= hours_weights.sum()
    pm_choices = np.array(['UPI', 'Card', 'Cash'])
    pm_probs = [0.55, 0.30, 0.15]
    
    # Pre-calculate transaction partitions per inventory row
    active_inv = [r for r in inventory_records if r['units_sold'] > 0]
    
    dates_col, stores_col, prods_col = [], [], []
    qtys_col, prices_col, discs_col, promos_col = [], [], [], []
    
    for row in active_inv:
        d_str = row['date']
        sid = row['store_id']
        pid = row['product_id']
        units = row['units_sold']
        
        mrp = price_map[pid]
        disc = disc_lookup.get((d_str, sid, pid), 0.0)
        promo = promo_lookup.get((d_str, sid, pid), 0)
        
        # Partition units into chunks of 3 to 8 (avg ~5.5) to keep transaction volume around ~1.1M rows
        rem = units
        while rem > 0:
            q = np.random.randint(3, 9)
            if q > rem:
                q = rem
            rem -= q
            
            dates_col.append(d_str)
            stores_col.append(sid)
            prods_col.append(pid)
            qtys_col.append(q)
            prices_col.append(mrp)
            discs_col.append(disc)
            promos_col.append(promo)
            
    n_tx = len(qtys_col)
    print(f"Constructing DataFrame for {n_tx:,} transactions...")
    
    # Vectorized generation of hours, payment modes, and customer IDs
    hours_col = np.random.choice(hours_dist, size=n_tx, p=hours_weights)
    pm_col = np.random.choice(pm_choices, size=n_tx, p=pm_probs)
    cust_ints = np.random.randint(10000, 99999, size=n_tx)
    cust_col = [f"CUST_{c}" for c in cust_ints]
    txn_ids = [f"TXN{i+1000001}" for i in range(n_tx)]
    
    qtys_arr = np.array(qtys_col, dtype=np.int32)
    prices_arr = np.array(prices_col, dtype=np.float64)
    discs_arr = np.array(discs_col, dtype=np.float64)
    
    # PS Trap 4: Impossible quantities (~0.2%)
    n_impossible = int(np.round(0.002 * n_tx))
    impossible_indices = sorted(np.random.choice(n_tx, size=n_impossible, replace=False).tolist())
    bad_quantities = np.random.choice([-5, -2, -1, 0], size=n_impossible).astype(np.int32)
    qtys_arr[impossible_indices] = bad_quantities
    
    totals_arr = np.round(qtys_arr * prices_arr * (1.0 - (discs_arr / 100.0)), 2)
    
    tx_df = pd.DataFrame({
        'transaction_id': txn_ids,
        'date': dates_col,
        'store_id': stores_col,
        'product_id': prods_col,
        'customer_id': cust_col,
        'quantity': qtys_arr,
        'unit_price': prices_arr,
        'discount_pct': discs_arr,
        'promotion_flag': promos_col,
        'total_amount': totals_arr,
        'payment_mode': pm_col,
        'hour': hours_col
    })

    # PS Trap 2: Duplicate transactions (~0.5%)
    n_dup = int(np.round(0.005 * n_tx))
    dup_indices = sorted(np.random.choice(n_tx, size=n_dup, replace=False).tolist())
    dup_rows = tx_df.iloc[dup_indices].copy()
    tx_df = pd.concat([tx_df, dup_rows], ignore_index=True)
    
    tx_df.to_csv("data/raw/transactions.csv", index=False)
    print(f"Saved data/raw/transactions.csv ({len(tx_df):,} rows, {n_impossible} impossible quantities, {n_dup} duplicates)")

    # -------------------------------------------------------------
    # 6. SAVE TRAPS MANIFEST
    # -------------------------------------------------------------
    manifest = {
        "seed": SEED,
        "blank_temperatures": {
            "description": "Missing temp_c in external_factors.csv (~3%)",
            "count": n_blank_temp,
            "target_pct": 3.0,
            "actual_pct": round(n_blank_temp / n_ext * 100, 2),
            "sample_indices": blank_temp_indices[:20]
        },
        "duplicate_transactions": {
            "description": "Identical transaction rows duplicated in transactions.csv (~0.5%)",
            "count": n_dup,
            "target_pct": 0.5,
            "actual_pct": round(n_dup / n_tx * 100, 2)
        },
        "category_spelling_variants": {
            "description": "Non-standard category casings injected into products.csv",
            "count": len(trap_category_variants),
            "variants": trap_category_variants
        },
        "impossible_quantities": {
            "description": "Negative or zero quantities injected into transactions.csv (~0.2%)",
            "count": n_impossible,
            "target_pct": 0.2,
            "actual_pct": round(n_impossible / n_tx * 100, 2),
            "sample_indices": impossible_indices[:20]
        },
        "inventory_arithmetic_mismatches": {
            "description": "Discrepancy closing != opening + received - sold in inventory.csv (~1.5%)",
            "count": n_mismatch,
            "target_pct": 1.5,
            "actual_pct": round(n_mismatch / n_inv * 100, 2),
            "sample_indices": mismatch_indices[:20]
        },
        "new_products_sparse_history": {
            "description": "Products with only 5 days of history",
            "product_ids": sparse_pids,
            "days_active": 5,
            "start_date": last_5_dates[0],
            "end_date": last_5_dates[-1]
        }
    }
    
    with open("data/raw/_traps_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved data/raw/_traps_manifest.json successfully.")
    
    # -------------------------------------------------------------
    # BUILD SUMMARY
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("NOVAMART BUILD SUMMARY")
    print("="*50)
    print(f"Stores count:         {len(stores_df)}")
    print(f"Products count:       {len(products_df)}")
    print(f"External factors:     {len(ext_df):,} rows ({ext_df['date'].min()} to {ext_df['date'].max()})")
    print(f"Inventory rows:       {len(inv_df):,}")
    print(f"Transactions rows:    {len(tx_df):,}")
    print(f"Stock-out Rate:       {achieved_stockout_rate*100:.2f}% (Target 6-12%)")
    print("Injected Traps:")
    print(f"  1. Blank temperatures:           {n_blank_temp} ({n_blank_temp/n_ext*100:.2f}%)")
    print(f"  2. Duplicate transactions:       {n_dup} ({n_dup/n_tx*100:.2f}%)")
    print(f"  3. Category spelling variants:   {len(trap_category_variants)} ({list(trap_category_variants.keys())})")
    print(f"  4. Impossible quantities:        {n_impossible} ({n_impossible/n_tx*100:.2f}%)")
    print(f"  5. Inventory arithmetic errors:  {n_mismatch} ({n_mismatch/n_inv*100:.2f}%)")
    print(f"  6. Sparse history products:      {sparse_pids} (5 days)")
    print("="*50)

if __name__ == "__main__":
    run()
