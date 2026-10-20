"""
data_processing.py
------------------
Automates processing and cleaning of large sales datasets using NumPy.
Replaces manual spreadsheet work with fast, vectorized operations.

Author: Nika Kvirkvaia
"""

import numpy as np
import time


np.random.seed(42)
NUM_RECORDS = 100_000
CURRENCY = "GEL"
UNKNOWN_REGION = 0   # missing regions are kept as "Unknown" instead of guessed

print(f"Generating raw dataset with {NUM_RECORDS:,} records...\n")

raw_data = {
    "product_id":   np.random.randint(1000, 9999, size=NUM_RECORDS).astype(float),
    "units_sold":   np.random.randint(0, 500, size=NUM_RECORDS).astype(float),
    "unit_price":   np.round(np.random.uniform(5.0, 500.0, size=NUM_RECORDS), 2),
    "discount_pct": np.random.choice([0, 5, 10, 15, 20, np.nan], size=NUM_RECORDS),
    "region_code":  np.random.choice([1, 2, 3, 4, 5, np.nan], size=NUM_RECORDS),
}


def report_missing(name: str, arr: np.ndarray) -> None:
    """Print the count and percentage of NaN values in an array."""
    n_missing = np.sum(np.isnan(arr))
    pct = n_missing / len(arr) * 100
    print(f"  {name:<15} missing: {n_missing:>6,}  ({pct:.1f}%)")


def region_label(code: int) -> str:
    return "Unknown" if code == UNKNOWN_REGION else str(code)


print("=== Missing Value Report (before cleaning) ===")
for col, arr in raw_data.items():
    report_missing(col, arr)


# Numeric field -> median imputation (not skewed by outliers)
median_discount = np.nanmedian(raw_data["discount_pct"])
raw_data["discount_pct"] = np.where(
    np.isnan(raw_data["discount_pct"]),
    median_discount,
    raw_data["discount_pct"],
)

# Categorical field -> explicit "Unknown" category.
# (An earlier version used the mode, which assigned all 16,645 missing rows to
#  region 2 and doubled its transaction count. Keeping them separate is honest.)
n_unknown = int(np.sum(np.isnan(raw_data["region_code"])))
raw_data["region_code"] = np.where(
    np.isnan(raw_data["region_code"]),
    UNKNOWN_REGION,
    raw_data["region_code"],
).astype(int)

print("\n=== Missing Value Report (after cleaning) ===")
for col, arr in raw_data.items():
    report_missing(col, arr.astype(float))
print(f"  ({n_unknown:,} missing regions labelled 'Unknown')")


print("\n=== Runtime Comparison: Loop vs Vectorized ===")

units = raw_data["units_sold"]
price = raw_data["unit_price"]
disc = raw_data["discount_pct"]

start = time.perf_counter()
revenue_loop = np.empty(NUM_RECORDS)
for i in range(NUM_RECORDS):
    revenue_loop[i] = units[i] * price[i] * (1 - disc[i] / 100)
loop_time = time.perf_counter() - start

start = time.perf_counter()
revenue_vec = units * price * (1 - disc / 100)
vec_time = time.perf_counter() - start

speedup = loop_time / vec_time
print(f"  Loop time:       {loop_time:.4f}s")
print(f"  Vectorized time: {vec_time:.6f}s")
print(f"  Speedup:         {speedup:.1f}x faster\n")

# A faster answer is worthless if it's wrong
assert np.allclose(revenue_loop, revenue_vec), "Mismatch between loop and vectorized results!"


# Rows with zero units sold produce zero revenue -> not a real sale
valid_mask = revenue_vec > 0
clean_revenue = revenue_vec[valid_mask]
clean_units = units[valid_mask]
clean_region = raw_data["region_code"][valid_mask]
clean_product_id = raw_data["product_id"][valid_mask].astype(int)

print(f"Records after filtering:  {len(clean_revenue):>8,}  "
      f"(removed {NUM_RECORDS - len(clean_revenue):,} rows with zero revenue)")


sort_idx = np.argsort(clean_revenue)[::-1]
sorted_revenue = clean_revenue[sort_idx]
sorted_units = clean_units[sort_idx]
sorted_region = clean_region[sort_idx]
sorted_product = clean_product_id[sort_idx]


REGIONS = np.array([1, 2, 3, 4, 5, UNKNOWN_REGION])
region_summary = np.zeros((len(REGIONS), 3))  # columns: revenue, units, transactions

for i, region in enumerate(REGIONS):
    mask = clean_region == region
    region_summary[i, 0] = np.sum(clean_revenue[mask])
    region_summary[i, 1] = np.sum(clean_units[mask])
    region_summary[i, 2] = np.count_nonzero(mask)


print("\n" + "=" * 58)
print("        BUSINESS REPORT — REGIONAL SALES SUMMARY")
print("=" * 58)
print(f"{'Region':<10} {'Revenue (' + CURRENCY + ')':>18} {'Units Sold':>12} {'Transactions':>14}")
print("-" * 58)

for i, region in enumerate(REGIONS):
    rev = region_summary[i, 0]
    units_r = int(region_summary[i, 1])
    txns = int(region_summary[i, 2])
    print(f"  {region_label(region):<8} {rev:>18,.2f} {units_r:>12,} {txns:>14,}")

total_rev = np.sum(region_summary[:, 0])
total_units = int(np.sum(region_summary[:, 1]))
total_txns = int(np.sum(region_summary[:, 2]))
print("-" * 58)
print(f"  {'TOTAL':<8} {total_rev:>18,.2f} {total_units:>12,} {total_txns:>14,}")

print("\n--- Top 5 Transactions by Revenue ---")
print(f"{'Rank':<6} {'Product ID':>12} {'Region':>8} {'Revenue (' + CURRENCY + ')':>16} {'Units':>8}")
print("-" * 54)
for rank in range(5):
    print(f"  {rank + 1:<4} {sorted_product[rank]:>12} {region_label(sorted_region[rank]):>8} "
          f"{sorted_revenue[rank]:>16,.2f} {int(sorted_units[rank]):>8,}")

print("\nScript completed successfully.")
