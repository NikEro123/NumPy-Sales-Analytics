# NumPy Sales Analytics

A vectorized data-cleaning and reporting pipeline built with NumPy. It generates a
100,000-row sales dataset, handles missing values, benchmarks vectorized math against a
plain Python loop, and prints a regional business report.

**The data is randomly generated, not real sales data.** The point of the project is the
processing and the performance comparison, not the numbers themselves.

## What it does

1. Generates 100,000 records with realistic missing values (~17% of `discount_pct` and
   `region_code` are NaN)
2. Reports missing values per column before and after cleaning
3. Imputes `discount_pct` with the **median** and `region_code` with the **mode**
4. Calculates revenue twice — once with a Python loop, once vectorized — and times both
5. Asserts the two results match, so the speed-up can't be hiding a bug
6. Filters invalid rows, sorts by revenue, and prints a regional summary plus the top 5
   transactions

## Result

```
=== Runtime Comparison: Loop vs Vectorized ===
  Loop time:       0.0552s
  Vectorized time: 0.001690s
  Speedup:         32.7x faster
```

Same arithmetic, same output, **32.7x less time** over 100,000 rows — verified identical
with `np.allclose()`.

## Sample output

```
=== Missing Value Report (before imputation) ===
  discount_pct    missing: 16,585  (16.6%)
  region_code     missing: 16,645  (16.6%)

Records after filtering:    99,800  (removed 200 invalid rows)

       BUSINESS REPORT — REGIONAL SALES SUMMARY
Region         Revenue (GEL)   Units Sold   Transactions
  1           941,896,519.82    4,171,789         16,639
  2         1,904,322,818.23    8,351,125         33,360
  3           943,689,143.10    4,143,427         16,642
  4           943,057,200.43    4,130,757         16,526
  5           944,153,612.75    4,151,272         16,633
```

## A finding worth noting

Look at region 2 — roughly double the transactions of every other region. That isn't in
the data; **it's an artifact of my own imputation.** Region 2 happened to be the mode, so
all 16,645 missing region codes were assigned to it, inflating the group by exactly that
amount.

It's a good illustration of why imputation is a decision with consequences rather than a
cleaning step you apply and forget. A safer approach for a categorical field like this
would be to keep an explicit "unknown" category instead of guessing.

## Run it

```bash
pip install numpy
python data_processing.py
```

## What I learned

- How much vectorized NumPy actually beats element-wise Python loops, measured rather
  than assumed
- Median vs mode imputation — and that mode imputation distorts category distributions
- Using boolean masks (`np.isnan`, `valid_mask`) to filter without looping
- Verifying an optimization with an assertion instead of trusting that it's correct
because it's faster

## Known limitations

- Data is synthetic; no real dataset is read in
- Results are printed to the terminal rather than exported to CSV or charted
- The filtering mask checks `region_code >= 1`, which can never be false after
  imputation — leftover from an earlier version
