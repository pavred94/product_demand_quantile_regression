# Product Demand Forecasting: XGBoost Quantile Regression

## Table of Contents

1. [Overview](#overview)
2. [Pre-processing](#pre-processing)
3. [Modeling](#modeling)

## Overview

Monthly product-level demand forecasting with 80% prediction intervals at a t+2 month horizon. Given historical order data, the model predicts demand two months ahead for each product-warehouse combination, outputting three values: P10 (10th percentile), P50 (median), and P90 (90th percentile). P10 and P90 form the 80% prediction interval used for procurement decisions, while P50 serves as the point forecast.

**Data:** ~1M daily transaction rows across 2,160 products, 4 warehouses, and 33 product categories, spanning 2011-2017 ([Kaggle source](https://www.kaggle.com/datasets/felixzhao/productdemandforecasting)).

**Final validation performance:**

| Metric | Value |
|---|---|
| Pinball Loss (avg P10/P50/P90) | 3,880 |
| Scaled vs. Naive Baseline | P10: 0.285, P50: 0.807, P90: 0.503 |
| Per-Quantile Calibration | P10: 10.8%, P50: 51.4%, P90: 90.0% |
| WAPE (P50) | 35.4% |
| 80% Interval Coverage | 79.2% (target 80%) |
| Mean Winkler Score | 57,540 |
| Quantile Crossing | 14 rows (0.05%) |

## Pre-processing

### Data Cleaning

**Missing dates.** 11,239 rows (1.1%) had missing date values, confined to Whse_A and Category_019. These were dropped since they could not be placed in the time series.

**Parenthesized values.** Approximately 10,500 rows (1%) contained values like `(100)`, which is accounting notation for negatives (returns, cancellations). These were corrected to negative integers. After monthly aggregation, only 0.51% of product-months net negative.

**Duplicate transactions.** 113,064 rows (10.9%) are exact duplicates across all columns. These are retained because identical transactions on the same day are plausible in order-level data (e.g. two separate orders of the same size). Note, in a real-world scenario, knowledge of data collection, domain knowledge and consultation with others would affect this decision. They are aggregated into monthly totals in the next step.

### Aggregation and Date Trimming

Daily transactions are summed to monthly totals per product-warehouse combination. The date range is trimmed to 2012-01 through 2016-12: 2011 is excluded because most products had no data until late in the year, creating artificial ramp-up patterns. The final month (2017-01) is excluded because it is incomplete.

### Stationarity and Target Transformation

An Augmented Dickey-Fuller (ADF) test confirmed raw monthly demand is non-stationary (p > 0.05) while 2-period differenced demand is stationary (p < 0.05). The 2-period differencing aligns with the t+2 forecast horizon, meaning the model predicts the change in demand between now and two months from now.

Before differencing, an arcsinh transformation is applied: `demand_diff = arcsinh(demand(t)) - arcsinh(demand(t-2))`. arcsinh compresses scale similarly to log for large values but, unlike log, is defined for negative values and linear near zero. This normalizes gradient magnitudes across demand scales so the model does not optimize exclusively for high-volume products. The raw target variable (demand differences) has a standard deviation of ~105,000; after arcsinh, the standard deviation drops to ~1.7.

Demand is reconstructed at prediction time via `predicted_demand = sinh(arcsinh(lag_2) + predicted_diff)`.

### Interior Gap Reindexing

Monthly aggregation only produces rows for months with orders. Since `groupby().shift(2)` shifts by position rather than calendar months, gaps between observed months cause lag features to reference the wrong time period. Each product-warehouse is reindexed to a contiguous monthly grid within its first-to-last observed date range, with gaps filled as zero demand. This added 24,575 rows (18% increase), bringing zero-demand to 15.5%.

### Sparse Product Filtering

Product-warehouse combinations with a fill rate below 50% (nonzero months / total months) are excluded from training. Their lag features would be dominated by zero-filled gaps rather than real demand patterns. Even the real observations of sparse products have unreliable features because their lags mostly pull from filled zeros. This removed 239 combinations (8.4%), leaving 148,623 rows with 11.7% zero-demand. Excluded products can still be forecast at inference time via cross-product features (Product_Code, Product_Category, Month).

### Feature Encoding

**Product_Code** is encoded using sklearn's `TargetEncoder` (cv=5, smooth="auto"), which replaces each product code with a smoothed mean of its demand values, blended toward the global mean to prevent overfitting for low-frequency products. This is necessary because Product_Code has ~2,000 unique values, too many for XGBoost's native categorical handling.

**Warehouse, Product_Category, and Month** use XGBoost's native categorical support (`enable_categorical=True`), which learns optimal categorical splits directly rather than relying on ordinal or one-hot encoding.

### Train/Validation Split

Date-boundary split at 2016-01 (80/20). All products for a given month are in the same set, preventing partial-month artifacts when aggregating predictions by date. Training covers 2012-01 to 2015-12 (118,981 rows), validation covers 2016-01 to 2016-12 (29,642 rows).

## Modeling
TODO
