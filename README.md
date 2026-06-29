# Product Demand Forecasting: XGBoost Quantile Regression

## Table of Contents

1. [Overview](#overview)
2. [Pre-processing](#pre-processing)

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
TODO
