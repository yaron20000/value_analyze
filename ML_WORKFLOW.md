# Machine Learning Workflow for Stock Prediction & Dividend Growth

## Overview

This project now has an **ML-optimized data structure** for predicting stock returns and dividend growth. The data flows through three stages:

```
Raw API Data (JSONB)  →  ML Tables (Flat)  →  Training/Prediction
   api_raw_data           ml_features              train_model.py
                          ml_targets
                          ml_stock_prices
```

---

## Quick Start

### 1. Set Up ML Schema

Create the ML-optimized tables:

```bash
psql -U postgres -d borsdata -f schema_ml.sql
```

This creates:
- `ml_features` - One row per (stock, year) with all 52+ KPIs as columns
- `ml_stock_prices` - Daily OHLCV data for technical features
- `ml_targets` - Future returns and dividend growth (what we predict)
- `ml_training_data` - View joining features + targets

### 2. Transform Raw Data to ML Format

After fetching data with `fetch_and_store.py`, transform it:

```bash
# Transform all instruments
python transform_to_ml.py --db-password yourpass

# Or transform specific instrument
python transform_to_ml.py --db-password yourpass --instrument-id 199
```

**What this does:**
- Extracts KPI time series from JSONB arrays
- Pivots from long format (one row per KPI-year) to wide format (one row per stock-year)
- Calculates target variables (future dividend growth, stock returns)
- Populates `ml_features`, `ml_targets`, and `ml_stock_prices` tables

### 3. Train Models

```bash
# Train dividend growth regression model
python train_model.py --db-password yourpass --model dividend-growth

# Train dividend increase classifier
python train_model.py --db-password yourpass --model dividend-classifier

# Train both
python train_model.py --db-password yourpass --model all
```

---

## Schema Design

### ML Features Table (Wide Format)

Each row = one training example (stock-year combination)

```sql
SELECT * FROM ml_features LIMIT 1;

┌───────────────┬──────┬────────┬─────────┬───────┬──────┬─────────┬───┐
│ instrument_id │ year │ period │ pe_ratio│ roe   │ ...  │ revenue │...│
├───────────────┼──────┼────────┼─────────┼───────┼──────┼─────────┼───┤
│ 199           │ 2024 │ 5      │ 18.5    │ 25.3  │ ...  │ 35865   │...│
└───────────────┴──────┴────────┴─────────┴───────┴──────┴─────────┴───┘
```

**Why this design?**
✅ Direct loading: `pd.read_sql()` → ready for scikit-learn
✅ No pivoting needed in training code
✅ Each column is a feature, each row is a sample
✅ Easy to add/remove features by adding/dropping columns

**Columns (60+ features):**
- **Identifiers**: instrument_id, year, period
- **Metadata**: company_name, sector, market, country
- **Valuation**: PE, PS, PB, EV/EBITDA, PEG, EV/Sales (13 features)
- **Profitability**: ROE, ROI, ROA, margins (11 features)
- **Financial Health**: Debt/Equity, liquidity ratios (12 features)
- **Growth**: Revenue, earnings, dividend growth (4 features)
- **Per-Share Metrics**: EPS, DPS, book value, FCF (5 features)
- **Cash Flow**: FCF margin, OCF, Capex (4 features)
- **Dividend**: Payout ratio (1 feature)
- **Absolute Metrics**: Revenue, earnings, assets, equity (7 features)

### ML Targets Table

```sql
SELECT * FROM ml_targets LIMIT 1;

┌───────────────┬──────┬───────────────────────────┬──────────────────┐
│ instrument_id │ year │ next_year_dividend_growth │ dividend_increased│
├───────────────┼──────┼───────────────────────────┼──────────────────┤
│ 199           │ 2024 │ 8.5                       │ true             │
└───────────────┴──────┴───────────────────────────┴──────────────────┘
```

**Target variables:**
- `next_year_dividend_growth` - % change in dividend (regression target)
- `next_3year_avg_dividend_growth` - Average growth over 3 years
- `dividend_increased` - Boolean (classification target)
- `next_year_return` - Stock price return % (future feature)
- `next_3year_return` - 3-year stock return (future feature)

### Training Data View (Features + Targets Joined)

```sql
SELECT * FROM ml_training_data WHERE year = 2023;
```

This view joins `ml_features` and `ml_targets` so you get everything in one query.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    API Raw Data (JSONB)                     │
│  Each KPI is a separate row with nested JSON array:        │
│  {kpiId: 6, values: [{y:2024,v:10.5}, {y:2023,v:9.8}...]}  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ transform_to_ml.py
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   ML Features (Flat Table)                  │
│  One row per (stock, year):                                 │
│  instrument_id │ year │ pe_ratio │ roe │ ... │ revenue      │
│  199           │ 2024 │ 18.5     │ 25  │ ... │ 35865        │
│  199           │ 2023 │ 20.1     │ 23  │ ... │ 38116        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ calculate_targets()
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                      ML Targets                             │
│  Future outcomes (shifted forward in time):                 │
│  inst_id │ year │ next_year_dividend_growth                 │
│  199     │ 2023 │ 8.5%  (= (2024_div - 2023_div)/2023_div)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ JOIN in ml_training_data view
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   Training Dataset                          │
│  Features (X) + Targets (y) in one table:                   │
│  year │ pe_ratio │ roe │ ... │ next_year_dividend_growth    │
│  2023 │ 20.1     │ 23  │ ... │ 8.5                          │
│  2022 │ 19.5     │ 22  │ ... │ 10.2                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ train_model.py
                       ↓
         ┌──────────────────────────────┐
         │  Trained ML Models           │
         │  - RandomForestRegressor     │
         │  - RandomForestClassifier    │
         │  - (your models here)        │
         └──────────────────────────────┘
```

---

## ML Training Pipeline

### Time-Based Train/Test Split

**CRITICAL**: Don't use random split for time-series data!

```python
# ❌ WRONG - Causes data leakage
X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ CORRECT - Split by time
split_year = 2020
train_mask = df['year'] < split_year  # Use old data for training
test_mask = df['year'] >= split_year   # Test on future data
```

**Why?** If you randomly split, you might train on 2024 data and test on 2020 data. This lets the model "cheat" by seeing the future.

### Example Training Code

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_sql("SELECT * FROM ml_training_data", conn)

# Features vs target
features = ['pe_ratio', 'roe', 'dividend_per_share', 'revenue_growth', ...]
X = df[features].fillna(df[features].median())
y = df['next_year_dividend_growth']

# Time-based split
train_mask = df['year'] < 2020
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# Train
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(f"Test RMSE: {rmse:.2f}%")
```

---

## Model Performance Expectations

### Dividend Growth Prediction (Regression)

**Baseline**: Predict "same as last year" → RMSE ≈ 15-20%

**Good model**: RMSE < 10%, R² > 0.3

**Why low R²?** Dividend growth has high variance and depends on unpredictable factors (board decisions, market conditions). Even R² of 0.3-0.5 is useful.

### Dividend Increase Classifier

**Baseline**: Always predict "increase" (if most dividends go up) → Accuracy ≈ 60-70%

**Good model**: Accuracy > 75%, Precision/Recall balanced

---

## Feature Engineering Tips

### Current Features (Already Included)

✅ Valuation ratios (PE, PS, PB, etc.)
✅ Profitability metrics (ROE, ROI, margins)
✅ Financial health (debt ratios, liquidity)
✅ Growth rates (revenue, earnings)
✅ Per-share metrics (EPS, DPS, FCF/share)

### Potential New Features

1. **Lagged features** - Previous year's values
   ```sql
   LAG(dividend_per_share, 1) OVER (PARTITION BY instrument_id ORDER BY year) as prev_year_dividend
   ```

2. **Moving averages** - 3-year average ROE, EPS, etc.
   ```sql
   AVG(roe) OVER (PARTITION BY instrument_id ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as roe_3yr_avg
   ```

3. **Trend indicators** - Is metric increasing/decreasing?
   ```sql
   CASE WHEN roe > LAG(roe) OVER (...) THEN 1 ELSE 0 END as roe_increasing
   ```

4. **Sector averages** - Compare to peer group
   ```sql
   AVG(pe_ratio) OVER (PARTITION BY sector, year) as sector_avg_pe
   ```

5. **Technical indicators** from `ml_stock_prices`:
   - 50-day / 200-day moving average
   - RSI, MACD, Bollinger Bands
   - Volatility (standard deviation of returns)

---

## Next Steps

### Immediate Actions

1. ✅ Run `schema_ml.sql` to create tables
2. ✅ Run `transform_to_ml.py` to populate tables
3. ✅ Run `train_model.py` to verify pipeline works
4. Review feature importance to understand what drives dividend growth

### Model Improvements

1. **Add more features** (lagged values, moving averages, sector comparisons)
2. **Try different models** (XGBoost, LightGBM, Neural Networks)
3. **Hyperparameter tuning** (GridSearchCV, Optuna)
4. **Ensemble methods** (blend RandomForest + XGBoost predictions)
5. **Add stock price predictions** (calculate `next_year_return` targets)

### Production Deployment

1. **Save trained models** (pickle, joblib)
2. **Create prediction endpoint** (Flask API)
3. **Automate retraining** (weekly/monthly cron job)
4. **Track prediction accuracy** over time
5. **Add backtesting** framework

---

## Comparison: Old vs New Schema

### Old Schema (JSONB - Bad for ML)

```sql
-- Each query requires complex JSON extraction
SELECT
    instrument_id,
    (raw_data->>'kpiId')::int as kpi_id,
    jsonb_array_elements(raw_data->'values')->>'y' as year,
    jsonb_array_elements(raw_data->'values')->>'v' as value
FROM api_raw_data
WHERE kpi_name = 'P/E'
-- Need to repeat for EACH KPI, then pivot... 😰
```

**Problems:**
- Requires complex pivoting for every training run
- Slow JSON extraction
- Hard to add features
- Can't directly use with scikit-learn

### New Schema (Flat - Great for ML)

```sql
-- Simple query, ready for ML
SELECT
    pe_ratio, roe, dividend_per_share, revenue_growth,
    next_year_dividend_growth
FROM ml_training_data
WHERE year >= 2020
```

**Benefits:**
- One query, done ✅
- Fast (indexed columns, no JSON parsing)
- Easy to add features (just add columns)
- Works with pandas, scikit-learn, PyTorch, TensorFlow

---

## FAQ

**Q: Do I still need the `api_raw_data` table?**

A: Yes! Keep it as your "data lake" for:
- Audit trail (what did the API return?)
- Re-processing if you change transformation logic
- Data you haven't extracted yet

Think of it as: Raw data → ML data (two-stage pipeline)

**Q: How often should I run `transform_to_ml.py`?**

A: After each data fetch. Add it to your workflow:
```bash
python fetch_and_store.py --db-password pass
python transform_to_ml.py --db-password pass  # ← Add this
```

**Q: Can I use this for real-time predictions?**

A: Not directly (data is annual/quarterly). For real-time, you'd need:
1. Daily stock prices → technical features
2. Latest quarterly report → fundamental features
3. Combine both → predict short-term returns

**Q: What if I want to predict stock returns instead of dividends?**

A: The schema already supports it! Just need to:
1. Calculate `next_year_return` targets (add to `transform_to_ml.py`)
2. Train a model using stock price data from `ml_stock_prices`

---

## References

- Original schema: [schema.sql](schema.sql)
- ML schema: [schema_ml.sql](schema_ml.sql)
- Transformation pipeline: [transform_to_ml.py](transform_to_ml.py)
- Example training: [train_model.py](train_model.py)
