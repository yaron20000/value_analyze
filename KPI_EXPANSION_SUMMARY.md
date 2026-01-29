# KPI Expansion Summary

## Overview

The Borsdata API fetcher has been expanded from **35 KPIs** to **80 KPIs** across 3 tiers, providing comprehensive coverage for ML/AI analysis.

## What Was Added

### Previous State
- **35 Tier 1 KPIs only**
- Basic coverage of valuation, profitability, growth, returns, financial health, and size metrics

### Current State
- **80 Total KPIs** (35 Tier 1 + 25 Tier 2 + 20 Tier 3)
- Comprehensive coverage including per-share metrics, quality scores, technical indicators, insider trading, and more

---

## Detailed Breakdown

### Tier 1 KPIs (35 - Already Implemented)

#### Valuation Metrics (8 KPIs)
1. Dividend Yield
2. P/E
3. P/S
4. P/B
10. EV/EBIT
11. EV/EBITDA
13. EV/FCF
15. EV/S

#### Profitability & Margins (6 KPIs)
28. Gross Margin
29. Operating Margin
30. Profit Margin
31. FCF Margin
32. EBITDA Margin
51. OCF Margin

#### Growth Metrics (6 KPIs)
94. Revenue Growth
96. EBIT Growth
97. Earnings Growth
98. Dividend Growth
99. Book Value Growth
100. Assets Growth

#### Return Metrics (4 KPIs)
33. Return on Equity (ROE)
34. Return on Assets (ROA)
36. Return on Capital (ROC)
37. Return on Invested Capital (ROIC)

#### Financial Health (6 KPIs)
39. Equity Ratio
40. Debt to Equity
41. Net Debt %
42. Net Debt/EBITDA
44. Current Ratio
46. Cash %

#### Size/Absolute Metrics (5 KPIs)
49. Enterprise Value
50. Market Cap
53. Revenue
56. Earnings
63. Free Cash Flow

---

### Tier 2 KPIs (25 - NEWLY ADDED)

#### Per-Share Metrics (6 KPIs)
5. Revenue per Share
6. Earnings per Share (EPS)
7. Dividend per Share
8. Book Value per Share
23. FCF per Share
68. OCF per Share

#### Additional Cash Flow (4 KPIs)
24. FCF Margin %
27. Earnings/FCF
62. Operating Cash Flow (absolute)
64. Capex

#### Quality Scores (3 KPIs)
167. F-Score (Piotroski)
163. Magic Formula
174. Earnings Stability

#### Technical Indicators (5 KPIs)
151. Performance (Price Change)
152. Total Return
159. RSI
311. Volatility H-L %
313. Volume

#### Insider/Ownership (7 KPIs)
237. Insider Net 1 Week
238. Insider Net 1 Month
239. Insider Net 3 Months
240. Insider Net 12 Months
241. Top 1 Shareholder %
242. Top 2 Shareholder %
243. Top 3 Shareholder %

---

### Tier 3 KPIs (20 - NEWLY ADDED)

#### Additional Valuation (2 KPIs)
19. PEG Ratio
20. Dividend Payout %

#### Additional Absolute Metrics (5 KPIs)
54. EBITDA
57. Total Assets
58. Total Equity
60. Net Debt
61. Number of Shares

#### Additional Technical (4 KPIs)
157. MA200 Rank
158. MA(50)/MA(200)
312. Volatility Std Dev
314. Volume Trend

#### Short Selling (4 KPIs)
207. Avg Short-Selling 1 Week
208. Avg Short-Selling 1 Month
209. Avg Short-Selling 3 Months
210. Avg Short-Selling 1 Year

#### Buybacks (3 KPIs)
213. Buyback 1 Month
214. Buyback 3 Months
215. Buyback 1 Year

#### Additional Quality (2 KPIs)
164. Graham Strategy
178. Cash Flow Stability

---

## Impact on Data Fetching

### Per Instrument:
- **Before:** 5 reports + 35 KPIs + 1 stock price = **41 API calls**
- **After:** 5 reports + 80 KPIs + 1 stock price = **86 API calls**
- **Increase:** +45 API calls per instrument (+110%)

### For 4 Default Instruments (3, 19, 199, 750):
- **Before:** ~180 API calls total
- **After:** ~360 API calls total
- **Duration:** ~72 seconds (with 0.2s rate limiting)

### Storage Impact:
- **JSON files:** 80 additional files per instrument
- **Database rows:** 80 additional rows per instrument per day
- **File size:** ~2-5 KB per KPI JSON file
- **Total additional storage per instrument:** ~200-400 KB

---

## Files Modified

### 1. [fetch_and_store.py](c:\Users\yaron\proj\value_analyze\fetch_and_store.py)
- **Lines 385-502:** Expanded `kpi_endpoints` list from 35 to 80 KPIs
- Added comprehensive comments for each tier and category
- Each KPI now includes: (ID, slug_name, display_name)

### 2. [README.md](c:\Users\yaron\proj\value_analyze\README.md)
- **Line 126:** Updated KPI description to show 80 KPIs across 3 tiers
- Added breakdown of what each tier includes

---

## JSON Output Format

Each KPI JSON file now includes the `kpi_name` field:

```json
{
  "timestamp": "2026-01-13T15:30:00.000000",
  "endpoint_name": "kpi_f_score_3",
  "kpi_name": "F-Score (Piotroski)",
  "success": true,
  "error": null,
  "data": {
    "kpiId": 167,
    "reportTime": "year",
    "priceValue": "mean",
    "values": [
      {"y": 2025, "p": 5, "v": 7.5},
      {"y": 2024, "p": 5, "v": 6.8}
    ]
  }
}
```

---

## Database Schema

The `api_raw_data` table now includes the `kpi_name` column:

```sql
CREATE TABLE api_raw_data (
    id SERIAL PRIMARY KEY,
    endpoint_name VARCHAR(100) NOT NULL,
    endpoint_path VARCHAR(500) NOT NULL,
    instrument_id INTEGER,
    kpi_name VARCHAR(100),  -- NEW: Human-readable KPI name
    fetch_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    raw_data JSONB,
    request_params JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### Example Database Queries

```sql
-- Get all KPIs for a specific instrument
SELECT kpi_name, raw_data->'values'
FROM api_raw_data
WHERE instrument_id = 3
  AND kpi_name IS NOT NULL
  AND success = true
ORDER BY kpi_name;

-- Get all quality scores
SELECT instrument_id, kpi_name, raw_data
FROM api_raw_data
WHERE kpi_name IN ('F-Score (Piotroski)', 'Magic Formula', 'Graham Strategy')
  AND success = true;

-- Get insider trading data
SELECT instrument_id, kpi_name, raw_data
FROM api_raw_data
WHERE kpi_name LIKE 'Insider Net%'
  AND success = true
ORDER BY instrument_id, kpi_name;

-- Count KPIs by tier (based on naming patterns)
SELECT
    CASE
        WHEN kpi_name IN ('Dividend Yield', 'P/E', 'P/S', 'P/B', 'EV/EBIT',
                         'EV/EBITDA', 'EV/FCF', 'EV/S', 'Gross Margin',
                         'Operating Margin', 'Profit Margin', 'FCF Margin',
                         'EBITDA Margin', 'OCF Margin') THEN 'Tier 1'
        WHEN kpi_name LIKE 'Insider Net%' OR kpi_name LIKE 'Top % Shareholder%'
             OR kpi_name = 'F-Score (Piotroski)' THEN 'Tier 2'
        WHEN kpi_name LIKE 'Short-Selling%' OR kpi_name LIKE 'Buyback%' THEN 'Tier 3'
        ELSE 'Other'
    END as tier,
    COUNT(DISTINCT kpi_name) as kpi_count
FROM api_raw_data
WHERE kpi_name IS NOT NULL
GROUP BY tier;
```

---

## Usage Example

### Fetch All 80 KPIs
```bash
# Using environment variables
python fetch_and_store.py

# With explicit API key and database
python fetch_and_store.py YOUR_API_KEY --db-password yourpass

# JSON files only (no database)
python fetch_and_store.py YOUR_API_KEY --no-db

# Specific instruments
python fetch_and_store.py YOUR_API_KEY --instruments "3,750" --no-db
```

### Expected Output
```
======================================================================
BORSDATA API FETCHER & POSTGRESQL STORER
======================================================================
API Key: abcd1234...xyz9
Database: borsdata @ localhost:5432
Instruments: [3, 19, 199, 750]
Time: 2026-01-13T15:30:00.000000
======================================================================

...

======================================================================
EXECUTION SUMMARY
======================================================================
Total endpoints: 360
Successful: 358
Failed: 2
Skipped (already exists): 0
Duration: 72.45 seconds
Results saved to: results/
Database: borsdata on localhost
======================================================================
```

---

## ML/AI Benefits

### Expanded Coverage Enables:

1. **Quality Factor Models**
   - F-Score, Magic Formula, Graham Strategy
   - Earnings and cash flow stability metrics

2. **Sentiment Analysis**
   - Insider trading patterns (4 timeframes)
   - Ownership concentration (top 3 shareholders)
   - Short selling trends (4 timeframes)
   - Buyback activity (3 timeframes)

3. **Technical Analysis**
   - RSI, moving averages, volatility metrics
   - Performance and total return
   - Volume and volume trends

4. **Per-Share Normalization**
   - EPS, Revenue/Share, Dividend/Share
   - FCF/Share, OCF/Share, Book Value/Share

5. **Comprehensive Valuation**
   - Multiple valuation ratios (P/E, P/S, P/B, EV ratios)
   - PEG ratio for growth-adjusted valuation

6. **Cash Flow Quality**
   - Operating cash flow vs earnings
   - Free cash flow metrics
   - Capex tracking

---

## Migration for Existing Databases

If you have an existing database, run the migration script:

```bash
psql -d borsdata -f migration_add_kpi_name.sql
```

This will:
- Add the `kpi_name` column to `api_raw_data` table
- Create an index on `kpi_name` for efficient queries
- Preserve all existing data

---

## Performance Considerations

### API Rate Limiting
- Borsdata API: 100 calls per 10 seconds
- With 0.2s delay: 5 calls per second = safe margin
- 360 total calls = ~72 seconds total runtime

### First Run vs Subsequent Runs
- **First run:** Fetches all 360 endpoints (~72 seconds)
- **Subsequent runs (same day):** Skips existing data (~5 seconds)
- Duplicate prevention via database constraints

### Database Query Performance
- JSONB indexing enabled for fast JSON queries
- `kpi_name` indexed for filtering
- Unique constraint prevents duplicate fetches

---

## Next Steps

### Optional Further Expansion

If you need even more KPIs, consider adding:

**Tier 4 - Specialized Metrics:**
- Additional insider trading timeframes (229-236)
- More technical indicators (155, 156, 160-162)
- Additional growth metrics (101-106)
- Sector-specific metrics

**Total Available:** 322 KPIs in Borsdata API

**Current Coverage:** 80 / 322 = **24.8%** of all available KPIs

---

## Summary

✅ **Expanded from 35 to 80 KPIs** (+129% increase)
✅ **Added KPI names to JSON and database**
✅ **Comprehensive ML/AI coverage across all categories**
✅ **Maintained backward compatibility**
✅ **Optimized with duplicate prevention**
✅ **Documented and tested**

The system now provides comprehensive fundamental, technical, and sentiment data for sophisticated machine learning models!
