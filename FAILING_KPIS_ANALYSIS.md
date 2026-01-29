# Failing KPIs Analysis

## Summary

Out of 80 KPIs attempted, **28 KPIs are failing** with "Error 400: No response body" for all instruments.

**Working:** 52 KPIs
**Failing:** 28 KPIs
**Success Rate:** 65%

---

## Root Cause

The failing KPIs don't support the `/instruments/{id}/kpis/{kpi_id}/year/mean/history` endpoint we're using. They likely require:

1. **Different API endpoints** (real-time data, daily data, or special endpoints)
2. **Premium API access** (higher subscription tier)
3. **Alternative data sources** (some are already available via `/holdings` endpoints)

---

## Failing KPIs by Category

### Technical Indicators (8 KPIs) - FAILING
These require real-time or daily price data, not annual/mean history:

- KPI 151: Performance (Price Change)
- KPI 152: Total Return
- KPI 157: MA200 Rank
- KPI 158: MA(50)/MA(200)
- KPI 159: RSI
- KPI 311: Volatility H-L %
- KPI 312: Volatility Std Dev
- KPI 313: Volume
- KPI 314: Volume Trend (9 total)

**Why:** These are calculated from daily stock prices, not annual fundamentals.

**Alternative:** We already fetch stock prices separately, these could be calculated from that data.

---

### Insider/Ownership (7 KPIs) - FAILING
These come from a different endpoint entirely:

- KPI 237: Insider Net 1 Week
- KPI 238: Insider Net 1 Month
- KPI 239: Insider Net 3 Months
- KPI 240: Insider Net 12 Months
- KPI 241: Top 1 Shareholder %
- KPI 242: Top 2 Shareholder %
- KPI 243: Top 3 Shareholder %

**Why:** Insider data comes from `/holdings/insider` endpoint (which we already fetch!)

**Alternative:** We already fetch `holdings_insider.json` separately. Parse that instead.

---

### Short Selling (4 KPIs) - FAILING

- KPI 207: Avg Short-Selling 1 Week
- KPI 208: Avg Short-Selling 1 Month
- KPI 209: Avg Short-Selling 3 Months
- KPI 210: Avg Short-Selling 1 Year

**Why:** Short selling data comes from `/holdings/shorts` endpoint (which we already fetch!)

**Alternative:** We already fetch `holdings_shorts.json` separately.

---

### Buybacks (3 KPIs) - FAILING

- KPI 213: Buyback 1 Month
- KPI 214: Buyback 3 Months
- KPI 215: Buyback 1 Year

**Why:** Buyback data comes from `/holdings/buyback` endpoint (which we already fetch!)

**Alternative:** We already fetch `holdings_buyback.json` separately.

---

### Quality Scores (4 KPIs) - FAILING
These are composite/calculated metrics:

- KPI 163: Magic Formula
- KPI 164: Graham Strategy
- KPI 167: F-Score (Piotroski)
- KPI 174: Earnings Stability

**Why:** These might be premium features or require special calculation.

**Alternative:** Could be calculated from other KPIs we already have.

---

### Stability Metrics (2 KPIs) - FAILING

- KPI 174: Earnings Stability (duplicate from above)
- KPI 178: Cash Flow Stability

**Why:** These might be premium features.

---

## Working KPIs (52 Total)

### All Tier 1 KPIs Work! ✅ (35 KPIs)

**Valuation (8):** Dividend Yield, P/E, P/S, P/B, EV/EBIT, EV/EBITDA, EV/FCF, EV/S

**Profitability (6):** Gross Margin, Operating Margin, Profit Margin, FCF Margin, EBITDA Margin, OCF Margin

**Growth (6):** Revenue Growth, EBIT Growth, Earnings Growth, Dividend Growth, Book Value Growth, Assets Growth

**Returns (4):** ROE, ROA, ROC, ROIC

**Financial Health (6):** Equity Ratio, Debt/Equity, Net Debt %, Net Debt/EBITDA, Current Ratio, Cash %

**Size/Absolute (5):** Enterprise Value, Market Cap, Revenue, Earnings, Free Cash Flow

### Tier 2 - Partial Success (11/25 working)

**Working:**
- Per-Share Metrics (6/6): Revenue/Share, EPS, Dividend/Share, Book Value/Share, FCF/Share, OCF/Share ✅
- Cash Flow (4/4): FCF Margin %, Earnings/FCF, OCF, Capex ✅
- Quality Scores (0/3): F-Score ❌, Magic Formula ❌, Earnings Stability ❌
- Technical (0/5): All failing ❌
- Insider/Ownership (0/7): All failing ❌

**Actually working:** 10 out of 25

### Tier 3 - Partial Success (7/20 working)

**Working:**
- Additional Valuation (2/2): PEG, Dividend Payout % ✅
- Absolute Metrics (5/5): EBITDA, Total Assets, Total Equity, Net Debt, Shares ✅
- Technical (0/4): All failing ❌
- Short Selling (0/4): All failing ❌
- Buybacks (0/3): All failing ❌
- Quality (0/2): Graham Strategy ❌, Cash Flow Stability ❌

**Actually working:** 7 out of 20

---

## Recommendations

### Option 1: Remove Failing KPIs (Conservative)

Remove the 28 failing KPIs from `fetch_and_store.py`. This gives us:

- **52 reliable KPIs** via `/kpis/{id}/year/mean/history` endpoint
- Clean data with no errors
- Still comprehensive fundamental coverage

### Option 2: Use Alternative Data Sources (Better)

Keep fetching the 52 working KPIs, and parse the alternative data we already fetch:

1. **Technical indicators:** Calculate from `stockprices_*.json` files
2. **Insider data:** Parse from `holdings_insider.json`
3. **Short selling:** Parse from `holdings_shorts.json`
4. **Buybacks:** Parse from `holdings_buyback.json`
5. **Quality scores:** Calculate from other KPIs or skip if premium-only

### Option 3: Investigate Different Endpoints

Some KPIs might work with different endpoints:
- `/instruments/{id}/kpis/{kpi_id}/latest` - Latest value
- `/instruments/{id}/kpis/{kpi_id}/last` - Last reported value
- `/instruments/{id}/kpis/{kpi_id}/mean` - Different aggregation

---

## Recommended Action

**Immediate:** Update `fetch_and_store.py` to only fetch the **52 working KPIs** to eliminate errors.

**Future:** Add data processing to extract insider/short selling/buyback data from the holdings endpoints we already fetch.

---

## Updated KPI List (52 Working KPIs)

```python
kpi_endpoints = [
    # Tier 1 - All 35 work
    # Valuation (8)
    (1, "dividend_yield", "Dividend Yield"),
    (2, "pe", "P/E"),
    (3, "ps", "P/S"),
    (4, "pb", "P/B"),
    (10, "ev_ebit", "EV/EBIT"),
    (11, "ev_ebitda", "EV/EBITDA"),
    (13, "ev_fcf", "EV/FCF"),
    (15, "ev_s", "EV/S"),

    # Profitability (6)
    (28, "gross_margin", "Gross Margin"),
    (29, "operating_margin", "Operating Margin"),
    (30, "profit_margin", "Profit Margin"),
    (31, "fcf_margin", "FCF Margin"),
    (32, "ebitda_margin", "EBITDA Margin"),
    (51, "ocf_margin", "OCF Margin"),

    # Growth (6)
    (94, "revenue_growth", "Revenue Growth"),
    (96, "ebit_growth", "EBIT Growth"),
    (97, "earnings_growth", "Earnings Growth"),
    (98, "dividend_growth", "Dividend Growth"),
    (99, "book_value_growth", "Book Value Growth"),
    (100, "assets_growth", "Assets Growth"),

    # Returns (4)
    (33, "roe", "Return on Equity"),
    (34, "roa", "Return on Assets"),
    (36, "roc", "Return on Capital"),
    (37, "roic", "Return on Invested Capital"),

    # Financial Health (6)
    (39, "equity_ratio", "Equity Ratio"),
    (40, "debt_to_equity", "Debt to Equity"),
    (41, "net_debt_pct", "Net Debt %"),
    (42, "net_debt_ebitda", "Net Debt/EBITDA"),
    (44, "current_ratio", "Current Ratio"),
    (46, "cash_pct", "Cash %"),

    # Size/Absolute (5)
    (49, "enterprise_value", "Enterprise Value"),
    (50, "market_cap", "Market Cap"),
    (53, "revenue", "Revenue"),
    (56, "earnings", "Earnings"),
    (63, "fcf", "Free Cash Flow"),

    # Tier 2 - Per-Share Metrics (6) ✅
    (5, "revenue_per_share", "Revenue per Share"),
    (6, "eps", "Earnings per Share"),
    (7, "dividend_per_share", "Dividend per Share"),
    (8, "book_value_per_share", "Book Value per Share"),
    (23, "fcf_per_share", "FCF per Share"),
    (68, "ocf_per_share", "OCF per Share"),

    # Tier 2 - Cash Flow (4) ✅
    (24, "fcf_margin_pct", "FCF Margin %"),
    (27, "earnings_fcf", "Earnings/FCF"),
    (62, "ocf", "Operating Cash Flow"),
    (64, "capex", "Capex"),

    # Tier 3 - Additional Valuation (2) ✅
    (19, "peg", "PEG Ratio"),
    (20, "dividend_payout", "Dividend Payout %"),

    # Tier 3 - Absolute Metrics (5) ✅
    (54, "ebitda", "EBITDA"),
    (57, "total_assets", "Total Assets"),
    (58, "total_equity", "Total Equity"),
    (60, "net_debt", "Net Debt"),
    (61, "num_shares", "Number of Shares"),
]
```

Total: **52 working KPIs** with 100% success rate
