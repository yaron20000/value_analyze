# KPI Analysis for ML System

## Currently Fetched KPIs (4 out of 322)

The script currently fetches only **4 KPIs** per instrument:

| KPI ID | Name | Category | Why It's Fetched |
|--------|------|----------|------------------|
| 2 | P/E (Price/Earnings) | Valuation | Classic valuation metric |
| 3 | P/S (Price/Sales) | Valuation | Revenue-based valuation |
| 4 | P/B (Price/Book) | Valuation | Book value metric |
| 33 | ROE (Return on Equity) | Profitability | Profitability measure |

**Note:** The code comment at line 388 says KPI 31 (FCF margin), but it's actually fetching KPI 33 (ROE).

## Critical Missing KPIs for ML Systems

### 1. **Fundamental Valuation Metrics** (High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 1 | Dividend Yield | Income investing, stability indicator |
| 10 | EV/EBIT | Enterprise value-based valuation |
| 11 | EV/EBITDA | Industry-standard valuation multiple |
| 13 | EV/FCF | Cash flow-based valuation |
| 15 | EV/S | Sales-based enterprise valuation |
| 74 | P/EBITDA | Alternative earnings metric |
| 75 | P/EBIT | Operating income valuation |
| 76 | P/FCF | Free cash flow valuation |

**ML Value:** Multiple valuation ratios allow the model to learn which metrics are most predictive for different sectors and market conditions.

### 2. **Profitability & Margins** (High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 28 | Gross Margin | Revenue quality indicator |
| 29 | Operating Margin | Operating efficiency |
| 30 | Profit Margin | Bottom-line profitability |
| 31 | FCF Margin | Cash generation efficiency |
| 32 | EBITDA Margin | Operating performance |
| 51 | OCF Margin | Operating cash flow efficiency |

**ML Value:** Margins are leading indicators of competitive advantage and pricing power. Critical for quality assessment.

### 3. **Growth Metrics** (High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 94 | Revenue Growth | Top-line expansion |
| 96 | EBIT Growth | Operating income growth |
| 97 | Earnings Growth | Bottom-line growth |
| 98 | Dividend Growth | Income growth trajectory |
| 99 | Book Value Growth | Equity accumulation |
| 100 | Assets Growth | Company expansion |

**ML Value:** Growth trends are fundamental to momentum and growth investing strategies. Essential for time-series forecasting.

### 4. **Return Metrics** (Medium-High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 34 | ROA (Return on Assets) | Asset efficiency |
| 36 | ROC (Return on Capital) | Capital allocation quality |
| 37 | ROIC (Return on Invested Capital) | Investment efficiency |
| 38 | Assets Turnover | Asset utilization |

**ML Value:** Return metrics indicate management quality and capital allocation efficiency. Key for quality factor models.

### 5. **Financial Health & Leverage** (High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 39 | Equity Ratio (Soliditet) | Financial stability |
| 40 | Debt to Equity | Leverage level |
| 41 | Net Debt | Absolute debt position |
| 42 | Net Debt/EBITDA | Debt serviceability |
| 44 | Current Ratio | Short-term liquidity |
| 46 | Cash-% | Cash position |

**ML Value:** Financial distress prediction requires leverage and liquidity metrics. Critical for risk assessment.

### 6. **Cash Flow Metrics** (High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 23 | FCF/Share | Per-share free cash flow |
| 24 | FCF Margin % | FCF efficiency |
| 27 | Earnings/FCF | Earnings quality |
| 62 | Operating Cash Flow (absolute) | Cash generation |
| 63 | Free Cash Flow (absolute) | Available cash |
| 68 | Operating Cash Flow/Share | Per-share OCF |

**ML Value:** Cash flow is harder to manipulate than earnings. Essential for quality and fraud detection models.

### 7. **Absolute Size Metrics** (Medium Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 50 | Market Cap | Company size |
| 49 | Enterprise Value | Total firm value |
| 53 | Revenue | Sales volume |
| 54 | EBITDA | Operating profit |
| 56 | Earnings | Net income |
| 57 | Total Assets | Asset base |
| 58 | Total Equity | Shareholder equity |
| 61 | Number of Shares | Share count |

**ML Value:** Size factors are fundamental to factor models. Market cap especially important for portfolio construction.

### 8. **Per-Share Metrics** (Medium Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 5 | Revenue/Share | Per-share sales |
| 6 | Earnings/Share | EPS |
| 7 | Dividend/Share | Dividend payout |
| 8 | Book Value/Share | BVPS |
| 70 | EBIT/Share | Operating income per share |
| 71 | EBITDA/Share | EBITDA per share |

**ML Value:** Per-share metrics normalize for company size and are used in relative valuations.

### 9. **Technical/Momentum Indicators** (Medium Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 151 | Performance (Price Change) | Momentum |
| 152 | Total Return | Total shareholder return |
| 159 | RSI | Overbought/oversold |
| 158 | MA(50)/MA(200) | Golden/death cross |
| 157 | MA200 Rank | Long-term trend position |
| 311 | Volatility H-L % | Price volatility |
| 312 | Volatility Std Dev | Statistical volatility |
| 313 | Volume | Trading activity |

**ML Value:** Technical indicators capture market sentiment and momentum. Important for trading strategies.

### 10. **Insider & Ownership Data** (Medium Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 229-232 | Insider Buy (1w, 1m, 3m, 12m) | Management confidence |
| 233-236 | Insider Sell (1w, 1m, 3m, 12m) | Management concerns |
| 237-240 | Insider Net (1w, 1m, 3m, 12m) | Net insider sentiment |
| 241-243 | Top 3 shareholders capital % | Ownership concentration |

**ML Value:** Insider trading is a strong signal. Ownership structure affects corporate governance and volatility.

### 11. **Short Selling & Buybacks** (Medium Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 207-210 | Avg Short-Selling (1w, 1m, 3m, 1y) | Market bearishness |
| 211-212 | Short-Selling Change (1w, 1m) | Sentiment shifts |
| 213-215 | Buyback (1m, 3m, 1y) | Share repurchases |

**ML Value:** Short interest predicts negative returns. Buybacks signal management confidence and support prices.

### 12. **Quality Scores** (High Priority)

| KPI ID | Name | Why Critical for ML |
|--------|------|---------------------|
| 167 | F-Score (Piotroski) | Financial strength composite |
| 163 | Magic Formula | Greenblatt's quality+value |
| 164 | Graham Strategy | Value investing score |
| 174 | Earnings Stability | Earnings consistency |
| 178 | Cash Flow Stability | CF consistency |

**ML Value:** Composite scores aggregate multiple signals. F-Score especially proven in academic literature.

## Recommended KPI Set for ML System

### Tier 1 - Essential (35 KPIs) - **Fetch These First**

**Valuation (8):**
- 1 (Dividend Yield), 10 (EV/EBIT), 11 (EV/EBITDA), 13 (EV/FCF), 15 (EV/S)
- 74 (P/EBITDA), 75 (P/EBIT), 76 (P/FCF)

**Profitability & Margins (6):**
- 28 (Gross Margin), 29 (Operating Margin), 30 (Profit Margin)
- 31 (FCF Margin), 32 (EBITDA Margin), 51 (OCF Margin)

**Growth (6):**
- 94 (Revenue Growth), 96 (EBIT Growth), 97 (Earnings Growth)
- 98 (Dividend Growth), 99 (Book Value Growth), 100 (Assets Growth)

**Returns (4):**
- 34 (ROA), 36 (ROC), 37 (ROIC), 38 (Assets Turnover)

**Financial Health (6):**
- 39 (Equity Ratio), 40 (Debt/Equity), 41 (Net Debt)
- 42 (Net Debt/EBITDA), 44 (Current Ratio), 46 (Cash-%)

**Size/Absolute (5):**
- 50 (Market Cap), 49 (Enterprise Value), 53 (Revenue)
- 56 (Earnings), 63 (Free Cash Flow)

### Tier 2 - Important (25 KPIs) - **Add These Next**

**Per-Share Metrics (6):**
- 5 (Revenue/Share), 6 (EPS), 7 (Dividend/Share)
- 8 (Book Value/Share), 23 (FCF/Share), 68 (OCF/Share)

**Cash Flow (4):**
- 24 (FCF Margin%), 27 (Earnings/FCF), 62 (OCF), 64 (Capex)

**Quality Scores (3):**
- 167 (F-Score), 163 (Magic Formula), 174 (Earnings Stability)

**Technical (5):**
- 151 (Performance), 152 (Total Return), 159 (RSI)
- 311 (Volatility H-L), 313 (Volume)

**Insider/Ownership (7):**
- 237-240 (Insider Net 1w, 1m, 3m, 12m)
- 241-243 (Top 3 shareholders)

### Tier 3 - Nice to Have (20 KPIs) - **Add for Completeness**

**Additional Valuation:**
- 19 (PEG), 20 (Dividend Payout)

**Additional Absolute:**
- 54 (EBITDA), 57 (Total Assets), 58 (Total Equity), 60 (Net Debt), 61 (Shares)

**Additional Technical:**
- 157 (MA200 Rank), 158 (MA50/MA200), 312 (Vol Std Dev), 314 (Volume Trend)

**Short Selling:**
- 207-210 (Avg Short-Selling 1w, 1m, 3m, 1y)

**Buybacks:**
- 213-215 (Buyback 1m, 3m, 1y)

**Other Quality:**
- 164 (Graham Strategy), 178 (Cash Flow Stability)

## Implementation Priority

### Phase 1: Core Fundamentals (35 KPIs)
```python
# Add to fetch_stock_data() in fetch_and_store.py
tier1_kpis = [
    # Valuation
    (1, "dividend_yield"), (10, "ev_ebit"), (11, "ev_ebitda"),
    (13, "ev_fcf"), (15, "ev_s"), (74, "p_ebitda"),
    (75, "p_ebit"), (76, "p_fcf"),

    # Profitability
    (28, "gross_margin"), (29, "operating_margin"), (30, "profit_margin"),
    (31, "fcf_margin"), (32, "ebitda_margin"), (51, "ocf_margin"),

    # Growth
    (94, "revenue_growth"), (96, "ebit_growth"), (97, "earnings_growth"),
    (98, "dividend_growth"), (99, "book_value_growth"), (100, "assets_growth"),

    # Returns
    (34, "roa"), (36, "roc"), (37, "roic"), (38, "asset_turnover"),

    # Financial Health
    (39, "equity_ratio"), (40, "debt_to_equity"), (41, "net_debt"),
    (42, "net_debt_ebitda"), (44, "current_ratio"), (46, "cash_pct"),

    # Size
    (50, "market_cap"), (49, "enterprise_value"), (53, "revenue"),
    (56, "earnings"), (63, "fcf")
]
```

### Phase 2: Enhanced Metrics (25 KPIs)
Add tier 2 KPIs for better model performance.

### Phase 3: Complete Dataset (20 KPIs)
Add tier 3 for comprehensive coverage.

## Why This Matters for ML

### 1. **Feature Engineering**
- Current: 4 KPIs = very limited feature space
- Recommended: 80 KPIs = rich feature space for learning patterns
- ML models need diverse, non-correlated features

### 2. **Factor Models**
Modern factor investing uses:
- **Value**: P/E, P/B, EV/EBITDA, P/FCF (need more than P/E, P/S, P/B)
- **Quality**: ROE, ROIC, Margins, Earnings Stability (only have ROE)
- **Growth**: Revenue growth, Earnings growth (missing)
- **Momentum**: Price performance, RSI (missing)
- **Size**: Market cap (missing)

### 3. **Time Series Analysis**
- Growth rates show trends
- Margins show operating leverage
- Multiple valuation metrics reduce noise

### 4. **Ensemble Learning**
Different KPIs work for different:
- Market conditions (bull/bear)
- Sectors (tech vs banks vs real estate)
- Company sizes (large vs small cap)

### 5. **Risk Management**
Financial health metrics essential for:
- Bankruptcy prediction
- Drawdown management
- Portfolio construction constraints

## Sector-Specific KPIs (Currently Missing)

### Real Estate Companies (KPI 277-289)
- 277 (Substansvärde), 278 (Räntetäckningsgrad), 280 (Uthyrningsgrad)
- **Why:** Real estate has unique metrics; using general KPIs misses key factors

### Banks (KPI 290-296)
- 290 (K/I-Tal), 291 (Kreditförluster), 292 (Kärnprimärkapital)
- **Why:** Bank profitability and risk differ fundamentally from other sectors

### Holding Companies (KPI 274-276)
- 274 (Substansvärde), 275 (Substansrabatt)
- **Why:** Holding companies trade at discounts to NAV - need these for valuation

## Data Quality Considerations

### Survivor Bias
- Need historical KPIs for delisted/failed companies
- Important for realistic backtesting

### Reporting Frequency
- Some KPIs update quarterly, others daily
- Need to handle mixed frequencies in ML pipeline

### Missing Data
- Not all companies report all KPIs
- ML model must handle sparse matrices
- Consider imputation strategies

## Estimated API Call Impact

### Current State
- 4 KPI endpoints × N instruments = 4N calls

### Tier 1 Implementation
- 35 KPI endpoints × N instruments = 35N calls
- For 100 instruments: 400 → 3,500 calls
- **Mitigation:** Use the duplicate prevention to avoid re-fetching

### Full Implementation
- 80 KPI endpoints × N instruments = 80N calls
- For 100 instruments: 8,000 calls
- **Mitigation:**
  - Fetch incrementally (Tier 1, then 2, then 3)
  - Skip existing data (24-hour window)
  - Use batch endpoints where available

## Conclusion

**Current Coverage: 1.2%** (4 out of 322 KPIs)

**Recommended for ML:**
- **Minimum viable:** 35 KPIs (Tier 1) = 11% coverage
- **Good coverage:** 60 KPIs (Tier 1 + 2) = 19% coverage
- **Comprehensive:** 80 KPIs (All tiers) = 25% coverage

**Next Steps:**
1. Implement Tier 1 KPIs first (35 KPIs)
2. Test ML model performance improvement
3. Add Tier 2 based on feature importance analysis
4. Consider sector-specific KPIs for specialized models
