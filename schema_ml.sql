-- ML-Optimized Schema for Stock Prediction & Dividend Growth
-- ============================================================
-- Designed for time-series machine learning tasks
-- Each table is optimized for direct loading into pandas/scikit-learn

-- Drop ML tables if they exist
DROP VIEW IF EXISTS ml_training_data CASCADE;
DROP TABLE IF EXISTS ml_pre_report_features CASCADE;
DROP TABLE IF EXISTS ml_targets CASCADE;
DROP TABLE IF EXISTS ml_stock_prices CASCADE;
DROP TABLE IF EXISTS ml_features CASCADE;

-- =============================================================================
-- 1. ML FEATURES TABLE - One row per (instrument, year)
-- =============================================================================
CREATE TABLE ml_features (
    -- Primary identifiers
    instrument_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    period INTEGER NOT NULL,  -- Number of months reported (3=Q1, 5=full year)

    -- Metadata (for filtering/grouping)
    company_name VARCHAR(100),
    sector VARCHAR(100),
    market VARCHAR(50),
    country VARCHAR(50),

    -- Valuation KPIs (13 features)
    pe_ratio NUMERIC(10,2),           -- P/E ratio
    ps_ratio NUMERIC(10,2),           -- P/S ratio
    pb_ratio NUMERIC(10,2),           -- P/B ratio
    ev_ebitda NUMERIC(10,2),          -- EV/EBITDA
    peg_ratio NUMERIC(10,2),          -- PEG ratio
    ev_sales NUMERIC(10,2),           -- EV/Sales

    -- Profitability KPIs (11 features)
    roe NUMERIC(5,2),                 -- Return on Equity %
    roi NUMERIC(5,2),                 -- Return on Investment %
    roa NUMERIC(5,2),                 -- Return on Assets %
    ebitda_margin NUMERIC(5,2),       -- EBITDA Margin %
    operating_margin NUMERIC(5,2),    -- Operating Margin %
    gross_margin NUMERIC(5,2),        -- Gross Margin %
    net_margin NUMERIC(5,2),          -- Net Margin %

    -- Financial Health KPIs (12 features)
    debt_equity NUMERIC(10,2),        -- Debt/Equity ratio
    equity_ratio NUMERIC(5,2),        -- Equity Ratio %
    current_ratio NUMERIC(5,2),       -- Current Ratio
    quick_ratio NUMERIC(5,2),         -- Quick Ratio
    interest_coverage NUMERIC(10,2),  -- Interest Coverage

    -- Growth KPIs (4 features)
    revenue_growth NUMERIC(5,2),      -- Revenue Growth %
    earnings_growth NUMERIC(5,2),     -- Earnings Growth %
    dividend_growth NUMERIC(5,2),     -- Dividend Growth % (TARGET RELATED!)

    -- Per-Share Metrics (5 features)
    eps NUMERIC(10,4),                -- Earnings per Share
    dividend_per_share NUMERIC(10,4), -- Dividend per Share (TARGET RELATED!)
    book_value_per_share NUMERIC(10,4), -- Book Value per Share
    fcf_per_share NUMERIC(10,4),      -- Free Cash Flow per Share
    ocf_per_share NUMERIC(10,4),      -- Operating Cash Flow per Share

    -- Cash Flow KPIs (4 features)
    fcf_margin NUMERIC(5,2),          -- FCF Margin %
    earnings_fcf NUMERIC(5,2),        -- Earnings/FCF ratio
    ocf NUMERIC(15,2),                -- Operating Cash Flow
    capex NUMERIC(15,2),              -- Capital Expenditures

    -- Dividend KPIs (1 feature)
    dividend_payout NUMERIC(5,2),     -- Dividend Payout % (TARGET RELATED!)

    -- Absolute Metrics (5 features for company size/scale)
    earnings NUMERIC(15,2),           -- Net Earnings
    revenue NUMERIC(15,2),            -- Revenue
    ebitda NUMERIC(15,2),             -- EBITDA
    total_assets NUMERIC(15,2),       -- Total Assets
    total_equity NUMERIC(15,2),       -- Total Equity
    net_debt NUMERIC(15,2),           -- Net Debt
    num_shares NUMERIC(15,0),         -- Number of Shares Outstanding

    -- Metadata
    fetch_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instrument_id, year, period)
);

-- Indexes for ML queries
CREATE INDEX idx_ml_features_instrument ON ml_features(instrument_id);
CREATE INDEX idx_ml_features_year ON ml_features(year);
CREATE INDEX idx_ml_features_sector ON ml_features(sector);
CREATE INDEX idx_ml_features_market ON ml_features(market);
CREATE INDEX idx_ml_features_year_range ON ml_features(year DESC);

-- =============================================================================
-- 2. STOCK PRICES TABLE - Daily prices for technical features
-- =============================================================================
CREATE TABLE ml_stock_prices (
    instrument_id INTEGER NOT NULL,
    date DATE NOT NULL,

    -- OHLCV data
    open NUMERIC(15,4),
    high NUMERIC(15,4),
    low NUMERIC(15,4),
    close NUMERIC(15,4),
    volume BIGINT,

    -- Derived technical features (can be calculated)
    daily_return NUMERIC(10,6),       -- (close - prev_close) / prev_close

    -- Metadata
    fetch_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instrument_id, date)
);

CREATE INDEX idx_ml_prices_instrument ON ml_stock_prices(instrument_id);
CREATE INDEX idx_ml_prices_date ON ml_stock_prices(date DESC);

-- =============================================================================
-- 3. PRE-REPORT PRICE FEATURES - Stock behavior before earnings reports
-- =============================================================================
CREATE TABLE ml_pre_report_features (
    instrument_id INTEGER NOT NULL,
    report_year INTEGER NOT NULL,
    report_period INTEGER NOT NULL,      -- 1=Q1, 2=Q2, 3=Q3, 4=Q4, 5=full year
    report_date DATE NOT NULL,           -- When report was published

    -- Price changes before report (% change)
    price_change_5d NUMERIC(10,4),       -- Price change 5 trading days before report
    price_change_10d NUMERIC(10,4),      -- Price change 10 trading days before report
    price_change_20d NUMERIC(10,4),      -- Price change 20 trading days before report
    price_change_30d NUMERIC(10,4),      -- Price change 30 trading days before report

    -- Volume features
    avg_volume_5d NUMERIC(15,2),         -- Average volume 5 days before report
    avg_volume_20d NUMERIC(15,2),        -- Average volume 20 days before report
    volume_ratio_5d_20d NUMERIC(10,4),   -- Volume 5d / Volume 20d (spike indicator)

    -- Volatility features
    volatility_5d NUMERIC(10,6),         -- Std dev of daily returns, 5 days before
    volatility_20d NUMERIC(10,6),        -- Std dev of daily returns, 20 days before

    -- Trend indicators (boolean)
    was_rising_5d BOOLEAN,               -- Price higher than 5 days ago?
    was_rising_10d BOOLEAN,              -- Price higher than 10 days ago?
    was_rising_20d BOOLEAN,              -- Price higher than 20 days ago?

    -- Price relative to period high/low
    pct_from_20d_high NUMERIC(10,4),     -- % below 20-day high
    pct_from_20d_low NUMERIC(10,4),      -- % above 20-day low

    -- Price on report date
    price_at_report NUMERIC(15,4),       -- Closing price on report date
    price_5d_before NUMERIC(15,4),       -- Price 5 days before
    price_20d_before NUMERIC(15,4),      -- Price 20 days before

    -- Metadata
    calculated_date TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instrument_id, report_year, report_period)
);

CREATE INDEX idx_ml_prereport_instrument ON ml_pre_report_features(instrument_id);
CREATE INDEX idx_ml_prereport_year ON ml_pre_report_features(report_year);
CREATE INDEX idx_ml_prereport_date ON ml_pre_report_features(report_date);

COMMENT ON TABLE ml_pre_report_features IS 'Price behavior before earnings reports - for detecting anticipation patterns';
COMMENT ON COLUMN ml_pre_report_features.price_change_5d IS 'Percentage price change from 5 trading days before report to day before report';
COMMENT ON COLUMN ml_pre_report_features.volume_ratio_5d_20d IS 'Ratio of recent volume to longer-term average - detects volume spikes before reports';
COMMENT ON COLUMN ml_pre_report_features.was_rising_5d IS 'True if stock price was trending upward in 5 days before report';

-- =============================================================================
-- 4. TARGET VARIABLES TABLE - What we're trying to predict
-- =============================================================================
CREATE TABLE ml_targets (
    instrument_id INTEGER NOT NULL,
    year INTEGER NOT NULL,

    -- Stock price targets (for regression)
    next_year_return NUMERIC(10,4),   -- % return in next year
    next_3year_return NUMERIC(10,4),  -- % return in next 3 years
    next_5year_return NUMERIC(10,4),  -- % return in next 5 years

    -- Dividend growth targets (for your specific use case)
    next_year_dividend_growth NUMERIC(10,4),   -- % dividend growth next year
    next_3year_avg_dividend_growth NUMERIC(10,4), -- Avg % dividend growth over 3 years
    next_5year_avg_dividend_growth NUMERIC(10,4), -- Avg % dividend growth over 5 years

    -- Classification targets (optional)
    next_year_outperformed BOOLEAN,   -- Did it beat market average?
    dividend_increased BOOLEAN,       -- Did dividend increase?

    -- Metadata
    calculated_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instrument_id, year)
);

CREATE INDEX idx_ml_targets_instrument ON ml_targets(instrument_id);
CREATE INDEX idx_ml_targets_year ON ml_targets(year);

-- =============================================================================
-- 5. TRAINING DATA VIEW - Ready for ML (join features + targets + pre-report)
-- =============================================================================
CREATE VIEW ml_training_data AS
SELECT
    f.*,
    t.next_year_return,
    t.next_3year_return,
    t.next_5year_return,
    t.next_year_dividend_growth,
    t.next_3year_avg_dividend_growth,
    t.next_5year_avg_dividend_growth,
    t.next_year_outperformed,
    t.dividend_increased,
    -- Pre-report features
    pr.report_date,
    pr.price_change_5d,
    pr.price_change_10d,
    pr.price_change_20d,
    pr.price_change_30d,
    pr.volume_ratio_5d_20d,
    pr.volatility_5d,
    pr.volatility_20d,
    pr.was_rising_5d,
    pr.was_rising_10d,
    pr.was_rising_20d,
    pr.pct_from_20d_high,
    pr.pct_from_20d_low
FROM ml_features f
LEFT JOIN ml_targets t ON f.instrument_id = t.instrument_id AND f.year = t.year
LEFT JOIN ml_pre_report_features pr ON f.instrument_id = pr.instrument_id
    AND f.year = pr.report_year AND f.period = pr.report_period
WHERE f.period = 5  -- Only use full-year data for training
ORDER BY f.instrument_id, f.year;

-- =============================================================================
-- COMMENTS
-- =============================================================================
COMMENT ON TABLE ml_features IS 'Flattened feature matrix for ML - one row per (stock, year)';
COMMENT ON TABLE ml_stock_prices IS 'Daily stock prices for technical features';
COMMENT ON TABLE ml_pre_report_features IS 'Price behavior before earnings reports - detects anticipation patterns';
COMMENT ON TABLE ml_targets IS 'Target variables for prediction (future returns, dividend growth)';
COMMENT ON VIEW ml_training_data IS 'Ready-to-use training data: features + targets + pre-report features joined';

COMMENT ON COLUMN ml_features.period IS '3=Q1, 5=full year - filter on 5 for annual models';
COMMENT ON COLUMN ml_targets.next_year_return IS 'Stock return from year Y to year Y+1 (%)';
COMMENT ON COLUMN ml_targets.next_year_dividend_growth IS 'Dividend growth from year Y to year Y+1 (%)';
COMMENT ON COLUMN ml_pre_report_features.was_rising_5d IS 'Key feature: was stock rising before report (anticipation signal)';
