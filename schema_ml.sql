-- ML-Optimized Schema for Stock Prediction & Dividend Growth
-- ============================================================
-- Designed for time-series machine learning tasks
-- Each table is optimized for direct loading into pandas/scikit-learn

-- Drop ML tables if they exist
DROP TABLE IF EXISTS ml_features CASCADE;
DROP TABLE IF EXISTS ml_stock_prices CASCADE;
DROP TABLE IF EXISTS ml_targets CASCADE;
DROP VIEW IF EXISTS ml_training_data CASCADE;

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
    open NUMERIC(10,2),
    high NUMERIC(10,2),
    low NUMERIC(10,2),
    close NUMERIC(10,2),
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
-- 3. TARGET VARIABLES TABLE - What we're trying to predict
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
-- 4. TRAINING DATA VIEW - Ready for ML (join features + targets)
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
    t.dividend_increased
FROM ml_features f
LEFT JOIN ml_targets t ON f.instrument_id = t.instrument_id AND f.year = t.year
WHERE f.period = 5  -- Only use full-year data for training
ORDER BY f.instrument_id, f.year;

-- =============================================================================
-- COMMENTS
-- =============================================================================
COMMENT ON TABLE ml_features IS 'Flattened feature matrix for ML - one row per (stock, year)';
COMMENT ON TABLE ml_stock_prices IS 'Daily stock prices for technical features';
COMMENT ON TABLE ml_targets IS 'Target variables for prediction (future returns, dividend growth)';
COMMENT ON VIEW ml_training_data IS 'Ready-to-use training data: features + targets joined';

COMMENT ON COLUMN ml_features.period IS '3=Q1, 5=full year - filter on 5 for annual models';
COMMENT ON COLUMN ml_targets.next_year_return IS 'Stock return from year Y to year Y+1 (%)';
COMMENT ON COLUMN ml_targets.next_year_dividend_growth IS 'Dividend growth from year Y to year Y+1 (%)';
