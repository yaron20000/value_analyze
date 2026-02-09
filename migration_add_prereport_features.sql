-- Migration: Add pre-report price features table
-- Run this to add the new table to an existing database

-- Create the new table
CREATE TABLE IF NOT EXISTS ml_pre_report_features (
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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ml_prereport_instrument ON ml_pre_report_features(instrument_id);
CREATE INDEX IF NOT EXISTS idx_ml_prereport_year ON ml_pre_report_features(report_year);
CREATE INDEX IF NOT EXISTS idx_ml_prereport_date ON ml_pre_report_features(report_date);

-- Update the training data view to include pre-report features
DROP VIEW IF EXISTS ml_training_data;

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

-- Add comments
COMMENT ON TABLE ml_pre_report_features IS 'Price behavior before earnings reports - for detecting anticipation patterns';
COMMENT ON COLUMN ml_pre_report_features.was_rising_5d IS 'Key feature: was stock rising before report (anticipation signal)';
