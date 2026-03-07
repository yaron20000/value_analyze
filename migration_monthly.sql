-- Migration: Add monthly tables for monthly walk-forward model
-- Run this after schema_ml.sql (or as an incremental migration)

-- =============================================================================
-- 1. MONTHLY TARGETS - One row per (instrument, year, month)
-- =============================================================================
DROP TABLE IF EXISTS ml_monthly_targets CASCADE;

CREATE TABLE ml_monthly_targets (
    instrument_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,

    month_end_date DATE,                        -- last trading day of the month
    month_end_price NUMERIC(15,4),              -- closing price on that day

    next_month_return NUMERIC(10,4),            -- % return from this month-end to next month-end
    market_median_monthly_return NUMERIC(10,4), -- median of all stocks' next_month_return
    next_month_excess_return NUMERIC(10,4),     -- next_month_return - market_median

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instrument_id, year, month)
);

CREATE INDEX idx_ml_monthly_targets_instrument ON ml_monthly_targets(instrument_id);
CREATE INDEX idx_ml_monthly_targets_yearmonth ON ml_monthly_targets(year, month);

COMMENT ON TABLE ml_monthly_targets IS 'Monthly stock returns for monthly walk-forward model';
COMMENT ON COLUMN ml_monthly_targets.next_month_return IS 'Percentage return from end of month M to end of month M+1';

-- =============================================================================
-- 2. MONTHLY PRICE FEATURES - Technical features as of each month-end
-- =============================================================================
DROP TABLE IF EXISTS ml_monthly_price_features CASCADE;

CREATE TABLE ml_monthly_price_features (
    instrument_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,

    month_end_date DATE,                     -- last trading day of the month

    -- Price momentum (% change relative to month-end)
    price_change_5d NUMERIC(10,4),
    price_change_10d NUMERIC(10,4),
    price_change_20d NUMERIC(10,4),
    price_change_30d NUMERIC(10,4),

    -- Volume
    volume_ratio_5d_20d NUMERIC(10,4),

    -- Volatility
    volatility_5d NUMERIC(10,6),
    volatility_20d NUMERIC(10,6),

    -- Trend
    was_rising_5d BOOLEAN,
    was_rising_10d BOOLEAN,
    was_rising_20d BOOLEAN,

    -- Relative position
    pct_from_20d_high NUMERIC(10,4),
    pct_from_20d_low NUMERIC(10,4),

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    PRIMARY KEY (instrument_id, year, month)
);

CREATE INDEX idx_ml_monthly_price_instrument ON ml_monthly_price_features(instrument_id);
CREATE INDEX idx_ml_monthly_price_yearmonth ON ml_monthly_price_features(year, month);

COMMENT ON TABLE ml_monthly_price_features IS 'Price-based features computed as of each month-end for monthly model';
