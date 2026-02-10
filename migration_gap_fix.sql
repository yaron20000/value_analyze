-- Migration: Add 17 missing KPI columns to ml_features + widen existing narrow columns
-- =====================================================================================
-- These KPIs are fetched from Borsdata but were not mapped into ml_features.
-- Uses ADD COLUMN IF NOT EXISTS to preserve existing data.
-- Also widens NUMERIC(5,2) columns to NUMERIC(10,2) to handle extreme values
-- (e.g., growth rates >1000%, negative ratios, etc.)

-- Drop the view first (it depends on ml_features columns and blocks ALTER TYPE)
DROP VIEW IF EXISTS ml_training_data;

-- Valuation additions
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS dividend_yield NUMERIC(10,4);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS ev_ebit NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS ev_fcf NUMERIC(10,2);

-- Profitability/Return additions
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS roc NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS roic NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS fcf_margin_pct NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS ocf_margin NUMERIC(10,2);

-- Financial Health additions
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS net_debt_pct NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS net_debt_ebitda NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS cash_pct NUMERIC(10,2);

-- Per-Share additions
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS revenue_per_share NUMERIC(10,4);

-- Size/Absolute additions
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS enterprise_value NUMERIC(15,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS market_cap NUMERIC(15,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS fcf NUMERIC(15,2);

-- Growth additions
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS ebit_growth NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS book_value_growth NUMERIC(10,2);
ALTER TABLE ml_features ADD COLUMN IF NOT EXISTS assets_growth NUMERIC(10,2);

-- Widen existing NUMERIC(5,2) columns that can overflow with real-world data
ALTER TABLE ml_features ALTER COLUMN roe TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN roa TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN ebitda_margin TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN operating_margin TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN gross_margin TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN net_margin TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN equity_ratio TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN current_ratio TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN revenue_growth TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN earnings_growth TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN dividend_growth TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN fcf_margin TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN earnings_fcf TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN dividend_payout TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN roi TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN quick_ratio TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN interest_coverage TYPE NUMERIC(10,2);

-- Also widen new columns in case they were created with wrong types by a previous migration run
ALTER TABLE ml_features ALTER COLUMN roc TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN roic TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN fcf_margin_pct TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN ocf_margin TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN net_debt_pct TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN net_debt_ebitda TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN cash_pct TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN ev_ebit TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN ev_fcf TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN ebit_growth TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN book_value_growth TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN assets_growth TYPE NUMERIC(10,2);
ALTER TABLE ml_features ALTER COLUMN dividend_yield TYPE NUMERIC(10,4);
ALTER TABLE ml_features ALTER COLUMN revenue_per_share TYPE NUMERIC(10,4);
ALTER TABLE ml_features ALTER COLUMN enterprise_value TYPE NUMERIC(15,2);
ALTER TABLE ml_features ALTER COLUMN market_cap TYPE NUMERIC(15,2);
ALTER TABLE ml_features ALTER COLUMN fcf TYPE NUMERIC(15,2);

-- Mark orphan columns that have no Borsdata KPI source
COMMENT ON COLUMN ml_features.roi IS 'UNUSED - No Borsdata KPI maps to this column';
COMMENT ON COLUMN ml_features.quick_ratio IS 'UNUSED - No Borsdata KPI maps to this column';
COMMENT ON COLUMN ml_features.interest_coverage IS 'UNUSED - No Borsdata KPI maps to this column';

-- Recreate the training data view
CREATE VIEW ml_training_data AS
SELECT
    f.*,
    t.next_year_return,
    t.next_3year_return,
    t.next_5year_return,
    t.next_year_excess_return,
    t.market_median_return,
    t.next_year_dividend_growth,
    t.next_3year_avg_dividend_growth,
    t.next_5year_avg_dividend_growth,
    t.next_year_outperformed,
    t.dividend_increased,
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
    pr.pct_from_20d_low,
    -- Holdings features
    h.insider_net_shares,
    h.insider_net_amount,
    h.insider_buy_count,
    h.insider_sell_count,
    h.insider_transaction_count,
    h.insider_buy_ratio,
    h.buyback_total_shares,
    h.buyback_total_amount,
    h.buyback_count,
    h.buyback_shares_pct
FROM ml_features f
LEFT JOIN ml_targets t ON f.instrument_id = t.instrument_id AND f.year = t.year
LEFT JOIN ml_pre_report_features pr ON f.instrument_id = pr.instrument_id
    AND f.year = pr.report_year AND f.period = pr.report_period
LEFT JOIN ml_holdings_features h ON f.instrument_id = h.instrument_id AND f.year = h.year
WHERE f.period = 5
ORDER BY f.instrument_id, f.year;
