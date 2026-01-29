-- Migration: Add unique constraint to prevent duplicate data
-- Run this on existing databases to add the duplicate prevention feature
--
-- Usage:
--   psql -U postgres -d borsdata -f migration_add_unique_constraint.sql

-- Add unique constraint to prevent duplicate data for same endpoint/instrument on same day
-- Uses a partial unique index that handles NULL instrument_id properly
-- This will fail if there are existing duplicates - clean them up first if needed
CREATE UNIQUE INDEX IF NOT EXISTS idx_api_raw_data_unique_daily ON api_raw_data(
    endpoint_name,
    COALESCE(instrument_id, -1),  -- Handle NULL instrument_id
    DATE(fetch_timestamp)
) WHERE success = true;

-- Verify the index was created
\d api_raw_data

-- Optional: Find existing duplicates (run before creating the index)
-- Uncomment to check for duplicates:
-- SELECT
--     endpoint_name,
--     COALESCE(instrument_id, -1) as inst_id,
--     DATE(fetch_timestamp) as fetch_date,
--     COUNT(*) as duplicate_count
-- FROM api_raw_data
-- WHERE success = true
-- GROUP BY endpoint_name, COALESCE(instrument_id, -1), DATE(fetch_timestamp)
-- HAVING COUNT(*) > 1
-- ORDER BY duplicate_count DESC;

-- Optional: Delete duplicate rows keeping only the most recent (run if you have duplicates)
-- Uncomment and run if the index creation fails due to duplicates:
-- DELETE FROM api_raw_data a
-- USING (
--     SELECT
--         endpoint_name,
--         instrument_id,
--         DATE(fetch_timestamp) as fetch_date,
--         MAX(id) as keep_id
--     FROM api_raw_data
--     WHERE success = true
--     GROUP BY endpoint_name, instrument_id, DATE(fetch_timestamp)
--     HAVING COUNT(*) > 1
-- ) b
-- WHERE a.endpoint_name = b.endpoint_name
-- AND (a.instrument_id = b.instrument_id OR (a.instrument_id IS NULL AND b.instrument_id IS NULL))
-- AND DATE(a.fetch_timestamp) = b.fetch_date
-- AND a.id != b.keep_id
-- AND a.success = true;
