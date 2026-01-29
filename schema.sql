-- PostgreSQL Schema for Borsdata API Raw Data Storage
-- This schema stores raw JSON data from various Borsdata API endpoints
-- for future AI processing and analysis

-- Drop tables if they exist (for development/reset)
DROP TABLE IF EXISTS api_raw_data CASCADE;
DROP TABLE IF EXISTS api_fetch_log CASCADE;

-- Main table for storing raw API responses
CREATE TABLE api_raw_data (
    id SERIAL PRIMARY KEY,
    endpoint_name VARCHAR(100) NOT NULL,
    endpoint_path VARCHAR(500) NOT NULL,
    instrument_id INTEGER,  -- NULL for metadata/global endpoints
    kpi_name VARCHAR(100),  -- KPI name for KPI endpoints, NULL for other endpoints
    fetch_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    raw_data JSONB,  -- Store the entire API response as JSONB for querying
    request_params JSONB,  -- Store request parameters
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table for tracking API fetch operations
CREATE TABLE api_fetch_log (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    total_endpoints INTEGER NOT NULL,
    successful_endpoints INTEGER NOT NULL,
    failed_endpoints INTEGER NOT NULL,
    instruments_fetched TEXT[],  -- Array of instrument IDs fetched
    duration_seconds NUMERIC(10, 2),
    notes TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Unique constraint to prevent duplicate data for same endpoint/instrument on same day
-- Uses a partial unique index that handles NULL instrument_id properly
CREATE UNIQUE INDEX idx_api_raw_data_unique_daily ON api_raw_data(
    endpoint_name,
    COALESCE(instrument_id, -1),  -- Handle NULL instrument_id
    DATE(fetch_timestamp)
) WHERE success = true;

-- Indexes for better query performance
CREATE INDEX idx_api_raw_data_endpoint ON api_raw_data(endpoint_name);
CREATE INDEX idx_api_raw_data_instrument ON api_raw_data(instrument_id);
CREATE INDEX idx_api_raw_data_timestamp ON api_raw_data(fetch_timestamp);
CREATE INDEX idx_api_raw_data_success ON api_raw_data(success);
CREATE INDEX idx_api_raw_data_jsonb ON api_raw_data USING GIN(raw_data);
CREATE INDEX idx_api_fetch_log_timestamp ON api_fetch_log(run_timestamp);

-- Comments for documentation
COMMENT ON TABLE api_raw_data IS 'Stores raw JSON responses from Borsdata API endpoints';
COMMENT ON TABLE api_fetch_log IS 'Logs each API data fetch operation for tracking and debugging';
COMMENT ON COLUMN api_raw_data.endpoint_name IS 'Friendly name for the endpoint (e.g., instruments, stockprices_single)';
COMMENT ON COLUMN api_raw_data.endpoint_path IS 'Actual API endpoint path (e.g., /instruments)';
COMMENT ON COLUMN api_raw_data.instrument_id IS 'Instrument ID for stock-specific endpoints, NULL for metadata endpoints';
COMMENT ON COLUMN api_raw_data.raw_data IS 'Complete JSON response from the API stored as JSONB';
