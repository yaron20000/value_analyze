-- Migration script to add kpi_name column to existing api_raw_data table
-- Run this if you have an existing database without the kpi_name column
-- Usage: psql -d borsdata -f migration_add_kpi_name.sql

-- Add kpi_name column to api_raw_data table
ALTER TABLE api_raw_data ADD COLUMN IF NOT EXISTS kpi_name VARCHAR(100);

-- Add comment for documentation
COMMENT ON COLUMN api_raw_data.kpi_name IS 'KPI name for KPI endpoints (e.g., P/E, ROE, Revenue Growth), NULL for non-KPI endpoints';

-- Create an index for better query performance
CREATE INDEX IF NOT EXISTS idx_api_raw_data_kpi_name ON api_raw_data(kpi_name);

-- Print confirmation message
DO $$
BEGIN
    RAISE NOTICE 'Migration complete: Added kpi_name column to api_raw_data table';
END $$;
