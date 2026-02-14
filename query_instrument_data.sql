-- ============================================================================
-- Show all raw data we have for each instrument (one row per instrument)
-- Combines: stock prices, financial reports, and all 52 KPIs
-- Uses DISTINCT ON to keep only the most recent fetch per instrument
-- ============================================================================

WITH instrument_info AS (
    -- Get instrument names from the most recent instruments metadata fetch
    SELECT DISTINCT ON ((inst->>'insId')::integer)
        (inst->>'insId')::integer AS instrument_id,
        inst->>'name' AS company_name,
        inst->>'ticker' AS ticker
    FROM api_raw_data,
         jsonb_array_elements(raw_data->'instruments') AS inst
    WHERE endpoint_name = 'instruments'
      AND success = true
    ORDER BY (inst->>'insId')::integer, fetch_timestamp DESC
),

-- Stock prices: most recent fetch per instrument
stock_prices AS (
    SELECT DISTINCT ON (instrument_id)
        instrument_id,
        jsonb_array_length(raw_data->'stockPricesList') AS price_days,
        (SELECT MIN(el->>'d') FROM jsonb_array_elements(raw_data->'stockPricesList') el) AS price_from,
        (SELECT MAX(el->>'d') FROM jsonb_array_elements(raw_data->'stockPricesList') el) AS price_to,
        fetch_timestamp AS prices_fetched_at
    FROM api_raw_data
    WHERE endpoint_name LIKE 'stockprices_%'
      AND endpoint_name NOT LIKE '%array%'
      AND endpoint_name NOT LIKE '%last%'
      AND endpoint_name NOT LIKE '%date%'
      AND instrument_id IS NOT NULL
      AND success = true
    ORDER BY instrument_id, fetch_timestamp DESC
),

-- Yearly reports: most recent fetch per instrument
reports_year AS (
    SELECT DISTINCT ON (instrument_id)
        instrument_id,
        jsonb_array_length(raw_data->'reports') AS yearly_reports_count,
        (SELECT MIN(el->>'year') FROM jsonb_array_elements(raw_data->'reports') el) AS year_from,
        (SELECT MAX(el->>'year') FROM jsonb_array_elements(raw_data->'reports') el) AS year_to,
        fetch_timestamp AS yearly_fetched_at
    FROM api_raw_data
    WHERE endpoint_name LIKE 'reports_year_%'
      AND instrument_id IS NOT NULL
      AND success = true
    ORDER BY instrument_id, fetch_timestamp DESC
),

-- R12 reports: most recent fetch per instrument
reports_r12 AS (
    SELECT DISTINCT ON (instrument_id)
        instrument_id,
        jsonb_array_length(raw_data->'reports') AS r12_reports_count
    FROM api_raw_data
    WHERE endpoint_name LIKE 'reports_r12_%'
      AND instrument_id IS NOT NULL
      AND success = true
    ORDER BY instrument_id, fetch_timestamp DESC
),

-- Quarterly reports: most recent fetch per instrument
reports_quarter AS (
    SELECT DISTINCT ON (instrument_id)
        instrument_id,
        jsonb_array_length(raw_data->'reports') AS quarter_reports_count
    FROM api_raw_data
    WHERE endpoint_name LIKE 'reports_quarter_%'
      AND instrument_id IS NOT NULL
      AND success = true
    ORDER BY instrument_id, fetch_timestamp DESC
),

-- KPI data per instrument (from batch KPI endpoints), aggregated to one row
kpi_summary AS (
    SELECT
        (inst_data->>'instrument')::integer AS instrument_id,
        COUNT(DISTINCT r.kpi_name) AS kpis_available,
        SUM(jsonb_array_length(inst_data->'values')) AS total_kpi_data_points,
        MIN((SELECT MIN(v->>'y') FROM jsonb_array_elements(inst_data->'values') v)) AS kpi_year_from,
        MAX((SELECT MAX(v->>'y') FROM jsonb_array_elements(inst_data->'values') v)) AS kpi_year_to,
        string_agg(DISTINCT r.kpi_name, ', ' ORDER BY r.kpi_name) AS kpi_names
    FROM api_raw_data r,
         jsonb_array_elements(r.raw_data->'kpisList') AS inst_data
    WHERE r.endpoint_name LIKE 'kpi_%_batch'
      AND r.success = true
    GROUP BY (inst_data->>'instrument')::integer
)

-- Final: one row per instrument with all available data
SELECT
    COALESCE(ii.instrument_id, sp.instrument_id, ry.instrument_id, ks.instrument_id) AS instrument_id,
    ii.company_name,
    ii.ticker,

    -- Stock price coverage
    sp.price_days,
    sp.price_from,
    sp.price_to,
    sp.prices_fetched_at,

    -- Report coverage
    ry.yearly_reports_count,
    ry.year_from AS report_year_from,
    ry.year_to AS report_year_to,
    rr.r12_reports_count,
    rq.quarter_reports_count,

    -- KPI coverage
    ks.kpis_available,
    ks.total_kpi_data_points,
    ks.kpi_year_from,
    ks.kpi_year_to,
    ks.kpi_names

FROM instrument_info ii
FULL OUTER JOIN stock_prices sp USING (instrument_id)
FULL OUTER JOIN reports_year ry USING (instrument_id)
FULL OUTER JOIN reports_r12 rr USING (instrument_id)
FULL OUTER JOIN reports_quarter rq USING (instrument_id)
FULL OUTER JOIN kpi_summary ks USING (instrument_id)

ORDER BY company_name NULLS LAST, instrument_id;
