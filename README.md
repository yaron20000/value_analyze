# Borsdata API Data Fetcher

This project fetches data from the Borsdata API and stores it in PostgreSQL for future AI processing.

## Features

- ✅ Fetches all Instrument Meta APIs (metadata)
- ✅ Fetches stock-specific data for configurable stocks
- ✅ Saves JSON responses to `results/` directory for debugging
- ✅ Stores raw data in PostgreSQL as JSONB
- ✅ Handles API rate limiting
- ✅ Comprehensive error handling and logging

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup PostgreSQL Database

Create a PostgreSQL database:

```bash
createdb borsdata
```

Run the schema setup:

```bash
psql -d borsdata -f schema.sql
```

### 3. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
BORSDATA_API_KEY=your_api_key_here
DB_PASSWORD=your_password_here
```

## Usage

### JSON Files Only (No Database)

If you don't want to use PostgreSQL and only want to save JSON files:

```bash
python fetch_and_store.py YOUR_API_KEY --no-db
```

This will fetch all API data and save it to the `results/` directory without requiring database credentials.

### With Database (Basic Usage)

Using environment variables from .env file:

```bash
python fetch_and_store.py
```

Or with command line:

```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```

### Fetch Specific Instruments

```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --instruments "3,750,97"
```

Or without database:

```bash
python fetch_and_store.py YOUR_API_KEY --no-db --instruments "3,750,97"
```

Default instruments are:
- 3 = ABB
- 750 = Securitas

### All Available Options

```bash
python fetch_and_store.py --help
```

Options:
- `--no-db`: Skip database storage and only save JSON files
- `--db-host`: PostgreSQL host (default: localhost)
- `--db-name`: Database name (default: borsdata)
- `--db-user`: Database user (default: postgres)
- `--db-password`: Database password (required for DB storage)
- `--db-port`: Database port (default: 5432)
- `--instruments`: Comma-separated instrument IDs (default: 3,750)

**Note**: If `--db-password` is not provided and `--no-db` is not set, the script will automatically run in JSON-only mode with a warning.

## What Data is Fetched

### Instrument Meta APIs (Metadata)
- Instruments list
- Instruments updated
- Markets
- Branches
- Sectors
- Countries
- Translation metadata
- KPI metadata
- KPI metadata updated

### Stock-Specific APIs (per instrument)
- Stock prices (last 100)
- Reports (year, R12, quarter, all)
- KPIs (P/E, P/S, P/B, ROE)

### Array-Based APIs (multiple stocks)
- Stock prices for all specified instruments
- Reports for all specified instruments

### Global APIs
- Latest stock prices (all stocks)
- Stock prices by date
- Report calendar
- Dividend calendar
- Stock splits
- Holdings (insider, shorts, buyback)

## Database Schema

The data is stored in two main tables:

### `api_raw_data`
Stores the raw JSON responses from each API endpoint:
- `endpoint_name`: Friendly name (e.g., "instruments", "stockprices_3")
- `endpoint_path`: API path (e.g., "/instruments")
- `instrument_id`: Stock ID (NULL for metadata endpoints)
- `raw_data`: Complete JSON response as JSONB
- `success`: Whether the API call succeeded
- `error_message`: Error details if failed

### `api_fetch_log`
Tracks each complete fetch operation:
- `run_timestamp`: When the fetch started
- `total_endpoints`: Total API calls made
- `successful_endpoints`: Successful calls
- `failed_endpoints`: Failed calls
- `instruments_fetched`: Array of instrument IDs
- `duration_seconds`: Total execution time

## Output Files

All responses are also saved to `results/` directory as JSON files:
- `instruments.json`
- `stockprices_3.json` (for instrument 3)
- `reports_year_750.json` (for instrument 750)
- etc.

Each JSON file contains:
```json
{
  "timestamp": "2026-01-06T...",
  "endpoint_name": "instruments",
  "success": true,
  "error": null,
  "data": { ... }
}
```

## Fixed Issues in api_test.py

- ✅ Removed broken `kpi_screener` endpoint (returned 404)
  - This endpoint required specific KPI calculation types that vary
  - Use `kpi_history` for individual instruments instead

## Querying the Data

Example PostgreSQL queries:

```sql
-- Get all successful fetches
SELECT endpoint_name, fetch_timestamp, instrument_id
FROM api_raw_data
WHERE success = true
ORDER BY fetch_timestamp DESC;

-- Get stock prices for a specific instrument
SELECT raw_data->'stockPricesList'
FROM api_raw_data
WHERE endpoint_name LIKE 'stockprices_%'
  AND instrument_id = 3;

-- Get latest reports for all instruments
SELECT instrument_id, raw_data
FROM api_raw_data
WHERE endpoint_name LIKE 'reports_year_%'
  AND success = true;

-- Query JSONB data directly
SELECT raw_data->'instruments'->0->>'name' as company_name
FROM api_raw_data
WHERE endpoint_name = 'instruments';

-- Get fetch statistics
SELECT * FROM api_fetch_log
ORDER BY run_timestamp DESC;
```

## Notes

- API is rate limited to 100 calls per 10 seconds
- Some endpoints require Pro+ membership (Holdings)
- Default instruments: ABB (3) and Securitas (750)
- All timestamps are in ISO format
- JSONB allows efficient querying of nested data

## License

MIT