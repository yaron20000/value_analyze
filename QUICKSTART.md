# Quick Start Guide

## What's Been Done

✅ Added duplicate prevention (no re-fetching within 24 hours)
✅ Added 2 more stocks: Atlas Copco (19), SEB (199)
✅ Added automatic batch API testing
✅ Total instruments: **4** (ABB, Atlas Copco, SEB, Securitas)

## Quick Commands

### 1. Test Batch API (Recommended First!)
```bash
# Quick test to see if batch endpoints work
python test_batch_api.py YOUR_API_KEY
```
**Time:** ~5 seconds
**Result:** Tells you if we can fetch multiple KPIs in one call

### 2. Fetch Data (No Database)
```bash
# Just create JSON files
python fetch_and_store.py YOUR_API_KEY --no-db
```
**Time:** ~1-2 minutes for 4 instruments
**Result:** JSON files in `results/` directory

### 3. Fetch Data (With Database)
```bash
# First time: setup PostgreSQL
setup-postgres.bat

# Then fetch data
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```
**Time:** ~1-2 minutes for 4 instruments
**Result:** Data in PostgreSQL + JSON files

### 4. Re-run (Skips Existing)
```bash
# Second run skips data from last 24 hours
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```
**Time:** ~5 seconds (most endpoints skipped)
**Result:** Only fetches new data

### 5. Force Full Refresh
```bash
# Force refetch everything
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --force-refetch
```

## Current Data Being Fetched

### Instruments (4)
- 3: ABB
- 19: Atlas Copco A
- 199: SEB A
- 750: Securitas

### Data Per Instrument
- Stock prices (last 100 days)
- Reports (year, quarter, R12)
- 4 KPIs: P/E, P/S, P/B, ROE

### Metadata (All Stocks)
- All instruments list
- Markets, sectors, branches
- KPI definitions

### Total API Calls
- **~56 calls** for 4 instruments
- **~11 seconds** at current rate

## What Happens When You Run

```
======================================================================
BORSDATA API FETCHER & POSTGRESQL STORER
======================================================================
API Key: 12345678...ABCD
Database: borsdata @ localhost:5432
Skip existing: Yes (within 24 hours)
Instruments: [3, 19, 199, 750]
Time: 2026-01-08T...
======================================================================

FETCHING INSTRUMENT META APIS (METADATA)
======================================================================
[1] Fetching instruments...
    → API: GET https://apiservice.borsdata.se/v1/instruments
    ← Response: 1234 instruments
    ✓ Saved to results/instruments.json
    ✓ Saved to database
...

TESTING BATCH KPI ENDPOINT
======================================================================
Test 1: /instruments/kpis with params {'instList': '3,19', 'kpiList': '2,3'}
    ✗ Failed: Error 404: Not Found
...
⚠ No batch KPI endpoint found - will use individual calls

FETCHING STOCK-SPECIFIC DATA FOR INSTRUMENTS: [3, 19, 199, 750]
======================================================================
--- Instrument ID: 3 ---
  [10] Fetching stockprices_3...
    ✓ Saved to results/stockprices_3.json
  [11] Fetching kpi_pe_3...
    ✓ Saved to results/kpi_pe_3.json
...

EXECUTION SUMMARY
======================================================================
Total endpoints: 56
Successful: 56
Failed: 0
Skipped (already exists): 0
Duration: 11.23 seconds
Results saved to: results/
Database: borsdata on localhost
Fetch log ID: 1
Skip existing: Enabled (within 24 hours)
======================================================================
```

## Files Created

### Code Files
- `fetch_and_store.py` - Main data fetcher (updated)
- `test_batch_api.py` - Batch endpoint tester (new)
- `schema.sql` - Database schema with duplicate prevention
- `setup-postgres.bat` - One-click PostgreSQL setup

### Documentation
- `BATCH_API_TESTING.md` - Batch testing explained
- `KPI_ANALYSIS.md` - 322 KPIs analyzed, recommendations
- `API_ENDPOINT_STRUCTURE.md` - How the API works
- `DUPLICATE_PREVENTION.md` - How duplicate prevention works
- `POSTGRES_SETUP.md` - PostgreSQL setup guide
- `QUICKSTART.md` - This file

### Data Files (after running)
- `results/*.json` - All API responses
- PostgreSQL database - Structured data

## Next Steps

### Option 1: Just Test (Safe)
```bash
python test_batch_api.py YOUR_API_KEY
```
See if batch endpoints work, no data fetched.

### Option 2: Fetch for 4 Stocks
```bash
python fetch_and_store.py YOUR_API_KEY --no-db
```
Get data for ABB, Atlas Copco, SEB, Securitas.

### Option 3: Add More Stocks
```bash
# Fetch for 10 stocks
python fetch_and_store.py YOUR_API_KEY --instruments 3,19,199,750,10,11,12,13,14,15 --no-db
```

### Option 4: Add More KPIs
After testing batch endpoints, add Tier 1 KPIs (35 total).
See [KPI_ANALYSIS.md](KPI_ANALYSIS.md) for recommendations.

## Troubleshooting

### "Error: API key is required"
```bash
# Set environment variable
export BORSDATA_API_KEY=your_key
python fetch_and_store.py
```

### "Error 403: Forbidden"
- Check API key is correct
- Check your Borsdata membership level

### "Error 429: Rate limited"
- Wait a few minutes
- Reduce rate: edit `time.sleep(0.2)` to `time.sleep(0.5)`

### "Skipped (already exists)"
- Normal! Means data was fetched within 24 hours
- Use `--force-refetch` to override

### Database connection failed
```bash
# Check PostgreSQL is running
docker ps

# Restart if needed
docker start postgres-borsdata
```

## Performance Tips

1. **Use duplicate prevention** (enabled by default)
   - Second run is 10x faster
   - Only fetches new data

2. **Test batch endpoints first**
   - Could reduce API calls by 90%
   - Run: `python test_batch_api.py YOUR_API_KEY`

3. **Fetch incrementally**
   - Start with 4 stocks
   - Add more once working
   - Database prevents duplicates

4. **Use database**
   - Faster duplicate checking
   - Better data querying
   - Persistence across runs

## Getting Help

- API issues: Check [Borsdata API docs](https://github.com/Borsdata-Sweden/API)
- Database issues: See [POSTGRES_SETUP.md](POSTGRES_SETUP.md)
- KPI questions: See [KPI_ANALYSIS.md](KPI_ANALYSIS.md)
- Batch testing: See [BATCH_API_TESTING.md](BATCH_API_TESTING.md)
