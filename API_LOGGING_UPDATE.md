# API Logging Update

## Changes Made

### 1. Detailed API Call Logging

Added comprehensive logging for all API requests and responses to make debugging easier and provide visibility into what the script is doing.

#### Request Logging
Every API call now prints:
```
→ API: GET https://apiservice.borsdata.se/v1/instruments?maxcount=100
```

The query parameters are shown (but the authKey is excluded for security).

#### Response Logging
After each successful request, the script logs the response size:
```
← Response: 823 instruments
← Response: 100 prices
← Response: 20 reports
← Response: 0 items in list
```

#### Save Status Logging
After processing each endpoint:
```
✓ Saved to results/instruments.json
✓ Saved to database
```

Or if there's an error:
```
✗ Error: Error 403: Forbidden - Check your API key or membership level
✓ Error logged to results/instruments.json
```

### 2. Holdings Endpoints Investigation

**Finding**: The holdings endpoints (`/holdings/insider`, `/holdings/shorts`, `/holdings/buyback`) do NOT accept any parameters.

- They are parameter-free endpoints that return data for ALL instruments
- Unlike `stockprices` or `reports` endpoints that accept `instList` parameter
- They require Pro+ membership
- Empty results for insider/buyback just mean no recent activity data is available

**Current Implementation**: All three endpoints are correctly called without parameters:
```python
("holdings_insider", "/holdings/insider", {}),
("holdings_shorts", "/holdings/shorts", {}),
("holdings_buyback", "/holdings/buyback", {}),
```

### 3. Improved Output Formatting

Changed from inline status indicators to multi-line output:

**Before:**
```
[1] Fetching instruments... ✓
[2] Fetching markets... ✓
```

**After:**
```
[1] Fetching instruments...
    → API: GET https://apiservice.borsdata.se/v1/instruments
    ← Response: 823 instruments
    ✓ Saved to results/instruments.json
    ✓ Saved to database

[2] Fetching markets...
    → API: GET https://apiservice.borsdata.se/v1/markets
    ← Response: 11 items in list
    ✓ Saved to results/markets.json
    ✓ Saved to database
```

## Example Output

When running the script, you'll now see detailed output like:

```
======================================================================
FETCHING INSTRUMENT META APIS (METADATA)
======================================================================

[1] Fetching instruments...
    → API: GET https://apiservice.borsdata.se/v1/instruments
    ← Response: 823 instruments
    ✓ Saved to results/instruments.json
    ✓ Saved to database

[2] Fetching instruments_updated...
    → API: GET https://apiservice.borsdata.se/v1/instruments/updated
    ← Response: 15 instruments
    ✓ Saved to results/instruments_updated.json
    ✓ Saved to database

======================================================================
FETCHING STOCK-SPECIFIC DATA FOR INSTRUMENTS: [3, 750]
======================================================================

--- Instrument ID: 3 ---

  [10] Fetching stockprices_3...
    → API: GET https://apiservice.borsdata.se/v1/instruments/3/stockprices?maxcount=100
    ← Response: 100 prices
    ✓ Saved to results/stockprices_3.json
    ✓ Saved to database

  [11] Fetching reports_year_3...
    → API: GET https://apiservice.borsdata.se/v1/instruments/3/reports/year?maxcount=10
    ← Response: 10 reports
    ✓ Saved to results/reports_year_3.json
    ✓ Saved to database
```

## Benefits

1. **Debugging**: Easy to see exactly what API calls are being made
2. **Monitoring**: Track response sizes and success rates in real-time
3. **Transparency**: Users can see exactly what's happening without checking log files
4. **Error Tracking**: Clear indication of which endpoints failed and why
5. **Performance**: Can see which endpoints return large amounts of data

## Testing

Run the script as usual:
```bash
python fetch_and_store.py YOUR_API_KEY --no-db
```

Or with database:
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --instruments "3,750"
```
