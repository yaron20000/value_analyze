# Batch API Testing Update

## Changes Made

### 1. Added More Instruments

Updated default instruments from 2 to 4:

**Before:**
- 3 (ABB)
- 750 (Securitas)

**After:**
- 3 (ABB)
- 19 (Atlas Copco A)
- 199 (SEB A)
- 750 (Securitas)

**File:** [fetch_and_store.py:536](fetch_and_store.py)

### 2. Added Batch API Testing

Added automatic testing to check if Borsdata supports batch KPI endpoints.

**New method:** `test_kpi_batch_endpoint()` [fetch_and_store.py:332-363](fetch_and_store.py)

This method tries different API patterns:
1. `/instruments/kpis?instList=3,19&kpiList=2,3` (both as query params)
2. `/instruments/kpis/2,3?instList=3,19` (KPI in path)
3. `/instruments/3,19/kpis?kpiList=2,3` (instrument in path)

**When it runs:**
- Automatically during `run()` [fetch_and_store.py:497](fetch_and_store.py)
- Tests with 2 instruments and 2 KPIs
- Prints results to show which pattern works (if any)

### 3. Standalone Test Script

Created `test_batch_api.py` for independent testing without running the full fetch process.

**Usage:**
```bash
# Test batch endpoints without fetching all data
python test_batch_api.py YOUR_API_KEY

# Or use environment variable
export BORSDATA_API_KEY=your_key
python test_batch_api.py
```

**What it does:**
- Tests 5 different batch endpoint patterns
- Shows exactly what works
- Displays sample response data
- No database needed
- Fast (only makes 5 test API calls)

## How to Test

### Option 1: Quick Test (Recommended First)

Run the standalone test script:
```bash
python test_batch_api.py YOUR_API_KEY
```

This will tell you if batch endpoints are available WITHOUT fetching all the data.

### Option 2: Full Run with Testing

Run the main script (will test batch, then fetch all data):
```bash
# Without database (JSON only)
python fetch_and_store.py YOUR_API_KEY --no-db

# With database
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```

The script will:
1. Fetch metadata (9 endpoints)
2. **Test batch KPI endpoints** (new!)
3. Fetch data for 4 instruments
4. Save everything to database

## Expected Results

### Scenario A: Batch Endpoints Work ✅

If you see:
```
TESTING BATCH KPI ENDPOINT
======================================================================

Test 1: /instruments/kpis with params {'instList': '3,19', 'kpiList': '2,3'}
    ✓ SUCCESS! Batch endpoint works: /instruments/kpis
    Response keys: ['kpiHistoryList']
```

**This means:**
- Can fetch multiple KPIs for multiple instruments in 1 call
- Huge optimization opportunity
- Next step: Update code to use batch endpoints

### Scenario B: Batch Endpoints Don't Work ✗

If you see:
```
TESTING BATCH KPI ENDPOINT
======================================================================

Test 1: /instruments/kpis with params {'instList': '3,19', 'kpiList': '2,3'}
    ✗ Failed: Error 404: Not Found

Test 2: /instruments/kpis/2,3 with params {'instList': '3,19'}
    ✗ Failed: Error 404: Not Found

Test 3: /instruments/3,19/kpis with params {'kpiList': '2,3'}
    ✗ Failed: Error 404: Not Found

⚠ No batch KPI endpoint found - will use individual calls
```

**This means:**
- Must fetch each KPI individually
- Current approach is correct
- Can optimize with parallel requests (ThreadPoolExecutor)

## Current API Call Count

### With 4 instruments (3, 19, 199, 750):

**Metadata (once):** 9 calls
**Batch test:** 3 calls (only during first run)

**Per instrument:**
- Stock prices: 1 call
- Reports (4 types): 4 calls
- KPIs (4 types): 4 calls
- **Subtotal: 9 calls × 4 instruments = 36 calls**

**Batch (all instruments):**
- Stock prices array: 1 call
- Reports array: 1 call
- **Subtotal: 2 calls**

**Global endpoints:** 8 calls

**Total: ~56 calls** (vs 37 for 2 instruments)

### If adding 35 KPIs (Tier 1):

**Without batch:**
- 4 instruments × 35 KPIs = 140 KPI calls
- Total: ~184 calls
- At 0.2s per call = ~37 seconds

**With batch (if available):**
- 1 call for all instruments and KPIs
- Total: ~54 calls
- At 0.2s per call = ~11 seconds

## Next Steps

### After Testing

1. **Run test script first:**
   ```bash
   python test_batch_api.py YOUR_API_KEY
   ```

2. **If batch works:**
   - I'll update the code to use batch endpoints
   - Massive performance improvement

3. **If batch doesn't work:**
   - Current approach is optimal
   - Can add parallel requests for speed
   - Will still work, just more API calls

### Then Add More KPIs

Once we know the batch situation, we can:
1. Add Tier 1 KPIs (35 total) from [KPI_ANALYSIS.md](KPI_ANALYSIS.md)
2. Optimize fetching based on batch availability
3. Scale to more instruments

## Files Modified

1. **[fetch_and_store.py](fetch_and_store.py)**
   - Line 536: Default instruments changed to `3,19,199,750`
   - Lines 332-363: New `test_kpi_batch_endpoint()` method
   - Line 497: Added batch testing to `run()` method

2. **[test_batch_api.py](test_batch_api.py)** (NEW)
   - Standalone testing script
   - Tests 5 batch endpoint patterns
   - No database required

3. **This file** (NEW)
   - Documentation of changes
   - Testing instructions

## Questions?

- **Q: Will this break existing functionality?**
  - A: No, the test is read-only and doesn't affect normal fetching

- **Q: Why test automatically?**
  - A: To know if we can optimize before fetching lots of data

- **Q: Can I skip the test?**
  - A: Yes, the test is informational. Data fetching continues regardless

- **Q: What if the test is slow?**
  - A: It only runs once at the start and makes 3 quick API calls
