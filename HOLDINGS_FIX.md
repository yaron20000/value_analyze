# Holdings Endpoints Fix - instList Parameter

## Issue

The holdings endpoints (`/holdings/insider`, `/holdings/shorts`, `/holdings/buyback`) were being called without the `instList` parameter, which resulted in empty responses for insider and buyback data.

## Root Cause

According to the Borsdata API documentation:
- `/v1/holdings/insider` - Returns Holdings insider **for array of instruments**
- `/v1/holdings/buyback` - Returns Holdings buyback **for array of instruments**
- `/v1/holdings/shorts` - Returns Holdings shorts **for array of instruments**

These endpoints **accept and likely require** the `instList` parameter to filter results by specific instruments.

## Solution

### 1. Updated `fetch_and_store.py`

Modified the `fetch_global_endpoints()` method to:
- Accept `instrument_ids` parameter
- Build `instList` parameter from the instrument IDs
- Pass `instList` to all three holdings endpoints

**Before:**
```python
global_endpoints = [
    ...
    ("holdings_insider", "/holdings/insider", {}),
    ("holdings_shorts", "/holdings/shorts", {}),
    ("holdings_buyback", "/holdings/buyback", {}),
]
```

**After:**
```python
# Build instrument list for holdings endpoints
inst_list = ",".join(str(i) for i in instrument_ids) if instrument_ids else None

# Holdings endpoints - can optionally filter by instrument list
if inst_list:
    global_endpoints.extend([
        ("holdings_insider", "/holdings/insider", {"instList": inst_list}),
        ("holdings_shorts", "/holdings/shorts", {"instList": inst_list}),
        ("holdings_buyback", "/holdings/buyback", {"instList": inst_list}),
    ])
```

### 2. Updated `api_test.py`

Added `instList` parameter to holdings endpoints:

**Before:**
```python
endpoints.extend([
    ("holdings_insider", "/holdings/insider", {}),
    ("holdings_shorts", "/holdings/shorts", {}),
    ("holdings_buyback", "/holdings/buyback", {}),
])
```

**After:**
```python
# Holdings endpoints accept instList parameter to filter by instruments
inst_list_sample = "3,750"  # ABB and Securitas
endpoints.extend([
    ("holdings_insider", "/holdings/insider", {"instList": inst_list_sample}),
    ("holdings_shorts", "/holdings/shorts", {"instList": inst_list_sample}),
    ("holdings_buyback", "/holdings/buyback", {"instList": inst_list_sample}),
])
```

## Expected Behavior

With this fix, the holdings endpoints will now:
1. **Request data for specific instruments** using the `instList` parameter
2. **Return filtered results** for only the requested instruments (e.g., 3, 750)
3. **Potentially return actual data** for insider trading and buyback activities for those instruments

## Example API Calls

The script will now make calls like:
```
GET https://apiservice.borsdata.se/v1/holdings/insider?instList=3,750&authKey=...
GET https://apiservice.borsdata.se/v1/holdings/shorts?instList=3,750&authKey=...
GET https://apiservice.borsdata.se/v1/holdings/buyback?instList=3,750&authKey=...
```

Instead of:
```
GET https://apiservice.borsdata.se/v1/holdings/insider?authKey=...
```

## Testing

Run the script to test the fix:
```bash
python fetch_and_store.py YOUR_API_KEY --no-db --instruments "3,750"
```

Watch for the API calls in the output:
```
[24] Fetching holdings_insider...
    → API: GET https://apiservice.borsdata.se/v1/holdings/insider?instList=3,750
    ← Response: X items in list
    ✓ Saved to results/holdings_insider.json
```

Check the JSON files to see if they now contain actual data:
- `results/holdings_insider.json`
- `results/holdings_buyback.json`

## Note

If the results are still empty, it may simply mean:
- There are no recent insider trading transactions for instruments 3 and 750
- There are no recent buyback activities for those instruments
- The data requires Pro+ membership and your account doesn't have access

But at least we're now correctly passing the instrument filter as documented!
