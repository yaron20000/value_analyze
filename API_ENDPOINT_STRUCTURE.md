# Borsdata API Endpoint Structure

## How the API Works

The Borsdata API has **two types of endpoints**:

### 1. **Per-Instrument Endpoints** (Individual API calls)
Each instrument requires a separate API call.

**Format:** `/instruments/{instrument_id}/...`

**Current Implementation:**
```python
for inst_id in [3, 750]:
    # Call 1: /instruments/3/kpis/2/year/mean/history  (P/E for ABB)
    # Call 2: /instruments/750/kpis/2/year/mean/history  (P/E for Securitas)
```

### 2. **Batch/Array Endpoints** (Multi-instrument in one call)
One API call can fetch data for multiple instruments.

**Format:** `/instruments/...?instList=3,750,123`

**Current Implementation:**
```python
# One call for all instruments:
# /instruments/stockprices?instList=3,750&maxcount=50
```

## Current API Call Count

### For 2 instruments (3, 750):

**Per-Instrument Calls:**
- Stock prices: 2 calls (1 per instrument)
- Reports (4 types): 8 calls (4 × 2 instruments)
- KPIs (4 types): 8 calls (4 × 2 instruments)
- **Subtotal: 18 calls**

**Batch Calls:**
- Stock prices array: 1 call
- Reports array: 1 call
- **Subtotal: 2 calls**

**Global/Metadata (once):**
- Metadata endpoints: 9 calls
- Global endpoints: 8 calls
- **Subtotal: 17 calls**

**Total for 2 instruments: 37 calls**

## Scaling: What Changes with More KPIs?

### Scenario 1: Add 31 more KPIs (Tier 1 from KPI_ANALYSIS.md)

**Current KPIs:** 4 per instrument
**New KPIs:** 35 per instrument

#### Option A: Individual Calls (Current Pattern)
```python
# For each instrument:
for kpi_id in [1, 2, 3, 4, 10, 11, 13, ... 35 total]:
    GET /instruments/{inst_id}/kpis/{kpi_id}/year/mean/history
```

**API calls for N instruments:**
- KPIs: 35 × N calls
- For 2 instruments: 70 calls (vs current 8)
- For 100 instruments: 3,500 calls
- For 1000 instruments: 35,000 calls

#### Option B: Check if Batch Endpoint Exists
Need to check Borsdata API docs if there's:
```
GET /instruments/kpis/history?instList=3,750&kpiList=1,2,3,4,10...
```

If this exists, it would be **1 call** instead of 35×N calls.

## Borsdata API Endpoint Types

### Type 1: Single Instrument, Single KPI (Most Common)
```
GET /instruments/{instrumentId}/kpis/{kpiId}/year/mean/history
```
**Returns:** Historical values for ONE KPI for ONE instrument
**Cost:** 1 call per KPI per instrument

### Type 2: Multiple Instruments, Single Data Type (Batch)
```
GET /instruments/stockprices?instList=3,750,123
```
**Returns:** Data for ALL specified instruments in one response
**Cost:** 1 call for all instruments

### Type 3: Global/Metadata (No instrument needed)
```
GET /instruments
GET /markets
GET /instruments/kpis/metadata
```
**Returns:** All instruments, all markets, all KPI definitions
**Cost:** 1 call total (not per instrument)

## Optimization Strategies

### Strategy 1: Use Batch Endpoints When Available ✅ (Already doing this)

**Currently using:**
```python
# Good! Uses batch endpoint
GET /instruments/stockprices?instList=3,750
GET /instruments/reports?instList=3,750
```

### Strategy 2: Check for KPI Batch Endpoints 🔍 (Need to verify)

**Need to test if this works:**
```python
# Does this exist?
GET /instruments/kpis/2?instList=3,750  # P/E for multiple instruments
```

If yes, we can reduce calls dramatically.

### Strategy 3: Parallel Requests (For per-instrument endpoints)

If batch endpoints don't exist, we can parallelize:
```python
# Instead of sequential:
for inst_id in instruments:
    fetch_kpi(inst_id, kpi_id)  # N calls, sequential

# Use parallel:
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(fetch_kpi, inst_id, kpi_id)
               for inst_id in instruments]
```

**Benefit:** Same number of API calls, but much faster execution
**Risk:** May hit rate limits faster

### Strategy 4: Smart Caching ✅ (Already implemented)

Already have duplicate prevention:
- Skip if data exists within 24 hours
- Database constraint prevents duplicates

## API Call Estimation

### Current State (4 KPIs)
| Instruments | Per-Instrument Calls | Batch Calls | Total |
|-------------|---------------------|-------------|-------|
| 2 | 18 | 2 | 20 |
| 10 | 90 | 2 | 92 |
| 100 | 900 | 2 | 902 |
| 1000 | 9,000 | 2 | 9,002 |

### With Tier 1 KPIs (35 KPIs) - Individual Calls
| Instruments | Per-Instrument Calls | Batch Calls | Total |
|-------------|---------------------|-------------|-------|
| 2 | 157 | 2 | 159 |
| 10 | 785 | 2 | 787 |
| 100 | 7,850 | 2 | 7,852 |
| 1000 | 78,500 | 2 | 78,502 |

### With Tier 1 KPIs (35 KPIs) - If Batch Available
| Instruments | Per-Instrument Calls | Batch Calls | Total |
|-------------|---------------------|-------------|-------|
| 2 | 87 | 37 | 124 |
| 10 | 87 | 37 | 124 |
| 100 | 87 | 37 | 124 |
| 1000 | 87 | 37 | 124 |

**Note:** The "Batch Calls" assumes we can batch KPIs. Need to verify this with API.

## Recommended Approach

### Phase 1: Research API Capabilities

Test these endpoints to see if they exist:
```bash
# Test 1: Multiple KPIs, single instrument
curl "https://apiservice.borsdata.se/v1/instruments/3/kpis/2,3,4/year/mean/history?authKey=XXX"

# Test 2: Single KPI, multiple instruments
curl "https://apiservice.borsdata.se/v1/instruments/kpis/2?instList=3,750&authKey=XXX"

# Test 3: Multiple KPIs, multiple instruments (ideal!)
curl "https://apiservice.borsdata.se/v1/instruments/kpis?instList=3,750&kpiList=2,3,4&authKey=XXX"
```

### Phase 2: Implement Based on API Support

**If batch KPI endpoints exist:**
```python
def fetch_kpis_batch(instrument_ids, kpi_ids):
    """Fetch multiple KPIs for multiple instruments in one call"""
    inst_list = ",".join(str(i) for i in instrument_ids)
    kpi_list = ",".join(str(k) for k in kpi_ids)
    endpoint = f"/instruments/kpis?instList={inst_list}&kpiList={kpi_list}"
    # 1 call for all data!
```

**If only per-instrument batch:**
```python
def fetch_kpis_per_instrument(instrument_id, kpi_ids):
    """Fetch multiple KPIs for one instrument"""
    kpi_list = ",".join(str(k) for k in kpi_ids)
    endpoint = f"/instruments/{instrument_id}/kpis?kpiList={kpi_list}"
    # N calls (one per instrument) instead of N×M (instruments × KPIs)
```

**If no batch support (worst case):**
```python
def fetch_kpis_individual(instrument_ids, kpi_ids):
    """Fetch each KPI individually with parallelization"""
    # Use ThreadPoolExecutor for parallel requests
    # N×M calls, but execute faster
```

### Phase 3: Optimize Existing Code

Current code already does some batching:
```python
# Line 402-405: Already uses batch endpoint for stock prices
("stockprices_array", "/instruments/stockprices",
 {"instList": inst_list, "maxcount": 50}),
```

We should extend this pattern to KPIs if possible.

## Rate Limiting Considerations

Borsdata API likely has rate limits. Current code has:
```python
time.sleep(0.2)  # 200ms between requests = max 5 req/sec
```

**Recommended:**
- Keep rate limiting with individual calls
- May be able to remove/reduce with batch calls
- Monitor for 429 errors and implement backoff

## Practical Example

### Example: Fetch 35 KPIs for 100 instruments

**Current Pattern (Individual calls):**
```python
for inst_id in instrument_ids:  # 100 iterations
    for kpi_id in kpi_ids:  # 35 iterations
        fetch(f"/instruments/{inst_id}/kpis/{kpi_id}/...")
        time.sleep(0.2)
# Total: 3,500 calls × 0.2s = 700 seconds = 11.7 minutes
```

**Optimized (Per-instrument batch):**
```python
for inst_id in instrument_ids:  # 100 iterations
    kpi_list = "1,2,3,4,10,11,..."  # 35 KPIs
    fetch(f"/instruments/{inst_id}/kpis?kpiList={kpi_list}")
    time.sleep(0.2)
# Total: 100 calls × 0.2s = 20 seconds
```

**Best Case (Full batch):**
```python
inst_list = ",".join(str(i) for i in instrument_ids)
kpi_list = "1,2,3,4,10,11,..."
fetch(f"/instruments/kpis?instList={inst_list}&kpiList={kpi_list}")
# Total: 1 call = <1 second
```

## Action Items

1. **Research Borsdata API documentation** for batch KPI endpoints
2. **Test batch endpoints** with actual API calls
3. **Update fetch_stock_data()** based on findings
4. **Implement parallel requests** if batch not available
5. **Monitor rate limits** and adjust sleep delays

## References

- Current implementation: [fetch_and_store.py:355-411](fetch_and_store.py)
- Batch example: [fetch_and_store.py:401-405](fetch_and_store.py)
- API responses: [results/stockprices_array.json](results/stockprices_array.json)
