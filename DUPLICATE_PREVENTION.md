# Duplicate Data Prevention

This document explains how the system prevents duplicate data when fetching from the Borsdata API.

## Overview

The system now prevents duplicate data insertion using a **two-layer approach**:

1. **Pre-fetch check**: Skips API calls if recent data already exists
2. **Database constraint**: Prevents duplicate inserts at the database level

## How It Works

### 1. Pre-Fetch Check (Application Level)

Before making an API call, the system checks if data already exists:

```python
def check_existing_data(name, instrument_id, hours_threshold=24):
    # Checks if successful data exists within the last 24 hours
    # Returns True if data exists, False otherwise
```

**Benefits:**
- Saves API calls (important for rate limiting)
- Reduces network traffic
- Faster execution when data already exists

**Behavior:**
- Default threshold: 24 hours
- Only checks successful fetches (success = true)
- Properly handles NULL instrument_id for metadata endpoints

### 2. Database Constraint (Database Level)

A unique index prevents duplicate entries:

```sql
CREATE UNIQUE INDEX idx_api_raw_data_unique_daily ON api_raw_data(
    endpoint_name,
    COALESCE(instrument_id, -1),  -- Handle NULL
    DATE(fetch_timestamp)
) WHERE success = true;
```

**Benefits:**
- Guarantees no duplicates even if pre-check fails
- Database-level data integrity
- Handles concurrent writes safely

**Behavior:**
- One successful record per endpoint/instrument/day
- Uses `ON CONFLICT DO NOTHING` for graceful handling
- Failed fetches (success = false) can have duplicates (for retry tracking)

## Usage

### Default Behavior (Skip Existing)

```bash
# Automatically skips data fetched within last 24 hours
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```

Output:
```
⏭ Skipping - recent data already exists (within 24 hours)
```

### Force Refetch

```bash
# Force refetch even if data exists
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --force-refetch
```

**Note:** The database constraint will still prevent actual duplicates on the same day.

### Statistics

The execution summary now includes skipped endpoints:

```
EXECUTION SUMMARY
======================================================================
Total endpoints: 45
Successful: 10
Failed: 0
Skipped (already exists): 35
Duration: 5.23 seconds
```

## Database Schema

### Unique Constraint

The constraint ensures uniqueness based on:
- `endpoint_name`: e.g., "instruments", "stockprices_3"
- `instrument_id`: Stock-specific ID (or NULL for metadata)
- `DATE(fetch_timestamp)`: Calendar day

### Why Only Successful Fetches?

Failed fetches (success = false) are NOT constrained, allowing you to:
- Retry failed fetches multiple times per day
- Track error patterns over time
- See error history without data loss

## Migration for Existing Databases

If you have an existing database, apply the migration:

```bash
psql -U postgres -d borsdata -f migration_add_unique_constraint.sql
```

### If You Have Existing Duplicates

The migration includes commented-out SQL to:
1. Find existing duplicates
2. Clean them up (keeping most recent)

See [migration_add_unique_constraint.sql](migration_add_unique_constraint.sql) for details.

## Examples

### Example 1: First Run
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```
- Fetches all endpoints
- Saves to database
- Total: 45, Successful: 45, Skipped: 0

### Example 2: Second Run (Same Day)
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```
- Skips all endpoints (data exists within 24 hours)
- No API calls made
- Total: 45, Successful: 0, Skipped: 45

### Example 3: Force Refetch
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --force-refetch
```
- Makes API calls for all endpoints
- Database constraint prevents duplicate inserts
- Total: 45, Successful: 0, Skipped: 0
- Output: "ℹ Data already exists in DB (skipped)"

### Example 4: Next Day
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass
```
- Previous day's data is now > 24 hours old
- Fetches all endpoints again
- New records created (different date)
- Total: 45, Successful: 45, Skipped: 0

## Technical Details

### Time-Based Deduplication

The system considers data "duplicate" if:
- Same endpoint_name
- Same instrument_id (or both NULL)
- Within the same calendar day (DATE)
- Previous fetch was successful

### Null Handling

Metadata endpoints (instruments, markets, etc.) have `instrument_id = NULL`. The constraint uses `COALESCE(instrument_id, -1)` to properly handle NULL values in the unique index.

### ON CONFLICT Behavior

```python
INSERT INTO api_raw_data (...) VALUES (...)
ON CONFLICT DO NOTHING
```

When a duplicate is detected:
- No error is raised
- `cursor.rowcount = 0`
- Application logs: "ℹ Data already exists in DB (skipped)"

## Performance Considerations

### Benefits
- **Reduced API calls**: Skip existing data saves money and respects rate limits
- **Faster execution**: No unnecessary network requests
- **Database efficiency**: Prevents table bloat from duplicates

### Trade-offs
- **Extra query per endpoint**: The existence check adds one SELECT query
- **Acceptable overhead**: Query is indexed and very fast (<1ms typically)

## Future Enhancements

Possible improvements:
- Configurable time threshold (currently hardcoded to 24 hours)
- Batch existence checks (check all endpoints at once)
- Smart refresh (only fetch if market is open)
- Differential updates (only fetch new data since last fetch)
