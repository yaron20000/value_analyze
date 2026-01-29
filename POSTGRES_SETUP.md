# PostgreSQL Setup Guide

Quick guide to run PostgreSQL locally for the Borsdata project.

## Quick Start (Docker - Recommended)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Windows - Double-click Setup
Simply run the batch script:
```bash
setup-postgres.bat
```

This automatically:
- Creates and starts PostgreSQL container
- Sets up the database schema
- Configures connection on localhost:5432

### Manual Docker Setup

```bash
# 1. Start PostgreSQL
docker run --name postgres-borsdata \
  -e POSTGRES_PASSWORD=yourpass \
  -e POSTGRES_DB=borsdata \
  -p 5432:5432 \
  -v postgres-borsdata-data:/var/lib/postgresql/data \
  -d postgres:16

# 2. Wait a few seconds for startup
timeout /t 5

# 3. Create schema
docker exec -i postgres-borsdata psql -U postgres -d borsdata < schema.sql

# 4. Verify it works
docker exec -it postgres-borsdata psql -U postgres -d borsdata -c "\dt"
```

### Managing Your Docker PostgreSQL

```bash
# Stop PostgreSQL
docker stop postgres-borsdata

# Start PostgreSQL (after stopping)
docker start postgres-borsdata

# View logs
docker logs postgres-borsdata

# Connect to database
docker exec -it postgres-borsdata psql -U postgres -d borsdata

# Backup database
docker exec postgres-borsdata pg_dump -U postgres borsdata > backup.sql

# Restore database
docker exec -i postgres-borsdata psql -U postgres -d borsdata < backup.sql

# Remove everything (including data)
docker stop postgres-borsdata
docker rm postgres-borsdata
docker volume rm postgres-borsdata-data
```

## Alternative: Docker Compose

Create [docker-compose.yml](docker-compose.yml):

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16
    container_name: postgres-borsdata
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: yourpass
      POSTGRES_DB: borsdata
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./schema.sql:/docker-entrypoint-initdb.d/schema.sql
    restart: unless-stopped

volumes:
  postgres-data:
```

Then:
```bash
# Start (creates schema automatically on first run)
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Stop and remove all data
docker-compose down -v
```

## Alternative: Native Windows Installation

### Option 1: Official Installer

1. Download from https://www.postgresql.org/download/windows/
2. Run installer (includes pgAdmin 4 GUI tool)
3. Choose installation directory (default: `C:\Program Files\PostgreSQL\16`)
4. Set superuser password
5. Use default port 5432
6. Complete installation

### Option 2: Chocolatey

```bash
choco install postgresql
```

### Setup Database

```bash
# Create database
psql -U postgres -c "CREATE DATABASE borsdata;"

# Create schema
psql -U postgres -d borsdata -f schema.sql

# Verify
psql -U postgres -d borsdata -c "\dt"
```

## Configuration

### Using .env file (Recommended)

1. Copy the example file:
```bash
copy .env.example .env
```

2. Edit `.env` with your values:
```
BORSDATA_API_KEY=your_actual_api_key
DB_PASSWORD=yourpass
```

3. Run the script (uses .env automatically):
```bash
python fetch_and_store.py
```

### Using Command Line Arguments

```bash
# Minimal (uses defaults)
python fetch_and_store.py YOUR_API_KEY --db-password yourpass

# Full configuration
python fetch_and_store.py YOUR_API_KEY \
  --db-host localhost \
  --db-port 5432 \
  --db-name borsdata \
  --db-user postgres \
  --db-password yourpass \
  --instruments 3,750
```

## Verify Installation

```bash
# Check PostgreSQL is running
docker ps

# Connect to database
docker exec -it postgres-borsdata psql -U postgres -d borsdata

# Inside psql, run:
\dt                          # List tables
\d api_raw_data             # Show table structure
\d+ api_raw_data            # Show table structure with details
\di                         # List indexes
SELECT version();           # Show PostgreSQL version
\q                          # Quit
```

## Common Issues

### Docker not running
```
ERROR: Docker is not running!
```
**Solution:** Start Docker Desktop

### Port 5432 already in use
```
Error: port is already allocated
```
**Solution:** Stop existing PostgreSQL or use different port:
```bash
docker run ... -p 5433:5432 ...  # Use port 5433 instead
# Then: python fetch_and_store.py YOUR_API_KEY --db-port 5433 --db-password yourpass
```

### Connection refused
```
psycopg2.OperationalError: could not connect to server
```
**Solution:**
- Check PostgreSQL is running: `docker ps`
- Wait a few seconds after starting
- Verify port: `docker port postgres-borsdata`

### Schema already exists error
```
ERROR: relation "api_raw_data" already exists
```
**Solution:** This is normal if re-running schema. To reset:
```bash
# Drop and recreate
docker exec -it postgres-borsdata psql -U postgres -d borsdata -c "DROP TABLE IF EXISTS api_raw_data CASCADE; DROP TABLE IF EXISTS api_fetch_log CASCADE;"
docker exec -i postgres-borsdata psql -U postgres -d borsdata < schema.sql
```

## GUI Tools

### pgAdmin 4 (Web-based)
- Included with Windows installer
- Access: http://localhost:5050 (if installed via installer)
- Docker version:
```bash
docker run -p 5050:80 \
  -e PGADMIN_DEFAULT_EMAIL=admin@admin.com \
  -e PGADMIN_DEFAULT_PASSWORD=admin \
  -d dpage/pgadmin4
```

### DBeaver (Free, cross-platform)
- Download: https://dbeaver.io/
- Supports PostgreSQL, MySQL, SQLite, etc.

### VS Code Extension
- Install "PostgreSQL" extension by Chris Kolkman
- Connect directly from VS Code

## Next Steps

Once PostgreSQL is running:

1. Test the connection:
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --instruments 3
```

2. Check the data:
```bash
docker exec -it postgres-borsdata psql -U postgres -d borsdata -c "SELECT COUNT(*) FROM api_raw_data;"
```

3. Run with your instruments:
```bash
python fetch_and_store.py YOUR_API_KEY --db-password yourpass --instruments 3,750,123
```

## Resources

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- Docker PostgreSQL: https://hub.docker.com/_/postgres
- psycopg2 (Python driver): https://www.psycopg.org/docs/
