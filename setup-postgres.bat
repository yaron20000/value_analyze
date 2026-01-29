@echo off
REM Quick PostgreSQL setup script for Windows
REM Requires Docker Desktop to be installed and running

echo ========================================
echo PostgreSQL Docker Setup for Borsdata
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/4] Stopping any existing postgres-borsdata container...
docker stop postgres-borsdata >nul 2>&1
docker rm postgres-borsdata >nul 2>&1

echo [2/4] Starting PostgreSQL container...
docker run --name postgres-borsdata ^
  -e POSTGRES_PASSWORD=yourpass ^
  -e POSTGRES_DB=borsdata ^
  -p 5432:5432 ^
  -v postgres-borsdata-data:/var/lib/postgresql/data ^
  -d postgres:16

if %errorlevel% neq 0 (
    echo ERROR: Failed to start PostgreSQL container
    pause
    exit /b 1
)

echo [3/4] Waiting for PostgreSQL to be ready...
timeout /t 5 /nobreak >nul

echo [4/4] Creating database schema...
docker exec -i postgres-borsdata psql -U postgres -d borsdata < schema.sql

if %errorlevel% neq 0 (
    echo WARNING: Schema creation may have failed. Run manually if needed:
    echo   docker exec -i postgres-borsdata psql -U postgres -d borsdata < schema.sql
) else (
    echo SUCCESS: Schema created successfully!
)

echo.
echo ========================================
echo PostgreSQL is ready!
echo ========================================
echo Connection details:
echo   Host: localhost
echo   Port: 5432
echo   Database: borsdata
echo   Username: postgres
echo   Password: thePapi4!
echo.
echo To connect: psql -h localhost -U postgres -d borsdata
echo To stop: docker stop postgres-borsdata
echo To start again: docker start postgres-borsdata
echo To remove: docker rm -f postgres-borsdata
echo ========================================
pause
