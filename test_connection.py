#!/usr/bin/env python3
"""
Simple PostgreSQL Connection Test
==================================
Tests the database connection using credentials from .env file.
"""

import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def test_connection():
    """Test PostgreSQL connection."""

    # Get connection parameters
    host = os.getenv('DB_HOST', 'localhost')
    port = os.getenv('DB_PORT', '5432')
    database = os.getenv('DB_NAME', 'borsdata')
    user = os.getenv('DB_USER', 'postgres')
    password = os.getenv('DB_PASSWORD')
    print (password)
    print("="*60)
    print("PostgreSQL Connection Test")
    print("="*60)
    print(f"Host:     {host}")
    print(f"Port:     {port}")
    print(f"Database: {database}")
    print(f"User:     {user}")
    print(f"Password: {'*' * len(password) if password else '(not set)'}")
    print("="*60)
    print()

    if not password:
        print("❌ ERROR: DB_PASSWORD not set in .env file")
        return False

    try:
        print("Attempting connection...")
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )

        print("✅ Connection successful!")

        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"\n📊 PostgreSQL version:")
        print(f"   {version}")

        # List tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()

        if tables:
            print(f"\n📋 Tables in database:")
            for table in tables:
                print(f"   - {table[0]}")
        else:
            print(f"\n⚠️  No tables found in database")

        cursor.close()
        conn.close()
        print("\n✅ Connection test completed successfully!")
        return True

    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed!")
        print(f"\nError details:")
        print(f"   {e}")
        print(f"\n💡 Common fixes:")
        print(f"   1. Check if PostgreSQL is running: docker ps")
        print(f"   2. Verify credentials in .env file")
        print(f"   3. Ensure DB_PASSWORD matches the one set in Docker")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    exit(0 if success else 1)
