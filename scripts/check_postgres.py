"""
Check that the data warehouse (PostgreSQL) is reachable from the application
container, using ONLY environment variables.

No credential is hardcoded: that is the project rule.

Run with: make check-db
"""

import os
import sys

import psycopg2


def main() -> int:
    params = {
        "host": os.environ["POSTGRES_HOST"],
        "port": os.environ["POSTGRES_PORT"],
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
    }
    # The password is never printed to the logs.
    print(f"Connecting to postgresql://{params['user']}@{params['host']}:{params['port']}/{params['dbname']}")

    try:
        with psycopg2.connect(**params) as conn, conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(f"  Version   : {cur.fetchone()[0].split(',')[0]}")

            cur.execute("SELECT current_database(), current_user;")
            db, user = cur.fetchone()
            print(f"  Database  : {db}   User: {user}")

            cur.execute("SELECT schema_name FROM information_schema.schemata"
                        " WHERE schema_name = 'gold';")
            print(f"  gold schema present: {'yes' if cur.fetchone() else 'NO'}")

            cur.execute("SELECT count(*) FROM gold._healthcheck;")
            print(f"  Rows in gold._healthcheck: {cur.fetchone()[0]}")
    except Exception as exc:
        print(f"\nCONNECTION FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nPOSTGRESQL IS OPERATIONAL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
