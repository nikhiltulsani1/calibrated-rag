#!/bin/bash
# Runs once, on first init, via postgres's /docker-entrypoint-initdb.d hook.
# Creates the Airflow metadata database in the same instance as the app's,
# owned by the same POSTGRES_USER — one credential, two databases, no
# second postgres container against a tight memory budget.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE airflow OWNER ${POSTGRES_USER}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
EOSQL
