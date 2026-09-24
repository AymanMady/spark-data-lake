-- Runs ONLY ONCE, when the postgres volume is first created.
-- (If you change this file later: make clean, then make up.)

-- The schema that will hold the Gold tables (phase 13).
CREATE SCHEMA IF NOT EXISTS gold;

-- A small witness table, to check right now that the database answers.
CREATE TABLE IF NOT EXISTS gold._healthcheck (
    id          SERIAL PRIMARY KEY,
    component   TEXT        NOT NULL,
    checked_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO gold._healthcheck (component) VALUES ('postgres-init');
