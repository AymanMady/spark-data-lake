# Release checklist

- [ ] Tests pass locally and in CI.
- [ ] Linting and type checks are clean.
- [ ] Schema changes are additive, or consumers have been warned.
- [ ] Migrations tested on a copy of production data.
- [ ] Backfill plan written when historical data is affected.
- [ ] New configuration documented in `.env.example`.
- [ ] Data-quality tests added for new columns.
- [ ] README and docs updated.
- [ ] Rollback procedure known.
- [ ] Monitoring checked after deploy (first run green, row counts normal).
