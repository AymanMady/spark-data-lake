# Development datasets

## Options

- **Synthetic data** from a generator script: safe, reproducible with a fixed
  seed, can include deliberate defects (duplicates, nulls, bad formats) to test
  the cleaning logic.
- **Sampled production data**: realistic distributions, but must be
  anonymised and access-controlled.
- **Seeds** (dbt `seeds/`): small static reference tables committed to git.

## Good practices

- Fix the random seed and the reference date so two runs produce the same
  files.
- Keep a small default size for fast feedback, with a flag for larger volumes
  when testing performance.
- Document which defects are injected and at which rate, so tests can assert
  on them.
