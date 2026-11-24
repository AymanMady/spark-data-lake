# Pinning dependencies

A pipeline that worked yesterday must work today with the same code. Unpinned
dependencies break that promise as soon as a library releases a new version.

## Strategy

- Pin **exact versions** (`pandas==2.2.2`) for applications and pipelines.
- Keep the direct dependencies in one file and generate the full lock (with
  `pip-compile` or `uv pip compile`) so transitive versions are pinned too.
- Use version ranges only in libraries meant to be installed by others.

## Docker images

Pin base images by tag **and** prefer a specific patch version
(`python:3.11.9-slim`), not `latest`. Rebuild on purpose, not by accident.

## Upgrading

Upgrade on a schedule (for example monthly), one ecosystem at a time, and run
the full test suite and a sample pipeline run before merging.
