---
status: "accepted"
date: 2026-06-18
decision-makers: [Zack]
consulted: [Tristan]
---

# Trigger the forecast monitor from the upstream ds-storms-pipeline

## Context and Problem Statement

The forecast monitor's correctness depends on the NHC tracks being present in
Postgres (`storms.nhc_tracks_geo`), which the upstream `ds-storms-pipeline` job
lands. If the monitor runs before the tracks arrive, it reads stale or missing
data. The observational monitor has no such upstream dependency. How should the
forecast monitor be triggered on Databricks?

## Decision Drivers

* Correctness — do not read before the tracks have landed.
* Avoid a time-guess; the upstream cadence varies and it has a `:30` late-WSP
  retry, so "30 min after the advisory" is not reliable.
* Keep the observational monitor simple (it has no data dependency).

## Considered Options

* Schedule the forecast monitor ~30 min after each advisory (status quo; what
  ds-storms-alerts does).
* The upstream `ds-storms-pipeline` triggers it via a downstream `run_job_task`
  once its tracks task completes.
* The Postgres database triggers it on row insert.

## Decision Outcome

The upstream `nhc_pipeline` job in `ds-storms-pipeline` runs the cub
`fcast_monitor` as a downstream `run_job_task`, after its tracks task. The cub
`fcast_monitor` therefore has **no schedule of its own**. The observational
monitor keeps a plain daily schedule.

Chosen because it is event-driven — the monitor runs exactly when its data is
ready, with no race. A schedule is only a guess at when the upstream finishes. A
database trigger is not natively available: Databricks job triggers are cron,
file-arrival (Unity Catalog volume / external location), or table-update (UC
Delta tables), and our data lands in an **external Azure Postgres**, which is
none of those (a marker-file + file-arrival shim would be extra plumbing on both
sides for no clear gain).

### Consequences

* Good — the forecast monitor runs only after the tracks have landed; no time
  guess, and it naturally follows the upstream cadence.
* Bad — couples the two repos/bundles: `ds-storms-pipeline` must reference the
  cub **prod** `fcast_monitor` job id, which does not exist until the cub bundle
  is deployed to prod. The wiring lives in the other repo and is therefore
  **deferred to go-live** (tracked as a follow-up issue on ds-storms-pipeline).
* Bad — the upstream's `:30` retry run could fire a second monitor run per
  advisory; mitigated by `max_concurrent_runs: 1` and the once-per-issuance
  email scoping.

## More Information

* The cub side is ready (the `fcast_monitor` job has no schedule). The upstream
  `run_job_task` is a follow-up in `ds-storms-pipeline`, to be wired against the
  prod job id at go-live.
* Compute/deployment model: see
  `0001-run-monitors-as-databricks-jobs-on-ephemeral-clusters.md`.
* `databricks/README.md` — "Why the forecast monitor is triggered, not scheduled".
