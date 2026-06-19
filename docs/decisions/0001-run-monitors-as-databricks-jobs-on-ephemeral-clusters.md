---
status: "accepted"
date: 2026-06-18
decision-makers: [Zack]
consulted: [Tristan]
---

# Run the Cuba hurricane monitors as Databricks jobs on ephemeral clusters

## Context and Problem Statement

The two monitors (forecast, observational) ran on GitHub Actions cron, sending
email via the humdata_email SMTP path. With email dispatch moving to listmonk and
the rest of the storms stack (ds-storms-pipeline, ds-storms-alerts) already on
Databricks — where GitHub Actions trigger latency had also degraded — we want to
run the monitors on Databricks too. How should they be defined, deployed, and on
what compute?

## Decision Drivers

* Align with the storms stack already on Databricks.
* Reproducible and **person-independent** — not tied to an individual's cluster.
* Reasonable cost; the monitors are plain Python, not Spark.
* Pipeline code stays DBX-agnostic, so the GitHub Actions fallback can run the
  exact same scripts.
* The forecast monitor drives anticipatory action with **multi-day lead times**,
  so per-run startup latency is not operationally critical.

## Considered Options

Deployment: a Databricks Asset Bundle (DAB) with `source: GIT` (clone the repo at
run time → push-to-ship) vs `source: WORKSPACE` (sync files at deploy).

Compute, for running the jobs:

* Ephemeral single-node job cluster per run
* Warm always-on all-purpose cluster with libraries pre-installed
* Instance pool
* Serverless jobs compute

## Decision Outcome

A DAB defines two jobs; each task is a thin wrapper (`databricks/run_monitor_job.py`)
that selects the listmonk backend and shells out to the unchanged monitor script.
`source: GIT` is used (push the branch to ship code; `bundle deploy` only for job
config). Each job runs on an **ephemeral single-node job cluster**, with the
`DSCI_AZ_*` and `DSCI_LISTMONK_*` credentials injected from the `dsci` secret
scope via `spark_env_vars`.

Chosen because: the multi-day AA lead times make the ~15–20 min per-run startup
(mostly library install) operationally negligible, while ephemeral clusters keep
the reproducible / person-independent / zero-idle-cost properties and match
ds-storms-pipeline's prod compute. A warm cluster would be faster per run but was
rejected (see cons). `source: GIT` is preferred over `source: WORKSPACE` for the
push-to-ship workflow; UI browsability is met separately by a Git folder under
`/Repos`, independent of how the jobs run.

### Consequences

* Good — reproducible, person-independent, no idle cost; library versions track
  `requirements.txt`; pipeline code stays pure Python (GHA fallback unaffected).
* Bad — ~15–20 min startup per run (dominated by library install on a bare
  cluster), which also slows dev iteration. Optimization options are documented
  in `databricks/README.md` ("Startup time").
* Implementation note — importing Python packages off the workspace FUSE mount
  (wsfs) is unreliable, so the wrapper copies `src` + `pipelines` to
  `/local_disk0` and runs from there.

### Confirmation

Both monitors were validated green (`TERMINATED SUCCESS`) via `dev`-target
dry-runs on the ephemeral cluster: DB read, library resolution, listmonk-backend
selection, and DRY_RUN simulation all worked end to end.

## Pros and Cons of the Options

### Ephemeral single-node job cluster (chosen)

* Good — isolated, reproducible, person-independent, zero idle cost.
* Bad — pays the full ~15 min library install every run.

### Warm all-purpose cluster, libraries pre-installed

* Good — fastest per run (libs already there).
* Bad — single-user ACLs (or a shared service principal to manage); idle cost;
  mutable shared state that other users/restarts can break; library set lives
  off-bundle and drifts from `requirements.txt`. Acceptable only as a dev-only
  convenience.

### Instance pool

* Neutral — shaves the ~4 min VM boot, not the ~15 min install; low ROI alone.

### Serverless jobs compute

* Good — fast start, env caching, scale-to-zero.
* Bad — needs serverless enabled, networking to reach Azure Postgres/blob, and a
  check that the geo stack installs cleanly; unproven for this stack here.

## More Information

* `databricks/README.md` — operations, gotchas, and startup-time optimizations.
* Forecast triggering is a separate decision: see
  `0002-trigger-forecast-monitor-from-upstream-pipeline.md`.
