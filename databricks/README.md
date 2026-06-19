# Databricks jobs — Cuba hurricane monitors

The two Cuba hurricane monitors run as **Databricks jobs** (defined in the
repo-root `databricks.yml`):

| Job | Trigger | Script |
|---|---|---|
| `fcast_monitor` | **Run-job task from `ds-storms-pipeline`'s `nhc_pipeline`** (once NHC tracks land in Postgres). No own schedule. | `pipelines/01_update_fcast_monitor.py` |
| `obsv_monitor`  | Schedule `15 17 * * *` (once daily, UTC) | `pipelines/02_update_obsv_monitor.py` |

This is the production deployment vehicle for the listmonk email backend: the
jobs run with `EMAIL_BACKEND=listmonk`. The GitHub Actions monitor workflows
(`Forecast Monitor`, `Observational Monitor`) stay disabled as the manual
fallback (they still send via the humdata_email SMTP path).

## Why the forecast monitor is triggered, not scheduled

The forecast trigger's correctness depends on the upstream `ds-storms-pipeline`
landing the NHC tracks in `storms.nhc_tracks_geo` **before** this monitor reads
them. A time-based schedule is only a guess at when that finishes (the upstream
even has a `:30` late-WSP retry). So instead, `nhc_pipeline` runs this monitor as
a downstream `run_job_task` once its tracks task has completed — event-driven, no
race. The observational monitor has no such dependency, so it stays on a plain
daily schedule.

## Architecture

- One bundle, two jobs, each a single `spark_python_task`.
- `source: GIT` clones this repo at `${var.git_branch}` **at run time**, so
  shipping a code change = pushing the branch (no redeploy needed). A
  `bundle deploy` is only needed when the *job config* (schedule, params,
  libraries, cluster) changes.
- Each task runs the thin wrapper `databricks/run_monitor_job.py`, which selects
  the monitor (`fcast`/`obsv`), sets `EMAIL_BACKEND=listmonk` and the run-mode
  env vars, **copies `src` + `pipelines` from the wsfs-mounted clone onto local
  disk**, and shells out to the monitor script from there. Pipeline code
  (`pipelines/`, `src/`) stays pure Python and DBX-agnostic — the GHA workflows
  run the same scripts. (The copy-to-local step is required because importing
  Python packages off the wsfs FUSE mount is unreliable — see Gotchas.)
- **Compute: an ephemeral single-node job cluster per run** (not anyone's
  personal interactive cluster). It starts bare, so every credential the
  pipeline reads is injected from the `dsci` secret scope via the cluster's
  `spark_env_vars`: `DSCI_AZ_*` (DB/blob — dev for NHC tracks, prod for IMERG,
  consumed by ocha_stratus) and `DSCI_LISTMONK_*` (sender creds, consumed by the
  listmonk dispatch). Ephemeral was a deliberate choice over a warm
  pre-installed cluster: the AA lead times are days, so the ~15–20 min cold
  start + library install per run is operationally negligible, and we keep the
  reproducible / person-independent / zero-idle-cost properties. See "Startup
  time" below for the trade-offs and possible optimizations.

## Target model — one live deploy, dev on demand

**Don't run two standing deploys.** Two targets on the same branch just double
up.

- **`prod`** (default target) is the only normally-deployed bundle. It serves the
  obsv schedule + the triggered fcast runs, plus ad-hoc manual runs. Live params:
  `dry_run=False`, `test_email=False`, `force_alert=False`.
- **`dev`** is deployed **on demand for feature work**, pointed at a feature
  branch. Development mode auto-pauses the obsv schedule and defaults to the test
  list, so it never competes with prod. Destroy it when the feature merges.

## Common commands

```bash
# --- Live jobs (prod is the default target) ---
databricks bundle validate -p DEFAULT
databricks bundle deploy   -p DEFAULT                                       # deploy/update the live jobs
databricks bundle run fcast_monitor -p DEFAULT                              # manual run — SENDS FOR REAL
databricks bundle run fcast_monitor -p DEFAULT --params dry_run=True,test_email=True   # safe manual test
databricks bundle run obsv_monitor  -p DEFAULT --params dry_run=True,test_email=True

# --- Feature development (throwaway dev jobs from your branch) ---
databricks bundle deploy  -t dev -p DEFAULT --var git_branch=my-feature
databricks bundle run     fcast_monitor -t dev -p DEFAULT   # paused schedule, test list
databricks bundle destroy -t dev -p DEFAULT                 # when the feature merges
```

## Run-mode params (the GHA-vars analog)

`dry_run`, `test_email`, and `force_alert` are bundle **variables** wired into
each job's parameter defaults, so the mode flips at deploy time without editing
files (`--var test_email=False`). Per run, override with `--params`.

| | `dry_run` | `test_email` | `force_alert` |
|---|---|---|---|
| `"True"` | run monitor but **skip** send/write | route all sends to the **[TEST] list** | inject **dummy storm data** (PRUEBA) to force an alert |
| `"False"` | **actually send/write** | send to the **real info/trigger audiences** | real data only |

⚠️ The default target is the **live** one: a bare `bundle run fcast_monitor`
**sends for real**. For a safe manual check pass
`--params dry_run=True,test_email=True`, or use a `dev` deploy.

## Prerequisites (one-time)

1. Workspace GitHub credentials for this repo (same mechanism as
   ds-storms-pipeline).
2. The `dsci` secret scope must carry the keys the cluster injects (all already
   present, shared with the rest of the storms stack): the `DSCI_AZ_*` DB/blob
   creds and the listmonk **sender** creds (`DSCI_LISTMONK_BASE_URL`,
   `DSCI_LISTMONK_API_USERNAME`, `DSCI_LISTMONK_API_KEY`).
3. The listmonk lists + dual-language template must exist on the instance the
   `DSCI_LISTMONK_BASE_URL` points at (run `pipelines/setup_cub_listmonk_lists.py`).

## Go-live checklist (nothing fires automatically on merge)

Merging this only adds the bundle definition — it does **not** deploy or
schedule anything. To take the monitors live on Databricks:

1. **Import the real subscribers** into the info/trigger lists
   (`python pipelines/setup_cub_listmonk_lists.py`). They are created with
   `attribs.type=mailing_list`, so the campaign template omits the unsubscribe
   link for them.
2. `databricks bundle deploy -p DEFAULT` — deploys both prod jobs (obsv schedule
   active; fcast trigger-only). Confirm `dry_run=False`, `test_email=False`.
   (Unattended sending already works: the wrapper sets
   `LISTMONK_SKIP_CONFIRMATION=true`.)
3. Deploy the matching `ds-storms-pipeline` change so `nhc_pipeline` triggers the
   prod `fcast_monitor` (point its `cub_fcast_job_id` var at the prod job id) —
   OCHA-DAP/ds-storms-pipeline#33.
4. Leave the GitHub Actions monitor workflows **disabled**. They are retired, not
   a live fallback: listmonk is now the default backend and GitHub carries no
   listmonk creds, so re-enabling one would fail rather than send.

There is a single listmonk instance; if a separate production instance is ever
added, repoint `dsci/DSCI_LISTMONK_BASE_URL` and re-run the setup script then.

## Gotchas / best practices (from the ds-storms-alerts/pipeline deploys)

- **Don't set `schedule.pause_status: UNPAUSED` in the resource.** It overrides
  development mode's auto-pause and makes the `dev` obsv job fire on the cron too.
  Leave it unset: production mode keeps prod running; development mode pauses dev.
- **The wrapper must not call `sys.exit()` at the top level.** `spark_python_task`
  treats a top-level `SystemExit` — *even code 0* — as a task failure. The wrapper
  raises an exception only on a non-zero child exit; success returns naturally.
- **The ephemeral job cluster starts bare.** Every credential the pipeline reads
  must be in the cluster's `spark_env_vars` (resolved from `dsci`). If a new env
  var is added to the pipeline, add it there too.
- **Verify runs from the task logs, not the CLI exit code.** `databricks bundle run`
  can report exit 0 while the task itself failed. Check the run output for the
  monitor's log lines (or `databricks jobs get-run-output <task_run_id>`).
- **`source: GIT` means runtime = branch HEAD.** Pushing the branch updates the
  next run automatically; only config changes need a `bundle deploy`.
- **Don't import Python code directly off the wsfs mount.** Under `source: GIT`
  the repo is cloned onto the workspace FUSE mount (wsfs), which raises hard
  filesystem errors (not a clean "not found") when the import machinery probes
  candidate filenames per module (`__init__.cpython-*.so`, `matplotlibrc`,
  `__pycache__`, …). This intermittently breaks `from src ...`. The wrapper
  copies `src` + `pipelines` to `/local_disk0` and runs from there; if you add a
  new top-level package the pipeline imports, add it to that copy list.
- **`matplotlib` needs a writable config dir** (`MPLCONFIGDIR=/tmp/...`).
- **Library versions track `requirements.txt`.** `numpy`/`pandas` are left to the
  cluster runtime (DBR base); the rest are pinned. If a pin conflicts with the
  runtime, loosen it in `databricks.yml` and document why.

## Startup time & possible optimizations

Each run is ~15–20 min before the monitor logic starts, almost all of it
**library install** on the bare ephemeral cluster (geopandas, rioxarray, xarray,
kaleido, `ocha-relay` from git), not cluster boot (~4 min). This is accepted on
purpose (see Architecture). If it ever needs cutting, in rough order of
leverage/risk:

- **Trim the dependency list** — lean on the DBR base where possible and drop
  `plotly`/`kaleido` if matplotlib can render the map/scatter plots (kaleido is a
  large chunk of the install). Stacks with any compute model.
- **Serverless jobs compute** — fast start + env caching, scale-to-zero; needs
  serverless enabled, network config to reach Azure Postgres/blob, and a check
  that the geo stack installs cleanly.
- **Custom container image / init script** with the libs baked in — near-zero
  install, still reproducible; costs an image to build and maintain.
- An **instance pool** alone is low ROI: it shaves the ~4 min boot, not the
  ~15 min install.
- A warm **all-purpose cluster** with libs pre-installed is fastest per-run but
  was rejected for prod: single-user ACLs, idle cost, mutable shared state, and
  the lib set living off-bundle (drifts from `requirements.txt`). Fine as a
  *dev-only* convenience if you want faster iteration.

## Rollback

Stop the DBX jobs: pause them in the workspace UI, or
`databricks bundle destroy -p DEFAULT`. Revert a deploy to the test list:
`databricks bundle deploy -p DEFAULT --var test_email=True`.

The humdata_email SMTP path remains as a manual escape hatch, but it is **not**
the GitHub Actions workflows: listmonk is the default backend and GitHub has no
listmonk creds, so re-enabling those workflows would fail, not send. To use the
SMTP fallback, run a monitor with `EMAIL_BACKEND=humdata_email` in an environment
that has the `DSCI_AWS_EMAIL_*` creds.
