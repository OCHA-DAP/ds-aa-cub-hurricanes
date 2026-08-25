# Databricks jobs — Cuba hurricane monitors

The two Cuba hurricane monitors run as **Databricks jobs** (defined in the
repo-root `databricks.yml`):

| Job | Trigger | Script |
|---|---|---|
| `fcast_monitor` | **Run-job task from `ds-storms-pipeline`'s `nhc_pipeline`** (once NHC tracks land in Postgres). No own schedule. | `pipelines/01_update_fcast_monitor.py` |
| `obsv_monitor`  | Schedule `15 17 * * *` (once daily, UTC) | `pipelines/02_update_obsv_monitor.py` |

This is the production deployment vehicle for the listmonk email backend: the
jobs run with `EMAIL_BACKEND=listmonk`. **Live in production since 2026-06-30**
(real audience: ~50 info / ~29 trigger recipients). The GitHub Actions monitor
workflows (`Forecast Monitor`, `Observational Monitor`) stay **disabled** —
they're retired, not a live fallback (listmonk is the default backend and GitHub
has no listmonk creds, so re-enabling one would fail rather than send). See
Rollback for the humdata_email escape hatch.

## Why the forecast monitor is triggered, not scheduled

The forecast trigger's correctness depends on the upstream `ds-storms-pipeline`
landing the NHC tracks in `storms.nhc_tracks_geo` **before** this monitor reads
them. A time-based schedule is only a guess at when that finishes (the upstream
even has a `:30` late-WSP retry). So instead, `nhc_pipeline` runs this monitor as
a downstream `run_job_task` once its tracks task has completed — event-driven, no
race. The upstream fires twice per cycle (`:00` + a `:30` WSP-retry), but only
triggers this monitor **once per new advisory** (see OCHA-DAP/ds-storms-pipeline#35).
The observational monitor has no such dependency, so it stays on a plain daily
schedule.

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
- **Compute: an ephemeral job cluster per run, under the "Job Compute" cluster
  policy** (`000C79D951EAF0D6`) — required since unrestricted compute is no
  longer available to the deployer. The policy forbids single-node and fixes
  `num_workers=1`, so each run is a **driver + 1 worker** (fine — the monitors
  are plain Python and run on the driver). It also **injects the DB/blob/IMERG
  credentials** (`DSCI_AZ_*`, `IMERG_*`, `STORAGE_ACCOUNT_*`, `CONTAINER_*`,
  consumed by ocha_stratus) as hidden `spark_env_vars`, so the bundle's cluster
  spec only adds `DSCI_LISTMONK_*` (the listmonk sender creds the policy doesn't
  provide). Ephemeral was a deliberate choice over a warm pre-installed cluster:
  the AA lead times are days, so the ~15–20 min cold start + library install per
  run is operationally negligible, and we keep the reproducible /
  person-independent / zero-idle-cost properties. See "Startup time" below.

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
| `"True"` | run monitor but **skip** send/write | route all sends to the **[TEST] list** | inject **dummy storm data** to force an alert (subject prefixed `[TEST]`) |
| `"False"` | **actually send/write** | send to the **real info/trigger audiences** | real data only |

⚠️ The default target is the **live** one: a bare `bundle run fcast_monitor`
**sends for real**. For a safe manual check pass
`--params dry_run=True,test_email=True`, or use a `dev` deploy.

## Prerequisites (one-time)

1. Workspace GitHub credentials for this repo (same mechanism as
   ds-storms-pipeline).
2. The `dsci` secret scope must carry the keys referenced by the Job Compute
   policy (`DSCI_AZ_*` DB/blob, `IMERG_*`, etc.) and by the bundle (the listmonk
   **sender** creds `DSCI_LISTMONK_BASE_URL` / `_API_USERNAME` / `_API_KEY`) —
   all already present, shared with the rest of the storms stack.
3. The listmonk lists + dual-language template must exist on the instance the
   `DSCI_LISTMONK_BASE_URL` points at (run `pipelines/setup_cub_listmonk_lists.py`).
4. The identity that triggers the forecast monitor (the `ds-storms-pipeline`
   `nhc_pipeline`'s run-as) needs `CAN_MANAGE_RUN` on the `fcast_monitor` job.
   The `dsci` group is granted this, which covers the dev (`adm.tdowning`) and
   prod (`adm.hker1`) run-as identities.

## Operations

The monitors are **live** (since 2026-06-30): `test_email=False`, sending to the
real info (`106`) / trigger (`107`) lists, fcast triggered by `ds-storms-pipeline`
and obsv on its daily schedule. Merging the bundle never deploys or sends anything
on its own — a `bundle deploy` is the only thing that updates the live jobs.

Common tasks:

- **Change the audience.** Re-populating the lists *is* the live switch — sends
  follow whatever is on `106`/`107`. The import is **additive**, so to *replace*
  the audience, clear the lists first, then
  `python pipelines/setup_cub_listmonk_lists.py` (reads `distribution_list.csv`).
  New subscribers get `attribs.type=mailing_list` so the template omits the
  unsubscribe link; pre-existing subscribers keep their own attribs (ADR-0004 —
  a known scope limit).
- **Soak-test the live path safely.** `… setup_cub_listmonk_lists.py --test`
  imports the safe `test_distribution_list.csv` recipients into the real lists,
  so a `test_email=False` run only reaches them. Clear before importing the real
  list afterwards.
- **Route everything to the `[TEST]` list** (config test) without touching the
  real lists: deploy `--var test_email=True`, or per run `--params test_email=True`.
- **Single listmonk instance.** If a separate prod instance is ever stood up,
  repoint `dsci/DSCI_LISTMONK_BASE_URL` and re-run the setup script there.

## Gotchas / best practices (from the ds-storms-alerts/pipeline deploys)

- **Don't set `schedule.pause_status: UNPAUSED` in the resource.** It overrides
  development mode's auto-pause and makes the `dev` obsv job fire on the cron too.
  Leave it unset: production mode keeps prod running; development mode pauses dev.
- **The wrapper must not call `sys.exit()` at the top level.** `spark_python_task`
  treats a top-level `SystemExit` — *even code 0* — as a task failure. The wrapper
  raises an exception only on a non-zero child exit; success returns naturally.
- **Credentials come from the Job Compute policy + the bundle.** The policy
  injects the DB/blob/IMERG creds (hidden `spark_env_vars`); the bundle's cluster
  spec adds only `DSCI_LISTMONK_*`. If the pipeline needs a new credential, add it
  to the bundle's `spark_env_vars` — or, if it's shared across the storms stack,
  to the policy. Also note the policy **fixes** some attributes (e.g.
  `spot_bid_max_price=100`, `num_workers=1`): the spec must match those or the
  deploy is rejected.
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
**library install** on the fresh policy job cluster (geopandas, rioxarray,
xarray, kaleido, `ocha-relay` from git), not cluster boot (~4 min). This is
accepted on purpose (see Architecture). If it ever needs cutting, in rough order
of leverage/risk:

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
