# Databricks jobs — Cuba hurricane monitors

The two Cuba hurricane monitors run as scheduled **Databricks jobs** (defined in
the repo-root `databricks.yml`), mirroring the two GitHub Actions monitor crons:

| Job | Schedule (UTC) | Script |
|---|---|---|
| `fcast_monitor` | `30 3,9,15,21 * * *` (6-hourly, ~30 min after each NHC advisory) | `pipelines/01_update_fcast_monitor.py` |
| `obsv_monitor`  | `15 17 * * *` (once daily) | `pipelines/02_update_obsv_monitor.py` |

This is the production deployment vehicle for the listmonk email backend: the
jobs run with `EMAIL_BACKEND=listmonk`. The GitHub Actions monitor workflows
(`Forecast Monitor`, `Observational Monitor`) stay disabled as the manual
fallback (they still send via the humdata_email SMTP path).

## Architecture

- One bundle, two jobs, each a single `spark_python_task`.
- `source: GIT` clones this repo at `${var.git_branch}` **at run time**, so
  shipping a code change = pushing the branch (no redeploy needed). A
  `bundle deploy` is only needed when the *job config* (schedule, params,
  libraries, cluster) changes.
- Each task runs the thin wrapper `databricks/run_monitor_job.py`, which selects
  the monitor (`fcast`/`obsv`), injects the listmonk creds from the `dsci`
  secret scope, sets `EMAIL_BACKEND=listmonk` and the run-mode env vars, and
  shells out to the monitor script. Pipeline code (`pipelines/`, `src/`) stays
  pure Python and DBX-agnostic — the GHA workflows run the same scripts.
- Compute: the existing interactive cluster `${var.existing_cluster_id}` (the
  one ds-storms-pipeline uses), which already carries the `DSCI_AZ_*` DB/blob
  env vars — including the `*_PROD_*` ones the IMERG raster path reads.

## Target model — one live deploy, dev on demand

**Don't run two standing deploys.** Two targets scheduled on the same branch
just fire twice per advisory.

- **`prod`** (default target) is the only normally-deployed bundle. It serves the
  scheduled runs *and* ad-hoc manual runs (override params per run). Live params:
  `dry_run=False`, `test_email=False`, `force_alert=False`.
- **`dev`** is deployed **on demand for feature work**, pointed at a feature
  branch. Development mode auto-pauses both schedules and defaults to the test
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
2. Listmonk **sender** config in the `dsci` secret scope (already present —
   shared with ds-storms-alerts):
   `DSCI_LISTMONK_BASE_URL`, `DSCI_LISTMONK_API_USERNAME`, `DSCI_LISTMONK_API_KEY`.
3. The listmonk lists + dual-language template must exist on the instance the
   `DSCI_LISTMONK_BASE_URL` points at (run `pipelines/setup_cub_listmonk_lists.py`).

## Go-live checklist (nothing fires automatically on merge)

Merging this only adds the bundle definition — it does **not** deploy or
schedule anything. To take the monitors live on Databricks:

1. **Provision the production listmonk instance** + recreate the lists/template
   and point `dsci/DSCI_LISTMONK_BASE_URL` at it (currently the demo instance).
2. **Import the real subscribers** to the info/trigger lists.
3. `databricks bundle deploy -p DEFAULT` — deploys both prod jobs (schedules
   active in production mode). Confirm `dry_run=False`, `test_email=False`.
4. Leave the GitHub Actions monitor workflows **disabled** (they're the manual
   fallback). Re-enabling them while the DBX jobs run would double-fire.

## Gotchas / best practices (from the ds-storms-alerts deploy)

- **Don't set `schedule.pause_status: UNPAUSED` in the resource.** It overrides
  development mode's auto-pause and makes the `dev` jobs fire on the cron too.
  Leave it unset: production mode keeps prod running; development mode pauses dev.
- **The wrapper must not call `sys.exit()` at the top level.** `spark_python_task`
  treats a top-level `SystemExit` — *even code 0* — as a task failure. The wrapper
  raises an exception only on a non-zero child exit; success returns naturally.
- **Verify runs from the task logs, not the CLI exit code.** `databricks bundle run`
  can report exit 0 while the task itself failed. Check the run output for the
  monitor's log lines (or `databricks jobs get-run-output <task_run_id>`).
- **`source: GIT` means runtime = branch HEAD.** Pushing the branch updates the
  next run automatically; only config changes need a `bundle deploy`.
- **`matplotlib` needs a writable config dir** on the cluster (`MPLCONFIGDIR=/tmp/...`),
  since the cloned repo lives on the read-only workspace FUSE mount.
- **Library versions track `requirements.txt`.** `numpy`/`pandas` are left to the
  cluster runtime (DBR base); the rest are pinned. If a pin conflicts with the
  runtime, loosen it in `databricks.yml` and document why.

## Rollback

Re-enable the GitHub Actions schedules (the humdata_email fallback):
```bash
gh workflow enable "Forecast Monitor"
gh workflow enable "Observational Monitor"
```
Pause the DBX jobs in the workspace UI, or `databricks bundle destroy -p DEFAULT`.
Revert a deploy to the test list: `databricks bundle deploy -p DEFAULT --var test_email=True`.
