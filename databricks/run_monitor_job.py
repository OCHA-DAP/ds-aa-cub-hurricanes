"""DBX entry wrapper for the Cuba hurricane monitors.

The bundle's ``spark_python_task`` passes the job parameters positionally:

    sys.argv[1] = monitor      # "fcast" | "obsv" (which monitor to run)
    sys.argv[2] = dry_run      # "True" | "False"
    sys.argv[3] = test_email   # "True" | "False"
    sys.argv[4] = force_alert  # "True" | "False"

The monitor scripts (pipelines/01_update_fcast_monitor.py /
02_update_obsv_monitor.py) and src/ stay pure Python — they don't know about
DBX (the GitHub Actions workflows run the same scripts). This wrapper is the
only DBX-specific glue and does two things:

1. Select the listmonk email backend and set the run-mode env vars the monitor
   reads at import (``EMAIL_BACKEND=listmonk`` so dispatch goes through
   ocha_relay rather than the humdata_email SMTP fallback, plus
   ``DRY_RUN`` / ``TEST_EMAIL`` / ``FORCE_ALERT``). The actual credentials
   (``DSCI_AZ_*`` DB/blob, ``DSCI_LISTMONK_*`` sender) are NOT set here — the
   job cluster injects them from the ``dsci`` secret scope via spark_env_vars
   (see databricks.yml), so they are already in the environment.

2. Shell out to the monitor script with ``PYTHONPATH`` set to the repo root so
   ``from src ...`` resolves — under ``source: GIT`` the repo is cloned but not
   pip-installed, and the scripts live in ``pipelines/`` rather than at the
   root.
"""

import os
import shutil
import subprocess
import sys
import tempfile

# monitor selector -> the pure-Python entry script it maps to.
_MONITOR_SCRIPTS = {
    "fcast": "pipelines/01_update_fcast_monitor.py",
    "obsv": "pipelines/02_update_obsv_monitor.py",
}


def _find_script_dir() -> str:
    """spark_python_task's exec context doesn't always define __file__."""
    try:
        return os.path.dirname(os.path.abspath(__file__))  # noqa: F821
    except NameError:
        pass
    if sys.argv and sys.argv[0]:
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.getcwd()


def _arg(i: int, default: str = "") -> str:
    return sys.argv[i] if len(sys.argv) > i else default


REPO_ROOT = os.path.abspath(os.path.join(_find_script_dir(), ".."))

MONITOR = _arg(1, "fcast")
DRY_RUN = _arg(2, "True")
TEST_EMAIL = _arg(3, "True")
FORCE_ALERT = _arg(4, "False")

if MONITOR not in _MONITOR_SCRIPTS:
    raise ValueError(
        f"unknown monitor {MONITOR!r}; expected one of "
        f"{sorted(_MONITOR_SCRIPTS)}"
    )

# Route email dispatch through listmonk (ocha_relay) rather than the legacy
# humdata_email SMTP backend. The DSCI_LISTMONK_* / DSCI_AZ_* credentials are
# supplied by the job cluster's spark_env_vars (resolved from the dsci secret
# scope), so they are already present in the environment here.
os.environ["EMAIL_BACKEND"] = "listmonk"
os.environ["DRY_RUN"] = DRY_RUN
os.environ["TEST_EMAIL"] = TEST_EMAIL
os.environ["FORCE_ALERT"] = FORCE_ALERT
# Headless: ocha_relay's send_campaign would otherwise prompt for typed
# confirmation and raise EOFError here. Skip the prompt so the job can send
# unattended (a local/manual run, which does not set this, stays interactive).
os.environ["LISTMONK_SKIP_CONFIRMATION"] = "true"

# Under source: GIT the repo is cloned onto the workspace FUSE mount (wsfs).
# Importing Python packages whose directories live on wsfs is unreliable: the
# import machinery probes several candidate filenames per module
# (__init__.cpython-*.so, __init__.py, ...) and wsfs raises a hard filesystem
# error for the non-existent candidates instead of a clean "not found",
# intermittently breaking `from src ...` (and likewise breaks __pycache__ writes
# and matplotlib's cwd scan). So copy the Python sources onto local disk and
# import/run from there — wsfs is then off the import path and the cwd entirely.
# The monitors read all their data from blob/DB, so src + pipelines is the whole
# footprint (templates live under src/).
LOCAL_ROOT = os.path.join(
    "/local_disk0" if os.path.isdir("/local_disk0") else tempfile.gettempdir(),
    "cub_monitor_run",
)
for _sub in ("src", "pipelines"):
    shutil.copytree(
        os.path.join(REPO_ROOT, _sub),
        os.path.join(LOCAL_ROOT, _sub),
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

env = dict(os.environ)
env["PYTHONPATH"] = LOCAL_ROOT + os.pathsep + env.get("PYTHONPATH", "")
# matplotlib needs a writable config/cache dir (and must not land on wsfs).
env["MPLCONFIGDIR"] = "/tmp/mplconfig"
# Unbuffered stdout/stderr: the child's stdout is block-buffered when piped (not
# a TTY), so print() output (e.g. the per-email "sending …" lines) can be
# dropped or reordered relative to logging in the captured task output. Force
# line-buffering so every send/skip line is reliably visible in the run logs.
env["PYTHONUNBUFFERED"] = "1"

cmd = [sys.executable, os.path.join(LOCAL_ROOT, _MONITOR_SCRIPTS[MONITOR])]

if __name__ == "__main__":
    print(
        f"[run_monitor_job] repo_root={REPO_ROOT} local_root={LOCAL_ROOT} "
        f"monitor={MONITOR} EMAIL_BACKEND=listmonk DRY_RUN={DRY_RUN} "
        f"TEST_EMAIL={TEST_EMAIL} FORCE_ALERT={FORCE_ALERT}"
    )
    rc = subprocess.run(cmd, cwd=LOCAL_ROOT, env=env, check=False).returncode
    # DBX treats a top-level sys.exit()/SystemExit (even code 0) as a task
    # failure. Raise only on non-zero; let success return naturally.
    if rc != 0:
        raise RuntimeError(f"{_MONITOR_SCRIPTS[MONITOR]} exited with code {rc}")
    print("[run_monitor_job] OK")
