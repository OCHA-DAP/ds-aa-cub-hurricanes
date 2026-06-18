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
import subprocess
import sys

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

# Make `src` importable for the child process (repo isn't pip-installed here).
env = dict(os.environ)
env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
# Give matplotlib a writable local config/cache dir (the cloned repo lives on
# the read-only workspace FUSE mount, which it can't use).
env["MPLCONFIGDIR"] = "/tmp/mplconfig"
# Don't write __pycache__/*.pyc next to the imported sources: under source: GIT
# the repo lives on the workspace FUSE mount (wsfs), which rejects __pycache__
# ("operation not supported") and would otherwise break every `from src ...`.
env["PYTHONDONTWRITEBYTECODE"] = "1"

cmd = [sys.executable, os.path.join(REPO_ROOT, _MONITOR_SCRIPTS[MONITOR])]

# Run from a normal local dir, NOT the cloned repo (which is on the wsfs FUSE
# mount). Libraries that scan the cwd on init — e.g. matplotlib looking for a
# ./matplotlibrc — hit a wsfs filesystem error there. The monitors import via
# PYTHONPATH and use __file__-based paths, so they don't depend on cwd; /tmp is
# also writable for any incidental temp files, unlike the read-only mount.
RUN_CWD = "/tmp"

if __name__ == "__main__":
    print(
        f"[run_monitor_job] repo_root={REPO_ROOT} monitor={MONITOR} "
        f"EMAIL_BACKEND=listmonk DRY_RUN={DRY_RUN} TEST_EMAIL={TEST_EMAIL} "
        f"FORCE_ALERT={FORCE_ALERT}"
    )
    rc = subprocess.run(cmd, cwd=RUN_CWD, env=env, check=False).returncode
    # DBX treats a top-level sys.exit()/SystemExit (even code 0) as a task
    # failure. Raise only on non-zero; let success return naturally.
    if rc != 0:
        raise RuntimeError(f"{_MONITOR_SCRIPTS[MONITOR]} exited with code {rc}")
    print("[run_monitor_job] OK")
