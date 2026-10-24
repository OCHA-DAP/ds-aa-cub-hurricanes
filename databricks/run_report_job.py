"""DBX entry wrapper for the Daily Hurricane Report (Report.qmd → email).

Bundle job ``daily_report`` (databricks.yml). Positional parameters:

    sys.argv[1] = recipients   # "" = the blob distribution list; else a
                               #      comma-separated override (test runs)

The report is a Quarto document executed with the Jupyter engine, so unlike
the monitors this needs the Quarto CLI on the cluster. The wrapper:

1. downloads the pinned Quarto CLI tarball onto local disk (once per
   cluster) and registers the ``ds-aa-cub-hurricanes`` Jupyter kernel the
   qmd front matter names, on the cluster's own Python;
2. pulls the SES sender creds (``DSCI_AWS_EMAIL_*``) from the ``dsci`` secret
   scope — the Job Compute policy injects the DB/blob creds the report's
   NHC reader (storms.nhc_* on prod) needs, but not the email ones;
3. copies the rendering footprint (``src``, ``pipelines``, ``Report.qmd``,
   the email template) off the wsfs checkout onto local disk — importing
   straight off the workspace FUSE mount is unreliable, see
   run_monitor_job.py — and runs ``pipelines/email_with_embedded_images.py``
   from there.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

QUARTO_VERSION = "1.10.18"
QUARTO_URL = (
    "https://github.com/quarto-dev/quarto-cli/releases/download/"
    f"v{QUARTO_VERSION}/quarto-{QUARTO_VERSION}-linux-amd64.tar.gz"
)
KERNEL_NAME = "ds-aa-cub-hurricanes"  # must match Report.qmd `jupyter: kernel:`
_COPY_DIRS = ("src", "pipelines")
_COPY_FILES = ("Report.qmd", "quarto-email-template.html")


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


def _local_root() -> str:
    base = "/local_disk0" if os.path.isdir("/local_disk0") else tempfile.gettempdir()
    return os.path.join(base, "cub_daily_report")


def install_quarto(dest: str) -> str:
    """Download + extract the Quarto CLI; return the bin dir. Idempotent."""
    bin_dir = os.path.join(dest, f"quarto-{QUARTO_VERSION}", "bin")
    if os.path.exists(os.path.join(bin_dir, "quarto")):
        return bin_dir
    os.makedirs(dest, exist_ok=True)
    tarball = os.path.join(dest, "quarto.tar.gz")
    print(f"[run_report_job] downloading {QUARTO_URL}")
    urllib.request.urlretrieve(QUARTO_URL, tarball)
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)
    os.remove(tarball)
    if not os.path.exists(os.path.join(bin_dir, "quarto")):
        raise RuntimeError(f"quarto binary not found under {bin_dir}")
    return bin_dir


def register_kernel(env: dict) -> None:
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--user",
         f"--name={KERNEL_NAME}", f"--display-name={KERNEL_NAME}"],
        check=True,
        env=env,
    )


def main() -> None:
    recipients = _arg(1, "").strip()
    repo_root = os.path.abspath(os.path.join(_find_script_dir(), ".."))
    local_root = _local_root()

    for sub in _COPY_DIRS:
        shutil.copytree(
            os.path.join(repo_root, sub),
            os.path.join(local_root, sub),
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in _COPY_FILES:
        shutil.copy2(os.path.join(repo_root, name), os.path.join(local_root, name))

    # SES sender creds from the dsci scope (not in the cluster policy env).
    try:
        from databricks.sdk.runtime import dbutils

        for key in (
            "DSCI_AWS_EMAIL_HOST",
            "DSCI_AWS_EMAIL_ADDRESS",
            "DSCI_AWS_EMAIL_USERNAME",
            "DSCI_AWS_EMAIL_PASSWORD",
        ):
            try:
                os.environ[key] = dbutils.secrets.get("dsci", key)
            except Exception as exc:  # noqa: BLE001
                print(f"[run_report_job] WARNING: dsci/{key} unavailable ({exc})")
    except ImportError:
        pass

    quarto_bin = install_quarto(os.path.join(local_root, "quarto"))

    env = dict(os.environ)
    env["PATH"] = quarto_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = local_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/mplconfig"
    env["HOME"] = env.get("HOME") or "/tmp"
    # Quarto's jupyter engine must use the cluster Python (where the libs are).
    env["QUARTO_PYTHON"] = sys.executable
    if recipients:
        env["REPORT_RECIPIENTS"] = recipients
    register_kernel(env)

    print(
        f"[run_report_job] repo_root={repo_root} local_root={local_root} "
        f"quarto={quarto_bin} recipients={'override' if recipients else 'distribution list'}"
    )
    cmd = [sys.executable, os.path.join(local_root, "pipelines", "email_with_embedded_images.py")]
    rc = subprocess.run(cmd, cwd=local_root, env=env, check=False).returncode
    # DBX treats a top-level SystemExit (even 0) as failure; raise only on
    # non-zero.
    if rc != 0:
        raise RuntimeError(f"email_with_embedded_images.py exited with code {rc}")
    print("[run_report_job] OK")


if __name__ == "__main__":
    main()
