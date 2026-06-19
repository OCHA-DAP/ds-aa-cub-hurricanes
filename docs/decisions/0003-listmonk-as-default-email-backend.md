---
status: "accepted"
date: 2026-06-19
decision-makers: [Zack]
consulted: [Tristan]
---

# Make listmonk the default email backend

## Context and Problem Statement

`EMAIL_BACKEND` selects the email system: `listmonk` (ocha_relay campaigns, the
path we're moving to) or `humdata_email` (the legacy AWS-SES SMTP path). It first
shipped defaulting to `humdata_email` so merging was behaviour-neutral. But
listmonk is now the intended primary path, and defaulting to the legacy system is
backwards. What should the default be, and what role does humdata_email keep?

## Decision Drivers

* Listmonk is the primary path going forward; the default should reflect that.
* We don't want the legacy SMTP path used implicitly.
* The GitHub Actions monitor workflows are being retired in favour of the
  Databricks jobs (see [[0001-run-monitors-as-databricks-jobs-on-ephemeral-clusters]]).

## Considered Options

* Keep `humdata_email` as the default; listmonk opt-in.
* Make `listmonk` the default; `humdata_email` only when explicitly selected.
* Automatic failover from listmonk to humdata_email on error.

## Decision Outcome

Default `EMAIL_BACKEND=listmonk`. `humdata_email` stays in the code as a **manual
escape hatch**, used only by explicitly setting `EMAIL_BACKEND=humdata_email` in
an environment that has the `DSCI_AWS_EMAIL_*` creds — and is deliberately **not**
wired into the GitHub Actions workflows. No automatic failover (a silent switch
between sending systems is worse than a visible failure).

### Consequences

* Good — the primary path is the default; the legacy system is never used
  implicitly.
* Good — still safe on merge: nothing auto-runs (monitor crons disabled, no
  bundle deployed), so flipping the default sends nothing by merging.
* Bad — the GHA monitor workflows are no longer a working fallback: with listmonk
  the default and no listmonk creds in GitHub, re-enabling one would fail rather
  than send via SMTP. Accepted — those workflows are retired; the SMTP fallback
  is the manual escape hatch above.
* The automated Databricks path sends unattended because its wrapper sets
  `LISTMONK_SKIP_CONFIRMATION=true`; local/manual runs keep ocha_relay's
  interactive typed-confirmation (it would otherwise `EOFError` headless and
  silently not send).

## More Information

* Backend routing: `src/email/backends.py` (selects at import on `EMAIL_BACKEND`).
* Unsubscribe handling for listmonk recipients:
  [[0004-suppress-unsubscribe-via-subscriber-attribute]].
