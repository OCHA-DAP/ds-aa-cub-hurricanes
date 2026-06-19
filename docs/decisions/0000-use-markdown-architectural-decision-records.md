---
status: "accepted"
date: 2026-06-18
decision-makers: [Zack]
---

# Use Markdown Architectural Decision Records

## Context and Problem Statement

This project makes technical decisions whose rationale is easy to lose — compute
models, data-source switches, triggering mechanisms, service and dependency
choices. Today that reasoning is scattered across commit messages, PR threads,
and code comments, or recorded nowhere, so the *rejected alternatives* and the
*why* disappear. How should we record significant decisions so they survive for
future maintainers (and for Claude)?

## Considered Options

* MADR (Markdown Any Decision Records)
* Another ADR template (e.g. Nygard, Tyree & Akerman)
* No formal records — status quo (commit messages / PR descriptions only)

## Decision Outcome

Chosen option: "MADR", because it is lightweight Markdown that lives in the repo
next to the code, and — unlike a commit message — it captures the considered
options and their trade-offs, not just the outcome. Records live in
`docs/decisions/`; new ones copy `template.md`. Scope is kept deliberately narrow
(see `CLAUDE.md`): only decisions with real trade-offs get an ADR, so the
practice stays non-intrusive.

### Consequences

* Good, because the rationale and the rejected alternatives are preserved where
  people will look for them.
* Good, because it is low-ceremony — a short Markdown file per decision.
* Bad, because it adds a small step when making a significant decision; mitigated
  by recording only decisions a maintainer would later question.

## More Information

* MADR: <https://adr.github.io/madr/>
* Adapted from developmentseed/stac-map's ADR 0000.
