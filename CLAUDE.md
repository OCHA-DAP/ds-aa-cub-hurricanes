# ds-aa-cub-hurricanes — project notes for Claude

## Architectural Decision Records (ADRs)

Significant or non-obvious technical decisions in this repo are recorded as
**ADRs** in `docs/decisions/`, using the MADR format. Copy
`docs/decisions/template.md`; the practice itself is recorded in
`docs/decisions/0000-use-markdown-architectural-decision-records.md`.

When you make or implement a decision here, apply this lightly:

- **Write an ADR when** the change settles something a future maintainer would
  ask *"why was it done this way?"* about — a decision with real trade-offs or
  rejected alternatives (e.g. compute model, a data-source switch, a triggering
  mechanism, choosing a service/dependency). Routine bug fixes, refactors,
  renames, dependency bumps, and obvious changes do **not** need one.
- **Keep it short** — a few sentences of context, the options weighed, the choice
  and why. The highest-value part is the *rejected* options and their cons; that
  is what commit messages and code lose.
- **Be non-intrusive.** Propose the ADR alongside the change; don't block work to
  write it, and don't manufacture decisions just to document them. One ADR per
  decision.
- **Naming:** `NNNN-short-title-with-dashes.md`, zero-padded next number.
- **Status** lives in the frontmatter (`proposed` → `accepted`). Supersede an old
  ADR (mark it, link the new one) rather than silently contradicting it.

If a decision is worth an ADR but now isn't the moment, say so and offer to write
it — don't drop it silently.
