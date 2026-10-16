---
status: "proposed"
date: 2026-10-16
decision-makers: Zack Arno
---

# Publish the analysis book to GitHub Pages from a local render on the freeze

## Context and Problem Statement

The Quarto book in `book_cub_hurricanes/` was published to Netlify. It is moving to
GitHub Pages under the team's landing-page convention (landing page at `/`, book at
`/book/`). Most chapters execute against the team Postgres, blob storage and Google Earth
Engine, some slowly. The executed results live in `_freeze/`, which is not in git. So the
question is where the book gets rendered and how it reaches the site.

## Considered Options

* Render locally on the freeze, push `_book/` into `book/` on a branch-served `gh-pages`
  (`publish_book.sh`, adapted from `ds-aa-lac-dry-corridor`)
* Render in a GitHub Actions workflow and deploy the Pages artifact
* Keep Netlify

## Decision Outcome

Chosen option: local render pushed to `gh-pages`. Nothing has to execute to publish, and
the script only touches `book/`, so the landing page and any later products are safe.

### Consequences

* Good, because a publish runs no code. The script refuses to render if any chapter's
  freeze hash no longer matches its `.qmd`, unless `--allow-execute` is passed.
* Good, because one URL space holds the book, the landing page and an external card for
  the trigger redesign app.
* Bad, because `_freeze/` exists only on the machine that last executed the chapters. If
  it is lost, the only way back is re-executing every chapter, which needs DB, blob and GEE
  access.
* Bad, because rendered output is committed to the `gh-pages` branch, so its history grows
  with every publish.

## Pros and Cons of the Options

### Render in GitHub Actions

* Bad, because CI has no `_freeze/`, so it would re-execute every chapter. That needs DB,
  blob and GEE credentials in CI, and the heavy chapters would run on every deploy.
* Bad, because committing `_freeze/` to make CI work would put generated outputs (including
  figures) in `main`'s history, which the KB advises against.

### Keep Netlify

* Bad, because it sits outside the team's Pages registry and landing-page convention.

## More Information

* KB: `methods/static-data-apps.md` ("One repo, one Pages site", "GitHub Pages gotchas").
* `quarto publish gh-pages` must not be used: it runs `git rm -r .` on the whole branch.
* The Netlify site gets a `_redirects` rule pointing every path to the same path under
  `/book/`.
* Prose edits to frozen chapters were applied to both the `.qmd` and the frozen markdown
  in `_freeze/<chapter>/execute-results/html.json`, with the stored hash updated, so they
  render without executing.
