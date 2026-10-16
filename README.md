# Pages site (`gh-pages` branch)

**Live:** https://ocha-dap.github.io/ds-aa-cub-hurricanes/

This branch *is* the site: GitHub Pages serves it as-is (branch mode, no build step).
It follows the team's landing-page convention (ds-knowledge-base
`methods/static-data-apps.md`): a landing page at `/`, each product under its own path.

| Path | What | Source |
|---|---|---|
| `/` | Landing page (`index.html`, `assets/`) | edited here |
| `/book/` | Quarto book: 2025 trigger development | `book_cub_hurricanes/` on `main`, published by `publish_book.sh` |

The landing page also links to the trigger redesign app on Azure
(`chd-ds-cub-trigger.azurewebsites.net`), which is hosted outside this site.

## Never run `quarto publish gh-pages`

It runs `git rm -r .` on this branch before copying the book in, so it deletes the landing
page. Publish the book with `book_cub_hurricanes/publish_book.sh` instead. It only touches
`book/` and refuses to push if anything else changed.

## The book renders from a local freeze

`book_cub_hurricanes/_freeze/` is not in git. The book can be re-rendered without running
any code only on a machine that has it. `publish_book.sh` checks this and refuses to render
if any chapter would re-execute, unless given `--allow-execute`.

## Adding a page

1. Commit a directory with an `index.html` (e.g. `my-product/index.html`).
2. Add a card to `index.html`: copy an existing `<a class="k">` block and change the href,
   title, blurb and foot.
3. Put a back-to-landing link at the top of the new page (snippet: ds-knowledge-base
   `methods/static-data-apps.md`, "Nested pages link back home"; the book's copy is
   `book_cub_hurricanes/home-link.html`).
4. Declare the URL in the KB: `surfaces:` on the `frameworks/cub-hurricanes/` page.
