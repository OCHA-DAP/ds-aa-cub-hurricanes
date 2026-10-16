#!/usr/bin/env bash
# Publish the rendered book to /book/ on the gh-pages branch.
#
# Use this instead of `quarto publish gh-pages`, which runs `git rm -r .` on the whole
# gh-pages branch before copying the book in. That would delete the landing page (layout:
# README.md on the gh-pages branch). This script only touches book/ and refuses to push if
# anything outside it changed. Adapted from ds-aa-lac-dry-corridor's publish_book.sh.
#
# The book renders from _freeze/, which is NOT in git and exists only on the machine that
# last executed the chapters. Before rendering, this script checks that every chapter with
# code has a freeze entry whose hash matches its .qmd; otherwise `freeze: auto` would
# re-execute it against the team DB / blob storage. It refuses unless --allow-execute.
#
# Usage (from anywhere in the repo):
#   book_cub_hurricanes/publish_book.sh                 check freeze, render, publish
#   book_cub_hurricanes/publish_book.sh --no-render     publish the existing _book/
#   add --dry-run to see what would change without committing or pushing
#   add --allow-execute to let stale chapters re-execute during the render
set -euo pipefail

render=1
dry_run=0
allow_execute=0
for arg in "$@"; do
  case "$arg" in
    --no-render) render=0 ;;
    --dry-run) dry_run=1 ;;
    --allow-execute) allow_execute=1 ;;
    *) echo "usage: $0 [--no-render] [--dry-run] [--allow-execute]" >&2; exit 2 ;;
  esac
done

book_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(git -C "$book_dir" rev-parse --show-toplevel)"
source_rev="$(git -C "$repo_root" describe --always --dirty)"

if [ "$render" = 1 ]; then
  # The project kernel (ds-aa-cub-hurricanes) points at the repo's .venv; quarto needs it
  # on PATH to resolve the kernel even when nothing executes.
  export PATH="$repo_root/.venv/bin:$PATH"
  export QUARTO_PYTHON="$repo_root/.venv/bin/python"

  # Exits non-zero, listing the chapters, if any would re-execute.
  if ! (cd "$book_dir" && python3 - "$allow_execute" <<'EOF'
import hashlib, json, pathlib, re, sys
chapters = re.findall(r"^\s+- (\S+\.qmd)", pathlib.Path("_quarto.yml").read_text(), re.M)
stale = []
for qmd in chapters:
    src = pathlib.Path(qmd).read_bytes()
    if b"\x60\x60\x60{python" not in src:  # no executable chunks
        continue
    frz = pathlib.Path("_freeze", qmd[:-4], "execute-results", "html.json")
    if not frz.exists():
        stale.append(f"{qmd}: no freeze entry")
    elif json.loads(frz.read_text())["hash"] != hashlib.md5(src).hexdigest():
        stale.append(f"{qmd}: freeze hash does not match source")
if stale and sys.argv[1] != "1":
    print("error: these chapters would re-execute during render:", *stale, sep="\n", file=sys.stderr)
    print("Re-run with --allow-execute if that is intended (needs DB/blob access).", file=sys.stderr)
    sys.exit(1)
EOF
  ); then
    exit 1
  fi
  quarto render "$book_dir"
fi
if [ ! -f "$book_dir/_book/index.html" ]; then
  echo "error: $book_dir/_book/index.html not found; render the book first" >&2
  exit 1
fi

git -C "$repo_root" fetch origin gh-pages
worktree="$(mktemp -d)/gh-pages"
git -C "$repo_root" worktree add --detach "$worktree" origin/gh-pages
trap 'cd "$repo_root" && git worktree remove --force "$worktree"' EXIT

rsync -a --delete --exclude .DS_Store "$book_dir/_book/" "$worktree/book/"

cd "$worktree"
# -f because quarto output (html/css/js/png) matches global ignore patterns in many setups
git add -A -f -- book
outside="$(git status --porcelain --untracked-files=all | grep -v '^.. book/' || true)"
if [ -n "$outside" ]; then
  echo "error: refusing to publish, changes outside book/:" >&2
  echo "$outside" >&2
  exit 1
fi
if git diff --cached --quiet; then
  echo "Nothing to publish: book/ on gh-pages already matches _book/."
  exit 0
fi

git diff --cached --stat | tail -1
if [ "$dry_run" = 1 ]; then
  echo "Dry run: not committing or pushing."
  exit 0
fi
git commit -q -m "Publish book to /book/ from $source_rev"
git push origin HEAD:gh-pages
echo "Published: https://ocha-dap.github.io/ds-aa-cub-hurricanes/book/"
