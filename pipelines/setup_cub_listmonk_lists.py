"""One-time setup: create the Cuba hurricane listmonk lists and migrate the
existing distribution-list subscribers into them.

We send Cuba hurricane emails to two audiences that are NOT the same set of
people (verified against the distribution list): ~49 "info" recipients and
~28 "trigger" recipients, with partial overlap. Listmonk has no to/cc concept,
so each audience becomes its own list (see src.constants.LISTMONK_LISTS).

ocha_relay's ListmonkClient can create lists (create_list / fetch_all_lists)
but does NOT expose subscriber creation, so the subscriber import goes through
listmonk's HTTP API directly, using the same DSCI_LISTMONK_* credentials that
ListmonkClient.from_env() reads.

The script is idempotent: existing lists (matched by tag) are reused, and
existing subscribers (matched by email) are added to the target lists rather
than recreated. Run --dry-run first to preview.

Requires ADMIN listmonk credentials (creating lists/subscribers is a write,
which the send-scoped sender_api key cannot do): DSCI_LISTMONK_BASE_URL,
DSCI_LISTMONK_ADMIN_API_USERNAME, DSCI_LISTMONK_ADMIN_API_KEY. (The dispatch in
Phase 2 uses the send-scoped DSCI_LISTMONK_API_* via ListmonkClient.from_env.)

Usage:
    python pipelines/setup_cub_listmonk_lists.py --dry-run   # preview only
    python pipelines/setup_cub_listmonk_lists.py             # apply
"""

import argparse
import os
import sys
from pathlib import Path

import ocha_stratus as stratus
import requests
from ocha_relay.listmonk import ListmonkClient

# Put the repo root on the path so `src` imports resolve when this script is
# run directly (the pipelines/ scripts otherwise rely on PYTHONPATH=<repo root>
# set in the GitHub workflows).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.constants import (  # noqa: E402
    LISTMONK_LISTS,
    LISTMONK_PROJECT_TAG,
    PROJECT_PREFIX,
)
from src.email.validation import is_valid_email  # noqa: E402

DIST_LIST_BLOB = f"{PROJECT_PREFIX}/email/distribution_list.csv"
# distribution_list.csv columns map "to"/"cc" per email type; for listmonk both
# simply mean "receives this email", so we treat them identically.
_RECEIVES = ["to", "cc"]


def _admin_client() -> ListmonkClient:
    """ListmonkClient built from the ADMIN credentials. Creating lists and
    subscribers is a write, which the send-scoped sender_api key (used by
    from_env) is forbidden to do."""
    return ListmonkClient(
        base_url=os.environ["DSCI_LISTMONK_BASE_URL"].rstrip("/"),
        username=os.environ["DSCI_LISTMONK_ADMIN_API_USERNAME"],
        password=os.environ["DSCI_LISTMONK_ADMIN_API_KEY"],
    )


def _listmonk_http():
    """(session, base_url) for direct listmonk subscriber calls, using the
    ADMIN credentials. base_url already includes the /api prefix."""
    base = os.environ["DSCI_LISTMONK_BASE_URL"].rstrip("/")
    session = requests.Session()
    session.auth = (
        os.environ["DSCI_LISTMONK_ADMIN_API_USERNAME"],
        os.environ["DSCI_LISTMONK_ADMIN_API_KEY"],
    )
    return session, base


def resolve_or_create_lists(client: ListmonkClient, dry_run: bool) -> dict:
    """Return {type: list_id} for the info/trigger lists, creating any that do
    not already exist (matched by their type tag)."""
    existing = client.fetch_all_lists(tag=LISTMONK_PROJECT_TAG)
    tag_to_id = {
        tag: lst["id"] for lst in existing for tag in lst.get("tags", [])
    }

    list_ids = {}
    for list_type, cfg in LISTMONK_LISTS.items():
        if cfg["tag"] in tag_to_id:
            lid = tag_to_id[cfg["tag"]]
            list_ids[list_type] = lid
            print(f"  ✓ list '{cfg['name']}' exists (id={lid})")
        elif dry_run:
            list_ids[list_type] = None
            print(f"  + would create list '{cfg['name']}' tag={cfg['tag']}")
        else:
            tags = [LISTMONK_PROJECT_TAG, cfg["tag"]]
            tags += cfg.get("extra_tags", [])
            new_id = client.create_list(name=cfg["name"], tags=tags)
            list_ids[list_type] = new_id
            print(f"  + created list '{cfg['name']}' (id={new_id})")
    return list_ids


def load_target_memberships() -> dict:
    """Read the distribution list and return {email: {"name", "types"}}, where
    ``types`` is the set of audiences ("info"/"trigger") that email belongs to.

    Audiences are kept as type labels (not list IDs) so the plan is meaningful
    even in a dry run before the lists exist. Invalid emails are skipped (via
    the shared src.email.validation.is_valid_email)."""
    df = stratus.load_csv_from_blob(DIST_LIST_BLOB)
    df["email"] = df["email"].str.strip()

    memberships: dict[str, dict] = {}
    skipped: set[str] = set()
    for list_type in ("info", "trigger"):
        for _, row in df[df[list_type].isin(_RECEIVES)].iterrows():
            email = str(row["email"]).lower()
            if not is_valid_email(email):
                skipped.add(str(row["email"]))
                continue
            # CSV name cells can be blank (NaN is truthy) -> fall back to the
            # email rather than letting a NaN through as the name.
            name = row.get("name")
            name = name if isinstance(name, str) and name.strip() else email
            entry = memberships.setdefault(
                email, {"name": name, "types": set()}
            )
            entry["types"].add(list_type)
    if skipped:
        print(f"  ! skipped {len(skipped)} invalid emails: {sorted(skipped)}")
    return memberships


def fetch_existing_subscribers(session, base) -> dict:
    """{email_lower: subscriber_id} for all current listmonk subscribers."""
    out, page = {}, 1
    while True:
        resp = session.get(
            f"{base}/subscribers",
            params={"page": page, "per_page": 1000},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        for row in data["results"]:
            out[row["email"].lower()] = row["id"]
        if page * data["per_page"] >= data["total"]:
            break
        page += 1
    return out


def import_subscribers(session, base, memberships, list_ids, dry_run):
    """Create new subscribers / add existing ones to the target lists.

    Existing subscribers are fetched in BOTH modes (a read), so the dry run
    reports an accurate new-vs-existing split. Only the writes are gated on
    dry_run."""
    existing = fetch_existing_subscribers(session, base)
    created = updated = 0
    for email, info in memberships.items():
        target_ids = [
            list_ids[t] for t in info["types"] if list_ids[t] is not None
        ]
        if email in existing:
            updated += 1
            if not dry_run and target_ids:
                session.put(
                    f"{base}/subscribers/lists",
                    json={
                        "ids": [existing[email]],
                        "action": "add",
                        "target_list_ids": target_ids,
                        "status": "confirmed",
                    },
                ).raise_for_status()
        else:
            created += 1
            if not dry_run and target_ids:
                session.post(
                    f"{base}/subscribers",
                    json={
                        "email": email,
                        "name": info["name"],
                        "status": "enabled",
                        "lists": target_ids,
                        "preconfirm_subscriptions": True,
                    },
                ).raise_for_status()
    verb = "would create" if dry_run else "created"
    verb2 = "would add to lists" if dry_run else "added to lists"
    print(f"  new subscribers ({verb}): {created}")
    print(f"  existing subscribers ({verb2}): {updated}")


def main(dry_run: bool = False, lists_only: bool = False):
    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"=== Cuba listmonk list setup ({mode}) ===")

    client = _admin_client()
    print("Resolving / creating lists:")
    list_ids = resolve_or_create_lists(client, dry_run)

    if lists_only:
        print("Lists only — skipping subscriber import.")
        print(f"List IDs: {list_ids}")
        print("Done.")
        return

    print("Mapping distribution-list subscribers:")
    memberships = load_target_memberships()
    n_info = sum("info" in m["types"] for m in memberships.values())
    n_trig = sum("trigger" in m["types"] for m in memberships.values())
    print(
        f"  {len(memberships)} unique subscribers "
        f"(info={n_info}, trigger={n_trig})"
    )

    print("Importing subscribers:")
    session, base = _listmonk_http()
    import_subscribers(session, base, memberships, list_ids, dry_run)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview lists/subscribers without creating anything.",
    )
    parser.add_argument(
        "--lists-only",
        action="store_true",
        help="Create the lists but skip the subscriber import.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, lists_only=args.lists_only)
