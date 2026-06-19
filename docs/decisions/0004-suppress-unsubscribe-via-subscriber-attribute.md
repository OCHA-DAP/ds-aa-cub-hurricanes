---
status: "accepted"
date: 2026-06-19
decision-makers: [Zack]
consulted: [Tristan, CHD listmonk team]
---

# Suppress the unsubscribe link via a per-subscriber attribute

## Context and Problem Statement

Listmonk campaigns inject an unsubscribe link by default. The Cuba hurricane
trigger/activation emails are **must-deliver** alerts to government and agency
contacts; we don't want a recipient to silently unsubscribe and miss an
activation. How do we suppress the unsubscribe link for these recipients?

## Decision Drivers

* Trigger/activation emails are operational, must-deliver.
* Avoid heavy template surgery (we *use* the shared campaign template and inject
  content into it — we don't want to fork its design).
* Recipients are operational contacts, not a general public newsletter audience.

## Considered Options

* Accept the unsubscribe link (recipients can opt out).
* Edit the template to hard-remove the unsubscribe link.
* Use listmonk's transactional API for triggers (no unsubscribe, bypasses
  subscription status).
* Per-subscriber attribute `type=mailing_list` — the campaign template already
  renders the unsubscribe link only when `attribs.type != "mailing_list"`.

## Decision Outcome

Set `attribs={"type": "mailing_list"}` on subscribers at import
(`pipelines/setup_cub_listmonk_lists.py`). The campaign template already gates the
unsubscribe link on `attribs.type != "mailing_list"`, so for these subscribers
the link is omitted — **no template change**, and it follows the CHD listmonk
team's existing convention for mailing-list-type recipients.

### Consequences

* Good — no template surgery; recipients can't self-unsubscribe from the alerts;
  the gate is per-subscriber, so a non-`mailing_list` subscriber still gets a link.
* Bad — `attribs` is **global per subscriber**, not per-list: the flag suppresses
  the unsubscribe link in *every* email that person receives on the shared
  listmonk instance (wherever a template checks it), not just Cuba's. Accepted
  because it's set only on new Cuba recipients, who are only on Cuba lists.
* Bad — the attribute is set only on the **new-subscriber** import path; a contact
  who already exists on the instance keeps their existing attribs (and so would
  still see the link). Accepted: overlap with other projects is negligible, and
  new-recipient scope was the chosen bound.

## More Information

* Mechanism: template footer conditional in
  `src/email/templates/listmonk/basecampaign_dual_language.html`; attribute set in
  the POST path of `import_subscribers`.
* Default backend / send behaviour:
  [[0003-listmonk-as-default-email-backend]].
