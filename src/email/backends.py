"""Email dispatch backend selector.

Routes the public send functions to one backend, chosen at import time by
src.constants.EMAIL_BACKEND:

    humdata_email -> src.email.send_emails      (legacy SMTP / AWS SES; the
                                                  preserved manual fallback)
    listmonk      -> src.email.listmonk_emails  (ocha_relay campaigns)

The switch is manual; there is no automatic failover between backends. Callers
(e.g. src.email.update_emails) import send_info_email / send_trigger_email from
here instead of from a specific backend.
"""

from src.constants import EMAIL_BACKEND

if EMAIL_BACKEND == "listmonk":
    from src.email.listmonk_emails import (  # noqa: F401
        send_info_email,
        send_trigger_email,
    )
else:
    from src.email.send_emails import (  # noqa: F401
        send_info_email,
        send_trigger_email,
    )
