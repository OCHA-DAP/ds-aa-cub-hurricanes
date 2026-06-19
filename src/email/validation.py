"""Email-address validation.

Kept in its own module (no import-time side effects) so it can be reused from
contexts that must not pull in src.email.utils, which reads SMTP settings at
import time. src.email.utils re-exports is_valid_email for backwards
compatibility.
"""

import re

# Pattern for validating an email address.
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"


def is_valid_email(email) -> bool:
    """Return True if `email` looks like a valid address."""
    return bool(isinstance(email, str) and re.match(EMAIL_REGEX, email))
