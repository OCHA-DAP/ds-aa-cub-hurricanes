import numpy as np
import os
import pytz
from datetime import datetime, timezone

PROJECT_PREFIX = "ds-aa-cub-hurricanes"
ISO3 = "cub"

# Monitoring start date - only process data from this date forward
# Set to Cuba timezone so that dummy emails show intended date
cuba_tz = pytz.timezone("America/Havana")
MONITORING_START_DATE = cuba_tz.localize(datetime(2025, 1, 1)).astimezone(
    timezone.utc
)


# Runtime control flags - centralized configuration
def _parse_bool_env(env_var: str, default: bool = False) -> bool:
    """Parse environment variable as boolean with proper defaults."""
    value = os.getenv(env_var)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


# Main control flags
DRY_RUN = _parse_bool_env("DRY_RUN", default=True)  # Safe default
TEST_EMAIL = _parse_bool_env("TEST_EMAIL", default=True)  # Safe default
FORCE_ALERT = _parse_bool_env("FORCE_ALERT", default=False)  # Off by default

# Saffir-Simpson scale (knots)
TS = 34
CAT1 = 64
CAT2 = 83
CAT3 = 96
CAT4 = 113
CAT5 = 137

CAT_LIMITS = [
    (TS, "Trop. Storm"),
    (CAT1, "Cat. 1"),
    (CAT2, "Cat. 2"),
    (CAT3, "Cat. 3"),
    (CAT4, "Cat. 4"),
    (CAT5, "Cat. 5"),
]

# specific storm SIDs for easy plotting / filtering
IKE = "2008245N17323"
GUSTAV = "2008238N13293"
IRMA = "2017242N16333"
IAN = "2022266N12294"
OSCAR = "2024293N21294"
RAFAEL = "2024309N13283"

D_THRESH = 230

THRESHS = {
    "readiness": {"s": 120, "lt_days": 5},
    "action": {"s": 120, "lt_days": 3},
    "obsv": {"p": 98.2, "s": 105},  # NEED TO UPDATE FOR PROD
}

MIN_EMAIL_DISTANCE = 1000

NUMERIC_NAME_REGEX = r"\b(?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|Thirteen|Fourteen|Fifteen|Sixteen|Seventeen|Eighteen|Nineteen|Twenty)\b"  # noqa: E501

# Temporary constants ported from haiti repo - will change to fit

CERF_SIDS = [
    "2016273N13300",  # Matthew
    "2008245N17323",  # Ike
    "2008238N13293",  # Gustav
    "2008241N19303",  # Hanna
    "2008229N18293",  # Fay
    "2012296N14283",  # Sandy
]

SPANISH_MONTHS = {
    "Jan": "ene.",
    "Feb": "feb.",
    "Mar": "mar.",
    "Apr": "abr.",
    "May": "may.",
    "Jun": "jun.",
    "Jul": "jul.",
    "Aug": "ago.",
    "Sep": "sep.",
    "Oct": "oct.",
    "Nov": "nov.",
    "Dec": "dic.",
}

CHD_GREEN = "#1bb580"

# Longitude zoom range for map plotting

LON_ZOOM_RANGE = np.array(
    [
        0.0007,
        0.0014,
        0.003,
        0.006,
        0.012,
        0.024,
        0.048,
        0.096,
        0.192,
        0.3712,
        0.768,
        1.536,
        3.072,
        6.144,
        11.8784,
        23.7568,
        47.5136,
        98.304,
        190.0544,
        360.0,
    ]
)


TEST_ATCF_ID = "TEST_ATCF_ID"
TEST_MONITOR_ID = "TEST_MONITOR_ID"
TEST_FCAST_MONITOR_ID = "TEST_FCAST_MONITOR_ID"
TEST_OBSV_MONITOR_ID = "TEST_OBSV_MONITOR_ID"
TEST_STORM_NAME = "TEST_STORM_NAME"
