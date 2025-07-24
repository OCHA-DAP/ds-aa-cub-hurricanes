PROJECT_PREFIX = "ds-aa-cub-hurricanes"
ISO3 = "cub"

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
