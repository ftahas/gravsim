"""Physical constants and unit conversions.

Internal units: length in AU, time in days, masses as GM in AU^3/day^2.
GM values are from JPL DE440 (Park et al. 2021, AJ 161:105); planets are *system*
GMs (planet + satellites) and Earth is the Earth-Moon barycenter, matching the
Horizons barycenter targets used in `gravsim.data`.
"""

from __future__ import annotations

import math

# --- defining constants (IAU 2012 / CODATA) -----------------------------------
AU_KM = 149_597_870.700  # astronomical unit [km]
C_KM_S = 299_792.458  # speed of light [km/s]
DAY_S = 86_400.0  # day [s]
JULIAN_YEAR_DAYS = 365.25
JULIAN_CENTURY_DAYS = 36_525.0

# --- derived unit conversions ---------------------------------------------------
KM3S2_TO_AU3D2 = DAY_S**2 / AU_KM**3  # km^3/s^2  ->  AU^3/day^2
C_AU_DAY = C_KM_S * DAY_S / AU_KM  # speed of light [AU/day]  (~173.1446)
ARCSEC_PER_RAD = 180.0 * 3600.0 / math.pi
RAD_PER_ARCSEC = 1.0 / ARCSEC_PER_RAD
DEG_PER_RAD = 180.0 / math.pi

# --- GM values, JPL DE440 [km^3/s^2] ----------------------------------------------
_GM_KM3S2 = {
    "Sun": 132_712_440_041.279419,
    "Mercury": 22_031.868551,
    "Venus": 324_858.592000,
    "Earth": 398_600.435507,
    "Moon": 4_902.800118,
    "Mars": 42_828.375816,  # Mars system
    "Jupiter": 126_712_764.100000,  # Jupiter system
    "Saturn": 37_940_584.841800,  # Saturn system
    "Uranus": 5_794_556.400000,  # Uranus system
    "Neptune": 6_836_527.100580,  # Neptune system
    "Pluto": 975.500000,  # Pluto system
}
_GM_KM3S2["EMB"] = _GM_KM3S2["Earth"] + _GM_KM3S2["Moon"]  # Earth-Moon barycenter

GM: dict[str, float] = {k: v * KM3S2_TO_AU3D2 for k, v in _GM_KM3S2.items()}
"""GM of each body in AU^3/day^2, keyed by name."""

GM_SUN = GM["Sun"]  # ~2.9591220828e-4 AU^3/day^2
GM_SUN_OVER_C2 = GM_SUN / C_AU_DAY**2  # gravitational radius of the Sun [AU] (~9.87e-9)
R_SUN_AU = 695_700.0 / AU_KM  # solar radius [AU] (IAU nominal)

# --- epochs and frames ------------------------------------------------------------
J2000_JD = 2_451_545.0  # 2000-01-01 12:00 TDB
OBLIQUITY_J2000_DEG = 84_381.448 / 3600.0  # mean obliquity at J2000 (IAU 1976) [deg]
OBLIQUITY_J2000_RAD = math.radians(OBLIQUITY_J2000_DEG)

# --- the eight planets, inner to outer, as used by the default system ----------------
PLANETS: tuple[str, ...] = (
    "Mercury",
    "Venus",
    "EMB",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
)
DEFAULT_BODIES: tuple[str, ...] = ("Sun",) + PLANETS

# Horizons target identifiers for the default bodies (system barycenters for planets).
HORIZONS_IDS: dict[str, str] = {
    "Sun": "10",
    "Mercury": "1",
    "Venus": "2",
    "EMB": "3",
    "Mars": "4",
    "Jupiter": "5",
    "Saturn": "6",
    "Uranus": "7",
    "Neptune": "8",
    "Pluto": "9",
    "Earth": "399",
    "Moon": "301",
}

# --- reference values used by the validation suite ------------------------------------
# Secular perihelion advance of Mercury in an inertial frame, arcsec per Julian century.
MERCURY_PRECESSION_GR_ARCSEC_CY = 42.98  # 1PN, Schwarzschild term
MERCURY_PRECESSION_NEWTON_ARCSEC_CY = 531.6  # planetary perturbations (Clemence 1947: 531.63)
MERCURY_PRECESSION_OBSERVED_ARCSEC_CY = 574.1
LIGHT_DEFLECTION_LIMB_ARCSEC = 1.751  # 4 GM / (c^2 R_sun)
