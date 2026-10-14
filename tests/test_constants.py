"""Tests for gravsim.constants: unit conversions, GM values, reference numbers, body tables.

Every check is against an independent, well-known number (IAU 2012 / CODATA definitions,
Gauss's constant, the DE440 header) so a typo in a constant is caught by a value it did not
come from.  No network, no integration.
"""

from __future__ import annotations

import math

import pytest

from gravsim import constants as C

# Gaussian gravitational constant k = 0.01720209895 AU^(3/2) / day (IAU 1976 value):
# with the AU and the day as units, GM_sun = k^2 AU^3/day^2 to about 1e-10 relative,
# the level at which the IAU 2012 AU and the DE440 solar GM stopped being defined by k.
GAUSS_K = 0.01720209895


# --------------------------------------------------------------------------- unit conversions
def test_defining_constants():
    assert C.AU_KM == 149_597_870.700
    assert C.C_KM_S == 299_792.458
    assert C.DAY_S == 86_400.0
    assert C.JULIAN_YEAR_DAYS == 365.25
    assert C.JULIAN_CENTURY_DAYS == 100.0 * C.JULIAN_YEAR_DAYS


def test_speed_of_light_in_au_per_day():
    assert C.C_AU_DAY == pytest.approx(173.1446, abs=5e-5)
    assert C.C_AU_DAY == pytest.approx(C.C_KM_S * C.DAY_S / C.AU_KM, rel=1e-15)
    # light crosses one AU in 499.0 s
    assert C.DAY_S / C.C_AU_DAY == pytest.approx(499.004784, abs=1e-6)


def test_km3s2_to_au3d2():
    assert C.KM3S2_TO_AU3D2 == pytest.approx(C.DAY_S**2 / C.AU_KM**3, rel=1e-15)
    # inverse check: 1 AU^3/day^2 is 4.4851e14 km^3/s^2
    assert 1.0 / C.KM3S2_TO_AU3D2 == pytest.approx(4.4851e14, rel=1e-4)


def test_angle_conversions():
    assert C.ARCSEC_PER_RAD == pytest.approx(206_264.806247, abs=1e-6)
    assert C.RAD_PER_ARCSEC * C.ARCSEC_PER_RAD == pytest.approx(1.0, rel=1e-15)
    assert C.DEG_PER_RAD == pytest.approx(57.29577951, abs=1e-8)
    assert math.radians(1.0) * C.DEG_PER_RAD == pytest.approx(1.0, rel=1e-15)


# --------------------------------------------------------------------------- GM values
def test_gm_sun_equals_gauss_k_squared():
    assert C.GM_SUN == pytest.approx(GAUSS_K**2, rel=5e-12)
    assert C.GM_SUN == pytest.approx(2.9591220828e-4, rel=1e-10)
    assert C.GM["Sun"] == C.GM_SUN


def test_gm_dictionary_is_in_au3_per_day2():
    for name, gm_km in C._GM_KM3S2.items():
        assert C.GM[name] == pytest.approx(gm_km * C.KM3S2_TO_AU3D2, rel=1e-15), name
    # the Earth-Moon barycenter carries both masses
    assert C.GM["EMB"] == pytest.approx(C.GM["Earth"] + C.GM["Moon"], rel=1e-15)
    assert C.GM["Earth"] / C.GM["Moon"] == pytest.approx(81.3005, abs=5e-4)


def test_mass_ratios_against_de440_header():
    """Sun/planet mass ratios from the DE440 header (Park et al. 2021), system masses."""
    ratios = {
        "Mercury": 6_023_657.9,
        "Venus": 408_523.72,
        "EMB": 328_900.56,
        "Mars": 3_098_703.5,
        "Jupiter": 1_047.3486,
        "Saturn": 3_497.90,
        "Uranus": 22_902.95,
        "Neptune": 19_412.26,
    }
    for name, ratio in ratios.items():
        assert C.GM_SUN / C.GM[name] == pytest.approx(ratio, rel=2e-5), name


def test_every_default_body_has_gm_and_horizons_id():
    for name in C.DEFAULT_BODIES:
        assert name in C.GM, name
        assert name in C.HORIZONS_IDS, name
        assert C.GM[name] > 0.0


def test_gm_values_are_positive_and_ordered():
    assert all(v > 0.0 for v in C.GM.values())
    assert C.GM["Jupiter"] > C.GM["Saturn"] > C.GM["Neptune"] > C.GM["Uranus"]
    assert C.GM["Uranus"] > C.GM["EMB"] > C.GM["Venus"] > C.GM["Mars"] > C.GM["Mercury"]
    assert C.GM["Mercury"] > C.GM["Pluto"]
    # the planets together are 1/745 of the Sun (Jupiter alone 1/1047)
    assert sum(C.GM[p] for p in C.PLANETS) / C.GM_SUN == pytest.approx(1.0 / 745.25, rel=1e-3)


# --------------------------------------------------------------------------- derived scales
def test_solar_gravitational_radius_and_solar_radius():
    assert C.GM_SUN_OVER_C2 == pytest.approx(C.GM_SUN / C.C_AU_DAY**2, rel=1e-15)
    assert C.GM_SUN_OVER_C2 * C.AU_KM == pytest.approx(1.47662, abs=2e-5)  # km
    assert C.R_SUN_AU * C.AU_KM == 695_700.0
    assert C.R_SUN_AU == pytest.approx(4.6505e-3, rel=1e-4)


def test_light_deflection_constant_from_constants_alone():
    delta = 4.0 * C.GM_SUN_OVER_C2 / C.R_SUN_AU * C.ARCSEC_PER_RAD
    assert delta == pytest.approx(C.LIGHT_DEFLECTION_LIMB_ARCSEC, abs=1e-3)


def test_mercury_reference_precession_numbers():
    assert C.MERCURY_PRECESSION_GR_ARCSEC_CY == pytest.approx(42.98, abs=1e-9)
    assert C.MERCURY_PRECESSION_NEWTON_ARCSEC_CY == pytest.approx(531.6, abs=1e-9)
    total = C.MERCURY_PRECESSION_NEWTON_ARCSEC_CY + C.MERCURY_PRECESSION_GR_ARCSEC_CY
    assert total == pytest.approx(C.MERCURY_PRECESSION_OBSERVED_ARCSEC_CY, abs=1.0)
    # the GR value from the formula with Mercury's a = 0.38710 AU, e = 0.20563, P = 87.969 d
    a, e, period = 0.38709893, 0.20563069, 87.969
    per_orbit = 6.0 * math.pi * C.GM_SUN / (C.C_AU_DAY**2 * a * (1.0 - e * e))
    per_cy = per_orbit * C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS / period
    assert per_cy == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.01)


# --------------------------------------------------------------------------- epochs and frames
def test_j2000_epoch_and_obliquity():
    assert C.J2000_JD == 2_451_545.0
    assert C.OBLIQUITY_J2000_DEG == pytest.approx(23.4392911, abs=1e-7)
    assert C.OBLIQUITY_J2000_RAD == pytest.approx(math.radians(C.OBLIQUITY_J2000_DEG), rel=1e-15)


# --------------------------------------------------------------------------- body tables
def test_default_bodies_are_sun_plus_eight_planets_inner_to_outer():
    assert C.DEFAULT_BODIES[0] == "Sun"
    assert C.DEFAULT_BODIES[1:] == C.PLANETS
    assert C.PLANETS == (
        "Mercury",
        "Venus",
        "EMB",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
    )
    assert len(C.DEFAULT_BODIES) == 9
    assert len(set(C.DEFAULT_BODIES)) == 9
    assert "Pluto" not in C.DEFAULT_BODIES and "Moon" not in C.DEFAULT_BODIES


def test_horizons_ids_are_barycenter_codes():
    """Planets use their system-barycenter codes 1..8; the Sun is 10; Earth/Moon are 399/301."""
    for k, planet in enumerate(C.PLANETS, start=1):
        assert C.HORIZONS_IDS[planet] == str(k), planet
    assert C.HORIZONS_IDS["Sun"] == "10"
    assert C.HORIZONS_IDS["Pluto"] == "9"
    assert C.HORIZONS_IDS["Earth"] == "399"
    assert C.HORIZONS_IDS["Moon"] == "301"
    assert all(isinstance(v, str) and v.isdigit() for v in C.HORIZONS_IDS.values())


def test_constants_are_plain_floats_and_tuples():
    for name in ("AU_KM", "C_AU_DAY", "GM_SUN", "ARCSEC_PER_RAD", "J2000_JD"):
        assert isinstance(getattr(C, name), float), name
    assert isinstance(C.PLANETS, tuple) and isinstance(C.DEFAULT_BODIES, tuple)
    assert isinstance(C.GM, dict) and all(isinstance(v, float) for v in C.GM.values())
