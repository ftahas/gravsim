"""Tests for gravsim.metric.null (photon geodesics, deflection, Shapiro delay). No network, fast.

Many of the quantities compared here are tiny in AU/day units (b_crit ~ 5e-8 AU, deflections
~ 1e-5 rad, Shapiro delays ~ 1e-9 days), far below pytest.approx's default abs=1e-12, so every
relative comparison on them passes ``abs=0`` to make the stated ``rel`` the tolerance enforced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gravsim.constants import (
    ARCSEC_PER_RAD,
    C_AU_DAY,
    DAY_S,
    GM_SUN,
    LIGHT_DEFLECTION_LIMB_ARCSEC,
    R_SUN_AU,
)
from gravsim.metric import null
from gravsim.metric import schwarzschild as sch

MU_SUN = GM_SUN / C_AU_DAY**2  # gravitational radius of the Sun, AU
EARTH_R = 1.0
MARS_R = 1.52
MICROSECOND_DAYS = 1e-6 / DAY_S


def limb_deflection_rad() -> float:
    """Einstein's 4 GM/(c^2 R_sun) in radians."""
    return 4.0 * GM_SUN / (C_AU_DAY**2 * R_SUN_AU)


# --------------------------------------------------------------------------- characteristic scales
def test_critical_impact_parameter_and_photon_sphere():
    b_crit = null.critical_impact_parameter(GM_SUN, C_AU_DAY)
    assert b_crit == pytest.approx(3.0 * math.sqrt(3.0) * MU_SUN, rel=1e-15, abs=0)
    # b_crit is the impact parameter of the photon sphere, and the turning point there is 3 mu
    r_ph = sch.photon_sphere_radius(GM_SUN, C_AU_DAY)
    assert null.impact_parameter(r_ph, GM_SUN, C_AU_DAY) == pytest.approx(b_crit, rel=1e-12, abs=0)
    assert null.closest_approach_radius(b_crit, GM_SUN, C_AU_DAY) == pytest.approx(
        r_ph, rel=1e-9, abs=0
    )
    # scales as gm/c^2
    assert null.critical_impact_parameter(2.0, 1.0) == pytest.approx(6.0 * math.sqrt(3.0))
    assert null.critical_impact_parameter(1.0, 10.0) == pytest.approx(0.03 * math.sqrt(3.0))


def test_closest_approach_inverts_impact_parameter():
    gm, c = 1.0, 1.0
    r0 = np.array([3.5, 5.0, 10.0, 100.0, 1e4])
    b = null.impact_parameter(r0, gm, c)
    assert np.all(b > r0)  # b^2 = r0^2 / (1 - rs/r0) > r0^2
    back = null.closest_approach_radius(b, gm, c)
    np.testing.assert_allclose(back, r0, rtol=1e-10)
    # b(r0) is stationary at the photon sphere, so the inverse is ill-conditioned close to it
    assert null.closest_approach_radius(null.impact_parameter(3.01, gm, c), gm, c) == (
        pytest.approx(3.01, rel=1e-6)
    )
    # weak field: r0 -> b - mu + ...
    assert null.closest_approach_radius(R_SUN_AU, GM_SUN, C_AU_DAY) == pytest.approx(
        R_SUN_AU - MU_SUN, rel=1e-6
    )
    # captured and below the photon sphere: NaN
    assert math.isnan(
        null.closest_approach_radius(0.5 * null.critical_impact_parameter(gm, c), gm, c)
    )
    assert math.isnan(null.impact_parameter(2.5, gm, c))


def test_isotropic_radius_round_trip():
    gm, c = 1.0, 1.0
    mu = gm / c**2
    r = np.array([2.0, 2.5, 3.0, 10.0, 1e3, 1e6])
    rho = null.isotropic_radius(r, gm, c)
    np.testing.assert_allclose(rho * (1.0 + mu / (2.0 * rho)) ** 2, r, rtol=1e-12)
    assert null.isotropic_radius(2.0 * mu, gm, c) == pytest.approx(0.5 * mu)  # horizon: rho = mu/2
    assert null.isotropic_radius(1.0, GM_SUN, C_AU_DAY) == pytest.approx(1.0 - MU_SUN, abs=1e-15)
    assert null.isotropic_radius(1.0, 0.0, C_AU_DAY) == 1.0


# --------------------------------------------------------------------------- analytic formulas
def test_weak_field_limb_value_and_ppn_scaling():
    weak = null.deflection_angle_weak(R_SUN_AU, GM_SUN, C_AU_DAY)
    assert weak == pytest.approx(limb_deflection_rad(), rel=1e-15)
    assert weak * ARCSEC_PER_RAD == pytest.approx(LIGHT_DEFLECTION_LIMB_ARCSEC, abs=1e-3)
    assert null.solar_limb_deflection_arcsec() == pytest.approx(weak * ARCSEC_PER_RAD, rel=1e-15)
    # gamma = 0 halves the deflection; gamma enters linearly as (1 + gamma)/2
    assert null.deflection_angle_weak(R_SUN_AU, GM_SUN, C_AU_DAY, gamma=0.0) == pytest.approx(
        0.5 * weak, rel=1e-15
    )
    assert null.solar_limb_deflection_arcsec(gamma=0.0) == pytest.approx(
        0.5 * null.solar_limb_deflection_arcsec(), rel=1e-15
    )
    assert null.deflection_angle_weak(1.0, 1.0, 1.0, gamma=0.5) == pytest.approx(3.0)
    # vectorised over b and ~ 1/b
    b = np.array([1.0, 2.0, 4.0]) * R_SUN_AU
    np.testing.assert_allclose(
        null.deflection_angle_weak(b, GM_SUN, C_AU_DAY), weak / np.array([1.0, 2.0, 4.0])
    )
    # scaling c down by 10 multiplies the deflection by 100
    assert null.deflection_angle_weak(R_SUN_AU, GM_SUN, C_AU_DAY / 10.0) == pytest.approx(
        100.0 * weak, rel=1e-12
    )


def test_newtonian_corpuscle_is_exactly_half_of_gr():
    b = np.array([0.3, 1.0, 7.0]) * R_SUN_AU
    corp = null.deflection_newtonian_corpuscle(b, GM_SUN, C_AU_DAY)
    weak = null.deflection_angle_weak(b, GM_SUN, C_AU_DAY)
    np.testing.assert_array_equal(corp, 0.5 * weak)
    assert null.deflection_newtonian_corpuscle(R_SUN_AU, GM_SUN, C_AU_DAY) == 0.5 * float(weak[1])
    # ... and equals the PPN formula with gamma = 0
    np.testing.assert_allclose(
        corp, null.deflection_angle_weak(b, GM_SUN, C_AU_DAY, gamma=0.0), rtol=1e-15
    )


# --------------------------------------------------------------------------- numeric deflection
def test_deflection_angle_at_the_solar_limb():
    numeric = null.deflection_angle(R_SUN_AU, GM_SUN, C_AU_DAY)
    weak = limb_deflection_rad()
    assert numeric == pytest.approx(weak, rel=1e-4)
    assert abs(numeric / weak - 1.0) < 1e-5
    assert numeric * ARCSEC_PER_RAD == pytest.approx(LIGHT_DEFLECTION_LIMB_ARCSEC, abs=1e-3)
    # the exact value exceeds Einstein's by the second-order term, relative (15 pi/16) mu/b
    excess = numeric / weak - 1.0
    assert excess > 0.0
    assert excess == pytest.approx(15.0 * math.pi / 16.0 * MU_SUN / R_SUN_AU, rel=0.05)


def test_second_order_coefficient_15_pi_over_4():
    # delta = 4 eps + (15 pi/4) eps^2 + (128/3) eps^3 + ...   with eps = mu/b (Keeton & Petters)
    gm, c = 1.0, 1.0
    for n, tol in ((50.0, 0.10), (1000.0, 0.01)):
        b = n * gm / c**2
        eps = 1.0 / n
        excess = null.deflection_angle(b, gm, c) - 4.0 * eps
        assert excess == pytest.approx(15.0 * math.pi / 4.0 * eps**2, rel=tol)
    # subtracting the third-order term tightens the agreement at b = 50 mu to < 1 percent
    eps = 1.0 / 50.0
    excess = null.deflection_angle(50.0, gm, c) - 4.0 * eps - 128.0 / 3.0 * eps**3
    assert excess == pytest.approx(15.0 * math.pi / 4.0 * eps**2, rel=0.01)


def test_deflection_depends_only_on_mu_over_b():
    ref = null.deflection_angle(R_SUN_AU, GM_SUN, C_AU_DAY)
    assert null.deflection_angle(10.0 * R_SUN_AU, 10.0 * GM_SUN, C_AU_DAY) == pytest.approx(
        ref, rel=1e-9, abs=0
    )
    assert null.deflection_angle(R_SUN_AU, GM_SUN / 4.0, C_AU_DAY / 2.0) == pytest.approx(
        ref, rel=1e-9, abs=0
    )
    # c/10 -> 100 times the deflection (still weak field)
    slow = null.deflection_angle(R_SUN_AU, GM_SUN, C_AU_DAY / 10.0)
    assert slow == pytest.approx(100.0 * ref, rel=1e-3)
    # no mass: the straight line, delta = 0
    assert abs(null.deflection_angle(1.0, 0.0, 1.0)) < 1e-12


def test_deflection_diverges_at_the_critical_impact_parameter():
    gm, c = 1.0, 1.0
    b_crit = null.critical_impact_parameter(gm, c)
    factors = [2.0, 1.1, 1.01, 1.001, 1.0001]
    deltas = [null.deflection_angle(f * b_crit, gm, c) for f in factors]
    assert all(d2 > d1 for d1, d2 in zip(deltas[:-1], deltas[1:], strict=True))
    assert deltas[factors.index(1.001)] > 2.0 * math.pi  # more than one full loop
    assert deltas[factors.index(1.01)] > math.pi  # (4.23 rad: not yet a full loop)
    # strong-field limit delta ~ -ln(b/b_crit - 1) - 0.4002 (Bozza 2002)
    bozza = -math.log(1e-4) - 0.4002
    assert deltas[-1] == pytest.approx(bozza, abs=5e-3)


def test_captured_rays_raise():
    gm, c = GM_SUN, C_AU_DAY
    b_crit = null.critical_impact_parameter(gm, c)
    for b in (b_crit, 0.5 * b_crit, 0.0, -R_SUN_AU):
        with pytest.raises(ValueError):
            null.deflection_angle(b, gm, c)
    with pytest.raises(ValueError):
        null.ray_path(b_crit, gm, c)
    with pytest.raises(ValueError):
        null.ray_path(0.0, gm, c)
    with pytest.raises(ValueError):
        null.ray_path(-0.9 * b_crit, gm, c)
    with pytest.raises(ValueError):
        null.shapiro_delay_numeric(1.0, 1.0, 0.9 * b_crit, gm, c)
    with pytest.raises(ValueError):
        null.null_transit(1.0, 1.0, b_crit, gm, c)


# --------------------------------------------------------------------------- ray paths
def outgoing_offset(x, y, delta):
    """Signed distance of (x, y) from the outgoing asymptote x sin(delta) + y cos(delta) = b."""
    return x * math.sin(delta) + y * math.cos(delta)


@pytest.mark.parametrize(
    "b, gm, c",
    [(R_SUN_AU, GM_SUN, C_AU_DAY), (20.0, 1.0, 1.0), (1.2 * 3.0 * math.sqrt(3.0), 1.0, 1.0)],
)
def test_ray_path_endpoints_on_asymptotes(b, gm, c):
    n = 1500
    x, y = null.ray_path(b, gm, c, n_points=n)
    assert x.shape == y.shape == (n,)
    r = np.hypot(x, y)
    r_max = max(1000.0 * b, 50.0)
    assert r[0] == pytest.approx(r_max, rel=1e-9)
    assert r[-1] == pytest.approx(r_max, rel=1e-9)
    # incoming asymptote y = b, entering from the left and travelling towards +x
    assert x[0] < 0.0 and x[1] > x[0]
    assert y[0] == pytest.approx(b, rel=1e-6)
    # outgoing asymptote: distance b from the origin, direction (cos delta, -sin delta)
    delta = null.deflection_angle(b, gm, c)
    assert outgoing_offset(x[-1], y[-1], delta) == pytest.approx(b, rel=1e-6)
    direction = np.array([x[-1] - x[-2], y[-1] - y[-2]])
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(direction, [math.cos(delta), -math.sin(delta)], atol=1e-4)
    # the mass is at the origin; the ray passes it at the turning point r_0 < b
    assert r.min() == pytest.approx(null.closest_approach_radius(b, gm, c), rel=1e-6)
    assert r.min() < b
    # the ray bends towards the mass: on the way out it dips below the incoming line
    assert y[-1] < b


def test_ray_path_straight_line_without_mass():
    x, y = null.ray_path(1.0, 0.0, 1.0, n_points=500)
    np.testing.assert_allclose(y, 1.0, rtol=1e-10)
    assert np.all(np.diff(x) > 0.0)


def test_ray_path_signed_b_mirrors_and_r_max_option():
    x_up, y_up = null.ray_path(2.0 * R_SUN_AU, GM_SUN, C_AU_DAY, n_points=400)
    x_dn, y_dn = null.ray_path(-2.0 * R_SUN_AU, GM_SUN, C_AU_DAY, n_points=400)
    np.testing.assert_array_equal(x_dn, x_up)
    np.testing.assert_array_equal(y_dn, -y_up)
    x, y = null.ray_path(R_SUN_AU, GM_SUN, C_AU_DAY, n_points=100, r_max=1.0)
    assert np.hypot(x[0], y[0]) == pytest.approx(1.0, rel=1e-9)
    assert np.hypot(x[-1], y[-1]) == pytest.approx(1.0, rel=1e-9)
    with pytest.raises(ValueError):
        null.ray_path(R_SUN_AU, GM_SUN, C_AU_DAY, r_max=0.5 * R_SUN_AU)
    with pytest.raises(ValueError):
        null.ray_path(R_SUN_AU, GM_SUN, C_AU_DAY, n_points=1)


def test_ray_bundle_matches_ray_path():
    b_values = np.array([-3.0, -1.0, 1.0, 3.0]) * R_SUN_AU
    bundle = null.ray_bundle(b_values, GM_SUN, C_AU_DAY, n_points=300)
    assert len(bundle) == 4
    r_max = max(1000.0 * 3.0 * R_SUN_AU, 50.0)
    for b, (x, y) in zip(b_values, bundle, strict=True):
        x_ref, y_ref = null.ray_path(b, GM_SUN, C_AU_DAY, n_points=300, r_max=r_max)
        np.testing.assert_array_equal(x, x_ref)
        np.testing.assert_array_equal(y, y_ref)
        assert np.sign(y[len(y) // 2]) == np.sign(b)
    assert null.ray_bundle([], GM_SUN, C_AU_DAY) == []


# --------------------------------------------------------------------------- Shapiro delay
def test_shapiro_delay_earth_mars_conjunction():
    k = 2.0 * GM_SUN / C_AU_DAY**3  # (1 + gamma) gm/c^3 for GR
    one_way = null.shapiro_delay(EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY)
    conj = null.shapiro_delay_conjunction(EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY)
    assert one_way == pytest.approx(125.0 * MICROSECOND_DAYS, rel=0.05, abs=0)
    assert 2.0 * one_way == pytest.approx(250.0 * MICROSECOND_DAYS, rel=0.05, abs=0)
    # b << r: the full form falls short of the conjunction form by the O(b^2/r^2) term
    # k (b^2/4)(1/r_e^2 + 1/r_r^2) inside the logarithm, 6e-7 of the delay at the limb
    second_order = k * R_SUN_AU**2 / 4.0 * (1.0 / EARTH_R**2 + 1.0 / MARS_R**2)
    assert one_way == pytest.approx(conj, rel=1e-5, abs=0)
    assert conj - one_way == pytest.approx(second_order, rel=1e-4, abs=0)
    # the exact geodesic agrees to 1e-3; the 1e-4 residual is dissected in the test below
    numeric = null.shapiro_delay_numeric(EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY)
    assert numeric == pytest.approx(one_way, rel=1e-3, abs=0)
    # further from the limb the delay drops logarithmically: between b = R_sun and 10 R_sun the
    # conjunction form drops by exactly k ln(100), the full form by 99 second-order terms less
    far = null.shapiro_delay(EARTH_R, MARS_R, 10.0 * R_SUN_AU, GM_SUN, C_AU_DAY)
    far_conj = null.shapiro_delay_conjunction(EARTH_R, MARS_R, 10.0 * R_SUN_AU, GM_SUN, C_AU_DAY)
    assert far < one_way
    assert conj - far_conj == pytest.approx(k * math.log(100.0), rel=1e-12, abs=0)
    assert one_way - far == pytest.approx(
        k * math.log(100.0) + 99.0 * second_order, rel=1e-6, abs=0
    )
    # at 10 R_sun the geodesic residual (~ delta^2) is a hundred times smaller still
    assert null.shapiro_delay_numeric(EARTH_R, MARS_R, 10.0 * R_SUN_AU, GM_SUN, C_AU_DAY) == (
        pytest.approx(far, rel=1e-4, abs=0)
    )


def test_shapiro_delay_b_enters_only_via_the_geometry():
    gm, c = GM_SUN, C_AU_DAY
    for b in (R_SUN_AU, 0.1, 0.9):
        big = math.sqrt(EARTH_R**2 - b**2) + math.sqrt(MARS_R**2 - b**2)
        expected = 2.0 * gm / c**3 * math.log((EARTH_R + MARS_R + big) / (EARTH_R + MARS_R - big))
        assert null.shapiro_delay(EARTH_R, MARS_R, b, gm, c) == pytest.approx(
            expected, rel=1e-9, abs=0
        )
    # symmetric in the two end points
    assert null.shapiro_delay(EARTH_R, MARS_R, 0.2, gm, c) == pytest.approx(
        null.shapiro_delay(MARS_R, EARTH_R, 0.2, gm, c), rel=1e-15, abs=0
    )
    for bad in (0.0, -0.1, 1.0, 1.3):
        with pytest.raises(ValueError):
            null.shapiro_delay(EARTH_R, MARS_R, bad, gm, c)
        with pytest.raises(ValueError):
            null.shapiro_delay_conjunction(EARTH_R, MARS_R, bad, gm, c)


def test_shapiro_delay_gamma_scaling():
    args = (EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY)
    gr = null.shapiro_delay(*args)
    assert null.shapiro_delay(*args, gamma=0.0) == pytest.approx(0.5 * gr, rel=1e-15, abs=0)
    assert null.shapiro_delay(*args, gamma=2.0) == pytest.approx(1.5 * gr, rel=1e-15, abs=0)
    conj = null.shapiro_delay_conjunction(*args)
    assert null.shapiro_delay_conjunction(*args, gamma=0.0) == pytest.approx(
        0.5 * conj, rel=1e-15, abs=0
    )


def test_shapiro_numeric_residual_is_end_point_geometry_minus_fermat_deficit():
    # shapiro_delay_numeric takes the ray's impact parameter b; shapiro_delay takes the closest
    # approach of the chord between the end points.  The bent ray's chord passes closer, at
    # d = b - delta r_e r_r/(r_e + r_r), which raises the weak-field value by delta^2 R/c with
    # R = r_e r_r/(r_e + r_r); Fermat's principle (least coordinate time between fixed end
    # points) then puts the exact delay below that straight-chord value by half as much.
    gm, c, b = GM_SUN, C_AU_DAY, R_SUN_AU
    delta = null.deflection_angle(b, gm, c)
    dphi, t = null.null_transit(EARTH_R, MARS_R, b, gm, c)
    rho_e = null.isotropic_radius(EARTH_R, gm, c)
    rho_r = null.isotropic_radius(MARS_R, gm, c)
    chord = math.sqrt(rho_e**2 + rho_r**2 - 2.0 * rho_e * rho_r * math.cos(dphi))
    d = rho_e * rho_r * math.sin(dphi) / chord
    reduced = EARTH_R * MARS_R / (EARTH_R + MARS_R)
    assert d < b
    assert b - d == pytest.approx(delta * reduced, rel=1e-3, abs=0)
    weak_b = null.shapiro_delay(EARTH_R, MARS_R, b, gm, c)
    weak_d = null.shapiro_delay(rho_e, rho_r, d, gm, c)
    numeric = null.shapiro_delay_numeric(EARTH_R, MARS_R, b, gm, c)
    kink = 0.5 * delta**2 * reduced / c  # extra length of two straight legs with a kink delta
    assert weak_d - weak_b == pytest.approx(2.0 * kink, rel=1e-2, abs=0)
    # Fermat: the geodesic beats the straight chord between the same end points (the deficit is
    # ~1e-13 days, resolved by the rtol = 1e-12 integration only to ~2e-14)
    assert numeric < weak_d
    assert weak_d - numeric == pytest.approx(kink, rel=0.3, abs=0)
    assert numeric - weak_b == pytest.approx(kink, rel=0.3, abs=0)
    assert numeric / weak_b - 1.0 == pytest.approx(1e-4, rel=0.2, abs=0)
    # the residual grows with r_e r_r/(r_e + r_r): 1.7e-3 of the delay for two ends at 30 AU
    far_weak = null.shapiro_delay(30.0, 30.0, b, gm, c)
    far_numeric = null.shapiro_delay_numeric(30.0, 30.0, b, gm, c)
    assert far_numeric - far_weak == pytest.approx(0.5 * delta**2 * 15.0 / c, rel=0.3, abs=0)
    assert far_numeric / far_weak - 1.0 == pytest.approx(1.7e-3, rel=0.2, abs=0)


def test_null_transit_flat_limit_and_scaling():
    # no mass: the ray is the straight line y = b, so t = R/c and the delay vanishes
    b = 0.3
    dphi, t = null.null_transit(EARTH_R, MARS_R, b, 0.0, C_AU_DAY)
    expected_phi = math.acos(b / EARTH_R) + math.acos(b / MARS_R)
    big = math.sqrt(EARTH_R**2 - b**2) + math.sqrt(MARS_R**2 - b**2)
    assert dphi == pytest.approx(expected_phi, rel=1e-10)
    assert t == pytest.approx(big / C_AU_DAY, rel=1e-10)
    assert abs(null.shapiro_delay_numeric(EARTH_R, MARS_R, b, 0.0, C_AU_DAY)) < 1e-12 * t
    # with the Sun: the swept angle exceeds the flat one by about the deflection 4 mu/b
    dphi_sun, t_sun = null.null_transit(EARTH_R, MARS_R, b, GM_SUN, C_AU_DAY)
    assert dphi_sun - expected_phi == pytest.approx(4.0 * MU_SUN / b, rel=0.3, abs=0)
    assert t_sun > big / C_AU_DAY
    # scaling c down by 10 multiplies the delay by ~1000 (gm/c^3, log nearly unchanged); the
    # end-point/Fermat residual scales with delta^2/mu ~ 1/c^2 and reaches 1 percent
    slow = null.shapiro_delay_numeric(EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY / 10.0)
    weak_slow = null.shapiro_delay(EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY / 10.0)
    assert slow == pytest.approx(weak_slow, rel=0.02, abs=0)
    fast = null.shapiro_delay_numeric(EARTH_R, MARS_R, R_SUN_AU, GM_SUN, C_AU_DAY)
    assert slow == pytest.approx(1000.0 * fast, rel=0.02, abs=0)
