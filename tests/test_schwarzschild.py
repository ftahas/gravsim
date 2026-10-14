"""Tests for gravsim.metric.schwarzschild (exact timelike geodesics). No network, all fast."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from gravsim.constants import AU_KM, C_AU_DAY, GM_SUN
from gravsim.metric import schwarzschild as sch

MERCURY_A = 0.38709893
MERCURY_E = 0.20563069
MERCURY_PERIOD = 2.0 * math.pi * math.sqrt(MERCURY_A**3 / GM_SUN)


# --------------------------------------------------------------------------- helpers / fixtures
def kepler_perihelion_state(a: float, e: float, gm: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity at perihelion of the Kepler ellipse (a, e): r_p = a(1-e), vis-viva."""
    r_p = a * (1.0 - e)
    v_p = math.sqrt(gm * (1.0 + e) / (a * (1.0 - e)))
    return np.array([r_p, 0.0]), np.array([0.0, v_p])


# Bound orbits used by the integrator tests: name -> (r0, v0, gm, c, tau_end, n_orbits).
# The first two start at an apsis (r0 . v0 = 0).  The others start at a generic point of the
# orbit (r0 . v0 != 0, as a Horizons state would), which exercises the Binet slope
# u'(phi0) = -v_r/|L| and the v_r^2 term of E; a sign error in either is invisible from an apsis.
_L_CIRC_10 = sch.circular_orbit_angular_momentum(10.0, 1.0, 1.0)  # circular orbit at r = 10 mu
ORBIT_CASES: dict[str, tuple[list[float], list[float], float, float, float, float]] = {
    "mercury_perihelion": (
        *(x.tolist() for x in kepler_perihelion_state(MERCURY_A, MERCURY_E, GM_SUN)),
        GM_SUN,
        C_AU_DAY,
        2.0 * MERCURY_PERIOD,
        2.1,
    ),
    "strong_apoapsis": ([10.0, 0.0], [0.0, 0.97 * _L_CIRC_10 / 10.0], 1.0, 1.0, 600.0, 5.0),
    "mercury_generic": ([0.30, 0.10], [0.012, 0.030], GM_SUN, C_AU_DAY, 160.0, 2.5),
    "strong_outward": ([10.0, 0.0], [0.03, 0.95 * _L_CIRC_10 / 10.0], 1.0, 1.0, 600.0, 6.0),
    "strong_inward": ([10.0, 0.0], [-0.03, 0.95 * _L_CIRC_10 / 10.0], 1.0, 1.0, 600.0, 6.0),
    "strong_retrograde_inward": (
        [10.0, 0.0],
        [-0.03, -0.95 * _L_CIRC_10 / 10.0],
        1.0,
        1.0,
        600.0,
        6.0,
    ),
}


def orbit_case(name: str) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    r0, v0, gm, c, tau_end, n_orbits = ORBIT_CASES[name]
    return np.asarray(r0, dtype=float), np.asarray(v0, dtype=float), gm, c, tau_end, n_orbits


def kepler_time_from_true_anomaly(nu: np.ndarray, a: float, e: float, gm: float) -> np.ndarray:
    """Time since perihelion for (unwrapped) true anomaly nu via Kepler's equation.

    E = nu - 2 atan(beta sin nu / (1 + beta cos nu)), beta = e / (1 + sqrt(1 - e^2)), is continuous
    in nu, so no unwrapping of the eccentric anomaly is needed.
    """
    beta = e / (1.0 + math.sqrt(1.0 - e * e))
    ecc_anom = nu - 2.0 * np.arctan(beta * np.sin(nu) / (1.0 + beta * np.cos(nu)))
    mean_anom = ecc_anom - e * np.sin(ecc_anom)
    return mean_anom / math.sqrt(gm / a**3)


def second_derivative(f, r: float, h: float) -> float:
    return (f(r + h) - 2.0 * f(r) + f(r - h)) / h**2


@pytest.fixture
def mercury() -> dict:
    r0, v0 = kepler_perihelion_state(MERCURY_A, MERCURY_E, GM_SUN)
    return {
        "a": MERCURY_A,
        "e": MERCURY_E,
        "gm": GM_SUN,
        "r0": r0,
        "v0": v0,
        "period": MERCURY_PERIOD,
    }


# --------------------------------------------------------------------------- characteristic radii
def test_characteristic_radii():
    gm, c = GM_SUN, C_AU_DAY
    rs = sch.schwarzschild_radius(gm, c)
    assert rs == pytest.approx(2.0 * gm / c**2)
    assert rs * AU_KM == pytest.approx(2.953, abs=2e-3)  # the Sun's Schwarzschild radius ~2.95 km
    assert sch.photon_sphere_radius(gm, c) == pytest.approx(1.5 * rs)
    assert sch.isco_radius(gm, c) == pytest.approx(3.0 * rs)
    # default c is C_AU_DAY
    assert sch.schwarzschild_radius(gm) == rs


# --------------------------------------------------------------------------- effective potential
def test_effective_potential_reduces_to_newton_for_large_c():
    r = np.linspace(0.3, 5.0, 7)
    L, gm = 0.01, GM_SUN
    v_gr = sch.effective_potential(r, L, gm, c=1e12)
    v_newton = sch.newton_effective_potential(r, L, gm)
    np.testing.assert_allclose(v_gr, v_newton, rtol=1e-14)
    # the GR term is exactly -gm L^2 / (c^2 r^3); check it where it is not swamped by cancellation
    r, L, gm, c = np.linspace(4.0, 50.0, 7), 4.0, 1.0, 1.0
    np.testing.assert_allclose(
        sch.effective_potential(r, L, gm, c) - sch.newton_effective_potential(r, L, gm),
        -gm * L**2 / (c**2 * r**3),
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "gm, c, L, radii",
    [
        (GM_SUN, C_AU_DAY, 0.0175, [0.2, 0.39, 1.0, 5.0]),
        (1.0, 1.0, 3.8, [4.0, 6.0, 10.0, 50.0]),
    ],
)
def test_radial_acceleration_is_minus_dV_dr(gm, c, L, radii):
    """d2r/dtau2 used by geodesic_orbit must equal -dV_eff/dr (central finite differences)."""
    for r in radii:
        h = 1e-5 * r
        v_plus = sch.effective_potential(r + h, L, gm, c)
        v_minus = sch.effective_potential(r - h, L, gm, c)
        dv_dr = (v_plus - v_minus) / (2.0 * h)
        assert sch.radial_acceleration(r, L, gm, c) == pytest.approx(-dv_dr, rel=1e-7)


@pytest.mark.parametrize("gm, c", [(1.0, 1.0), (GM_SUN, C_AU_DAY)])
def test_circular_orbits_and_isco(gm, c):
    mu = gm / c**2
    # stable circular orbit at 6.5 mu: minimum of V_eff
    r_s = 6.5 * mu
    L_s = sch.circular_orbit_angular_momentum(r_s, gm, c)
    assert L_s**2 > 12.0 * (gm / c) ** 2
    r_stable, r_unstable = sch.circular_orbit_radii(L_s, gm, c)
    assert r_stable == pytest.approx(r_s, rel=1e-12)
    assert sch.photon_sphere_radius(gm, c) < r_unstable < sch.isco_radius(gm, c)
    assert sch.radial_acceleration(r_s, L_s, gm, c) == pytest.approx(0.0, abs=1e-12 * gm / r_s**2)
    curv = second_derivative(lambda r: sch.effective_potential(r, L_s, gm, c), r_s, 1e-3 * r_s)
    assert curv > 0.0
    # unstable circular orbit at 5.5 mu: maximum of V_eff
    r_u = 5.5 * mu
    L_u = sch.circular_orbit_angular_momentum(r_u, gm, c)
    r_stable, r_unstable = sch.circular_orbit_radii(L_u, gm, c)
    assert r_unstable == pytest.approx(r_u, rel=1e-12)
    assert r_stable > sch.isco_radius(gm, c)
    curv = second_derivative(lambda r: sch.effective_potential(r, L_u, gm, c), r_u, 1e-3 * r_u)
    assert curv < 0.0
    # ISCO: both radii merge at 6 mu when L^2 = 12 (gm/c)^2 (tiny offset avoids rounding to NaN)
    L_isco = math.sqrt(12.0) * gm / c * (1.0 + 1e-9)
    r_stable, r_unstable = sch.circular_orbit_radii(L_isco, gm, c)
    assert r_stable == pytest.approx(sch.isco_radius(gm, c), rel=1e-4)
    assert r_unstable == pytest.approx(sch.isco_radius(gm, c), rel=1e-4)
    assert sch.circular_orbit_angular_momentum(sch.isco_radius(gm, c), gm, c) == pytest.approx(
        math.sqrt(12.0) * gm / c, rel=1e-12
    )
    # below the ISCO angular momentum there is no circular orbit at all
    below_isco = sch.circular_orbit_radii(0.99 * math.sqrt(12.0) * gm / c, gm, c)
    assert all(math.isnan(x) for x in below_isco)
    # inside the photon sphere no timelike circular orbit exists
    assert math.isnan(sch.circular_orbit_angular_momentum(2.9 * mu, gm, c))


def test_circular_orbit_radii_vectorised_and_newtonian_limit():
    gm, c = 1.0, 1.0
    L = np.array([3.0, math.sqrt(12.0) * 1.001, 4.0, 10.0])
    r_stable, r_unstable = sch.circular_orbit_radii(L, gm, c)
    assert r_stable.shape == (4,) and r_unstable.shape == (4,)
    assert np.isnan(r_stable[0]) and np.isnan(r_unstable[0])
    assert np.all(r_stable[1:] > r_unstable[1:])
    np.testing.assert_allclose((r_stable * r_unstable)[1:], 3.0 * L[1:] ** 2 / c**2, rtol=1e-12)
    # large L: stable radius -> Newtonian L^2/gm, unstable radius -> photon sphere 3 gm/c^2
    r_stable, r_unstable = sch.circular_orbit_radii(1e4, gm, c)
    assert r_stable == pytest.approx(1e8 / gm, rel=1e-6)
    assert r_unstable == pytest.approx(sch.photon_sphere_radius(gm, c), rel=1e-6)


# --------------------------------------------------------------------------- initial data
def test_conserved_quantities_weak_field(mercury):
    r0, v0, gm = mercury["r0"], mercury["v0"], mercury["gm"]
    E, L = sch.conserved_quantities(r0, v0, gm, C_AU_DAY)
    assert L == pytest.approx(r0[0] * v0[1], rel=1e-15)
    eps_newton = 0.5 * v0[1] ** 2 - gm / r0[0]  # = -gm / (2a)
    assert eps_newton == pytest.approx(-gm / (2.0 * MERCURY_A), rel=1e-12)
    # E = c^2 + eps - gm L^2/(c^2 r^3) - eps^2/(2 c^2) + ...: the corrections are ~1e-7 relative
    assert E - C_AU_DAY**2 == pytest.approx(eps_newton, rel=1e-6)
    assert E - C_AU_DAY**2 != pytest.approx(eps_newton, rel=1e-8)
    # retrograde start flips the sign of L and keeps E
    E_r, L_r = sch.conserved_quantities(r0, -v0, gm, C_AU_DAY)
    assert L_r == -L and E_r == E


@pytest.mark.parametrize("case", ["mercury_generic", "strong_outward", "strong_retrograde_inward"])
def test_conserved_quantities_off_apsis(case):
    """E and L for a start with r0 . v0 != 0, where the v_r^2 term of E is live.

    (i) The exact energy equation (1/2) v_r^2 + V_eff(r0) = (E^2/c^2 - c^2)/2 (an identity between
    conserved_quantities and effective_potential that only holds with the v_r^2 term); (ii) the
    weak-field limit E - c^2 -> |v|^2/2 - gm/r with the *full* speed; (iii) E and L recomputed
    from later points of the proper-time geodesic equal the initial values, which cannot happen
    with a wrong E formula because v_r differs from point to point.
    """
    r0, v0, gm, c, tau_end, _ = orbit_case(case)
    r_init = float(np.hypot(*r0))
    v_r = float(r0 @ v0) / r_init
    assert abs(v_r) > 0.05 * float(np.linalg.norm(v0))  # genuinely off an apsis
    E, L = sch.conserved_quantities(r0, v0, gm, c)
    assert L == pytest.approx(r0[0] * v0[1] - r0[1] * v0[0], rel=1e-15)
    kinetic = (E**2 / c**2 - c**2) / 2.0  # (1/2)(dr/dtau)^2 + V_eff along the geodesic
    v_eff = float(sch.effective_potential(r_init, L, gm, c))
    assert kinetic == pytest.approx(0.5 * v_r**2 + v_eff, rel=1e-6)
    assert 0.5 * v_r**2 > 1e-3 * abs(kinetic)  # the tolerance above resolves the v_r^2 term
    if gm / (c**2 * r_init) < 1e-7:  # weak field: E = c^2 + eps + O(eps^2/c^2, gm eps/(c^2 r))
        eps_newton = 0.5 * float(v0 @ v0) - gm / r_init
        assert E - c**2 == pytest.approx(eps_newton, rel=1e-6)
    tau, _t, r, phi = sch.geodesic_orbit(r0, v0, gm, c, tau_end=tau_end)
    dr_dtau = CubicSpline(tau, r).derivative()
    dphi_dtau = CubicSpline(tau, phi).derivative()
    for k in (tau.size // 3, tau.size // 2, (4 * tau.size) // 5):
        radial = np.array([math.cos(phi[k]), math.sin(phi[k])])
        azimuthal = np.array([-radial[1], radial[0]])
        pos = r[k] * radial
        vel = float(dr_dtau(tau[k])) * radial + r[k] * float(dphi_dtau(tau[k])) * azimuthal
        E_k, L_k = sch.conserved_quantities(pos, vel, gm, c)
        assert E_k - c**2 == pytest.approx(E - c**2, rel=1e-6)
        assert L_k == pytest.approx(L, rel=1e-8)


# --------------------------------------------------------------------------- Newtonian limit
def test_newtonian_limit_reproduces_kepler_ellipse(mercury):
    """With c = 1e9 the Binet orbit is the Kepler ellipse p/(1 + e cos phi) and t obeys Kepler."""
    a, e, gm = mercury["a"], mercury["e"], mercury["gm"]
    phi, r, t = sch.binet_orbit(mercury["r0"], mercury["v0"], gm, c=1e9, n_orbits=3)
    assert phi.shape == r.shape == t.shape == (3 * 2000 + 1,)
    assert phi[0] == 0.0 and phi[-1] == pytest.approx(6.0 * math.pi)
    p = a * (1.0 - e * e)
    np.testing.assert_allclose(r, p / (1.0 + e * np.cos(phi)), rtol=1e-10)
    t_kepler = kepler_time_from_true_anomaly(phi, a, e, gm)
    np.testing.assert_allclose(t, t_kepler, atol=1e-9)
    assert t[-1] == pytest.approx(3.0 * mercury["period"], rel=1e-10)


# --------------------------------------------------------------------------- perihelion advance
def test_periapsis_angles_recovers_exact_ellipse_minima():
    a, e = 1.0, 0.3
    p = a * (1.0 - e * e)
    phi = np.linspace(-0.5, 3.0 * 2.0 * math.pi + 0.5, 7001)
    # perturb the grid so the parabola must cope with non-uniform spacing
    phi = phi + 0.3 * (phi[1] - phi[0]) * np.sin(7.0 * phi)
    r = p / (1.0 + e * np.cos(phi))
    peri = sch.periapsis_angles(phi, r)
    np.testing.assert_allclose(peri, 2.0 * math.pi * np.arange(0, 4), atol=1e-9)
    # monotonic radius: no interior minimum
    assert sch.periapsis_angles(phi, np.exp(phi)).size == 0
    with pytest.raises(ValueError):
        sch.periapsis_angles(phi, r[:-1])


def test_precession_1pn_formula():
    assert sch.precession_per_orbit_1pn(MERCURY_A, MERCURY_E, GM_SUN, C_AU_DAY) == pytest.approx(
        6.0 * math.pi * GM_SUN / (C_AU_DAY**2 * MERCURY_A * (1.0 - MERCURY_E**2))
    )
    # 42.98"/century for Mercury
    period_days = 2.0 * math.pi * math.sqrt(MERCURY_A**3 / GM_SUN)
    per_century = sch.precession_per_orbit_1pn(MERCURY_A, MERCURY_E, GM_SUN) * 36525.0 / period_days
    assert per_century * 180.0 * 3600.0 / math.pi == pytest.approx(42.98, abs=0.01)


def test_precession_mercury_matches_1pn():
    """Exact advance vs 1PN for Mercury.

    Physics: the next-order term is ~7.5 mu/p ~ 2e-7 relative.  Numerics: DOP853 at rtol 1e-12
    accumulates ~1e-10 rad of phase over 50 orbits, i.e. ~2e-6 relative on the mean advance, which
    sets the 1e-5 tolerance.
    """
    num = sch.precession_per_orbit_numeric(MERCURY_A, MERCURY_E, GM_SUN, C_AU_DAY, n_orbits=50)
    pn1 = sch.precession_per_orbit_1pn(MERCURY_A, MERCURY_E, GM_SUN, C_AU_DAY)
    assert num == pytest.approx(pn1, rel=1e-5)
    assert num == pytest.approx(5.02e-7, abs=1e-9)


@pytest.mark.parametrize("scale", [100, 300])
def test_precession_scaled_c_shows_next_order_term(scale):
    """With c/100 and c/300 the exact advance exceeds 1PN by the second-order term.

    Expanding the exact Schwarzschild advance for an orbit started at perihelion with the Newtonian
    velocity of the osculating ellipse (a, e) gives, with mu = gm/c^2 and p = a(1 - e^2),

        Delta phi = 6 pi (mu/p) [1 + (30 + 5 e^2)/4 (mu/p) + O((mu/p)^2)],

    i.e. a relative excess of ~7.55 mu/p for Mercury's eccentricity (2.0e-3 at c/100, 1.8e-2 at
    c/300, with a ~2% third-order correction at c/300).
    """
    c = C_AU_DAY / scale
    a, e, gm = MERCURY_A, MERCURY_E, GM_SUN
    mu_over_p = gm / (c**2 * a * (1.0 - e * e))
    num = sch.precession_per_orbit_numeric(a, e, gm, c, n_orbits=50)
    pn1 = sch.precession_per_orbit_1pn(a, e, gm, c)
    rel = num / pn1 - 1.0
    assert abs(rel) < 10.0 * mu_over_p  # agreement within "a few" mu/p
    assert rel == pytest.approx((30.0 + 5.0 * e * e) / 4.0 * mu_over_p, rel=0.1)


# --------------------------------------------------------------------------- orbit integrators
@pytest.mark.parametrize("case", list(ORBIT_CASES))
def test_binet_and_geodesic_agree(case):
    """r(phi) and t(phi) from the Binet and the proper-time integrations coincide.

    The generic (off-apsis) cases exercise the Binet initial slope u'(phi0) = -v_r/|L|: a sign
    error there would send the two integrations in opposite radial directions.  The retrograde
    case checks the sense of rotation with v_r != 0.
    """
    r0, v0, gm, c, tau_end, n_orbits = orbit_case(case)
    _E, L = sch.conserved_quantities(r0, v0, gm, c)
    sense = 1.0 if L > 0.0 else -1.0
    r_init = float(np.hypot(*r0))
    v_r = float(r0 @ v0) / r_init
    phi0 = math.atan2(r0[1], r0[0])
    tau, t_geo, r_geo, phi_geo = sch.geodesic_orbit(r0, v0, gm, c, tau_end=tau_end)
    assert tau.shape == t_geo.shape == r_geo.shape == phi_geo.shape == (20000,)
    assert tau[0] == 0.0 and tau[-1] == pytest.approx(tau_end)
    assert np.all(np.diff(t_geo) > 0) and np.all(sense * np.diff(phi_geo) > 0)
    phi_b, r_b, t_b = sch.binet_orbit(r0, v0, gm, c, n_orbits=n_orbits)
    assert phi_b[0] == pytest.approx(phi0) and phi_geo[0] == pytest.approx(phi0)
    assert sense * phi_b[-1] > sense * phi_geo[-1]  # the Binet run must cover the azimuth range
    r_of_phi = CubicSpline(sense * phi_b, r_b)  # sense * phi is increasing for either rotation
    t_of_phi = CubicSpline(sense * phi_b, t_b)
    np.testing.assert_allclose(r_of_phi(sense * phi_geo), r_geo, rtol=1e-8)
    np.testing.assert_allclose(t_of_phi(sense * phi_geo), t_geo, rtol=1e-8, atol=1e-10 * tau_end)
    # initial slope dr/dphi = (dr/dtau)/(dphi/dtau) = v_r r0^2 / L (signed; zero at an apsis)
    slope = sense * float(r_of_phi.derivative()(sense * phi_b[0]))
    assert slope == pytest.approx(v_r * r_init**2 / L, rel=1e-6, abs=1e-6 * r_init)
    # coordinate time runs faster than proper time (gravitational + kinematic time dilation)
    assert np.all(t_geo[1:] > tau[1:])


def test_strong_field_orbit_large_advance_outside_photon_sphere():
    """Orbit launched at r0 = 10 gm/c^2 around the Sun: huge advance, never inside r = 3 gm/c^2."""
    gm, c = GM_SUN, C_AU_DAY
    mu = gm / c**2
    r0 = 10.0 * mu
    # slightly sub-circular angular momentum: the start is the apoapsis of an eccentric orbit
    L = 0.97 * sch.circular_orbit_angular_momentum(r0, gm, c)
    phi, r, t = sch.binet_orbit([r0, 0.0], [0.0, L / r0], gm, c, n_orbits=8)
    assert r.min() > sch.photon_sphere_radius(gm, c)
    assert r.max() == pytest.approx(r0, rel=1e-10)
    peri = sch.periapsis_angles(phi, r)
    assert peri.size >= 4
    advances = np.diff(peri) - 2.0 * math.pi
    assert np.all(advances > 0.5)
    np.testing.assert_allclose(advances, advances[0], rtol=1e-8)  # the same every radial period
    assert np.all(np.diff(t) > 0)


def test_capture_orbit_terminates_above_horizon():
    gm, c = 1.0, 1.0
    rs = sch.schwarzschild_radius(gm, c)
    r0, v0 = [10.0, 0.0], [0.0, 0.3]  # L = 3 < sqrt(12): no barrier, the particle plunges
    tau, t, r, phi = sch.geodesic_orbit(r0, v0, gm, c, tau_end=1000.0, n_samples=4000)
    assert tau.shape == t.shape == r.shape == phi.shape
    assert 2 < tau.size < 4000
    assert rs < r[-1] < sch.photon_sphere_radius(gm, c)
    assert np.all(np.diff(r) < 0)
    phi_b, r_b, t_b = sch.binet_orbit(r0, v0, gm, c, n_orbits=5)
    assert phi_b.shape == r_b.shape == t_b.shape and phi_b.size < 5 * 2000 + 1
    assert rs < r_b[-1] < sch.photon_sphere_radius(gm, c)
    # custom stopping radius
    tau, t, r, phi = sch.geodesic_orbit(r0, v0, gm, c, tau_end=1000.0, n_samples=4000, r_stop=4.0)
    assert 4.0 <= r[-1] < 4.2 and r.size < 4000


def test_retrograde_orbit_mirrors_prograde(mercury):
    r0, v0, gm = mercury["r0"], mercury["v0"], mercury["gm"]
    phi_p, r_p, t_p = sch.binet_orbit(r0, v0, gm, C_AU_DAY, n_orbits=1.5)
    phi_r, r_r, t_r = sch.binet_orbit(r0, -v0, gm, C_AU_DAY, n_orbits=1.5)
    np.testing.assert_allclose(phi_r, -phi_p)
    np.testing.assert_array_equal(r_r, r_p)
    np.testing.assert_array_equal(t_r, t_p)
    period = mercury["period"]
    tau, t, r, phi = sch.geodesic_orbit(r0, -v0, gm, C_AU_DAY, tau_end=period, n_samples=50)
    assert np.all(np.diff(phi) < 0) and np.all(np.diff(t) > 0)


def test_orbit_functions_do_not_mutate_inputs(mercury):
    r0, v0 = mercury["r0"].copy(), mercury["v0"].copy()
    r0_ref, v0_ref = r0.copy(), v0.copy()
    sch.binet_orbit(r0, v0, GM_SUN, C_AU_DAY, n_orbits=0.5)
    sch.geodesic_orbit(r0, v0, GM_SUN, C_AU_DAY, tau_end=10.0, n_samples=100)
    sch.conserved_quantities(r0, v0, GM_SUN, C_AU_DAY)
    np.testing.assert_array_equal(r0, r0_ref)
    np.testing.assert_array_equal(v0, v0_ref)
    r = np.array([0.3, 0.4, 0.5])
    r_ref = r.copy()
    sch.effective_potential(r, 0.01, GM_SUN, C_AU_DAY)
    sch.radial_acceleration(r, 0.01, GM_SUN, C_AU_DAY)
    sch.circular_orbit_angular_momentum(r, GM_SUN, C_AU_DAY)
    np.testing.assert_array_equal(r, r_ref)


def test_invalid_inputs_raise():
    gm, c = 1.0, 1.0
    with pytest.raises(ValueError):  # inside the horizon
        sch.conserved_quantities([1.5, 0.0], [0.0, 0.1], gm, c)
    with pytest.raises(ValueError):  # radial infall has L = 0: not a Binet orbit
        sch.binet_orbit([10.0, 0.0], [-0.1, 0.0], gm, c, n_orbits=1)
    with pytest.raises(ValueError):  # 3-D vectors are not accepted
        sch.binet_orbit([10.0, 0.0, 0.0], [0.0, 0.3, 0.0], gm, c, n_orbits=1)
    with pytest.raises(ValueError):
        sch.geodesic_orbit([10.0, 0.0], [0.0, 0.3], gm, c, tau_end=-1.0)
    with pytest.raises(ValueError):
        sch.binet_orbit([10.0, 0.0], [0.0, 0.3], gm, c, n_orbits=0)
    with pytest.raises(ValueError):  # start below the stopping radius
        sch.geodesic_orbit([3.0, 0.0], [0.0, 0.3], gm, c, tau_end=1.0, r_stop=5.0)
    with pytest.raises(ValueError):  # circular orbit has no perihelion
        sch.precession_per_orbit_numeric(MERCURY_A, 0.0, GM_SUN, C_AU_DAY)
