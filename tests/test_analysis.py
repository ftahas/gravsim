"""Tests for gravsim.analysis (kepler, elements, precession, conserved, compare).

Everything here is analytic: Kepler orbits built from elements, synthetic trajectories whose
argument of perihelion grows at a known rate, and closed-form energies.  No integrator, no network.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gravsim.analysis import compare, conserved, elements, kepler, precession
from gravsim.analysis.precession import SecularFit
from gravsim.bodies import Trajectory
from gravsim.constants import ARCSEC_PER_RAD, C_AU_DAY, GM, GM_SUN, JULIAN_CENTURY_DAYS

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- helpers / fixtures
def angle_diff(x, y):
    """Smallest signed difference between two angles (radians)."""
    return elements.wrap_angle(np.asarray(x) - np.asarray(y))


@pytest.fixture
def rng():
    return np.random.default_rng(20261013)


def random_elements(rng, n, inc_range=(0.05, np.pi - 0.05), e_range=(0.01, 0.95)):
    """Well-conditioned random elliptic elements (dict of arrays of length n)."""
    return {
        "a": rng.uniform(0.3, 30.0, n),
        "e": rng.uniform(*e_range, n),
        "inc": rng.uniform(*inc_range, n),
        "raan": rng.uniform(-np.pi, np.pi, n),
        "argp": rng.uniform(-np.pi, np.pi, n),
        "nu": rng.uniform(-np.pi, np.pi, n),
    }


def make_precessing_trajectory(
    a=0.38709893,
    e=0.20563069,
    inc=0.1222,
    raan=0.8435,
    argp0=0.5088,
    argp_rate=1e-4,
    gm=GM_SUN,
    n_orbits=30,
    n_per_orbit=997,
    phase=0.3,
    M0=None,
    body="Mercury",
) -> tuple[Trajectory, float]:
    """Kepler orbit whose argument of perihelion grows linearly: argp(t) = argp0 + argp_rate t.

    The velocity is the osculating Keplerian one, so the osculating elements at every sample
    are exactly (a, e, inc, raan, argp(t)).  The central body moves uniformly (its offset must
    cancel in `Trajectory.relative`).

    With `n_per_orbit` samples per period every perihelion passage sits at the same grid phase.
    By default (`M0=None`) the passages fall `phase` steps after a sample, at
    ``t = (k n_per_orbit + n_per_orbit // 2 + phase) h``.  Because |r(t)| is even about
    perihelion, `phase = 0` (on a sample) and `phase = 0.5` (exactly between two samples) make
    the three-point parabola exact by symmetry; the default 0.3 leaves its truncation error
    visible.  An explicit `M0` overrides `phase`.  The expected passage times strictly inside
    the sampled interval are stored in ``meta["t_peri"]``.
    """
    period = TWO_PI / kepler.mean_motion(a, gm)
    if M0 is None:
        M0 = TWO_PI * (1.0 - (n_per_orbit // 2 + phase) / n_per_orbit)
    t = np.linspace(0.0, n_orbits * period, n_orbits * n_per_orbit + 1)
    M = M0 + TWO_PI * t / period
    E = kepler.solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r, v = kepler.elements_to_state(a, e, inc, raan, argp0 + argp_rate * t, nu, gm)
    sun_v = np.array([1e-5, -2e-5, 3e-6])
    sun_r = np.array([0.003, -0.002, 0.001]) + sun_v * t[:, None]
    pos = np.stack([sun_r, sun_r + r], axis=1)
    vel = np.stack([np.broadcast_to(sun_v, r.shape), sun_v + v], axis=1)
    t_peri = (np.arange(n_orbits + 2) - M0 / TWO_PI) * period
    t_peri = t_peri[(t_peri > 0.0) & (t_peri < t[-1])]
    traj = Trajectory(
        t=t,
        pos=pos,
        vel=vel,
        gm=np.array([gm, 0.0]),
        names=("Sun", body),
        meta={"synthetic": True, "argp_rate": argp_rate, "period": period, "t_peri": t_peri},
    )
    return traj, period


# =========================================================================== kepler.solve_kepler
@pytest.mark.parametrize("e", [0.0, 0.3, 0.9, 0.99])
def test_solve_kepler_residual(e):
    M = np.linspace(-4 * np.pi, 4 * np.pi, 4001)
    E = kepler.solve_kepler(M, e)
    assert E.shape == M.shape
    assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-13


def test_solve_kepler_branch_scalar_and_broadcast():
    assert kepler.solve_kepler(0.0, 0.5) == 0.0
    assert np.isclose(kepler.solve_kepler(np.pi, 0.5), np.pi)
    E1 = kepler.solve_kepler(1.3, 0.4)
    E2 = kepler.solve_kepler(1.3 + 6 * np.pi, 0.4)
    assert np.isclose(E2 - E1, 6 * np.pi, atol=1e-13)
    assert np.ndim(E1) == 0
    e = np.array([0.1, 0.5, 0.9])
    E = kepler.solve_kepler(np.array([[0.5], [2.0]]), e)
    assert E.shape == (2, 3)
    assert np.allclose(E - e * np.sin(E), np.array([[0.5], [2.0]]), atol=1e-13)


def test_solve_kepler_rejects_unbound():
    with pytest.raises(ValueError):
        kepler.solve_kepler(1.0, 1.0)
    with pytest.raises(ValueError):
        kepler.solve_kepler(1.0, -0.1)
    with pytest.raises(ValueError):
        kepler.solve_kepler(np.nan, 0.1)


NEAR_PARABOLIC_E = [
    0.0, 0.3, 0.9, 0.99, 1 - 1e-4, 1 - 1e-6, 1 - 1e-8, 1 - 1e-10, 1 - 1e-12, 1 - 1e-15
]


def kepler_reference(E, M, e):
    """Polish a float64 solution of Kepler's equation with Newton steps in `np.longdouble`."""
    ld = np.longdouble
    E, M, e = (np.asarray(x).astype(ld) for x in (E, M, e))
    for _ in range(3):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def test_solve_kepler_near_parabolic_stops_at_the_rounding_floor():
    # e -> 1 with M -> 0 is where f' = 1 - e cos E -> 0: at the rounding floor of
    # f = E - e sin E - M the Halley step is noise / f', far above any absolute tolerance, so
    # the solver must recognise the floor instead of stalling (it used to raise RuntimeError).
    # One vectorised call mixes benign and ill-conditioned samples: each must stop on its own.
    e = np.array(NEAR_PARABOLIC_E)[:, None]
    m = np.logspace(-16, np.log10(np.pi), 60)
    M = np.concatenate([-m[::-1], [0.0], m, np.linspace(-np.pi, np.pi, 101)])[None, :]
    E = kepler.solve_kepler(M, e)
    assert E.shape == (e.size, M.size) and np.all(np.isfinite(E))
    eps = np.finfo(np.float64).eps
    scale = np.maximum(np.abs(E), np.abs(M))
    ld = np.longdouble
    El = E.astype(ld)
    resid = np.abs(El - e.astype(ld) * np.sin(El) - M.astype(ld))
    assert np.all(resid <= 12 * eps * scale)  # the true residual is at the float64 floor
    if np.finfo(ld).eps < eps:  # extended precision available: check E itself
        fp = 1.0 - e * np.cos(E)  # conditioning: dE/dM = 1 / f'
        err = np.abs(El - kepler_reference(E, M, e))
        assert np.all(err <= 12 * eps * scale / fp + 4 * eps * np.abs(E))


@pytest.mark.parametrize(
    "e, M",
    [(1 - 1e-6, 1e-9), (1 - 1e-6, 1e-7), (1 - 1e-7, 1e-10), (1 - 1e-9, 1e-12), (1 - 1e-15, 1e-7)],
)
def test_solve_kepler_near_parabolic_scalars_regression(e, M):
    E = kepler.solve_kepler(M, e)  # used to raise RuntimeError (|dE| stuck just above tol)
    assert np.ndim(E) == 0 and 0.0 < E < 0.1
    assert abs(E - e * np.sin(E) - M) <= 8 * np.finfo(np.float64).eps * E


# =========================================================================== elements <-> state
def test_elements_state_round_trip_generic(rng):
    el = random_elements(rng, 300)
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    assert r.shape == (300, 3) and v.shape == (300, 3)
    out = kepler.state_to_elements(r, v, GM_SUN)
    assert np.allclose(out["a"], el["a"], rtol=1e-10)
    assert np.allclose(out["e"], el["e"], atol=1e-10)
    assert np.allclose(out["inc"], el["inc"], atol=1e-10)
    for key in ("raan", "argp", "nu"):
        assert np.max(np.abs(angle_diff(out[key], el[key]))) < 1e-9
    assert np.max(np.abs(angle_diff(out["varpi"], el["raan"] + el["argp"]))) < 1e-9
    assert np.allclose(out["period"], TWO_PI * np.sqrt(el["a"] ** 3 / GM_SUN), rtol=1e-10)
    E = 2 * np.arctan2(
        np.sqrt(1 - el["e"]) * np.sin(el["nu"] / 2), np.sqrt(1 + el["e"]) * np.cos(el["nu"] / 2)
    )
    assert np.max(np.abs(angle_diff(out["M"], E - el["e"] * np.sin(E)))) < 1e-9
    # every angle wrapped to (-pi, pi]
    for key in ("raan", "argp", "nu", "M", "varpi"):
        assert np.all(out[key] > -np.pi) and np.all(out[key] <= np.pi)
    assert np.all((out["inc"] >= 0) & (out["inc"] <= np.pi))
    # and back to the state
    r2, v2 = kepler.elements_to_state(
        out["a"], out["e"], out["inc"], out["raan"], out["argp"], out["nu"], GM_SUN
    )
    assert np.allclose(r2, r, rtol=0, atol=1e-11 * np.max(np.abs(r)))
    assert np.allclose(v2, v, rtol=0, atol=1e-11 * np.max(np.abs(v)))


@pytest.mark.parametrize("inc", [0.0, 1e-9, 1e-6, np.pi - 1e-6, np.pi - 1e-9, np.pi])
def test_elements_state_round_trip_near_equatorial(rng, inc):
    el = random_elements(rng, 50, inc_range=(inc, inc))
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    out = kepler.state_to_elements(r, v, GM_SUN)
    assert np.allclose(out["a"], el["a"], rtol=1e-10)
    assert np.allclose(out["e"], el["e"], atol=1e-10)
    assert np.allclose(out["inc"], inc, atol=1e-12)
    # raan and argp are ill-conditioned individually; the longitudes of perihelion and of the
    # body (raan + argp [+ nu] prograde, raan - argp [- nu] retrograde) are not.
    sign = 1.0 if inc < np.pi / 2 else -1.0
    assert np.max(np.abs(angle_diff(out["raan"] + sign * out["argp"],
                                    el["raan"] + sign * el["argp"]))) < 1e-8
    assert np.max(np.abs(angle_diff(out["raan"] + sign * (out["argp"] + out["nu"]),
                                    el["raan"] + sign * (el["argp"] + el["nu"])))) < 1e-8
    assert np.max(np.abs(angle_diff(out["nu"], el["nu"]))) < 1e-8
    if inc == 0.0 or inc == np.pi:
        assert np.all(out["raan"] == 0.0)
    # state round trip is exact regardless of the angle conditioning
    r2, v2 = kepler.elements_to_state(
        out["a"], out["e"], out["inc"], out["raan"], out["argp"], out["nu"], GM_SUN
    )
    assert np.allclose(r2, r, rtol=0, atol=1e-11 * np.max(np.abs(r)))
    assert np.allclose(v2, v, rtol=0, atol=1e-11 * np.max(np.abs(v)))


@pytest.mark.parametrize("e", [0.0, 1e-10, 1e-6])
def test_elements_state_round_trip_near_circular(rng, e):
    el = random_elements(rng, 50, e_range=(e, e))
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    out = kepler.state_to_elements(r, v, GM_SUN)
    assert np.allclose(out["a"], el["a"], rtol=1e-10)
    assert np.allclose(out["e"], e, atol=1e-12)
    assert np.allclose(out["inc"], el["inc"], atol=1e-10)
    assert np.max(np.abs(angle_diff(out["raan"], el["raan"]))) < 1e-9
    # argp is ill-conditioned; argp + nu (argument of latitude) is not
    assert np.max(np.abs(angle_diff(out["argp"] + out["nu"], el["argp"] + el["nu"]))) < 1e-8
    if e == 0.0:
        assert np.all(out["argp"] == 0.0)
    r2, v2 = kepler.elements_to_state(
        out["a"], out["e"], out["inc"], out["raan"], out["argp"], out["nu"], GM_SUN
    )
    assert np.allclose(r2, r, rtol=0, atol=1e-11 * np.max(np.abs(r)))
    assert np.allclose(v2, v, rtol=0, atol=1e-11 * np.max(np.abs(v)))


def test_state_to_elements_scalar_and_time_axis(rng):
    el = random_elements(rng, 7)
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    out_all = kepler.state_to_elements(r, v, GM_SUN)
    for i in range(7):
        out_i = kepler.state_to_elements(r[i], v[i], GM_SUN)
        for key, val in out_i.items():
            assert np.ndim(val) == 0
            assert np.isclose(val, out_all[key][i], rtol=0, atol=1e-14)
    # (T, N, 3)-style leading axes also work
    out_2d = kepler.state_to_elements(r.reshape(7, 1, 3), v.reshape(7, 1, 3), GM_SUN)
    assert out_2d["a"].shape == (7, 1)
    assert np.allclose(out_2d["a"][:, 0], out_all["a"])


def test_state_to_elements_rejects_unbound():
    r = np.array([1.0, 0.0, 0.0])
    v_esc = np.array([0.0, np.sqrt(2 * GM_SUN) * 1.01, 0.0])
    with pytest.raises(ValueError):
        kepler.state_to_elements(r, v_esc, GM_SUN)
    with pytest.raises(ValueError):
        kepler.elements_to_state(1.0, 1.2, 0.0, 0.0, 0.0, 0.0, GM_SUN)


def test_elements_to_state_matches_textbook_perifocal():
    a, e, gm = 1.3, 0.4, GM_SUN
    p = a * (1 - e * e)
    nu = 0.9
    r, v = kepler.elements_to_state(a, e, 0.0, 0.0, 0.0, nu, gm)
    rmag = p / (1 + e * np.cos(nu))
    assert np.allclose(r, rmag * np.array([np.cos(nu), np.sin(nu), 0.0]))
    assert np.allclose(v, np.sqrt(gm / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0]))
    # vis-viva
    assert np.isclose(np.dot(v, v), gm * (2 / rmag - 1 / a))


# =========================================================================== kepler.propagate
def test_propagate_one_period_returns_initial_state(rng):
    el = random_elements(rng, 40)
    r0, v0 = kepler.elements_to_state(gm=GM_SUN, **el)
    # The period of the orbit the state actually encodes: `a` from a state near perihelion of
    # an eccentric orbit is ill-conditioned (da/a = 2a/r dr/r), so the nominal period differs
    # from it by O(eps/(1-e)) and only the self-consistent one closes the orbit to 1e-12.
    period = kepler.state_to_elements(r0, v0, GM_SUN)["period"]
    r1, v1 = kepler.propagate(r0, v0, GM_SUN, period)
    assert np.max(np.abs(r1 - r0) / np.linalg.norm(r0, axis=-1)[:, None]) < 1e-12
    assert np.max(np.abs(v1 - v0) / np.linalg.norm(v0, axis=-1)[:, None]) < 1e-12
    nominal = TWO_PI * np.sqrt(el["a"] ** 3 / GM_SUN)
    assert np.allclose(period, nominal, rtol=1e-13)
    r1, v1 = kepler.propagate(r0, v0, GM_SUN, nominal)
    assert np.max(np.abs(r1 - r0) / np.linalg.norm(r0, axis=-1)[:, None]) < 1e-10
    assert np.max(np.abs(v1 - v0) / np.linalg.norm(v0, axis=-1)[:, None]) < 1e-10
    # zero propagation is the identity
    r2, v2 = kepler.propagate(r0, v0, GM_SUN, 0.0)
    assert np.max(np.abs(r2 - r0) / np.linalg.norm(r0, axis=-1)[:, None]) < 1e-13


def test_propagate_half_period_reaches_aphelion():
    a, e, inc, raan, argp = 0.7, 0.35, 0.3, 1.1, -0.4
    r_p, v_p = kepler.elements_to_state(a, e, inc, raan, argp, 0.0, GM_SUN)
    period = TWO_PI * np.sqrt(a**3 / GM_SUN)
    r_a, v_a = kepler.propagate(r_p, v_p, GM_SUN, period / 2)
    r_a_ref, v_a_ref = kepler.elements_to_state(a, e, inc, raan, argp, np.pi, GM_SUN)
    assert np.allclose(r_a, r_a_ref, atol=1e-13)
    assert np.allclose(v_a, v_a_ref, atol=1e-15)
    assert np.isclose(np.linalg.norm(r_a), a * (1 + e))
    assert np.isclose(np.linalg.norm(v_a), np.sqrt(GM_SUN / (a * (1 - e * e))) * (1 - e))
    # and the quarter-period point obeys Kepler's equation
    r_q, _ = kepler.propagate(r_p, v_p, GM_SUN, period / 4)
    E = kepler.solve_kepler(np.pi / 2, e)
    assert np.isclose(np.linalg.norm(r_q), a * (1 - e * np.cos(E)))


def test_propagate_composes_and_vectorises_over_dt(rng):
    el = random_elements(rng, 1)
    r0, v0 = kepler.elements_to_state(gm=GM_SUN, **el)
    r0, v0 = r0[0], v0[0]
    dts = np.array([0.0, 3.7, 50.0, 400.0, -120.0])
    r_vec, v_vec = kepler.propagate(r0, v0, GM_SUN, dts)
    assert r_vec.shape == (5, 3)
    for k, dt in enumerate(dts):
        r_k, v_k = kepler.propagate(r0, v0, GM_SUN, dt)
        assert np.allclose(r_k, r_vec[k], atol=1e-14)
        assert np.allclose(v_k, v_vec[k], atol=1e-16)
    r_ab, v_ab = kepler.propagate(*kepler.propagate(r0, v0, GM_SUN, 123.4), GM_SUN, 456.7)
    r_c, v_c = kepler.propagate(r0, v0, GM_SUN, 123.4 + 456.7)
    assert np.allclose(r_ab, r_c, atol=1e-12)
    assert np.allclose(v_ab, v_c, atol=1e-14)


def test_propagate_matches_numerical_integration():
    a, e = 0.5, 0.6
    r0, v0 = kepler.elements_to_state(a, e, 0.4, 0.2, 1.5, 2.0, GM_SUN)

    def rhs(t, y):
        r = y[:3]
        return np.concatenate([y[3:], -GM_SUN * r / np.linalg.norm(r) ** 3])

    t_end = 0.37 * TWO_PI * np.sqrt(a**3 / GM_SUN)
    sol = solve_ivp(rhs, (0, t_end), np.concatenate([r0, v0]), method="DOP853",
                    rtol=1e-13, atol=1e-16)
    r1, v1 = kepler.propagate(r0, v0, GM_SUN, t_end)
    assert np.allclose(r1, sol.y[:3, -1], atol=1e-10)
    assert np.allclose(v1, sol.y[3:, -1], atol=1e-12)


def test_propagate_comet_through_perihelion():
    # A vectorised sweep across perihelion of a near-parabolic orbit puts some samples in the
    # M -> 0, e -> 1 corner of Kepler's equation; one stalled sample used to abort the whole call.
    a, e = 5.0e5, 1 - 1e-6  # q = 0.5 AU
    r_p, v_p = kepler.elements_to_state(a, e, 0.3, 1.0, 2.0, 0.0, GM_SUN)
    dt = np.linspace(-50.0, 50.0, 2001)
    r, v = kepler.propagate(r_p, v_p, GM_SUN, dt)
    assert r.shape == (2001, 3) and np.all(np.isfinite(r)) and np.all(np.isfinite(v))
    rmag = np.linalg.norm(r, axis=-1)
    assert np.argmin(rmag) == 1000 and np.isclose(rmag[1000], a * (1 - e), rtol=1e-9)
    assert np.allclose(rmag[1001:], rmag[999::-1], rtol=1e-9)  # |r(t)| is even about perihelion
    assert np.allclose(elements.lrl_vector(r, v, GM_SUN), elements.lrl_vector(r_p, v_p, GM_SUN),
                       atol=1e-9)


# =========================================================================== elements.py
def test_lrl_vector_magnitude_and_direction(rng):
    el = random_elements(rng, 200)
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    evec = elements.lrl_vector(r, v, GM_SUN)
    assert evec.shape == (200, 3)
    assert np.allclose(np.linalg.norm(evec, axis=-1), el["e"], atol=1e-12)
    r_peri, _ = kepler.elements_to_state(
        el["a"], el["e"], el["inc"], el["raan"], el["argp"], 0.0, GM_SUN
    )
    r_peri_hat = r_peri / np.linalg.norm(r_peri, axis=-1)[:, None]
    assert np.allclose(evec / el["e"][:, None], r_peri_hat, atol=1e-10)
    # constant along the orbit: propagate and recompute
    r_t, v_t = kepler.propagate(r[0], v[0], GM_SUN, np.linspace(0, 1000, 50))
    e_t = elements.lrl_vector(r_t, v_t, GM_SUN)
    assert np.allclose(e_t, evec[0], atol=1e-12)


def test_specific_h_and_node_vector(rng):
    el = random_elements(rng, 20)
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    h = elements.specific_h(r, v)
    assert np.allclose(h, np.cross(r, v))
    p = el["a"] * (1 - el["e"] ** 2)
    assert np.allclose(np.linalg.norm(h, axis=-1), np.sqrt(GM_SUN * p))
    n = elements.node_vector(h)
    assert np.allclose(n[:, 2], 0.0)
    assert np.allclose(np.arctan2(n[:, 1], n[:, 0]), el["raan"])
    assert np.allclose(np.sum(n * h, axis=-1), 0.0, atol=1e-18)
    # equatorial orbit: node along +x
    h_eq = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    assert np.allclose(elements.node_vector(h_eq), [[1, 0, 0], [1, 0, 0]])


def test_perihelion_longitude_equals_raan_plus_argp(rng):
    el = random_elements(rng, 300)
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    varpi = elements.perihelion_longitude(r, v, GM_SUN)
    assert varpi.shape == (300,)
    assert np.all(varpi > -np.pi) and np.all(varpi <= np.pi)
    assert np.max(np.abs(angle_diff(varpi, el["raan"] + el["argp"]))) < 1e-10
    assert np.allclose(varpi, kepler.state_to_elements(r, v, GM_SUN)["varpi"])
    # scalar input gives a scalar
    assert np.ndim(elements.perihelion_longitude(r[0], v[0], GM_SUN)) == 0


def test_perihelion_longitude_equatorial_cases():
    # prograde equatorial: atan2(e_y, e_x)
    for raan, argp in [(0.3, 1.1), (2.5, 2.0), (-2.0, -2.5)]:
        r, v = kepler.elements_to_state(1.0, 0.3, 0.0, raan, argp, 0.9, GM_SUN)
        evec = elements.lrl_vector(r, v, GM_SUN)
        varpi = elements.perihelion_longitude(r, v, GM_SUN)
        assert np.isclose(varpi, np.arctan2(evec[1], evec[0]))
        assert np.isclose(angle_diff(varpi, raan + argp), 0.0, atol=1e-12)
    # retrograde equatorial is consistent with state_to_elements
    r, v = kepler.elements_to_state(1.0, 0.3, np.pi, 0.3, 1.1, 0.9, GM_SUN)
    el = kepler.state_to_elements(r, v, GM_SUN)
    assert np.isclose(elements.perihelion_longitude(r, v, GM_SUN), el["raan"] + el["argp"])


def test_wrap_angle():
    x = np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi, 7.0, -7.0, 1e3])
    w = elements.wrap_angle(x)
    assert np.all(w > -np.pi) and np.all(w <= np.pi)
    assert np.allclose(np.sin(w), np.sin(x)) and np.allclose(np.cos(w), np.cos(x))
    assert w[1] == np.pi and w[2] == np.pi
    assert np.ndim(elements.wrap_angle(4.0)) == 0


# =========================================================================== precession.py
@pytest.fixture(scope="module")
def precessing():
    return make_precessing_trajectory()


def test_perihelion_longitude_series_recovers_linear_drift(precessing):
    traj, _ = precessing
    rate = traj.meta["argp_rate"]
    t, varpi = precession.perihelion_longitude_series(traj, "Mercury")
    assert t.shape == varpi.shape == traj.t.shape
    assert t is not traj.t and np.array_equal(t, traj.t)
    expected = elements.wrap_angle(0.8435 + 0.5088 + rate * t)
    assert np.max(np.abs(angle_diff(varpi, expected))) < 1e-10
    fit = precession.fit_secular_rate(t, varpi)
    assert abs(fit.rate - rate) / rate < 1e-6
    assert abs(fit.rate - rate) / rate < 1e-10  # it is in fact machine precision
    assert fit.rate_err < 1e-9 * rate
    assert np.isclose(angle_diff(fit.intercept, 0.8435 + 0.5088), 0.0, atol=1e-10)
    assert np.max(np.abs(fit.residuals)) < 1e-10
    assert isinstance(fit, SecularFit)
    assert np.isclose(fit.rate_arcsec_per_century, precession.rate_arcsec_per_century(rate))


def test_perihelion_passages_count_times_and_longitudes(precessing):
    traj, period = precessing
    t_peri, varpi_peri = precession.perihelion_passages(traj, "Mercury")
    expected = traj.meta["t_peri"]
    assert expected.shape == (30,)
    assert t_peri.shape == varpi_peri.shape == (30,)
    assert np.allclose(np.diff(expected), period)
    h = traj.t[1] - traj.t[0]
    # The passages sit 0.3 h past a sample (an unrefined time would be off by 0.3 h).  The
    # parabolic vertex carries the truncation error of three-point interpolation, ~2e-6 h at
    # 997 samples per orbit for e = 0.206 (it scales as h^2 in units of h: see the next test).
    assert np.max(np.abs(t_peri - expected)) < 1e-5 * h
    rate = traj.meta["argp_rate"]
    expected_varpi = elements.wrap_angle(0.8435 + 0.5088 + rate * expected)
    # varpi is taken at the nearest sample, at most half a step from the passage
    assert np.max(np.abs(angle_diff(varpi_peri, expected_varpi))) < rate * h
    fit = precession.fit_secular_rate(t_peri, varpi_peri)
    assert abs(fit.rate - rate) / rate < 1e-3
    assert fit.rate_err < 1e-2 * rate


@pytest.mark.parametrize("phase", [0.15, 0.3, 0.8])
def test_perihelion_passages_refinement_error_scales_as_h_cubed(phase):
    # |r(t)| is even about perihelion, so the cubic term of its Taylor series vanishes and the
    # vertex of the parabola through three samples is off by O(f4 h^3 / f2), f = |r|: halving h
    # divides the error by ~8.  (At phase 0 or 0.5 the parabola is exact by symmetry.)
    errs, steps = [], []
    for n in (50, 100, 200):
        traj, _ = make_precessing_trajectory(n_orbits=4, n_per_orbit=n, phase=phase)
        t_peri, _ = precession.perihelion_passages(traj, "Mercury")
        assert t_peri.shape == traj.meta["t_peri"].shape == (4,)
        errs.append(np.max(np.abs(t_peri - traj.meta["t_peri"])))
        steps.append(traj.t[1] - traj.t[0])
    assert errs[0] < 2e-3 * steps[0]  # unrefined would be ~phase * h
    ratios = np.array(errs[:-1]) / np.array(errs[1:])
    assert np.all((6.0 < ratios) & (ratios < 10.0))
    for symmetric_phase in (0.0, 0.5):
        traj, _ = make_precessing_trajectory(n_orbits=4, n_per_orbit=50, phase=symmetric_phase)
        t_peri, _ = precession.perihelion_passages(traj, "Mercury")
        assert t_peri.shape == (4,)
        assert np.max(np.abs(t_peri - traj.meta["t_peri"])) < 1e-9 * steps[0]


def test_perihelion_passages_exact_tie_counted_once():
    # Two samples straddling perihelion with bit-identical |r|, built by mirroring one half of a
    # planar orbit (y -> -y, v_x -> -v_x is exact in floating point, so this does not rely on
    # libm symmetry).  The tie must count once, and the parabola through two equal ordinates
    # has its vertex exactly at their midpoint.
    a, e, gm = 0.5, 0.4, GM_SUN
    period = TWO_PI / kepler.mean_motion(a, gm)
    h = period / 60
    t_pos = h * (0.5 + np.arange(8))
    E = kepler.solve_kepler(TWO_PI * t_pos / period, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r_pos, v_pos = kepler.elements_to_state(a, e, 0.0, 0.0, 0.0, nu, gm)  # xy-plane, peri on +x
    r_neg = (r_pos * np.array([1.0, -1.0, 1.0]))[::-1]
    v_neg = (v_pos * np.array([-1.0, 1.0, -1.0]))[::-1]
    t = np.concatenate([-t_pos[::-1], t_pos])
    # cyclic axis permutation (a rotation): inc = pi/2, raan = pi/2, argp = 0 -> varpi = pi/2
    r = np.concatenate([r_neg, r_pos])[:, [2, 0, 1]]
    v = np.concatenate([v_neg, v_pos])[:, [2, 0, 1]]
    el = kepler.state_to_elements(r, v, gm)  # the mirrored half is the same Kepler orbit
    assert np.allclose(el["e"], e, atol=1e-12) and np.allclose(el["a"], a, rtol=1e-12)
    assert np.max(np.abs(angle_diff(el["nu"], np.concatenate([-nu[::-1], nu])))) < 1e-12
    rmag = np.sqrt(np.sum(r * r, axis=-1))
    assert rmag[7] == rmag[8] and rmag[7] < rmag[6] and rmag[8] < rmag[9]
    traj = Trajectory(
        t, np.stack([np.zeros_like(r), r], axis=1), np.stack([np.zeros_like(v), v], axis=1),
        np.array([gm, 0.0]), ("Sun", "comet"),
    )
    t_peri, varpi_peri = precession.perihelion_passages(traj, "comet")
    assert t_peri.shape == varpi_peri.shape == (1,)
    assert abs(t_peri[0]) < 1e-12 * h
    assert np.isclose(angle_diff(varpi_peri[0], np.pi / 2), 0.0, atol=1e-12)


def test_perihelion_passages_edge_cases():
    traj, period = make_precessing_trajectory(n_orbits=2, n_per_orbit=50, M0=0.0)
    # starting at perihelion: t=0 and t=2P are minima at the ends -> only the interior one
    t_peri, varpi_peri = precession.perihelion_passages(traj, "Mercury")
    assert t_peri.shape == (1,)
    assert np.isclose(t_peri[0], period, atol=1e-2 * period / 50)
    # too short a series
    short = Trajectory(traj.t[:2], traj.pos[:2], traj.vel[:2], traj.gm, traj.names)
    tp, vp = precession.perihelion_passages(short, "Mercury")
    assert tp.size == 0 and vp.size == 0
    # non-increasing times are refused
    bad = Trajectory(traj.t[::-1], traj.pos, traj.vel, traj.gm, traj.names)
    with pytest.raises(ValueError):
        precession.perihelion_passages(bad, "Mercury")


def test_fit_secular_rate_handles_wraparound():
    rate = 0.01  # rad/day: ~4 wraps over 2640 days
    t = np.linspace(0.0, 2640.0, 5000)
    angle = elements.wrap_angle(np.pi - 0.01 + rate * t)
    assert np.sum(np.abs(np.diff(angle)) > np.pi) >= 4
    fit = precession.fit_secular_rate(t, angle)
    assert abs(fit.rate - rate) / rate < 1e-12
    assert fit.rate_err < 1e-12
    # a negative rate, starting just above -pi
    angle = elements.wrap_angle(-np.pi + 0.01 - rate * t)
    fit = precession.fit_secular_rate(t, angle)
    assert abs(fit.rate + rate) / rate < 1e-12


def test_fit_secular_rate_error_estimate_and_validation(rng):
    t = np.linspace(0.0, 1000.0, 2001)
    sigma = 1e-3
    noise = rng.normal(0.0, sigma, t.size)
    fit = precession.fit_secular_rate(t, 0.2 + 5e-5 * t + noise)
    sxx = np.sum((t - t.mean()) ** 2)
    assert np.isclose(fit.rate, 5e-5, atol=5 * sigma / np.sqrt(sxx))
    assert np.isclose(fit.rate_err, sigma / np.sqrt(sxx), rtol=0.1)
    assert np.isclose(np.sqrt(np.mean(fit.residuals**2)), sigma, rtol=0.1)
    with pytest.raises(ValueError):
        precession.fit_secular_rate([0.0, 1.0, 2.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        precession.fit_secular_rate([1.0, 1.0], [0.0, 1.0])
    two = precession.fit_secular_rate([0.0, 1.0], [0.0, 0.5])
    assert two.rate == 0.5 and np.isnan(two.rate_err)


def test_rate_conversion():
    assert np.isclose(precession.rate_arcsec_per_century(1.0), ARCSEC_PER_RAD * JULIAN_CENTURY_DAYS)
    rad_per_day = 42.98 / ARCSEC_PER_RAD / JULIAN_CENTURY_DAYS
    assert np.isclose(precession.rate_arcsec_per_century(rad_per_day), 42.98)
    out = precession.rate_arcsec_per_century(np.array([1.0, 2.0]))
    assert out.shape == (2,)


def test_gr_precession_mercury_and_scalings():
    a, e = 0.38709893, 0.20563069
    gr = precession.gr_precession_arcsec_per_century(a, e, GM_SUN)
    assert abs(gr - 42.98) < 0.02
    # PPN factor (2 + 2 gamma - beta)/3 and 1/c^2 scaling
    assert np.isclose(precession.gr_precession_arcsec_per_century(a, e, GM_SUN, beta=1, gamma=0),
                      gr / 3)
    assert np.isclose(precession.gr_precession_arcsec_per_century(a, e, GM_SUN, beta=0, gamma=1),
                      gr * 4 / 3)
    assert np.isclose(precession.gr_precession_arcsec_per_century(a, e, GM_SUN, c=C_AU_DAY / 10),
                      gr * 100)
    # vectorised over planets: Venus, Earth, Mars (ROADMAP: 8.62, 3.84, 1.35)
    planets = precession.gr_precession_arcsec_per_century(
        np.array([0.72333199, 1.00000011, 1.52366231]),
        np.array([0.00677323, 0.01671022, 0.09341233]),
        GM_SUN,
    )
    assert np.allclose(planets, [8.62, 3.84, 1.35], atol=0.02)


# =========================================================================== conserved.py
def two_body_barycentric(gm1, gm2, r_rel, v_rel, v_cm=None):
    """Barycentric (N=2, 3) positions/velocities from a relative state (leading axes allowed)."""
    mu = gm1 + gm2
    r1, r2 = -gm2 / mu * r_rel, gm1 / mu * r_rel
    v1, v2 = -gm2 / mu * v_rel, gm1 / mu * v_rel
    if v_cm is not None:
        v1, v2 = v1 + v_cm, v2 + v_cm
    return np.stack([r1, r2], axis=-2), np.stack([v1, v2], axis=-2)


def test_energy_two_body_closed_form(rng):
    gm1, gm2 = GM_SUN, GM["Jupiter"]
    el = random_elements(rng, 25)
    r_rel, v_rel = kepler.elements_to_state(gm=gm1 + gm2, **el)
    pos, vel = two_body_barycentric(gm1, gm2, r_rel, v_rel)
    gm = np.array([gm1, gm2])
    # G*E = -gm1 gm2 / (2a) in the barycentric frame
    E = conserved.energy(pos, vel, gm)
    assert E.shape == (25,)
    assert np.allclose(E, -gm1 * gm2 / (2 * el["a"]), rtol=1e-12)
    # a centre-of-mass velocity adds 1/2 (gm1 + gm2) |V|^2
    v_cm = np.array([1e-3, -2e-3, 5e-4])
    pos2, vel2 = two_body_barycentric(gm1, gm2, r_rel, v_rel, v_cm)
    E2 = conserved.energy(pos2, vel2, gm)
    assert np.allclose(E2, -gm1 * gm2 / (2 * el["a"]) + 0.5 * (gm1 + gm2) * np.dot(v_cm, v_cm),
                       rtol=1e-12)
    # single-state call agrees with the time-series call and returns a scalar
    e0 = conserved.energy(pos[0], vel[0], gm)
    assert np.ndim(e0) == 0 and np.isclose(e0, E[0], rtol=1e-14)


def test_angular_momentum_two_body_closed_form(rng):
    gm1, gm2 = GM_SUN, GM["Saturn"]
    el = random_elements(rng, 25)
    r_rel, v_rel = kepler.elements_to_state(gm=gm1 + gm2, **el)
    pos, vel = two_body_barycentric(gm1, gm2, r_rel, v_rel)
    gm = np.array([gm1, gm2])
    L = conserved.angular_momentum(pos, vel, gm)
    assert L.shape == (25, 3)
    assert np.allclose(L, gm1 * gm2 / (gm1 + gm2) * np.cross(r_rel, v_rel), rtol=1e-12)
    L0 = conserved.angular_momentum(pos[0], vel[0], gm)
    assert L0.shape == (3,) and np.allclose(L0, L[0])


def test_energy_and_angular_momentum_conserved_along_kepler_orbit():
    gm1, gm2 = GM_SUN, GM["Mercury"]
    r0, v0 = kepler.elements_to_state(0.387, 0.2056, 0.12, 0.84, 0.5, 2.3, gm1 + gm2)
    t = np.linspace(0.0, 2000.0, 300)
    r_rel, v_rel = kepler.propagate(r0, v0, gm1 + gm2, t)
    pos, vel = two_body_barycentric(gm1, gm2, r_rel, v_rel)
    gm = np.array([gm1, gm2])
    E = conserved.energy(pos, vel, gm)
    assert np.max(np.abs(E - E[0]) / abs(E[0])) < 1e-12
    L = conserved.angular_momentum(pos, vel, gm)
    assert np.max(np.abs(L - L[0])) < 1e-12 * np.linalg.norm(L[0])
    traj = Trajectory(t, pos, vel, gm, ("Sun", "Mercury"))
    err = conserved.relative_energy_error(traj)
    assert err[0] == 0.0 and np.max(np.abs(err)) < 1e-12


def test_energy_test_particles_and_many_bodies(rng):
    # a test particle (gm = 0) contributes nothing
    pos = rng.normal(size=(4, 3))
    vel = rng.normal(size=(4, 3)) * 1e-2
    gm = np.array([1e-4, 3e-7, 0.0, 2e-8])
    E_full = conserved.energy(pos, vel, gm)
    E_no_tp = conserved.energy(pos[[0, 1, 3]], vel[[0, 1, 3]], gm[[0, 1, 3]])
    assert np.isclose(E_full, E_no_tp, rtol=1e-14)
    # explicit double loop for a 4-body state
    kin = 0.5 * sum(gm[i] * np.dot(vel[i], vel[i]) for i in range(4))
    pot = -sum(
        gm[i] * gm[j] / np.linalg.norm(pos[i] - pos[j]) for i in range(4) for j in range(i + 1, 4)
    )
    assert np.isclose(E_full, kin + pot, rtol=1e-14)
    # chunked path: a long series equals the per-sample values
    T = 3 * conserved._CHUNK + 17
    pos_t = np.repeat(pos[None], T, axis=0) * (1 + 1e-3 * np.arange(T))[:, None, None]
    vel_t = np.repeat(vel[None], T, axis=0)
    E_t = conserved.energy(pos_t, vel_t, gm)
    for k in (0, conserved._CHUNK - 1, conserved._CHUNK, T - 1):
        assert np.isclose(E_t[k], conserved.energy(pos_t[k], vel_t[k], gm), rtol=1e-14)
    with pytest.raises(ValueError):
        conserved.energy(pos, vel, gm[:3])


def test_relative_energy_error_zero_for_constant_trajectory():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.0, 0.0], [0.0, 0.017, 0.0]])
    T = 50
    traj = Trajectory(
        t=np.arange(T, dtype=float),
        pos=np.repeat(pos[None], T, axis=0),
        vel=np.repeat(vel[None], T, axis=0),
        gm=np.array([GM_SUN, GM["Jupiter"]]),
        names=("Sun", "p"),
    )
    err = conserved.relative_energy_error(traj)
    assert err.shape == (T,)
    assert np.all(err == 0.0)
    # a Sun plus test particles only has zero total energy: no scale to measure against
    zero = Trajectory(traj.t, traj.pos, traj.vel, np.array([GM_SUN, 0.0]), traj.names)
    with pytest.raises(ValueError):
        conserved.relative_energy_error(zero)


# =========================================================================== compare.py
def test_separation_constant_offset_and_grid_check(precessing):
    trajA, _ = precessing
    offset = np.array([1e-6, -2e-6, 2e-6])
    posB = trajA.pos.copy()
    posB[:, 1, :] += offset
    trajB = Trajectory(trajA.t, posB, trajA.vel, trajA.gm, trajA.names)
    sep = compare.separation(trajA, trajB, "Mercury")
    assert sep.shape == trajA.t.shape
    assert np.allclose(sep, np.linalg.norm(offset), rtol=1e-9)
    assert np.all(compare.separation(trajA, trajB, "Sun") == 0.0)
    # mismatched grids raise
    shifted = Trajectory(trajA.t + 1e-6, trajA.pos, trajA.vel, trajA.gm, trajA.names)
    with pytest.raises(ValueError):
        compare.separation(trajA, shifted, "Mercury")
    assert np.allclose(compare.separation(trajA, shifted, "Mercury", t_atol=1e-5), 0.0)
    shorter = Trajectory(trajA.t[:-1], trajA.pos[:-1], trajA.vel[:-1], trajA.gm, trajA.names)
    with pytest.raises(ValueError):
        compare.separation(trajA, shorter, "Mercury")


def test_perihelion_rate_difference_of_two_synthetic_runs(precessing):
    trajA, _ = precessing
    rate_a = trajA.meta["argp_rate"]
    rate_b = 1.3e-4
    trajB, _ = make_precessing_trajectory(argp_rate=rate_b)
    assert np.array_equal(trajA.t, trajB.t)
    rate, err = compare.perihelion_rate_difference(trajA, trajB, "Mercury")
    expected = precession.rate_arcsec_per_century(rate_a - rate_b)
    assert abs(rate - expected) / abs(expected) < 1e-6
    assert err < 1e-9 * abs(expected)
    # the separation grows from zero as the two perihelia drift apart
    sep = compare.separation(trajA, trajB, "Mercury")
    assert sep[0] < 1e-15 and sep[-1] > 1e-3
    # a run against itself: zero rate
    rate0, err0 = compare.perihelion_rate_difference(trajA, trajA, "Mercury")
    assert rate0 == 0.0 and err0 == 0.0
    # grids must match
    shorter = Trajectory(trajB.t[:-1], trajB.pos[:-1], trajB.vel[:-1], trajB.gm, trajB.names)
    with pytest.raises(ValueError):
        compare.perihelion_rate_difference(trajA, shorter, "Mercury")


def test_perihelion_rate_difference_cancels_wrap(precessing):
    # Each series wraps several times; the difference must not.
    trajA, _ = make_precessing_trajectory(argp_rate=0.01, argp0=np.pi - 0.3)
    trajB, _ = make_precessing_trajectory(argp_rate=0.01 - 2e-5, argp0=np.pi - 0.3)
    rate, err = compare.perihelion_rate_difference(trajA, trajB, "Mercury")
    expected = precession.rate_arcsec_per_century(2e-5)
    assert abs(rate - expected) / expected < 1e-6


# =========================================================================== purity
def test_functions_do_not_mutate_inputs(rng):
    el = random_elements(rng, 10)
    r, v = kepler.elements_to_state(gm=GM_SUN, **el)
    r_copy, v_copy = r.copy(), v.copy()
    gm = np.array([GM_SUN, 0.0])
    kepler.state_to_elements(r, v, GM_SUN)
    kepler.propagate(r, v, GM_SUN, 10.0)
    elements.perihelion_longitude(r, v, GM_SUN)
    elements.lrl_vector(r, v, GM_SUN)
    pos = np.stack([np.zeros_like(r), r], axis=1)
    vel = np.stack([np.zeros_like(v), v], axis=1)
    pos_copy, vel_copy = pos.copy(), vel.copy()
    conserved.energy(pos, vel, gm)
    conserved.angular_momentum(pos, vel, gm)
    traj = Trajectory(np.arange(10.0), pos, vel, gm, ("Sun", "p"))
    precession.perihelion_longitude_series(traj, "p")
    precession.perihelion_passages(traj, "p")
    assert np.array_equal(r, r_copy) and np.array_equal(v, v_copy)
    assert np.array_equal(pos, pos_copy) and np.array_equal(vel, vel_copy)
    angle = elements.wrap_angle(np.linspace(0, 20, 10))
    angle_copy = angle.copy()
    precession.fit_secular_rate(np.arange(10.0), angle)
    assert np.array_equal(angle, angle_copy)
