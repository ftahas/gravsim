"""Tests for ``gravsim.forces.eih`` (Einstein-Infeld-Hoffmann 1PN N-body law). No network.

Physics being checked
---------------------
* Newtonian limit: ``c -> infinity`` must give ``accel_newton`` (the law returns the Newtonian
  acceleration plus ``1/c^2`` times a correction, so this holds to roundoff).
* Test-particle limit (the key correctness check).  For one massive body ``0`` at rest at the
  origin and test particles ``i`` (``mu_i = 0``) every EIH term reduces to the ``pn1`` form:
  ``sum_{l != i} mu_l/r_il = mu_0/r`` (the other particles are massless);
  ``sum_{k != 0} mu_k/r_0k = 0``; ``v_0 = 0`` kills ``|v_0|^2``, ``v_i . v_0`` and
  ``((r_i - r_0) . v_0)^2``; ``a_0^N = 0`` because nothing attracts body 0, which kills the
  ``(r_0 - r_i) . a_0^N`` term and the whole third line.  The bracket becomes
  ``1 - 2(beta+gamma) mu_0/(c^2 r) + gamma v^2/c^2`` and the second line
  ``(mu_0/(c^2 r^3)) [ r . (2+2gamma) v ] v``, i.e. exactly
  ``-mu_0 r/r^3 + (mu_0/(c^2 r^3)) [ (2(beta+gamma) mu_0/r - gamma v^2) r + 2(1+gamma) (r.v) v ]``
  = ``accel_newton + make_pn1(c, central=0, beta, gamma)``, for every ``beta, gamma, c``.
  The identity is algebraic, so it is checked at ``c = 0.1`` and ``c = 1`` AU/day as well as at
  the physical ``c``: there the correction is 1e-8 of the total and recovering it by
  subtraction costs ~1e-8 of its own size in roundoff, so only the full acceleration is held
  to 1e-12 at the physical ``c``.
* Symmetry: the law is a function of the set of bodies, so relabelling bodies permutes the
  output; swapping two bodies with identical ``gm`` swaps their rows exactly.
* Mercury at J2000: the EIH correction differs from ``pn1`` by ~1e-3 of itself, and the
  difference is the Sun-pair terms that ``pn1`` lacks (the Sun's barycentric velocity and
  Newtonian acceleration, the planets' potentials at Mercury and at the Sun); the direct 1PN
  terms of the planets as sources are ~1e-3 of that again.
* Oracle: REBOUNDx ``gr_full`` (Tamayo et al. 2020) implements the same equations with
  ``beta = gamma = 1`` (iterating ``a_j`` to the full acceleration, an ``O(1/c^4)``
  difference); accelerations at t = 0 and positions after two years of Sun + 8 planets must
  agree (1e-10 relative, < 1 km).
* Precession: the run-vs-run differences against Newton give 42.98 (Mercury), 8.62 (Venus),
  3.84 (EMB), 1.35 (Mars) arcsec/century over a century (slow), the EIH and ``pn1`` Mercury
  rates agree to < 0.02 arcsec/century, and the two-body PPN rate scales as
  ``(2 + 2 gamma - beta)/3`` for both laws (10-year runs).
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
import pytest

from gravsim.analysis import (
    gr_precession_arcsec_per_century,
    perihelion_rate_difference,
    separation,
    state_to_elements,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import (
    AU_KM,
    C_AU_DAY,
    GM_SUN,
    JULIAN_YEAR_DAYS,
    MERCURY_PRECESSION_GR_ARCSEC_CY,
)
from gravsim.data.horizons import load_j2000
from gravsim.forces import accel_newton, make_eih, make_newton_pn1, make_pn1
from gravsim.integrators import integrate_scipy

RTOL, ATOL = 1e-12, 1e-15
GR_RATES = {"Mercury": MERCURY_PRECESSION_GR_ARCSEC_CY, "Venus": 8.62, "EMB": 3.84, "Mars": 1.35}
PPN_CASES = [(1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (2.0, 1.0)]


# --------------------------------------------------------------------------- helpers
def random_system(rng: np.random.Generator, n: int, n_test: int = 0):
    """Random positions in a ~10 AU box, velocities ~0.01 AU/day, log-uniform GMs."""
    pos = rng.uniform(-10.0, 10.0, size=(n, 3))
    vel = rng.normal(0.0, 0.01, size=(n, 3))
    gm = 10.0 ** rng.uniform(-11.0, -4.0, size=n)
    if n_test:
        gm[-n_test:] = 0.0
    return pos, vel, gm


def pair_1pn_terms(i, j, pos, vel, gm, a_newt, c, beta, gamma) -> np.ndarray:
    """All ``1/c^2`` terms of body ``j`` acting on body ``i`` (Moyer 2003, eq. 4-26).

    Plain-Python transcription of the formula in the ``eih`` module docstring, one pair at a
    time; the sums over ``l`` and ``k`` run over the whole system.  The Newtonian part
    ``mu_j (r_j - r_i)/r^3`` is *not* included.
    """
    n = len(gm)
    c2 = c * c
    d = pos[i] - pos[j]
    r = math.sqrt(float(d @ d))
    phi_i = sum(gm[k] / np.linalg.norm(pos[i] - pos[k]) for k in range(n) if k != i)
    phi_j = sum(gm[k] / np.linalg.norm(pos[j] - pos[k]) for k in range(n) if k != j)
    vi, vj, aj = vel[i], vel[j], a_newt[j]
    bracket_minus_one = (
        -2.0 * (beta + gamma) / c2 * phi_i
        - (2.0 * beta - 1.0) / c2 * phi_j
        + gamma * float(vi @ vi) / c2
        + (1.0 + gamma) * float(vj @ vj) / c2
        - 2.0 * (1.0 + gamma) / c2 * float(vi @ vj)
        - 1.5 / c2 * (float(d @ vj) / r) ** 2
        + 0.5 / c2 * float((-d) @ aj)
    )
    line1 = gm[j] * (-d) / r**3 * bracket_minus_one
    line2 = (
        gm[j]
        / (c2 * r**3)
        * float(d @ ((2.0 + 2.0 * gamma) * vi - (1.0 + 2.0 * gamma) * vj))
        * (vi - vj)
    )
    line3 = (3.0 + 4.0 * gamma) / (2.0 * c2) * gm[j] * aj / r
    return line1 + line2 + line3


def eih_scalar(pos, vel, gm, c, beta, gamma) -> np.ndarray:
    """Hand-written EIH acceleration: Newton plus the pair terms, triple Python loop."""
    n = len(gm)
    a_newt = accel_newton(pos, vel, gm, 0.0)
    out = a_newt.copy()
    for i in range(n):
        for j in range(n):
            if j != i:
                out[i] += pair_1pn_terms(i, j, pos, vel, gm, a_newt, c, beta, gamma)
    return out


def daily_grid(t0: float, years: float) -> np.ndarray:
    return t0 + np.arange(int(round(years * JULIAN_YEAR_DAYS)) + 1, dtype=np.float64)


def dop853(state: SystemState, accel, t_eval: np.ndarray) -> Trajectory:
    return integrate_scipy(state, accel, t_eval, method="DOP853", rtol=RTOL, atol=ATOL)


def rebound_gr_full(state: SystemState, c: float):
    """REBOUND simulation (G = 1, masses = gm, IAS15) with the REBOUNDx ``gr_full`` force."""
    rebound = pytest.importorskip("rebound")
    reboundx = pytest.importorskip("reboundx")
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.t = state.t
    sim.integrator = "ias15"
    for i in range(state.n):
        x, y, z = (float(q) for q in state.pos[i])
        vx, vy, vz = (float(q) for q in state.vel[i])
        sim.add(m=float(state.gm[i]), x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    rebx = reboundx.Extras(sim)
    gr_full = rebx.load_force("gr_full")
    rebx.add_force(gr_full)
    gr_full.params["c"] = float(c)
    return sim, rebx, gr_full  # rebx and the force must outlive the simulation


def rebound_positions(sim) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in sim.particles])


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261014)


@pytest.fixture(scope="module")
def j2000() -> SystemState:
    return load_j2000()


@pytest.fixture(scope="module")
def two_body_state() -> SystemState:
    return load_j2000().subset(("Sun", "Mercury")).to_barycentric()


@pytest.fixture(scope="module")
def two_body_newton_10yr(two_body_state) -> Trajectory:
    return dop853(two_body_state, accel_newton, daily_grid(two_body_state.t, 10.0))


@pytest.fixture(scope="module")
def nbody_100yr(j2000) -> dict[str, Trajectory]:
    """Century runs of Sun + 8 planets: Newton, EIH and Newton + pn1 (slow tests only)."""
    t_eval = daily_grid(j2000.t, 100.0)
    sun = j2000.index("Sun")
    return {
        "newton": dop853(j2000, accel_newton, t_eval),
        "eih": dop853(j2000, make_eih(c=C_AU_DAY), t_eval),
        "pn1": dop853(j2000, make_newton_pn1(c=C_AU_DAY, central=sun), t_eval),
    }


# --------------------------------------------------------------------------- formula
def test_exported_from_forces_package():
    import gravsim.forces as forces

    assert "make_eih" in forces.__all__
    assert forces.make_eih is make_eih


@pytest.mark.parametrize("n,n_test", [(2, 0), (4, 1), (7, 2)])
def test_matches_hand_written_scalar(rng, n, n_test):
    """Vectorised law vs a plain-Python triple loop over the same formula (c lowered)."""
    pos, vel, gm = random_system(rng, n, n_test)
    gm[0] = GM_SUN
    for c, beta, gamma in [(30.0, 1.0, 1.0), (10.0, 0.7, 1.4), (1.0, 2.0, 0.0)]:
        a = make_eih(c=c, beta=beta, gamma=gamma)(pos, vel, gm, 0.0)
        ref = eih_scalar(pos, vel, gm, c, beta, gamma)
        assert a.shape == (n, 3) and a.dtype == np.float64 and a.flags.c_contiguous
        np.testing.assert_allclose(a, ref, rtol=1e-13, atol=1e-13 * np.abs(ref).max())


def test_matches_hand_written_scalar_at_j2000(j2000):
    s = j2000
    a = make_eih()(s.pos, s.vel, s.gm, s.t)
    ref = eih_scalar(s.pos, s.vel, s.gm, C_AU_DAY, 1.0, 1.0)
    np.testing.assert_allclose(a, ref, rtol=1e-13, atol=0.0)
    # the GR correction for Mercury is ~1e-7 of its Newtonian acceleration
    a_n = accel_newton(s.pos, s.vel, s.gm, s.t)
    i = s.index("Mercury")
    ratio = np.linalg.norm(a[i] - a_n[i]) / np.linalg.norm(a_n[i])
    assert 3e-8 < ratio < 3e-7


# --------------------------------------------------------------------------- (1) Newton limit
def test_huge_c_reproduces_newton(rng, j2000):
    pos, vel, gm = random_system(rng, 8, n_test=2)
    gm[0] = GM_SUN
    for p, v, g in ((pos, vel, gm), (j2000.pos, j2000.vel, j2000.gm)):
        a_eih = make_eih(c=1e12)(p, v, g, 0.0)
        a_n = accel_newton(p, v, g, 0.0)
        np.testing.assert_allclose(a_eih, a_n, rtol=1e-13, atol=0.0)
        # and at the physical c the two differ (the correction is really there)
        assert np.abs(make_eih()(p, v, g, 0.0) - a_n).max() > 1e-13 * np.abs(a_n).max()


# --------------------------------------------------------------------------- (2) test particles
@pytest.mark.parametrize(
    "beta,gamma", [(1.0, 1.0), (0.3, 1.7), (2.2, 0.4), (-0.5, 1.1), (1.6, 2.3)]
)
def test_test_particle_limit_equals_newton_plus_pn1(rng, beta, gamma):
    """Sun at rest at the origin plus massless particles: EIH == Newton + pn1 exactly.

    See the module docstring for the derivation (every N-body term either reduces to the
    central-mass form or vanishes with ``mu_i = 0``, ``v_0 = 0`` and ``a_0^N = 0``).
    """
    n = 5
    pos, vel, gm = random_system(rng, n, n_test=n - 1)
    pos[0] = 0.0
    vel[0] = 0.0
    gm[0] = GM_SUN
    pos[1:] *= 0.1  # bring the particles to ~1 AU so the 1PN terms are not tiny
    for c in (C_AU_DAY, 1.0, 0.1):
        eih = make_eih(c=c, beta=beta, gamma=gamma)
        pn1 = make_pn1(c=c, central=0, beta=beta, gamma=gamma)
        a_eih = eih(pos, vel, gm, 0.0)
        a_n = accel_newton(pos, vel, gm, 0.0)
        a_pn1 = pn1(pos, vel, gm, 0.0)
        np.testing.assert_allclose(a_eih, a_n + a_pn1, rtol=1e-12, atol=0.0)
        assert np.all(a_eih[0] == 0.0)  # nothing attracts the Sun
        # the correction recovered by subtraction carries the roundoff of the Newtonian term
        # (~1e-16 |a_N|), which at the physical c is ~1e-8 of the correction itself
        corr_rtol = 1e-12 if c <= 1.0 else 1e-8
        np.testing.assert_allclose(
            a_eih - a_n, a_pn1, rtol=corr_rtol, atol=1e-15 * np.abs(a_n).max()
        )
    # the correction is not trivially zero: at c = 0.1 it is a large fraction of Newton
    a_c = make_eih(c=0.1, beta=beta, gamma=gamma)(pos, vel, gm, 0.0)
    assert np.abs(a_c - accel_newton(pos, vel, gm, 0.0)).max() > 1e-3 * np.abs(a_c).max()


def test_test_particle_limit_moving_sun_differs_from_pn1(rng):
    """With the Sun moving the EIH velocity terms are barycentric, pn1's relative: they differ.

    This is the known frame dependence of the 1PN velocity terms (the equations are Lorentz,
    not Galilei, invariant), which is why the symmetry test below replaces a boost test.
    """
    pos, vel, gm = random_system(rng, 3, n_test=2)
    gm[0] = GM_SUN
    pos[1:] *= 0.1
    vel[0] = np.array([0.02, -0.01, 0.005])
    a_eih = make_eih(c=1.0)(pos, vel, gm, 0.0)
    a_ref = accel_newton(pos, vel, gm, 0.0) + make_pn1(c=1.0)(pos, vel, gm, 0.0)
    assert np.abs(a_eih - a_ref).max() > 1e-6 * np.abs(a_ref).max()


# --------------------------------------------------------------------------- (3) symmetry
def test_swapping_identical_bodies_permutes_output(rng):
    pos, vel, gm = random_system(rng, 6, n_test=1)
    gm[0] = GM_SUN
    gm[3] = gm[2]  # two identical bodies
    fn = make_eih(c=20.0, beta=0.9, gamma=1.2)
    a = fn(pos, vel, gm, 0.0)
    swap = np.arange(6)
    swap[[2, 3]] = [3, 2]
    a_swapped = fn(pos[swap], vel[swap], gm[swap], 0.0)
    np.testing.assert_array_equal(a_swapped, a[swap])
    assert np.linalg.norm(a[2] - a[3]) > 0.0  # distinct states, distinct accelerations


def test_general_permutation_equivariance(rng):
    pos, vel, gm = random_system(rng, 7, n_test=2)
    gm[0] = GM_SUN
    fn = make_eih(c=20.0)
    a = fn(pos, vel, gm, 0.0)
    perm = rng.permutation(7)
    a_perm = fn(pos[perm], vel[perm], gm[perm], 0.0)
    np.testing.assert_allclose(a_perm, a[perm], rtol=1e-14, atol=1e-14 * np.abs(a).max())


# --------------------------------------------------------------------------- (4) Mercury vs pn1
def test_mercury_correction_close_to_pn1_and_dominated_by_sun_pair_terms(j2000):
    s = j2000
    sun, merc = s.index("Sun"), s.index("Mercury")
    a_n = accel_newton(s.pos, s.vel, s.gm, s.t)
    corr_eih = make_eih()(s.pos, s.vel, s.gm, s.t) - a_n
    corr_pn1 = make_pn1(central=sun)(s.pos, s.vel, s.gm, s.t)
    diff = corr_eih[merc] - corr_pn1[merc]
    rel = np.linalg.norm(diff) / np.linalg.norm(corr_pn1[merc])
    assert rel < 1e-2, f"EIH-minus-pn1 for Mercury is {rel:.2e} of the 1PN correction"
    assert rel > 1e-5  # it is not zero either: the Sun moves and the planets have potentials

    # Split the difference into the Sun-pair terms pn1 lacks (Sun's barycentric velocity and
    # Newtonian acceleration, planets' potentials at Mercury and at the Sun) and the direct
    # 1PN terms with the planets as sources.
    sun_pair = pair_1pn_terms(merc, sun, s.pos, s.vel, s.gm, a_n, C_AU_DAY, 1.0, 1.0)
    sun_motion_and_potentials = sun_pair - corr_pn1[merc]
    planets_direct = sum(
        pair_1pn_terms(merc, j, s.pos, s.vel, s.gm, a_n, C_AU_DAY, 1.0, 1.0)
        for j in range(s.n)
        if j not in (sun, merc)
    )
    # diff is ~3e-4 of a correction that is itself ~1e-7 of Newton, so it carries ~1e-5 of
    # Newtonian roundoff relative to its own size; the decomposition holds to that level
    np.testing.assert_allclose(sun_motion_and_potentials + planets_direct, diff, rtol=1e-4)
    ratio = np.linalg.norm(sun_motion_and_potentials) / np.linalg.norm(diff)
    remainder = np.linalg.norm(planets_direct) / np.linalg.norm(diff)
    assert remainder < 0.05 and abs(ratio - 1.0) < 0.05, (
        f"Sun-motion + planet-potential terms are {ratio:.4f} of the EIH-minus-pn1 difference; "
        f"direct planetary 1PN terms are {remainder:.2e} of it; "
        f"the difference is {rel:.2e} of the pn1 correction"
    )


# --------------------------------------------------------------------------- (5) REBOUNDx oracle
def test_rebound_gr_full_acceleration_at_t0(j2000):
    rebound = pytest.importorskip("rebound")
    s = j2000
    sim, rebx, gr_full = rebound_gr_full(s, C_AU_DAY)
    rebound.clibrebound.reb_simulation_update_acceleration(ctypes.byref(sim))
    a_rb = np.array([[p.ax, p.ay, p.az] for p in sim.particles])
    a_eih = make_eih()(s.pos, s.vel, s.gm, s.t)
    a_n = accel_newton(s.pos, s.vel, s.gm, s.t)
    per_body = np.linalg.norm(a_rb - a_eih, axis=1) / np.linalg.norm(a_eih, axis=1)
    assert per_body.max() < 1e-10, per_body
    # the agreement is on the GR term, not only on Newton: the corrections themselves agree
    # far better than the correction differs from zero (subtraction roundoff ~1e-7 of it)
    corr_rb, corr_g = a_rb - a_n, a_eih - a_n
    merc = s.index("Mercury")
    assert np.linalg.norm(corr_rb[merc] - corr_g[merc]) < 1e-6 * np.linalg.norm(corr_g[merc])
    assert np.linalg.norm(corr_g[merc]) > 1e-8 * np.linalg.norm(a_n[merc])


def test_rebound_gr_full_two_year_positions(j2000):
    s = j2000
    years = 2.0
    t_eval = s.t + np.linspace(0.0, years * JULIAN_YEAR_DAYS, 74)  # every ~10 days
    sim, rebx, gr_full = rebound_gr_full(s, C_AU_DAY)
    pos_rb = [rebound_positions(sim)]
    for t in t_eval[1:]:
        sim.integrate(float(t))
        pos_rb.append(rebound_positions(sim))
    pos_rb = np.array(pos_rb)
    eih = dop853(s, make_eih(c=C_AU_DAY), t_eval)
    sep_km = np.linalg.norm(eih.pos - pos_rb, axis=2) * AU_KM  # (T, N)
    assert sep_km[0].max() == 0.0
    worst = sep_km.max(axis=0)
    assert worst.max() < 1.0, dict(zip(s.names, worst, strict=True))
    # teeth: Newton drifts from the GR run by hundreds of km for Mercury over the same span
    newton = dop853(s, accel_newton, t_eval)
    drift_km = separation(newton, eih, "Mercury")[-1] * AU_KM
    assert drift_km > 50.0, drift_km
    assert eih.meta["nfev"] > 1000


# --------------------------------------------------------------------------- (6) century
@pytest.mark.slow
def test_nbody_century_gr_rates_vs_newton(nbody_100yr):
    newton, eih = nbody_100yr["newton"], nbody_100yr["eih"]
    for body, expected in GR_RATES.items():
        rate, err = perihelion_rate_difference(eih, newton, body)
        assert rate == pytest.approx(expected, abs=0.10), f"{body}: {rate:.3f} +- {err:.3f}"
        assert err < 0.05, body


@pytest.mark.slow
def test_nbody_century_eih_vs_pn1_mercury(nbody_100yr):
    eih, pn1 = nbody_100yr["eih"], nbody_100yr["pn1"]
    rate, err = perihelion_rate_difference(eih, pn1, "Mercury")
    assert abs(rate) < 0.02, f"EIH minus Newton+pn1 Mercury rate: {rate:.5f} +- {err:.5f}"


# --------------------------------------------------------------------------- (7) PPN scaling
@pytest.mark.parametrize("beta,gamma", PPN_CASES)
def test_ppn_scaling_two_body(two_body_state, two_body_newton_10yr, beta, gamma):
    """Rate difference vs Newton = (2 + 2 gamma - beta)/3 * 42.98 for both laws (10 yr)."""
    state, newton = two_body_state, two_body_newton_10yr
    sun = state.index("Sun")
    expected = (2.0 + 2.0 * gamma - beta) / 3.0 * MERCURY_PRECESSION_GR_ARCSEC_CY
    r, v = state.relative_to("Sun")
    i = state.index("Mercury")
    gm = float(state.gm[sun] + state.gm[i])
    el = state_to_elements(r[i], v[i], gm)
    analytic = float(gr_precession_arcsec_per_century(el["a"], el["e"], gm, beta=beta, gamma=gamma))
    assert analytic == pytest.approx(expected, abs=0.02)
    laws = {
        "eih": make_eih(c=C_AU_DAY, beta=beta, gamma=gamma),
        "pn1": make_newton_pn1(c=C_AU_DAY, central=sun, beta=beta, gamma=gamma),
    }
    for name, law in laws.items():
        run = dop853(state, law, newton.t)
        rate, err = perihelion_rate_difference(run, newton, "Mercury")
        assert rate == pytest.approx(expected, abs=0.15), f"{name}: {rate:.3f} +- {err:.3f}"
        assert err < 0.1, name  # nominal OLS error of the 10-year differenced series


# --------------------------------------------------------------------------- protocol / purity
def test_params_pure_and_ignores_t(rng):
    pos, vel, gm = random_system(rng, 5, n_test=1)
    gm[0] = GM_SUN
    pos0, vel0, gm0 = pos.copy(), vel.copy(), gm.copy()
    fn = make_eih(c=40.0, beta=1.1, gamma=0.9)
    a1 = fn(pos, vel, gm, 0.0)
    a2 = fn(pos, vel, gm, 36525.0)
    a3 = fn(pos, vel, gm)
    np.testing.assert_array_equal(pos, pos0)
    np.testing.assert_array_equal(vel, vel0)
    np.testing.assert_array_equal(gm, gm0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(a1, a3)
    assert not np.shares_memory(a1, pos) and not np.shares_memory(a1, vel)
    assert fn.params == {"law": "eih", "c": 40.0, "beta": 1.1, "gamma": 0.9}
    assert fn.__name__ == "accel_eih"
    assert make_eih().params["c"] == C_AU_DAY


def test_correction_scales_as_inverse_c_squared(rng):
    pos, vel, gm = random_system(rng, 4)
    gm[0] = GM_SUN
    a_n = accel_newton(pos, vel, gm, 0.0)
    corr_1 = make_eih(c=1.0)(pos, vel, gm, 0.0) - a_n
    corr_10 = make_eih(c=10.0)(pos, vel, gm, 0.0) - a_n
    # the corrections are recovered by subtraction, so the smallest components carry
    # roundoff of the Newtonian term: compare relative to the largest component
    np.testing.assert_allclose(
        corr_1, 100.0 * corr_10, rtol=1e-9, atol=1e-12 * np.abs(corr_1).max()
    )


def test_test_particle_feels_but_does_not_exert(rng):
    pos, vel, gm = random_system(rng, 5, n_test=1)
    gm[0] = GM_SUN
    fn = make_eih(c=20.0)
    a = fn(pos, vel, gm, 0.0)
    a_wo = fn(pos[:-1], vel[:-1], gm[:-1], 0.0)
    np.testing.assert_allclose(a[:-1], a_wo, rtol=1e-15, atol=0.0)
    assert np.linalg.norm(a[-1]) > 0.0


def test_rejects_bad_c_and_bad_shapes(rng):
    with pytest.raises(ValueError):
        make_eih(c=0.0)
    with pytest.raises(ValueError):
        make_eih(c=-1.0)
    pos, vel, gm = random_system(rng, 4)
    fn = make_eih()
    with pytest.raises(ValueError, match="gm must have shape"):
        fn(pos, vel, gm[:-1], 0.0)
    with pytest.raises(ValueError, match="vel shape"):
        fn(pos, vel[:-1], gm, 0.0)
    with pytest.raises(ValueError, match="pos must have shape"):
        fn(pos[:, :2], vel[:, :2], gm, 0.0)


def test_single_body_and_empty_system():
    fn = make_eih()
    one = fn(np.zeros((1, 3)), np.ones((1, 3)), np.array([GM_SUN]), 0.0)
    assert one.shape == (1, 3) and np.all(one == 0.0)
    none = fn(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), 0.0)
    assert none.shape == (0, 3)


def test_accepts_non_float_non_contiguous_inputs(rng):
    pos_i = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=np.int64)
    vel = rng.normal(0.0, 0.01, size=(3, 3))
    gm = np.array([GM_SUN, 1e-6, 0.0])
    big = np.zeros((3, 6))
    big[:, ::2] = vel
    fn = make_eih(c=10.0)
    a_ref = fn(pos_i.astype(float), vel, gm, 0.0)
    np.testing.assert_allclose(fn(pos_i, big[:, ::2], gm, 0.0), a_ref, rtol=1e-15)
