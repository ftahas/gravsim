"""Validation targets (ROADMAP section 4) as tests: Newton vs General Relativity, end to end.

Each test reproduces one row of ``scripts/validate.py`` with the package's own force laws,
integrators and measurement toolkit, never touching the network (initial conditions and the
Horizons reference series come from the cached files in ``gravsim.data``).

Physics
-------
* Noise floor: a Newtonian two-body orbit has a fixed perihelion, so any fitted drift of the
  longitude of perihelion ``varpi`` (direction of the Laplace-Runge-Lenz vector) is numerical.
  DOP853 at rtol 1e-12 gives ~1e-3 arcsec/century, far below the 0.5 target; the fixed-step
  Leapfrog and Yoshida4 at dt = 0.5 d give thousands of arcsec/century (spurious precession of
  order ``(2 pi dt / P)^order`` per orbit, P = 88 d), which is why they are not used for GR.
* 1PN advance: ``dvarpi = 6 pi GM / (c^2 a (1 - e^2))`` per orbit = 42.98 arcsec/century for
  Mercury.  It is measured as the secular rate of the *difference* of the ``varpi`` series of a
  Newton + 1PN run and a Newton run on the same grid, which cancels the periodic terms and the
  shared integrator error (``analysis.compare.perihelion_rate_difference``).
* Exact Schwarzschild geodesic (Binet equation) vs 1PN: the 2PN remainder is ~1e-15 relative
  for Mercury, so they agree to far better than 0.01 arcsec/century.
* Exaggerated c: with ``c -> c/100`` the advance grows 1e4-fold to 0.29 deg/orbit while
  ``v/c ~ 0.013`` keeps the 1PN expansion valid; the fit matches the formula to < 1 %.
* N-body: Sun + 8 planets.  Planetary perturbations advance Mercury's perihelion by
  531.6 arcsec/century (Clemence 1947); with 1PN the total is 574.6; the 1PN-minus-Newton
  differences are 42.98 (Mercury), 8.62 (Venus), 3.84 (EMB), 1.35 (Mars).  Absolute rates
  need a century because the OLS slope of the osculating series is biased by long-period
  planetary terms over short spans; the differences are clean already after ten years.
* Horizons: the DE441-based reference includes GR, so the Newton run drifts from it by
  thousands of km per century while the 1PN run stays within a few hundred km.
* Energy: DOP853 is not symplectic but at rtol 1e-12 its drift is ~1e-13 per century;
  Yoshida4 at dt = 0.5 d has a bounded error ~2e-9.
* Light deflection at the solar limb: ``4 GM / (c^2 R_sun)`` = 1.751 arcsec.
* EIH N-body (Phase 5): with every body relativistic (``forces.eih``) the run-vs-run
  differences against Newton are the same 42.98 / 8.62 / 3.84 / 1.35, the EIH-minus-pn1
  secular difference for Mercury is below 0.02 arcsec/century (measured -2e-5: the Sun's
  barycentric motion and the planets' potentials only add periodic terms), and REBOUNDx
  ``gr_full`` (IAS15) reproduces the EIH positions to < 10 km over ten years (measured 0.04 km).
* PPN scaling (Phase 5): Sun + Mercury under ``make_eih(beta, gamma)`` precesses at
  ``(2 + 2 gamma - beta)/3`` times the GR rate to 1 percent (measured 3e-4 over ten years).
* Light (Phase 6): the exact null geodesic grazing the Sun is bent by 1.751 arcsec (+- 0.001),
  Soldner's corpuscle by half; the one-way Earth-Mars Shapiro delay at the limb is
  ``(1 + gamma) GM/c^3 ln[(r_e + r_r + R)/(r_e + r_r - R)]`` = 124 microseconds and the exact
  null-geodesic transit agrees with it to 1 percent (measured 1e-4).

* Competing effects (Phase 8): Sun + Mercury with the solar quadrupole (``forces.extras.make_j2``,
  ``J2 = 2.2e-7``) advances the perihelion by +0.028 +- 0.005 arcsec/century relative to Newton
  (measured 0.0278 already over ten years, 0.0278 over the century); with ``J2 x 1000`` the
  node and argument rates in the solar-equator frame match Kaula's first-order formulas to 3 %
  (measured 1e-4).  Lense-Thirring frame dragging with the solar spin ``x 1e5`` gives the node
  ``2 GJ/(c^2 a^3 (1-e^2)^1.5)`` and the argument ``-6 GJ cos i/(...)`` to 3 % (1e-4); at the
  real size the difference against Newton over a century is -0.0020 arcsec/century (slow).
* The Moon (Phase 8): Sun + planets + Earth + Moon on the daily grid of the cached Horizons
  Moon series: sidereal month 27.32 +- 0.05 d, perigee +40.7 +- 1.5 and node -19.3 +- 1.0 deg/yr
  over ten years (+-3 / +-2 over two: the long-period lunar inequalities bias short spans, and
  the Horizons series fitted over the same span shows the same bias to 0.3 deg/yr); the Moon
  within 500 km of Horizons over two years (38 km with 1PN, 157 km Newton); the EMB point mass
  drifts from the real Earth-Moon barycentre by ``2 eps n a t`` ~ 110 km/yr (reported); Mercury's
  GR-minus-Newton rate with the Moon present equals the EMB value (1e-7 over ten years, held to
  0.1) while its Newtonian rate shifts by -0.0025 arcsec/century.

Fast tests use ten-year (or 50-orbit) runs and finish in ~30 s; the century-long versions are
marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gravsim.analysis import (
    element_series,
    fit_secular_rate,
    gr_precession_arcsec_per_century,
    inclination_to_axis,
    perihelion_longitude_series,
    perihelion_passages,
    perihelion_rate_difference,
    rate_arcsec_per_century,
    relative_energy_error,
    separation,
    state_to_elements,
    wrap_angle,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.constants import (
    ARCSEC_PER_RAD,
    AU_KM,
    C_AU_DAY,
    DAY_S,
    DEG_PER_RAD,
    GJ_SUN,
    GM_SUN,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
    LIGHT_DEFLECTION_LIMB_ARCSEC,
    MERCURY_PRECESSION_GR_ARCSEC_CY,
    MERCURY_PRECESSION_J2_ARCSEC_CY,
    MERCURY_PRECESSION_LT_ARCSEC_CY,
    MERCURY_PRECESSION_NEWTON_ARCSEC_CY,
    MERCURY_PRECESSION_OBSERVED_ARCSEC_CY,
    MOON_NODE_RATE_DEG_YR,
    MOON_PERIGEE_RATE_DEG_YR,
    R_SUN_AU,
    SUN_J2,
    SUN_POLE_ECLIPTIC,
)
from gravsim.data.horizons import load_j2000, load_moon_reference_series, load_reference_series
from gravsim.forces import (
    accel_newton,
    combine,
    make_eih,
    make_j2,
    make_lense_thirring,
    make_newton_pn1,
)
from gravsim.integrators import integrate, integrate_scipy
from gravsim.metric import null
from gravsim.metric.schwarzschild import precession_per_orbit_1pn, precession_per_orbit_numeric

RTOL, ATOL = 1e-12, 1e-15
GR_RATES = {"Mercury": MERCURY_PRECESSION_GR_ARCSEC_CY, "Venus": 8.62, "EMB": 3.84, "Mars": 1.35}
NBODY_PN1_MERCURY = 574.6
FAST_YEARS = 10.0
SLOW_YEARS = 100.0
FIXED_DT = 0.5
PPN_CASES = [(0.0, 1.0), (1.0, 0.0), (2.0, 1.0)]
EARTH_AU, MARS_AU = 1.0, 1.5237  # mean distances for the Shapiro geometry
MICROSECOND_DAYS = 1e-6 / DAY_S
J2_AMPLIFICATION, LT_AMPLIFICATION = 1000.0, 1e5  # Phase 8 amplified two-body runs
SIDEREAL_MONTH_DAYS = 27.321662
DEG_PER_YEAR = DEG_PER_RAD * JULIAN_YEAR_DAYS


# ----------------------------------------------------------------------------- helpers
def daily_grid(t0: float, years: float) -> np.ndarray:
    return t0 + np.arange(int(round(years * JULIAN_YEAR_DAYS)) + 1, dtype=np.float64)


def dop853(state: SystemState, accel, t_eval: np.ndarray) -> Trajectory:
    return integrate_scipy(state, accel, t_eval, method="DOP853", rtol=RTOL, atol=ATOL)


def dense_rate(traj: Trajectory, body: str = "Mercury") -> tuple[float, float]:
    """OLS secular rate of the osculating varpi series and its error [arcsec/century]."""
    t, varpi = perihelion_longitude_series(traj, body)
    fit = fit_secular_rate(t, varpi)
    return float(fit.rate_arcsec_per_century), float(fit.rate_err_arcsec_per_century)


def passage_rate(traj: Trajectory, body: str = "Mercury") -> float:
    """Secular rate of varpi sampled at the perihelion passages [arcsec/century]."""
    t_p, varpi_p = perihelion_passages(traj, body)
    return float(fit_secular_rate(t_p, varpi_p).rate_arcsec_per_century)


def mercury_elements(state: SystemState) -> dict[str, float]:
    r, v = state.relative_to("Sun")
    i = state.index("Mercury")
    gm = float(state.gm[state.index("Sun")] + state.gm[i])
    el = state_to_elements(r[i], v[i], gm)
    return {"a": float(el["a"]), "e": float(el["e"]), "period": float(el["period"]), "gm": gm}


def two_body_runs(years: float) -> tuple[Trajectory, Trajectory]:
    state = load_j2000().subset(("Sun", "Mercury")).to_barycentric()
    t_eval = daily_grid(state.t, years)
    newton = dop853(state, accel_newton, t_eval)
    pn1 = dop853(state, make_newton_pn1(c=C_AU_DAY, central=state.index("Sun")), t_eval)
    return newton, pn1


def nbody_runs(years: float) -> tuple[Trajectory, Trajectory]:
    state = load_j2000()
    t_eval = daily_grid(state.t, years)
    newton = dop853(state, accel_newton, t_eval)
    pn1 = dop853(state, make_newton_pn1(c=C_AU_DAY, central=state.index("Sun")), t_eval)
    return newton, pn1


def rebound_gr_full_positions(state: SystemState, t_eval: np.ndarray) -> np.ndarray:
    """Positions (T, N, 3) from REBOUND (G = 1, IAS15) with the REBOUNDx ``gr_full`` force."""
    rebound = pytest.importorskip("rebound")
    reboundx = pytest.importorskip("reboundx")
    sim = rebound.Simulation()
    sim.G = 1.0
    sim.t = float(state.t)
    sim.integrator = "ias15"
    for i in range(state.n):
        x, y, z = (float(q) for q in state.pos[i])
        vx, vy, vz = (float(q) for q in state.vel[i])
        sim.add(m=float(state.gm[i]), x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)
    rebx = reboundx.Extras(sim)
    gr_full = rebx.load_force("gr_full")
    rebx.add_force(gr_full)
    gr_full.params["c"] = float(C_AU_DAY)
    out = [np.array([[p.x, p.y, p.z] for p in sim.particles])]
    for t in t_eval[1:]:
        sim.integrate(float(t))
        out.append(np.array([[p.x, p.y, p.z] for p in sim.particles]))
    del rebx, gr_full  # must outlive the integration
    return np.array(out)


def horizons_separations_km(
    newton: Trajectory, pn1: Trajectory, ref: Trajectory
) -> tuple[np.ndarray, np.ndarray]:
    """Mercury's |r_run - r_Horizons| at the reference epochs inside the runs' span [km]."""
    mask = ref.t <= newton.t[-1] + 1e-9
    ref_sub = Trajectory(ref.t[mask], ref.pos[mask], ref.vel[mask], ref.gm, ref.names)
    idx = np.searchsorted(newton.t, ref_sub.t)
    assert np.allclose(newton.t[idx], ref_sub.t, rtol=0.0, atol=1e-9)

    def sub(tr: Trajectory) -> Trajectory:
        return Trajectory(tr.t[idx], tr.pos[idx], tr.vel[idx], tr.gm, tr.names)

    sep_n = separation(sub(newton), ref_sub, "Mercury", t_atol=1e-9) * AU_KM
    sep_g = separation(sub(pn1), ref_sub, "Mercury", t_atol=1e-9) * AU_KM
    return sep_n, sep_g


def equator_frame(pole=SUN_POLE_ECLIPTIC) -> np.ndarray:
    """Rotation (rows x, y, z) into the frame whose z axis is the solar pole; r_eq = R @ r."""
    z = np.asarray(pole, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def equatorial_rate_differences(traj: Trajectory, newton: Trajectory) -> dict[str, float]:
    """Fitted rates [rad/day] of the differenced node and argument of Mercury, solar equator."""
    rot = equator_frame()
    out = {}
    for key in ("raan", "argp"):
        series = []
        for tr in (traj, newton):
            tr_eq = Trajectory(tr.t, tr.pos @ rot.T, tr.vel @ rot.T, tr.gm, tr.names)
            series.append(element_series(tr_eq, "Mercury", key)[1])
        out[key] = fit_secular_rate(traj.t, wrap_angle(series[0] - series[1])).rate
    return out


def mercury_equator_inclination(state: SystemState) -> float:
    r, v = state.relative_to("Sun")
    i = state.index("Mercury")
    return float(inclination_to_axis(r[i], v[i], SUN_POLE_ECLIPTIC))


def lunar_fits(traj: Trajectory) -> dict[str, float]:
    """Sidereal month [d] and perigee/node rates [deg/yr] from the geocentric elements."""
    r, v = traj.relative("Moon", "Earth")
    gm = float(traj.gm[traj.index("Earth")] + traj.gm[traj.index("Moon")])
    el = state_to_elements(r, v, gm)
    n_sidereal = fit_secular_rate(traj.t, el["varpi"] + el["M"]).rate
    return {
        "month": 2.0 * np.pi / n_sidereal,
        "perigee": fit_secular_rate(traj.t, el["varpi"]).rate * DEG_PER_YEAR,
        "node": fit_secular_rate(traj.t, el["raan"]).rate * DEG_PER_YEAR,
    }


def earth_moon_barycentre(traj: Trajectory) -> np.ndarray:
    i_e, i_m = traj.index("Earth"), traj.index("Moon")
    w = traj.gm[[i_e, i_m]] / traj.gm[[i_e, i_m]].sum()
    return w[0] * traj.pos[:, i_e] + w[1] * traj.pos[:, i_m]


def moon_runs(years: float) -> dict[str, Trajectory]:
    """Split (Earth + Moon) and EMB systems, Newton and Newton + 1PN, on the Horizons grid."""
    ref = load_moon_reference_series()
    n = int(round(years * JULIAN_YEAR_DAYS))
    t_eval = ref.t[: n + 1]
    split, emb = load_j2000(with_moon=True), load_j2000()
    gr_law = make_newton_pn1(c=C_AU_DAY, central=split.index("Sun"))
    return {
        "horizons": Trajectory(t_eval, ref.pos[: n + 1], ref.vel[: n + 1], ref.gm, ref.names),
        "split_newton": dop853(split, accel_newton, t_eval),
        "split_gr": dop853(split, gr_law, t_eval),
        "emb_newton": dop853(emb, accel_newton, t_eval),
        "emb_gr": dop853(emb, gr_law, t_eval),
    }


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def two_body_state() -> SystemState:
    return load_j2000().subset(("Sun", "Mercury")).to_barycentric()


@pytest.fixture(scope="module")
def two_body_10yr() -> tuple[Trajectory, Trajectory]:
    return two_body_runs(FAST_YEARS)


@pytest.fixture(scope="module")
def two_body_100yr() -> tuple[Trajectory, Trajectory]:
    return two_body_runs(SLOW_YEARS)


@pytest.fixture(scope="module")
def nbody_10yr() -> tuple[Trajectory, Trajectory]:
    return nbody_runs(FAST_YEARS)


@pytest.fixture(scope="module")
def nbody_100yr() -> tuple[Trajectory, Trajectory]:
    return nbody_runs(SLOW_YEARS)


@pytest.fixture(scope="module")
def nbody_eih_10yr() -> Trajectory:
    state = load_j2000()
    return dop853(state, make_eih(c=C_AU_DAY), daily_grid(state.t, FAST_YEARS))


@pytest.fixture(scope="module")
def nbody_eih_100yr() -> Trajectory:
    state = load_j2000()
    return dop853(state, make_eih(c=C_AU_DAY), daily_grid(state.t, SLOW_YEARS))


@pytest.fixture(scope="module")
def horizons() -> Trajectory:
    return load_reference_series()


@pytest.fixture(scope="module")
def moon_2yr() -> dict[str, Trajectory]:
    return moon_runs(2.0)


@pytest.fixture(scope="module")
def moon_10yr() -> dict[str, Trajectory]:
    return moon_runs(10.0)


# ----------------------------------------------------------------------------- check 1
def test_two_body_state_is_barycentric_sun_mercury(two_body_state):
    assert two_body_state.names == ("Sun", "Mercury")
    r0, v0 = two_body_state.barycenter()
    assert np.abs(r0).max() < 1e-16 and np.abs(v0).max() < 1e-18  # roundoff of ~0.3 AU
    el = mercury_elements(two_body_state)
    assert el["a"] == pytest.approx(0.3871, abs=5e-4)
    assert el["e"] == pytest.approx(0.2056, abs=5e-4)
    assert el["period"] == pytest.approx(87.97, abs=0.02)


def test_two_body_newton_noise_floor_10yr(two_body_10yr):
    newton, _ = two_body_10yr
    rate, err = dense_rate(newton)
    assert abs(rate) < 0.5, f"spurious precession {rate} arcsec/cy"
    assert abs(rate) < 0.01  # measured ~1.2e-3 arcsec/cy: three orders below the target
    assert abs(passage_rate(newton)) < 0.5


def test_two_body_fixed_step_spurious_precession(two_body_state):
    """Leapfrog and Yoshida4 at dt = 0.5 d precess spuriously by >> 43 arcsec/century."""
    t_end = two_body_state.t + FAST_YEARS * JULIAN_YEAR_DAYS
    rates = {}
    for name in ("leapfrog", "yoshida4"):
        traj = integrate(two_body_state, accel_newton, t_end, FIXED_DT, name, save_every=2)
        assert traj.meta["dt"] == pytest.approx(FIXED_DT, rel=1e-3)  # driver rounds n_steps
        rates[name], _ = dense_rate(traj)
    assert abs(rates["leapfrog"]) > 1e4  # measured ~ -2e5 arcsec/cy
    assert abs(rates["yoshida4"]) > 1e2  # measured ~ 2.6e3 arcsec/cy
    assert abs(rates["yoshida4"]) < abs(rates["leapfrog"]) / 10  # higher order helps
    # Yoshida4's spurious rate scales as dt^4: halving the step divides it by ~16.
    traj = integrate(two_body_state, accel_newton, t_end, FIXED_DT / 2, "yoshida4", save_every=4)
    half, _ = dense_rate(traj)
    assert 8 < rates["yoshida4"] / half < 32


# ----------------------------------------------------------------------------- check 2
def test_two_body_pn1_advance_10yr(two_body_10yr, two_body_state):
    newton, pn1 = two_body_10yr
    rate, err = perihelion_rate_difference(pn1, newton, "Mercury")
    assert rate == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.10)
    assert err < 0.05
    el = mercury_elements(two_body_state)
    analytic = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"]))
    assert analytic == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.01)
    assert rate == pytest.approx(analytic, abs=0.10)
    absolute, _ = dense_rate(pn1)
    assert absolute == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.10)


# ----------------------------------------------------------------------------- check 3
def test_schwarzschild_exact_vs_1pn_mercury(two_body_state):
    el = mercury_elements(two_body_state)
    exact = precession_per_orbit_numeric(el["a"], el["e"], el["gm"], C_AU_DAY, n_orbits=50)
    pn1 = precession_per_orbit_1pn(el["a"], el["e"], el["gm"], C_AU_DAY)
    to_cy = ARCSEC_PER_RAD * JULIAN_CENTURY_DAYS / el["period"]
    assert pn1 * to_cy == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.01)
    assert abs(exact - pn1) * to_cy < 0.01  # measured ~1e-4 arcsec/cy


# ----------------------------------------------------------------------------- check 4
def test_exaggerated_c_rosette_matches_formula(two_body_state):
    """c/100: the advance is 1e4 times larger and still matches 1PN to 1 percent."""
    el = mercury_elements(two_body_state)
    c = C_AU_DAY / 100.0
    t_end = 50 * el["period"]
    t_eval = two_body_state.t + np.linspace(0.0, t_end, int(round(t_end / 0.25)) + 1)
    sun = two_body_state.index("Sun")
    pn1 = dop853(two_body_state, make_newton_pn1(c=c, central=sun), t_eval)
    newton = dop853(two_body_state, accel_newton, t_eval)
    predicted = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"], c=c))
    assert predicted / MERCURY_PRECESSION_GR_ARCSEC_CY == pytest.approx(1e4, rel=1e-3)
    v = np.linalg.norm(two_body_state.relative_to("Sun")[1][two_body_state.index("Mercury")])
    assert v / c < 0.02  # 1PN still valid
    fitted, _ = dense_rate(pn1)
    differenced, _ = perihelion_rate_difference(pn1, newton, "Mercury")
    assert fitted == pytest.approx(predicted, rel=0.01)
    assert differenced == pytest.approx(predicted, rel=0.01)
    assert passage_rate(pn1) == pytest.approx(predicted, rel=0.01)
    # the perihelion really turns: ~0.29 deg per orbit, 14 deg over the run
    _, varpi = perihelion_longitude_series(pn1, "Mercury")
    total_deg = np.degrees(np.unwrap(varpi)[-1] - varpi[0])
    assert total_deg == pytest.approx(0.2875 * 50, rel=0.02)


# ----------------------------------------------------------------------------- checks 5-6
def test_nbody_10yr_planetary_perturbations_present(nbody_10yr):
    """Ten years is too short for the +-3 target (long-period terms), but the order is right."""
    newton, _ = nbody_10yr
    rate, _ = dense_rate(newton)
    assert abs(rate - MERCURY_PRECESSION_NEWTON_ARCSEC_CY) < 100.0  # measured ~569


def test_nbody_10yr_pn1_minus_newton(nbody_10yr):
    newton, pn1 = nbody_10yr
    rate, err = perihelion_rate_difference(pn1, newton, "Mercury")
    assert rate == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.10)
    assert err < 0.05
    for body in ("Venus", "EMB", "Mars"):
        rate, _ = perihelion_rate_difference(pn1, newton, body)
        # periodic terms of these slower orbits need the century run for +-0.1
        assert rate == pytest.approx(GR_RATES[body], abs=1.0)


# ----------------------------------------------------------------------------- check 7
def test_horizons_reference_mercury_rate(horizons):
    assert horizons.names == load_j2000().names
    assert horizons.t[0] == 0.0 and horizons.t[-1] >= 36500.0
    assert np.allclose(np.diff(horizons.t), 20.0)
    rate, err = dense_rate(horizons)
    assert rate == pytest.approx(MERCURY_PRECESSION_OBSERVED_ARCSEC_CY, abs=5.0)
    assert passage_rate(horizons) == pytest.approx(MERCURY_PRECESSION_OBSERVED_ARCSEC_CY, abs=5.0)


def test_horizons_separation_10yr_pn1_closer_than_newton(nbody_10yr, horizons):
    newton, pn1 = nbody_10yr
    sep_n, sep_g = horizons_separations_km(newton, pn1, horizons)
    assert sep_n[0] == 0.0 and sep_g[0] == 0.0
    assert sep_g[-1] < sep_n[-1]
    assert sep_g.mean() < sep_n.mean()
    assert sep_n[-1] > 500.0  # Newton drifts (measured ~1500 km after 10 yr)
    assert sep_g[-1] < 50.0  # 1PN stays close (measured ~5 km)


# ----------------------------------------------------------------------------- check 8
def test_energy_error_10yr(nbody_10yr):
    newton, _ = nbody_10yr
    assert np.abs(relative_energy_error(newton)).max() < 1e-9  # measured ~6e-14
    state = load_j2000()
    t_end = state.t + 10.0 * JULIAN_YEAR_DAYS
    yoshida = integrate(state, accel_newton, t_end, FIXED_DT, "yoshida4", save_every=2)
    assert yoshida.meta["dt"] == pytest.approx(FIXED_DT, rel=1e-3)
    assert np.abs(relative_energy_error(yoshida)).max() < 1e-8  # measured ~2.2e-9


# ----------------------------------------------------------------------------- check 9
def test_light_deflection_constant():
    delta = 4.0 * GM_SUN / (C_AU_DAY**2 * R_SUN_AU) * ARCSEC_PER_RAD
    assert delta == pytest.approx(LIGHT_DEFLECTION_LIMB_ARCSEC, abs=0.001)


# ----------------------------------------------------------------------------- checks 10-12
def test_nbody_10yr_eih_minus_newton(nbody_10yr, nbody_eih_10yr):
    newton, _ = nbody_10yr
    eih = nbody_eih_10yr
    np.testing.assert_array_equal(eih.t, newton.t)
    assert eih.meta["method"] == "DOP853"
    rate, err = perihelion_rate_difference(eih, newton, "Mercury")
    assert rate == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.10)  # measured 42.986
    assert err < 0.05
    for body in ("Venus", "EMB", "Mars"):
        rate, _ = perihelion_rate_difference(eih, newton, body)
        assert rate == pytest.approx(GR_RATES[body], abs=1.0), body  # +-0.1 needs the century


def test_nbody_10yr_eih_minus_pn1_mercury(nbody_10yr, nbody_eih_10yr):
    """The Sun-only 1PN law already carries the secular term; EIH adds only periodic km."""
    _, pn1 = nbody_10yr
    eih = nbody_eih_10yr
    rate, err = perihelion_rate_difference(eih, pn1, "Mercury")
    assert abs(rate) < 0.02, f"EIH minus pn1: {rate:.2e} +- {err:.2e} arcsec/cy"  # ~ -2e-5
    sep_km = separation(eih, pn1, "Mercury") * AU_KM
    assert 0.1 < sep_km.max() < 5.0  # measured 0.5 km after ten years
    assert separation(eih, pn1, "Sun").max() * AU_KM < 0.1  # 0.03 km


def test_rebound_gr_full_oracle_10yr(nbody_10yr, nbody_eih_10yr):
    """REBOUNDx gr_full (same EIH equations, IAS15) reproduces the EIH run to < 10 km."""
    state = load_j2000()
    eih = nbody_eih_10yr
    idx = np.arange(0, eih.t.size, 10)  # every ten days
    pos_rb = rebound_gr_full_positions(state, eih.t[idx])
    sep_km = np.linalg.norm(eih.pos[idx] - pos_rb, axis=2) * AU_KM  # (T, N)
    assert sep_km[0].max() == 0.0
    worst = sep_km.max(axis=0)
    assert worst.max() < 10.0, dict(zip(state.names, worst, strict=True))
    assert worst[state.index("Mercury")] < 1.0  # measured 0.035 km
    # teeth: Newton drifts from both GR runs by hundreds of km over the same span
    newton, _ = nbody_10yr
    assert separation(newton, eih, "Mercury")[-1] * AU_KM > 500.0


# ----------------------------------------------------------------------------- check 13
@pytest.mark.parametrize("beta,gamma", PPN_CASES)
def test_ppn_two_body_scaling_10yr(two_body_10yr, two_body_state, beta, gamma):
    """Sun + Mercury under make_eih(beta, gamma): rate = (2 + 2 gamma - beta)/3 x 42.98."""
    newton, _ = two_body_10yr
    run = dop853(two_body_state, make_eih(c=C_AU_DAY, beta=beta, gamma=gamma), newton.t)
    rate, err = perihelion_rate_difference(run, newton, "Mercury")
    factor = (2.0 + 2.0 * gamma - beta) / 3.0
    predicted = factor * MERCURY_PRECESSION_GR_ARCSEC_CY
    assert rate / predicted == pytest.approx(1.0, abs=0.01)  # measured 1.0001 - 1.0004
    assert err < 0.1
    el = mercury_elements(two_body_state)
    analytic = gr_precession_arcsec_per_century(
        el["a"], el["e"], el["gm"], beta=beta, gamma=gamma
    )
    assert float(analytic) == pytest.approx(predicted, abs=0.02)


# ----------------------------------------------------------------------------- checks 14-15
def test_light_deflection_numeric_limb_and_soldner():
    mu = GM_SUN / C_AU_DAY**2
    exact = null.deflection_angle(R_SUN_AU, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD
    weak = null.deflection_angle_weak(R_SUN_AU, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD
    soldner = null.deflection_newtonian_corpuscle(R_SUN_AU, GM_SUN, C_AU_DAY) * ARCSEC_PER_RAD
    assert exact == pytest.approx(LIGHT_DEFLECTION_LIMB_ARCSEC, abs=0.001)  # 1.7512
    assert weak == pytest.approx(LIGHT_DEFLECTION_LIMB_ARCSEC, abs=0.001)
    assert soldner == pytest.approx(0.5 * weak, rel=1e-12)  # 0.8756: half of GR
    assert soldner == pytest.approx(0.875, abs=0.001)
    # the exact geodesic exceeds 4 GM/(c^2 b) by the second-order (15 pi/16) mu/b
    assert exact / weak - 1.0 == pytest.approx(15.0 * np.pi / 16.0 * mu / R_SUN_AU, rel=0.05)


def test_shapiro_delay_earth_mars_conjunction():
    weak = null.shapiro_delay(EARTH_AU, MARS_AU, R_SUN_AU, GM_SUN, C_AU_DAY)
    conj = null.shapiro_delay_conjunction(EARTH_AU, MARS_AU, R_SUN_AU, GM_SUN, C_AU_DAY)
    numeric = null.shapiro_delay_numeric(EARTH_AU, MARS_AU, R_SUN_AU, GM_SUN, C_AU_DAY)
    one_way_us = weak / MICROSECOND_DAYS
    assert one_way_us == pytest.approx(124.0, abs=1.0)  # measured 123.6; Viking round trip 247
    assert 2.0 * one_way_us == pytest.approx(250.0, abs=5.0)
    assert conj == pytest.approx(weak, rel=1e-5, abs=0)
    assert numeric / weak == pytest.approx(1.0, abs=0.01)  # measured 1.0001
    # gamma = 0 halves it; the delay is a 1e-7 fraction of the light time
    half = null.shapiro_delay(EARTH_AU, MARS_AU, R_SUN_AU, GM_SUN, C_AU_DAY, gamma=0.0)
    assert half == pytest.approx(0.5 * weak, rel=1e-12, abs=0)
    assert weak / ((EARTH_AU + MARS_AU) / C_AU_DAY) == pytest.approx(1e-7, rel=0.05)


# ----------------------------------------------------------------------------- checks 16-17
def test_j2_amplified_kaula_rates_10yr(two_body_10yr, two_body_state):
    """J2 x 1000: node and argument rates in the solar-equator frame vs Kaula to 3 percent."""
    newton, _ = two_body_10yr
    j2 = J2_AMPLIFICATION * SUN_J2
    sun = two_body_state.index("Sun")
    run = dop853(two_body_state, combine(accel_newton, make_j2(j2=j2, central=sun)), newton.t)
    rates = equatorial_rate_differences(run, newton)
    el = mercury_elements(two_body_state)
    inc = mercury_equator_inclination(two_body_state)
    assert np.degrees(inc) == pytest.approx(3.38, abs=0.05)
    n_mean = np.sqrt(el["gm"] / el["a"] ** 3)
    k = n_mean * j2 * (R_SUN_AU / (el["a"] * (1.0 - el["e"] ** 2))) ** 2
    pred_raan = -1.5 * k * np.cos(inc)
    pred_argp = 0.75 * k * (5.0 * np.cos(inc) ** 2 - 1.0)
    assert rates["raan"] / pred_raan == pytest.approx(1.0, abs=0.03)  # measured 0.99996
    assert rates["argp"] / pred_argp == pytest.approx(1.0, abs=0.03)  # measured 0.99991
    assert rates["raan"] < 0.0 < rates["argp"]  # the node regresses, the argument advances
    varpi = rate_arcsec_per_century(rates["raan"] + rates["argp"]) / J2_AMPLIFICATION
    assert varpi == pytest.approx(MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.002)


def test_j2_real_size_advance_10yr(two_body_10yr, two_body_state):
    """Real J2 = 2.2e-7 differenced against Newton: +0.028 +- 0.005 arcsec/century."""
    newton, _ = two_body_10yr
    sun = two_body_state.index("Sun")
    run = dop853(two_body_state, combine(accel_newton, make_j2(central=sun)), newton.t)
    rate, err = perihelion_rate_difference(run, newton, "Mercury")
    assert rate == pytest.approx(MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.005)  # measured 0.0278
    assert err < 1e-3


def test_lense_thirring_amplified_rates_10yr(two_body_10yr, two_body_state):
    """Solar spin x 1e5: node and argument rates vs the Lense-Thirring formulas to 3 percent."""
    newton, _ = two_body_10yr
    sun = two_body_state.index("Sun")
    law = combine(accel_newton, make_lense_thirring(central=sun, scale=LT_AMPLIFICATION))
    run = dop853(two_body_state, law, newton.t)
    rates = equatorial_rate_differences(run, newton)
    el = mercury_elements(two_body_state)
    inc = mercury_equator_inclination(two_body_state)
    k = LT_AMPLIFICATION * GJ_SUN / (C_AU_DAY**2 * el["a"] ** 3 * (1.0 - el["e"] ** 2) ** 1.5)
    assert rates["raan"] / (2.0 * k) == pytest.approx(1.0, abs=0.03)  # measured 0.99996
    assert rates["argp"] / (-6.0 * k * np.cos(inc)) == pytest.approx(1.0, abs=0.03)  # 0.99995
    assert rates["argp"] < 0.0 < rates["raan"]  # node dragged forward, argument driven back
    varpi = rate_arcsec_per_century(rates["raan"] + rates["argp"]) / LT_AMPLIFICATION
    assert varpi == pytest.approx(MERCURY_PRECESSION_LT_ARCSEC_CY, abs=0.0005)


# ----------------------------------------------------------------------------- check 18
def test_moon_lunar_orbit_2yr(moon_2yr):
    """Sidereal month, perigee and node rates over two years (short-span tolerances)."""
    fit = lunar_fits(moon_2yr["split_newton"])
    ref = lunar_fits(moon_2yr["horizons"])
    assert fit["month"] == pytest.approx(SIDEREAL_MONTH_DAYS, abs=0.05)  # measured 27.3211
    assert fit["perigee"] == pytest.approx(MOON_PERIGEE_RATE_DEG_YR, abs=3.0)  # 40.24
    assert fit["node"] == pytest.approx(MOON_NODE_RATE_DEG_YR, abs=2.0)  # -19.43
    assert fit["perigee"] == pytest.approx(ref["perigee"], abs=0.3)  # same bias as Horizons
    assert fit["node"] == pytest.approx(ref["node"], abs=0.3)


def test_moon_vs_horizons_and_emb_drift_2yr(moon_2yr):
    sep_newton = separation(moon_2yr["split_newton"], moon_2yr["horizons"], "Moon") * AU_KM
    sep_gr = separation(moon_2yr["split_gr"], moon_2yr["horizons"], "Moon") * AU_KM
    assert sep_gr.max() < 500.0  # measured 38 km
    assert sep_newton.max() < 500.0  # measured 157 km (mostly the Earth's own 1PN advance)
    assert sep_gr.max() < sep_newton.max()
    bc = earth_moon_barycentre(moon_2yr["split_gr"])
    vs_horizons = np.linalg.norm(bc - earth_moon_barycentre(moon_2yr["horizons"]), axis=1)
    drift = np.linalg.norm(bc - moon_2yr["emb_gr"].body("EMB")[0], axis=1) * AU_KM
    assert vs_horizons.max() * AU_KM < 100.0  # 0.9 km: the split run follows the ephemeris
    assert 100.0 < drift[-1] < 500.0  # 234 km: the EMB point mass lags the real barycentre
    assert drift[drift.size // 2] == pytest.approx(0.5 * drift[-1], rel=0.2)  # linear growth


def test_mercury_rates_with_the_moon_2yr(moon_2yr):
    rate_split, _ = perihelion_rate_difference(
        moon_2yr["split_gr"], moon_2yr["split_newton"], "Mercury"
    )
    rate_emb, _ = perihelion_rate_difference(moon_2yr["emb_gr"], moon_2yr["emb_newton"], "Mercury")
    shift, _ = perihelion_rate_difference(
        moon_2yr["split_newton"], moon_2yr["emb_newton"], "Mercury"
    )
    assert rate_split == pytest.approx(rate_emb, abs=0.3)  # measured 1e-7 apart
    assert rate_split == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.5)  # 2-yr span
    assert -0.01 < shift < 0.0  # measured -0.0024: the pair slows the Newtonian advance


# ============================================================================= slow
@pytest.mark.slow
def test_j2_and_lense_thirring_real_size_century(two_body_100yr, two_body_state):
    """Sun + Mercury + J2 (real) minus Newton over a century: +0.028 +- 0.005; LT reported."""
    newton, _ = two_body_100yr
    sun = two_body_state.index("Sun")
    j2 = dop853(two_body_state, combine(accel_newton, make_j2(central=sun)), newton.t)
    lt = dop853(two_body_state, combine(accel_newton, make_lense_thirring(central=sun)), newton.t)
    rate_j2, err_j2 = perihelion_rate_difference(j2, newton, "Mercury")
    rate_lt, err_lt = perihelion_rate_difference(lt, newton, "Mercury")
    report = f"J2 {rate_j2:.5f} +- {err_j2:.1e}, LT {rate_lt:.5f} +- {err_lt:.1e} arcsec/cy"
    assert rate_j2 == pytest.approx(MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.005), report  # 0.0278
    assert rate_lt == pytest.approx(MERCURY_PRECESSION_LT_ARCSEC_CY, abs=0.001), report  # -0.0020


@pytest.mark.slow
def test_moon_lunar_orbit_10yr(moon_10yr):
    fit = lunar_fits(moon_10yr["split_newton"])
    ref = lunar_fits(moon_10yr["horizons"])
    report = (
        f"10 yr: month {fit['month']:.4f} d, perigee {fit['perigee']:+.3f}, node {fit['node']:+.3f}"
    )
    assert fit["month"] == pytest.approx(SIDEREAL_MONTH_DAYS, abs=0.05), report  # 27.3217
    assert fit["perigee"] == pytest.approx(MOON_PERIGEE_RATE_DEG_YR, abs=1.5), report  # 40.59
    assert fit["node"] == pytest.approx(MOON_NODE_RATE_DEG_YR, abs=1.0), report  # -19.36
    assert fit["perigee"] == pytest.approx(ref["perigee"], abs=0.3), report
    assert fit["node"] == pytest.approx(ref["node"], abs=0.3), report


@pytest.mark.slow
def test_mercury_gr_rate_with_the_moon_10yr(moon_10yr):
    rate_split, _ = perihelion_rate_difference(
        moon_10yr["split_gr"], moon_10yr["split_newton"], "Mercury"
    )
    rate_emb, _ = perihelion_rate_difference(
        moon_10yr["emb_gr"], moon_10yr["emb_newton"], "Mercury"
    )
    shift, _ = perihelion_rate_difference(
        moon_10yr["split_newton"], moon_10yr["emb_newton"], "Mercury"
    )
    assert rate_split == pytest.approx(rate_emb, abs=0.1)  # measured 6e-8 apart
    assert rate_split == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.1)
    assert shift == pytest.approx(-0.0025, abs=0.001)
    sep_gr = separation(moon_10yr["split_gr"], moon_10yr["horizons"], "Moon") * AU_KM
    assert sep_gr.max() < 500.0  # measured 161 km over ten years


@pytest.mark.slow
def test_two_body_newton_noise_floor_century(two_body_100yr):
    newton, _ = two_body_100yr
    rate, _ = dense_rate(newton)
    assert abs(rate) < 0.5
    assert abs(passage_rate(newton)) < 0.5


@pytest.mark.slow
def test_two_body_pn1_advance_century(two_body_100yr, two_body_state):
    newton, pn1 = two_body_100yr
    rate, err = perihelion_rate_difference(pn1, newton, "Mercury")
    assert rate == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.10)
    assert err < 0.01
    el = mercury_elements(two_body_state)
    analytic = float(gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"]))
    assert rate == pytest.approx(analytic, abs=0.10)
    absolute, _ = dense_rate(pn1)
    assert absolute == pytest.approx(MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.10)


@pytest.mark.slow
def test_nbody_newton_mercury_century(nbody_100yr):
    newton, _ = nbody_100yr
    rate, err = dense_rate(newton)
    assert rate == pytest.approx(MERCURY_PRECESSION_NEWTON_ARCSEC_CY, abs=3.0)
    assert err < 1.0
    assert passage_rate(newton) == pytest.approx(MERCURY_PRECESSION_NEWTON_ARCSEC_CY, abs=3.0)


@pytest.mark.slow
def test_nbody_pn1_mercury_century(nbody_100yr):
    newton, pn1 = nbody_100yr
    rate, _ = dense_rate(pn1)
    assert rate == pytest.approx(NBODY_PN1_MERCURY, abs=3.0)
    assert passage_rate(pn1) == pytest.approx(NBODY_PN1_MERCURY, abs=3.0)
    for body, expected in GR_RATES.items():
        diff, err = perihelion_rate_difference(pn1, newton, body)
        assert diff == pytest.approx(expected, abs=0.10), body
        assert err < 0.05, body


@pytest.mark.slow
def test_horizons_separation_century_pn1_closer_than_newton(nbody_100yr, horizons):
    newton, pn1 = nbody_100yr
    sep_n, sep_g = horizons_separations_km(newton, pn1, horizons)
    assert sep_g[-1] < sep_n[-1]
    assert sep_g.mean() < sep_n.mean()
    assert sep_n[-1] > 2000.0  # measured ~5700 km
    assert sep_g[-1] < 1000.0  # measured ~180 km


@pytest.mark.slow
def test_energy_error_century(nbody_100yr):
    newton, _ = nbody_100yr
    assert np.abs(relative_energy_error(newton)).max() < 1e-9  # measured ~7e-14


@pytest.mark.slow
def test_nbody_eih_century(nbody_100yr, nbody_eih_100yr):
    newton, pn1 = nbody_100yr
    eih = nbody_eih_100yr
    rate, _ = dense_rate(eih)
    assert rate == pytest.approx(NBODY_PN1_MERCURY, abs=3.0)  # same -2.4 bias as check 6
    assert passage_rate(eih) == pytest.approx(NBODY_PN1_MERCURY, abs=3.0)
    for body, expected in GR_RATES.items():
        diff, err = perihelion_rate_difference(eih, newton, body)
        assert diff == pytest.approx(expected, abs=0.10), body
        assert err < 0.05, body
    diff, err = perihelion_rate_difference(eih, pn1, "Mercury")
    assert abs(diff) < 0.02, f"EIH minus pn1: {diff:.2e} +- {err:.2e} arcsec/cy"


@pytest.mark.slow
def test_horizons_separation_century_eih_closer_than_newton(nbody_100yr, nbody_eih_100yr, horizons):
    newton, pn1 = nbody_100yr
    sep_n, sep_e = horizons_separations_km(newton, nbody_eih_100yr, horizons)
    _, sep_g = horizons_separations_km(newton, pn1, horizons)
    assert sep_e[-1] < sep_n[-1] and sep_e.mean() < sep_n.mean()
    assert sep_e[-1] < 1000.0  # measured ~180 km, like the pn1 run
    assert abs(sep_e[-1] - sep_g[-1]) < 20.0  # the two GR laws differ by km, not hundreds
