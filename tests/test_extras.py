"""Tests for ``gravsim.forces.extras`` (J2, Lense-Thirring) and the Phase 8 analysis helpers
``precession.element_series`` and ``elements.inclination_to_axis``.  No network.

Physics being checked
---------------------
* J2 is a potential force: the closed-form acceleration equals ``-grad U`` with
  ``U = gm j2 r_eq^2 (3 (r_hat.n)^2 - 1) / (2 r^3)``, checked against a five-point central
  finite difference for a tilted axis, plus the limits on the axis (``+3k n/r^4``) and on the
  equator (``-3k r_hat/(2 r^4)``).
* Both laws conserve momentum through the reaction on the central body
  (``sum_i gm_i a_i = 0``) and vanish exactly for ``j2 = 0`` / ``scale = 0``.
* Lense-Thirring: the vector formula against a scalar transcription, ``a . v = 0`` (work-free),
  purely radial and outward for a prograde circular equatorial orbit.
* Secular rates (Kaula for J2, Lense-Thirring for frame dragging) measured on Sun + Mercury
  two-body runs with the effect amplified (``j2 x 1000``, ``J x 1e5``) over 10 years and
  compared with the first-order formulas to 3 %.  The formulas refer the inclination ``i``,
  node ``Omega`` and argument ``omega`` to the *equator of the central body*, so the trajectory
  is rotated into the solar-equator frame (rows of ``equator_frame``: ``x`` toward the
  ascending node of the solar equator on the ecliptic, ``z`` = ``SUN_POLE_ECLIPTIC``) before
  ``element_series`` is applied; the inclination is cross-checked with ``inclination_to_axis``
  on the unrotated state.
* Real-size effects over a century (slow): run-vs-run differences against Newton give
  dvarpi/dt = +0.028"/cy (J2) and -0.002"/cy (Lense-Thirring) for Mercury.
* Mercury's inclination to the solar equator at J2000 is 3.38 deg; ``element_series`` with
  ``key="varpi"`` reproduces ``perihelion_longitude_series``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gravsim.analysis import (
    element_series,
    fit_secular_rate,
    inclination_to_axis,
    perihelion_longitude_series,
    perihelion_rate_difference,
    rate_arcsec_per_century,
    state_to_elements,
)
from gravsim.bodies import Trajectory
from gravsim.constants import (
    C_AU_DAY,
    GJ_SUN,
    GM_SUN,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
    MERCURY_PRECESSION_J2_ARCSEC_CY,
    MERCURY_PRECESSION_LT_ARCSEC_CY,
    R_SUN_AU,
    SUN_J2,
    SUN_POLE_ECLIPTIC,
)
from gravsim.data.horizons import load_j2000
from gravsim.forces import accel_newton, combine, make_j2, make_lense_thirring
from gravsim.integrators import integrate_scipy

# --------------------------------------------------------------------------- helpers / fixtures


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261014)


@pytest.fixture(scope="module")
def sun_mercury():
    """Sun + Mercury two-body system at J2000, barycentric."""
    return load_j2000().subset(("Sun", "Mercury")).to_barycentric()


def random_axis(rng: np.random.Generator) -> np.ndarray:
    """A random unit vector, generic (tilted with respect to every coordinate axis)."""
    n = rng.normal(size=3)
    return n / np.linalg.norm(n)


def random_system(rng: np.random.Generator, n: int, central: int = 0, n_test: int = 0):
    """Central body plus ``n - 1`` others at 0.3-5 AU from it, log-uniform GMs."""
    pos = rng.uniform(-2.0, 2.0, size=(n, 3))
    vel = rng.normal(0.0, 0.01, size=(n, 3))
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    pos = pos[central] + rng.uniform(0.3, 5.0, size=n)[:, None] * directions
    gm = 10.0 ** rng.uniform(-10.0, -6.0, size=n)
    gm[central] = GM_SUN
    if n_test:
        others = [i for i in range(n) if i != central]
        gm[others[-n_test:]] = 0.0
    return pos, vel, gm


def j2_potential(r_rel: np.ndarray, gm_c: float, j2: float, r_eq: float, axis: np.ndarray):
    """``U = gm j2 r_eq^2 (3 (r_hat . n)^2 - 1) / (2 r^3)`` per unit mass (plain floats)."""
    rx, ry, rz = (float(x) for x in r_rel)
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    cos_theta = (rx * axis[0] + ry * axis[1] + rz * axis[2]) / rmag
    return gm_c * j2 * r_eq**2 * (3.0 * cos_theta**2 - 1.0) / (2.0 * rmag**3)


def gradient_fd(f, x: np.ndarray, h: float) -> np.ndarray:
    """Five-point central finite-difference gradient (error O(h^4))."""
    grad = np.empty(3)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        grad[k] = (
            -f(x + 2 * e) + 8.0 * f(x + e) - 8.0 * f(x - e) + f(x - 2 * e)
        ) / (12.0 * h)
    return grad


def lt_scalar(r_rel, v_rel, GJ, c) -> np.ndarray:
    """Hand-written ``(2/(c^2 r^3)) [ (3/r^2)(r.GJ)(r x v) + (v x GJ) ]`` with plain floats."""
    rx, ry, rz = (float(x) for x in r_rel)
    vx, vy, vz = (float(x) for x in v_rel)
    jx, jy, jz = (float(x) for x in GJ)
    rmag = math.sqrt(rx * rx + ry * ry + rz * rz)
    rJ = rx * jx + ry * jy + rz * jz
    rxv = (ry * vz - rz * vy, rz * vx - rx * vz, rx * vy - ry * vx)
    vxJ = (vy * jz - vz * jy, vz * jx - vx * jz, vx * jy - vy * jx)
    pref = 2.0 / (c * c * rmag**3)
    coef = 3.0 * rJ / rmag**2
    return np.array([pref * (coef * rxv[k] + vxJ[k]) for k in range(3)])


def equator_frame(pole) -> np.ndarray:
    """Rotation matrix (rows ``x, y, z``) of a frame whose ``z`` axis is ``pole``.

    ``x`` points to the ascending node of the equator (plane perpendicular to ``pole``) on the
    reference ``xy`` plane, i.e. ``z_ref x pole``; ``y = z x x``.  ``r_eq = R @ r``.
    """
    z = np.asarray(pole, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def rotate_trajectory(traj: Trajectory, R: np.ndarray) -> Trajectory:
    """The same trajectory with every position and velocity rotated by ``R``."""
    return Trajectory(
        t=traj.t,
        pos=traj.pos @ R.T,
        vel=traj.vel @ R.T,
        gm=traj.gm,
        names=traj.names,
        meta=dict(traj.meta),
    )


def ten_year_run(state, law) -> Trajectory:
    """Newton + ``law`` for 10 years, sampled daily, DOP853 at rtol 1e-12."""
    t_eval = state.t + np.arange(0.0, 10.0 * JULIAN_YEAR_DAYS + 0.5, 1.0)
    return integrate_scipy(state, combine(accel_newton, law), t_eval, rtol=1e-12)


def equatorial_rates(traj: Trajectory, body: str = "Mercury", central: str = "Sun"):
    """Fitted ``raan`` and ``argp`` rates [rad/day] in the solar-equator frame, plus mean a, e, i.

    The elements are taken after rotating the trajectory into the frame whose ``z`` axis is
    ``SUN_POLE_ECLIPTIC`` (``equator_frame``), which is what the Kaula and Lense-Thirring
    formulas refer to; ``i`` comes from ``inclination_to_axis`` on the unrotated state and is
    checked against the rotated ``inc`` element.
    """
    traj_eq = rotate_trajectory(traj, equator_frame(SUN_POLE_ECLIPTIC))
    t, raan = element_series(traj_eq, body, "raan", central)
    _, argp = element_series(traj_eq, body, "argp", central)
    _, a = element_series(traj_eq, body, "a", central)
    _, e = element_series(traj_eq, body, "e", central)
    _, inc_eq = element_series(traj_eq, body, "inc", central)
    r, v = traj.relative(body, central)
    inc = inclination_to_axis(r, v, SUN_POLE_ECLIPTIC)
    assert np.abs(inc - inc_eq).max() < 1e-12
    return fit_secular_rate(t, raan), fit_secular_rate(t, argp), a.mean(), e.mean(), inc.mean()


# --------------------------------------------------------------------------- J2


def test_j2_is_minus_gradient_of_potential(rng):
    axis = random_axis(rng)
    j2, r_eq = 1e-3, 0.05  # arbitrary sizes; the check is relative
    pos, vel, gm = random_system(rng, 7, central=0, n_test=2)
    acc = make_j2(j2=j2, r_eq=r_eq, central=0, axis=axis)(pos, vel, gm, 0.0)
    assert acc.shape == (7, 3) and acc.dtype == np.float64 and acc.flags.c_contiguous
    for i in range(1, 7):
        r_rel = pos[i] - pos[0]
        rmag = np.linalg.norm(r_rel)
        grad = gradient_fd(lambda x: j2_potential(x, gm[0], j2, r_eq, axis), r_rel, 1e-4 * rmag)
        assert np.abs(acc[i] + grad).max() <= 1e-8 * np.abs(acc[i]).max()


def test_j2_axis_and_equator_limits(rng):
    axis = random_axis(rng)
    j2, r_eq, rmag = 2e-4, 0.1, 0.7
    k = GM_SUN * j2 * r_eq**2
    perp = np.cross(axis, random_axis(rng))
    perp /= np.linalg.norm(perp)
    pos = np.array([np.zeros(3), rmag * axis, rmag * perp])
    vel = np.zeros((3, 3))
    gm = np.array([GM_SUN, 0.0, 0.0])
    acc = make_j2(j2=j2, r_eq=r_eq, axis=axis)(pos, vel, gm, 0.0)
    assert np.allclose(acc[1], 3.0 * k / rmag**4 * axis, rtol=1e-13, atol=0.0)  # weaker pull
    assert np.allclose(acc[2], -1.5 * k / rmag**4 * perp, rtol=1e-13, atol=0.0)  # stronger
    assert np.all(acc[0] == 0.0)  # test particles exert no reaction


@pytest.mark.parametrize("central", [0, 2])
def test_j2_and_lt_conserve_momentum(rng, central):
    pos, vel, gm = random_system(rng, 6, central=central)
    axis = random_axis(rng)
    for law in (
        make_j2(j2=1e-3, r_eq=0.05, central=central, axis=axis),
        make_lense_thirring(gj=1e-6, c=10.0, central=central, axis=axis),
    ):
        acc = law(pos, vel, gm, 0.0)
        momentum_rate = np.einsum("i,ij->j", gm, acc)
        budget = np.abs(gm[:, None] * acc).sum(axis=0)
        assert np.abs(momentum_rate).max() <= 1e-14 * budget.max()
        assert np.any(acc[central] != 0.0)  # the reaction is really there
        # the reaction is the only thing in the central row
        others = [i for i in range(6) if i != central]
        expected = -np.einsum("i,ij->j", gm[others], acc[others]) / gm[central]
        assert np.allclose(acc[central], expected, rtol=1e-14, atol=0.0)


def test_massless_central_gets_no_reaction(rng):
    pos, vel, gm = random_system(rng, 4)
    gm[0] = 0.0
    acc = make_lense_thirring(gj=1e-6, c=10.0, central=0)(pos, vel, gm, 0.0)
    assert np.all(acc[0] == 0.0) and np.all(np.isfinite(acc))
    acc = make_j2(j2=1e-3, r_eq=0.05, central=0)(pos, vel, gm, 0.0)
    assert np.all(acc == 0.0)  # the J2 field is proportional to gm_c


def test_zero_strength_gives_exact_zero(rng):
    pos, vel, gm = random_system(rng, 5)
    assert np.all(make_j2(j2=0.0)(pos, vel, gm, 0.0) == 0.0)
    assert np.all(make_lense_thirring(scale=0.0)(pos, vel, gm, 0.0) == 0.0)
    assert np.all(make_lense_thirring(gj=0.0)(pos, vel, gm, 0.0) == 0.0)


def test_inputs_are_not_mutated_and_params(rng):
    pos, vel, gm = random_system(rng, 5)
    pos0, vel0, gm0 = pos.copy(), vel.copy(), gm.copy()
    j2 = make_j2()
    lt = make_lense_thirring(scale=10.0)
    j2(pos, vel, gm, 0.0)
    lt(pos, vel, gm, 0.0)
    assert np.array_equal(pos, pos0) and np.array_equal(vel, vel0) and np.array_equal(gm, gm0)
    assert j2.params["law"] == "j2" and j2.params["j2"] == SUN_J2
    assert j2.params["r_eq"] == R_SUN_AU and j2.params["central"] == 0
    assert np.allclose(j2.params["axis"], SUN_POLE_ECLIPTIC, rtol=1e-15, atol=0.0)
    assert lt.params["law"] == "lense_thirring" and lt.params["gj"] == GJ_SUN
    assert lt.params["scale"] == 10.0 and lt.params["c"] == C_AU_DAY
    combined = combine(accel_newton, j2, lt)  # plain functions are named by __name__
    assert combined.params["law"] == "accel_newton+j2+lense_thirring"


def test_argument_validation(rng):
    pos, vel, gm = random_system(rng, 3)
    with pytest.raises(TypeError):
        make_j2(central="Sun")
    with pytest.raises(TypeError):
        make_lense_thirring(central="Sun")
    with pytest.raises(IndexError):
        make_j2(central=3)(pos, vel, gm, 0.0)
    with pytest.raises(IndexError):
        make_lense_thirring(central=-4)(pos, vel, gm, 0.0)
    with pytest.raises(ValueError):
        make_j2(axis=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        make_lense_thirring(c=0.0)
    with pytest.raises(ValueError):
        make_j2()(pos, vel, gm[:2], 0.0)
    with pytest.raises(ValueError):
        make_lense_thirring()(pos[:, :2], vel, gm, 0.0)
    # negative index resolves like numpy's and axis is normalised
    a_neg = make_j2(central=-3, axis=(0.0, 0.0, 5.0))(pos, vel, gm, 0.0)
    a_pos = make_j2(central=0, axis=(0.0, 0.0, 1.0))(pos, vel, gm, 0.0)
    assert np.array_equal(a_neg, a_pos)


# --------------------------------------------------------------------------- Lense-Thirring


def test_lense_thirring_matches_scalar_formula(rng):
    axis = random_axis(rng)
    gj, scale, c = 3e-7, 2.5, 20.0
    GJ = gj * scale * axis
    pos, vel, gm = random_system(rng, 6, central=1, n_test=2)
    acc = make_lense_thirring(gj=gj, c=c, central=1, axis=axis, scale=scale)(pos, vel, gm, 0.0)
    for i in [0, 2, 3, 4, 5]:
        expected = lt_scalar(pos[i] - pos[1], vel[i] - vel[1], GJ, c)
        assert np.allclose(acc[i], expected, rtol=1e-13, atol=1e-13 * np.abs(expected).max())
        # work-free: the gravitomagnetic force is perpendicular to the (relative) velocity
        v_rel = vel[i] - vel[1]
        assert abs(acc[i] @ v_rel) <= 1e-14 * np.linalg.norm(acc[i]) * np.linalg.norm(v_rel)


def test_lense_thirring_radial_for_circular_equatorial_orbit(rng):
    axis = random_axis(rng)
    e1 = np.cross(axis, random_axis(rng))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)  # e1 x e2 = axis: prograde when v is along +e2
    rmag = 0.4
    vmag = math.sqrt(GM_SUN / rmag)
    center, v_center = rng.normal(size=3), rng.normal(size=3) * 1e-3
    pos = np.array([center, center + rmag * e1])
    vel = np.array([v_center, v_center + vmag * e2])
    gm = np.array([GM_SUN, 0.0])
    acc = make_lense_thirring(gj=1e-6, c=10.0, axis=axis)(pos, vel, gm, 0.0)
    a = acc[1]
    assert np.linalg.norm(np.cross(a, e1)) <= 1e-13 * np.linalg.norm(a)  # purely radial
    assert abs(a @ axis) <= 1e-13 * np.linalg.norm(a)
    assert a @ e1 > 0.0  # outward for prograde motion: parallel mass currents repel
    expected = 2.0 * 1e-6 * vmag / (10.0**2 * rmag**3)
    assert np.isclose(a @ e1, expected, rtol=1e-13)
    # retrograde: inward
    vel_retro = np.array([v_center, v_center - vmag * e2])
    assert make_lense_thirring(gj=1e-6, c=10.0, axis=axis)(pos, vel_retro, gm, 0.0)[1] @ e1 < 0.0


# --------------------------------------------------------------------------- secular rates


def test_j2_kaula_rates_mercury(sun_mercury):
    """Node and argument rates of Mercury with j2 x 1000 vs Kaula's first-order formulas."""
    amplified = 1000.0 * SUN_J2
    traj = ten_year_run(sun_mercury, make_j2(j2=amplified))
    fit_raan, fit_argp, a, e, inc = equatorial_rates(traj)
    gm = float(sun_mercury.gm.sum())
    n_mean = math.sqrt(gm / a**3)
    p = a * (1.0 - e * e)
    k = n_mean * amplified * (R_SUN_AU / p) ** 2
    pred_raan = -1.5 * k * math.cos(inc)
    pred_argp = 0.75 * k * (5.0 * math.cos(inc) ** 2 - 1.0)
    assert math.degrees(inc) == pytest.approx(3.38, abs=0.05)
    assert fit_raan.rate == pytest.approx(pred_raan, rel=0.03), (
        f"dOmega/dt {rate_arcsec_per_century(fit_raan.rate):.4g} vs "
        f"{rate_arcsec_per_century(pred_raan):.4g} arcsec/cy"
    )
    assert fit_argp.rate == pytest.approx(pred_argp, rel=0.03), (
        f"domega/dt {rate_arcsec_per_century(fit_argp.rate):.4g} vs "
        f"{rate_arcsec_per_century(pred_argp):.4g} arcsec/cy"
    )
    # the node regresses and the argument advances; their sum (/1000) is the +0.028"/cy of
    # constants.MERCURY_PRECESSION_J2_ARCSEC_CY
    assert fit_raan.rate < 0.0 < fit_argp.rate
    varpi_rate = rate_arcsec_per_century(fit_raan.rate + fit_argp.rate) / 1000.0
    assert varpi_rate == pytest.approx(MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.002)


def test_lense_thirring_rates_mercury(sun_mercury):
    """Node and argument rates of Mercury with J x 1e5 vs the Lense-Thirring formulas."""
    scale = 1e5
    traj = ten_year_run(sun_mercury, make_lense_thirring(scale=scale))
    fit_raan, fit_argp, a, e, inc = equatorial_rates(traj)
    k = GJ_SUN * scale / (C_AU_DAY**2 * a**3 * (1.0 - e * e) ** 1.5)
    pred_raan = 2.0 * k
    pred_argp = -6.0 * k * math.cos(inc)
    assert fit_raan.rate == pytest.approx(pred_raan, rel=0.03), (
        f"dOmega/dt {rate_arcsec_per_century(fit_raan.rate):.4g} vs "
        f"{rate_arcsec_per_century(pred_raan):.4g} arcsec/cy"
    )
    assert fit_argp.rate == pytest.approx(pred_argp, rel=0.03), (
        f"domega/dt {rate_arcsec_per_century(fit_argp.rate):.4g} vs "
        f"{rate_arcsec_per_century(pred_argp):.4g} arcsec/cy"
    )
    # frame dragging advances the node in the sense of the spin and drives the argument back
    assert fit_argp.rate < 0.0 < fit_raan.rate
    varpi_rate = rate_arcsec_per_century(fit_raan.rate + fit_argp.rate) / scale
    assert varpi_rate == pytest.approx(MERCURY_PRECESSION_LT_ARCSEC_CY, abs=0.0005)


@pytest.mark.slow
def test_real_j2_and_lense_thirring_mercury_century(sun_mercury):
    """Run-vs-run perihelion rate differences against Newton over a century, real sizes.

    J2: +0.028 +- 0.005 arcsec/century.  Lense-Thirring: -0.002 +- 0.002 (reported; the
    signal is at the level of the numerical noise of a century-long absolute fit, but the
    differencing against the Newton run on the same grid cancels most of it).
    """
    t_eval = sun_mercury.t + np.arange(0.0, JULIAN_CENTURY_DAYS + 0.5, 1.0)
    newton = integrate_scipy(sun_mercury, accel_newton, t_eval, rtol=1e-12)
    j2 = integrate_scipy(sun_mercury, combine(accel_newton, make_j2()), t_eval, rtol=1e-12)
    lt = integrate_scipy(
        sun_mercury, combine(accel_newton, make_lense_thirring()), t_eval, rtol=1e-12
    )
    rate_j2, err_j2 = perihelion_rate_difference(j2, newton, "Mercury")
    rate_lt, err_lt = perihelion_rate_difference(lt, newton, "Mercury")
    report = f"J2 {rate_j2:.4f} +- {err_j2:.4f}, LT {rate_lt:.4f} +- {err_lt:.4f} arcsec/cy"
    assert rate_j2 == pytest.approx(MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.005), report
    assert rate_lt == pytest.approx(MERCURY_PRECESSION_LT_ARCSEC_CY, abs=0.002), report


# --------------------------------------------------------------------------- analysis helpers


def test_mercury_inclination_to_solar_equator():
    state = load_j2000()
    r, v = state.relative_to("Sun")
    i = state.index("Mercury")
    inc_eq = inclination_to_axis(r[i], v[i], SUN_POLE_ECLIPTIC)
    assert math.degrees(inc_eq) == pytest.approx(3.38, abs=0.05)
    # to the ecliptic Mercury is inclined 7.0 deg, and that is the classical element
    inc_ecl = inclination_to_axis(r[i], v[i], (0.0, 0.0, 1.0))
    assert math.degrees(inc_ecl) == pytest.approx(7.005, abs=0.01)
    assert inc_ecl == pytest.approx(state_to_elements(r[i], v[i], GM_SUN)["inc"], abs=1e-14)


def test_inclination_to_axis_vectorised_and_validated(rng):
    r = rng.normal(size=(4, 5, 3))
    v = rng.normal(size=(4, 5, 3)) * 0.01
    axis = rng.normal(size=3) * 3.0  # not unit: normalised internally
    inc = inclination_to_axis(r, v, axis)
    assert inc.shape == (4, 5) and np.all((inc >= 0.0) & (inc <= np.pi))
    assert np.allclose(inc, inclination_to_axis(r, v, axis / np.linalg.norm(axis)))
    assert np.allclose(inc[2, 3], inclination_to_axis(r[2, 3], v[2, 3], axis))
    # reversing the axis gives the supplementary angle; swapping r and v (h -> -h) too
    assert np.allclose(inclination_to_axis(r, v, -axis), np.pi - inc)
    assert np.allclose(inclination_to_axis(v, r, axis), np.pi - inc)
    with pytest.raises(ValueError):
        inclination_to_axis(r, v, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        inclination_to_axis(r, v, (1.0, 0.0))


def test_element_series_matches_perihelion_longitude_series(sun_mercury):
    t_eval = sun_mercury.t + np.linspace(0.0, 200.0, 401)
    traj = integrate_scipy(sun_mercury, accel_newton, t_eval)
    t_ref, varpi_ref = perihelion_longitude_series(traj, "Mercury")
    t, varpi = element_series(traj, "Mercury", "varpi")
    assert np.array_equal(t, t_ref) and np.array_equal(t, traj.t)
    assert t is not traj.t  # a copy, like perihelion_longitude_series
    assert varpi.shape == (401,) and np.allclose(varpi, varpi_ref, rtol=0.0, atol=1e-12)
    # every other key is the matching entry of state_to_elements on the relative state
    r, v = traj.relative("Mercury", "Sun")
    el = state_to_elements(r, v, float(traj.gm.sum()))
    for key in ("a", "e", "inc", "raan", "argp", "nu", "M", "period"):
        _, series = element_series(traj, "Mercury", key, central="Sun")
        assert np.array_equal(series, el[key]), key
    assert np.allclose(el["a"], el["a"][0], rtol=1e-9)  # Keplerian: constant elements
    with pytest.raises(KeyError):
        element_series(traj, "Mercury", "omega")
    with pytest.raises(KeyError):
        element_series(traj, "Pluto", "a")
