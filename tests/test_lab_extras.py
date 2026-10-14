"""Tests for the competing-effects lab (``app/lab_extras.py``): back end and a headless run.

``lab_extras.py`` is imported from its path (``app/`` is not a package); its back end needs
only the gravsim package.  The smoke test runs ``render()`` headlessly through
``streamlit.testing.v1.AppTest`` with the default widget values and is skipped when streamlit
or plotly is not installed.  No network.

Physics being checked
---------------------
* ``kaula_rates``: on the equator ``dOmega/dt = -(3/2) n J2 (R/p)^2`` and
  ``domega/dt = 3 n J2 (R/p)^2``; ``domega/dt`` vanishes at the critical inclination
  ``cos^2 i = 1/5``; for Mercury (``i`` = 3.38 deg to the solar equator) the sum is the
  +0.028"/cy of ``constants.MERCURY_PRECESSION_J2_ARCSEC_CY``; linear in ``J2``, ``a^-7/2``.
* ``lense_thirring_rates``: ``dOmega/dt = 2 GJ/(c^2 a^3 (1-e^2)^1.5)`` independent of ``i``,
  ``domega/dt = -3 cos i dOmega/dt``; Mercury -0.002"/cy; scales with ``GJ`` and ``1/c^2``.
* ``moon_tidal_rates``: the first-order ``(3/4) m^2 n`` is 20.1 deg/yr for the Moon, the series
  to ``m^5`` gives 40.3 (perigee) and -19.2 (node) deg/yr.
* ``ring_precession_rate``: ``(3/4) n (m/M) (a/a_ring)^3``, ~0.0014"/cy for the main belt.
* ``equator_frame``: a proper rotation with ``z`` along the solar pole and ``x`` in the ecliptic.
* ``extras_runs`` with ``J2 x 1000`` and ``J x 1e5`` over three years: the differenced node and
  argument rates of Mercury agree with Kaula + Lense-Thirring to 1 %, with the expected signs;
  zero scales reproduce the Newtonian run exactly.
* With the Moon toggle the system is ``DEFAULT_BODIES_MOON`` and the lunar fits over two years
  give the sidereal month 27.32 +- 0.05 d, the perigee advance 40.7 +- 3 and the node
  regression -19.3 +- 2 deg/yr (the long-period lunar inequalities bias short spans).
* The precession budget sums to the observed 574.6"/cy with the modelled terms and scales its
  solar rows with the amplification factors.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from gravsim import constants as C
from gravsim.analysis import rate_arcsec_per_century
from gravsim.bodies import Trajectory

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _load_lab():
    spec = importlib.util.spec_from_file_location(
        "gravsim_app_lab_extras", APP_DIR / "lab_extras.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


lab = _load_lab()

A_MERCURY, E_MERCURY = 0.3871, 0.2056
INC_MERCURY_EQ = math.radians(3.38)  # to the solar equator


# --------------------------------------------------------------------------- formulas
def test_kaula_rates_formula_and_mercury():
    n = math.sqrt(C.GM_SUN / A_MERCURY**3)
    k = n * C.SUN_J2 * (C.R_SUN_AU / (A_MERCURY * (1.0 - E_MERCURY**2))) ** 2
    d_raan, d_argp = lab.kaula_rates(A_MERCURY, E_MERCURY, 0.0, C.GM_SUN)
    assert d_raan == pytest.approx(-1.5 * k, rel=1e-12)
    assert d_argp == pytest.approx(3.0 * k, rel=1e-12)
    assert isinstance(d_raan, float) and isinstance(d_argp, float)
    # critical inclination: no secular motion of the argument
    _, d_crit = lab.kaula_rates(A_MERCURY, E_MERCURY, math.acos(1.0 / math.sqrt(5.0)), C.GM_SUN)
    assert abs(d_crit) < 1e-12 * k
    # Mercury: the node regresses, the argument advances, the sum is +0.028"/cy
    d_raan, d_argp = lab.kaula_rates(A_MERCURY, E_MERCURY, INC_MERCURY_EQ, C.GM_SUN)
    assert d_raan < 0.0 < d_argp
    assert rate_arcsec_per_century(d_raan + d_argp) == pytest.approx(
        C.MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.001
    )
    # linear in J2 and in n (R/p)^2 ~ a^-7/2, broadcasting over arrays
    big = lab.kaula_rates(A_MERCURY, E_MERCURY, INC_MERCURY_EQ, C.GM_SUN, j2=10.0 * C.SUN_J2)
    assert big[0] == pytest.approx(10.0 * d_raan) and big[1] == pytest.approx(10.0 * d_argp)
    arr = lab.kaula_rates(np.array([A_MERCURY, 2.0 * A_MERCURY]), E_MERCURY, 0.0, C.GM_SUN)
    assert arr[0].shape == (2,)
    assert arr[0][1] == pytest.approx(arr[0][0] * 2.0**-3.5)


def test_lense_thirring_rates_formula_and_mercury():
    k = C.GJ_SUN / (C.C_AU_DAY**2 * A_MERCURY**3 * (1.0 - E_MERCURY**2) ** 1.5)
    d_raan, d_argp = lab.lense_thirring_rates(A_MERCURY, E_MERCURY, 0.0)
    assert d_raan == pytest.approx(2.0 * k, rel=1e-12)
    assert d_argp == pytest.approx(-6.0 * k, rel=1e-12)
    for inc in (0.3, 1.0, 2.5):
        r, w = lab.lense_thirring_rates(A_MERCURY, E_MERCURY, inc)
        assert r == pytest.approx(d_raan, rel=1e-12)  # the node rate does not depend on i
        assert w == pytest.approx(-3.0 * math.cos(inc) * r, rel=1e-12)
    d_raan, d_argp = lab.lense_thirring_rates(A_MERCURY, E_MERCURY, INC_MERCURY_EQ)
    assert d_argp < 0.0 < d_raan
    assert rate_arcsec_per_century(d_raan + d_argp) == pytest.approx(
        C.MERCURY_PRECESSION_LT_ARCSEC_CY, abs=0.0003
    )
    # scaling with the spin and with 1/c^2
    assert lab.lense_thirring_rates(A_MERCURY, E_MERCURY, 0.0, gj=2.0 * C.GJ_SUN)[0] == (
        pytest.approx(4.0 * k)
    )
    assert lab.lense_thirring_rates(A_MERCURY, E_MERCURY, 0.0, c=0.5 * C.C_AU_DAY)[0] == (
        pytest.approx(8.0 * k)
    )


def test_moon_tidal_rates_first_order_and_series():
    n_moon = 2.0 * math.pi / 27.3217
    n_sun = 2.0 * math.pi / 365.2564
    t = lab.moon_tidal_rates(n_moon, n_sun)
    assert t.m == pytest.approx(0.0748, abs=1e-3)
    assert t.leading == pytest.approx(0.75 * t.m**2 * n_moon, rel=1e-12)
    assert t.leading * lab.DEG_PER_YEAR == pytest.approx(20.1, abs=0.3)
    # Clairaut: the higher orders double the perigee rate and barely touch the node
    assert t.perigee * lab.DEG_PER_YEAR == pytest.approx(40.3, abs=0.5)
    assert t.node * lab.DEG_PER_YEAR == pytest.approx(-19.2, abs=0.3)
    assert t.perigee > t.leading and -t.leading < t.node < 0.0
    assert abs(t.perigee * lab.DEG_PER_YEAR - C.MOON_PERIGEE_RATE_DEG_YR) < 1.0
    assert abs(t.node * lab.DEG_PER_YEAR - C.MOON_NODE_RATE_DEG_YR) < 0.3
    with pytest.raises(ValueError):
        lab.moon_tidal_rates(0.0, n_sun)
    with pytest.raises(ValueError):
        lab.moon_tidal_rates(n_moon, -n_sun)


def test_ring_precession_rate_scaling():
    gm_ring, a_ring = lab.ASTEROID_RING_GM, lab.ASTEROID_RING_AU
    rate = lab.ring_precession_rate(gm_ring, a_ring, A_MERCURY)
    assert 0.001 < rate < 0.002  # the main belt is a hundred times below the ranging precision
    n = math.sqrt(C.GM_SUN / A_MERCURY**3)
    expected = rate_arcsec_per_century(0.75 * n * (gm_ring / C.GM_SUN) * (A_MERCURY / a_ring) ** 3)
    assert rate == pytest.approx(expected, rel=1e-12)
    assert lab.ring_precession_rate(2.0 * gm_ring, a_ring, A_MERCURY) == pytest.approx(2.0 * rate)
    assert lab.ring_precession_rate(gm_ring, 2.0 * a_ring, A_MERCURY) == pytest.approx(rate / 8.0)
    # (3/4) n (a/a_ring)^3 with n ~ a^-3/2: rate ~ a^3/2
    assert lab.ring_precession_rate(gm_ring, a_ring, 4.0 * A_MERCURY) == pytest.approx(8.0 * rate)


def test_equator_frame_and_rotation():
    rot = lab.equator_frame()
    pole = np.asarray(C.SUN_POLE_ECLIPTIC)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    np.testing.assert_allclose(rot[2], pole / np.linalg.norm(pole), atol=1e-15)
    assert rot[0, 2] == 0.0  # x in the ecliptic, at the node of the solar equator
    np.testing.assert_allclose(rot @ pole, [0.0, 0.0, 1.0], atol=1e-15)
    # the solar equator's ascending node on the ecliptic is at longitude 75.76 deg
    assert math.degrees(math.atan2(rot[0, 1], rot[0, 0])) == pytest.approx(C.SUN_POLE_NODE_DEG)
    with pytest.raises(ValueError):
        lab.equator_frame((0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        lab.equator_frame((0.0, 0.0, 0.0))
    # rotating a trajectory preserves distances and leaves the input untouched
    rng = np.random.default_rng(9)
    pos, vel = rng.normal(size=(5, 2, 3)), rng.normal(size=(5, 2, 3))
    traj = Trajectory(np.arange(5.0), pos, vel, np.array([1.0, 0.0]), ("Sun", "p"), {"k": 1})
    turned = lab.rotate_trajectory(traj, rot)
    np.testing.assert_allclose(np.linalg.norm(turned.pos, axis=2), np.linalg.norm(pos, axis=2))
    np.testing.assert_array_equal(turned.t, traj.t)
    assert turned.names == traj.names and turned.meta == traj.meta
    assert np.array_equal(traj.pos, pos)


# --------------------------------------------------------------------------- runs
@pytest.fixture(scope="module")
def amplified():
    return lab.extras_runs(3.0, 1000.0, 1.0e5)


def test_extras_runs_shapes_grid_and_validation(amplified):
    assert isinstance(amplified.newton, Trajectory) and isinstance(amplified.extras, Trajectory)
    assert amplified.newton.names == amplified.extras.names == ("Sun", "Mercury")
    np.testing.assert_array_equal(amplified.newton.t, amplified.extras.t)
    assert amplified.newton.t[0] == 0.0
    assert amplified.newton.t[-1] == pytest.approx(3.0 * C.JULIAN_YEAR_DAYS, abs=1.0)
    assert amplified.newton.meta["method"] == amplified.extras.meta["method"] == "DOP853"
    assert amplified.j2 == 1000.0 * C.SUN_J2 and amplified.gj == 1.0e5 * C.GJ_SUN
    assert amplified.with_moon is False and amplified.years == 3.0
    # the amplified terms really act: frame dragging x 1e5 turns the ellipse by ~6" in 3 years
    r_x, r_n = amplified.extras.body("Mercury")[0], amplified.newton.body("Mercury")[0]
    sep_km = np.linalg.norm(r_x - r_n, axis=1) * C.AU_KM
    assert 100.0 < sep_km[-1] < 1.0e5
    for bad in ({"years": 0.0}, {"step": -1.0}, {"j2_scale": -1.0}, {"lt_scale": -0.5}):
        with pytest.raises(ValueError):
            lab.extras_runs(**({"years": 1.0, "step": 1.0} | bad))


def test_zero_scales_reproduce_newton_exactly():
    runs = lab.extras_runs(1.0, 0.0, 0.0)
    assert runs.j2 == 0.0 and runs.gj == 0.0
    np.testing.assert_array_equal(runs.newton.pos, runs.extras.pos)
    np.testing.assert_array_equal(runs.newton.vel, runs.extras.vel)
    res = lab.node_perigee_rates(runs)
    assert res.raan_rate == res.argp_rate == res.varpi_rate == 0.0
    assert res.predicted["raan"] == res.predicted["argp"] == res.predicted["varpi"] == 0.0


def test_node_perigee_rates_match_formulas(amplified):
    res = lab.node_perigee_rates(amplified)
    assert res.t.shape == res.d_raan.shape == res.d_argp.shape == amplified.newton.t.shape
    assert np.all(np.abs(res.d_raan) <= np.pi) and np.all(np.abs(res.d_argp) <= np.pi)
    pred = res.predicted
    assert res.raan_rate == pytest.approx(pred["raan"], rel=0.01)
    assert res.argp_rate == pytest.approx(pred["argp"], rel=0.01)
    assert res.varpi_rate == pytest.approx(pred["varpi"], rel=0.01)
    assert 0.0 < res.raan_err < 0.01 * abs(res.raan_rate)
    assert 0.0 < res.argp_err < 0.01 * abs(res.argp_rate)
    # the oblate Sun regresses the node and advances the argument; frame dragging the opposite
    assert pred["raan_j2"] < 0.0 < pred["raan_lt"] and pred["argp_lt"] < 0.0 < pred["argp_j2"]
    assert pred["raan"] == pytest.approx(pred["raan_j2"] + pred["raan_lt"])
    assert pred["varpi"] == pytest.approx(pred["raan"] + pred["argp"])
    # the sizes: J2 x 1000 gives -27.9" on the node, J x 1e5 gives +100.9"
    assert pred["raan_j2"] == pytest.approx(-27.9, abs=0.1)
    assert pred["raan_lt"] == pytest.approx(100.9, abs=0.2)
    assert pred["argp_j2"] == pytest.approx(55.6, abs=0.2)
    assert pred["argp_lt"] == pytest.approx(-302.2, abs=0.5)
    el = res.elements
    assert el["a"] == pytest.approx(A_MERCURY, abs=5e-4)
    assert el["e"] == pytest.approx(E_MERCURY, abs=5e-4)
    assert math.degrees(el["inc_equator"]) == pytest.approx(3.38, abs=0.05)
    assert math.degrees(el["inc"]) == pytest.approx(7.005, abs=0.02)


def test_moon_runs_and_rates():
    runs = lab.extras_runs(2.0, 0.0, 0.0, with_moon=True)
    assert runs.with_moon and runs.newton.names == C.DEFAULT_BODIES_MOON
    moon = lab.moon_rates(runs)
    assert moon.t.shape == moon.varpi.shape == moon.raan.shape == runs.newton.t.shape
    assert moon.sidereal_days == pytest.approx(27.3217, abs=0.05)
    assert moon.perigee_rate == pytest.approx(C.MOON_PERIGEE_RATE_DEG_YR, abs=3.0)
    assert moon.node_rate == pytest.approx(C.MOON_NODE_RATE_DEG_YR, abs=2.0)
    assert 0.0 < moon.perigee_err < 2.0 and 0.0 < moon.node_err < 0.5
    assert moon.perigee_reference == C.MOON_PERIGEE_RATE_DEG_YR
    assert moon.node_reference == C.MOON_NODE_RATE_DEG_YR
    assert moon.mean_distance_km == pytest.approx(384_400.0, abs=3000.0)
    assert moon.e_mean == pytest.approx(0.055, abs=0.01)
    assert moon.inc_mean_deg == pytest.approx(5.15, abs=0.3)
    assert moon.tidal.leading * lab.DEG_PER_YEAR == pytest.approx(20.1, abs=0.3)
    assert moon.tidal.m == pytest.approx(0.0748, abs=1e-3)
    # Mercury's differenced node/argument in the full system vanish with the scales at zero
    res = lab.node_perigee_rates(runs)
    assert res.raan_rate == res.argp_rate == 0.0
    with pytest.raises(ValueError):
        lab.moon_rates(lab.extras_runs(0.5, 0.0, 0.0))


# --------------------------------------------------------------------------- the budget
def test_precession_budget_entries_and_scaling():
    budget = lab.precession_budget()
    assert len(budget) == 6
    names = [e.name for e in budget]
    assert names[0].startswith("planets") and names[1] == "General Relativity"
    assert [e.modelled for e in budget] == [True] * 5 + [False]
    assert budget[0].rate == C.MERCURY_PRECESSION_NEWTON_ARCSEC_CY
    assert budget[1].rate == C.MERCURY_PRECESSION_GR_ARCSEC_CY
    assert budget[2].rate == C.MERCURY_PRECESSION_J2_ARCSEC_CY
    assert budget[3].rate == lab.MOON_EMB_SHIFT_ARCSEC_CY
    assert budget[4].rate == C.MERCURY_PRECESSION_LT_ARCSEC_CY
    assert 0.001 < budget[5].rate < 0.002
    assert sum(e.rate for e in budget[:3]) == pytest.approx(574.6, abs=0.01)
    # sorted by decreasing size, every entry annotated
    sizes = [abs(e.rate) for e in budget]
    assert sizes == sorted(sizes, reverse=True)
    assert all(e.note for e in budget)
    amplified = lab.precession_budget(1000.0, 1.0e5)
    assert amplified[2].rate == pytest.approx(1000.0 * C.MERCURY_PRECESSION_J2_ARCSEC_CY)
    assert amplified[4].rate == pytest.approx(1.0e5 * C.MERCURY_PRECESSION_LT_ARCSEC_CY)
    assert "1000" in amplified[2].name and "100000" in amplified[4].name
    assert amplified[0].rate == budget[0].rate and amplified[3].rate == budget[3].rate


# --------------------------------------------------------------------------- the front end
def test_cached_wrapper_is_streamlit_cache_when_available():
    if lab.st is None:
        pytest.skip("streamlit not installed")
    assert hasattr(lab._extras_lab, "clear")  # st.cache_data wraps the back-end call


def test_render_headless():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "import lab_extras\n"
        "lab_extras.render()\n"
    )
    at = AppTest.from_string(script, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert at.header[0].value.startswith("Competing effects")
    assert len(at.toggle) == 3 and len(at.slider) == 3
    assert len(at.metric) >= 4
    # defaults: J2 x 1000 and J x 1e5 over ten years; the node formula reads +73.03"/cy
    assert at.metric[1].value.startswith("+73.0")
    assert at.metric[1].value.endswith("″/cy")
    # the Moon toggle adds the lunar readouts: sidereal month, perigee and node rates
    at.toggle[0].set_value(True).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 8
    assert at.metric[4].value.startswith("27.32")
