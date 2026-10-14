"""Tests for the Streamlit labs: the pure back end (``app/labs.py``) and a headless smoke run.

``labs.py`` is imported from its path (``app/`` is not a package).  The back-end tests need
only the gravsim package; the ``AppTest`` smoke tests run the actual script headlessly with
default widget values and are skipped when streamlit or plotly is not installed.  No network.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from gravsim import constants as C
from gravsim.bodies import Trajectory
from gravsim.metric import schwarzschild as sch

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _load_labs():
    spec = importlib.util.spec_from_file_location("gravsim_app_labs", APP_DIR / "labs.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


labs = _load_labs()


# --------------------------------------------------------------------------- lab 1
def test_integrator_lab_leapfrog_vs_yoshida():
    lf = labs.integrator_lab("leapfrog", 0.5, 2.0)
    y4 = labs.integrator_lab("yoshida4", 0.5, 2.0)
    assert isinstance(lf.traj, Trajectory) and lf.traj.names == ("Sun", "Mercury")
    assert lf.energy_error.shape == (lf.traj.n_steps,) and lf.energy_error[0] == 0.0
    assert lf.steps_per_orbit == pytest.approx(87.97 / 0.5, rel=2e-3)
    assert lf.period == pytest.approx(87.97, abs=0.02)
    # bounded bands: leapfrog ~3e-4 at 176 steps/orbit (peaks at perihelion), yoshida4 far smaller
    assert np.abs(lf.energy_error).max() < 1e-3
    assert np.abs(y4.energy_error).max() < np.abs(lf.energy_error).max() / 10
    # spurious precession: leapfrog ~ -2e5 arcsec/cy, yoshida4 ~ +2.6e3 at this step
    assert abs(lf.spurious_rate) > 1e4 and abs(y4.spurious_rate) < abs(lf.spurious_rate) / 10
    assert lf.spurious_rate_err >= 0.0


def test_integrator_lab_rejects_unknown_name():
    with pytest.raises(KeyError):
        labs.integrator_lab("verlet-ish", 1.0, 1.0)


# --------------------------------------------------------------------------- lab 2
@pytest.fixture(scope="module")
def two_body_runs():
    return labs.solar_system_runs(3.0, 1.0, ("Sun", "Mercury"))


def test_solar_system_runs_shapes_and_grid(two_body_runs):
    newton, gr = two_body_runs
    assert newton.names == gr.names == ("Sun", "Mercury")
    np.testing.assert_array_equal(newton.t, gr.t)
    assert newton.t[0] == 0.0 and newton.t[-1] == pytest.approx(3.0 * C.JULIAN_YEAR_DAYS, abs=1.0)
    assert newton.meta["method"] == "DOP853" and gr.meta["method"] == "DOP853"
    with pytest.raises(ValueError):
        labs.solar_system_runs(1.0, 1.0, ("Mercury",))


def test_amplified_difference_and_separation(two_body_runs):
    newton, gr = two_body_runs
    r_n, r_g, r_amp = labs.amplified_difference(newton, gr, "Mercury", 1e5)
    assert r_n.shape == r_g.shape == r_amp.shape == (newton.n_steps, 3)
    np.testing.assert_allclose(r_amp, r_n + 1e5 * (r_g - r_n))
    np.testing.assert_array_equal(labs.amplified_difference(newton, gr, "Mercury", 1.0)[2], r_g)
    sep = labs.separation_km(newton, gr, "Mercury")
    assert sep.shape == (newton.n_steps,) and sep[0] == 0.0
    assert 10.0 < sep[-1] < 2000.0  # GR moves Mercury by ~500 km in 3 years


# --------------------------------------------------------------------------- lab 3
def test_precession_track_recovers_gr_rate(two_body_runs):
    newton, gr = two_body_runs
    track = labs.precession_track(newton, gr, "Mercury", 1.0)
    assert track.t.shape == track.varpi_newton.shape == track.varpi_gr.shape
    assert track.difference.shape == track.t.shape
    assert track.analytic == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.01)
    assert track.rate_difference == pytest.approx(track.analytic, abs=0.3)  # 3-year span
    assert track.rate_difference_err < 0.3
    assert abs(track.rate_newton) * C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS < 0.5  # noise floor
    assert track.elements["a"] == pytest.approx(0.3871, abs=5e-4)


def test_precession_track_scaled_c_uses_same_c_in_analytic():
    newton, gr = labs.solar_system_runs(2.0, 100.0, ("Sun", "Mercury"), step=0.5)
    track = labs.precession_track(newton, gr, "Mercury", 100.0)
    assert track.analytic == pytest.approx(1e4 * C.MERCURY_PRECESSION_GR_ARCSEC_CY, rel=1e-3)
    assert track.rate_difference == pytest.approx(track.analytic, rel=0.02)


# --------------------------------------------------------------------------- lab 4
def test_potential_curves_circular_orbits():
    curves = labs.potential_curves(4.0)
    assert curves.r.shape == curves.v_gr.shape == curves.v_newton.shape
    r_st, r_un = sch.circular_orbit_radii(4.0, 1.0, 1.0)
    assert curves.r_stable == pytest.approx(r_st) and curves.r_unstable == pytest.approx(r_un)
    assert curves.r_unstable == pytest.approx(4.0) and curves.r_stable == pytest.approx(12.0)
    assert curves.v_top > curves.v_well
    # the sampled curve has its maximum near the unstable orbit and its minimum near the stable one
    inner = (curves.r > 3.0) & (curves.r < 8.0)  # the curve dives to -inf at the horizon
    outer = curves.r > 6.0
    assert curves.r[inner][np.argmax(curves.v_gr[inner])] == pytest.approx(4.0, abs=0.1)
    assert curves.r[outer][np.argmin(curves.v_gr[outer])] == pytest.approx(12.0, abs=0.1)
    low = labs.potential_curves(3.0)
    assert math.isnan(low.r_stable) and math.isnan(low.v_top)


def test_strong_field_orbit_bound_escaping_and_captured():
    curves = labs.potential_curves(4.0)
    whirl = labs.strong_field_orbit(4.0, curves.v_top - 1e-4, 30.0, tau_end=1500.0)
    assert not whirl.captured and 4.0 < whirl.r_min < 4.5  # bounces off the barrier at r = 4
    assert whirl.turns > 2.0  # ... after whirling around the unstable circular orbit
    assert whirl.x.shape == whirl.y.shape == whirl.r.shape == whirl.phi.shape == whirl.tau.shape
    np.testing.assert_allclose(np.hypot(whirl.x, whirl.y), whirl.r)
    assert np.all(np.diff(whirl.t) > 0.0) and whirl.t[-1] > whirl.tau[-1]  # time dilation
    plunge = labs.strong_field_orbit(4.0, curves.v_top + 1e-3, 30.0, tau_end=1500.0)
    # the capture event stops the integration at 1.001 r_s; the last *sample* sits just above
    assert plunge.captured and plunge.r_min < 2.1 and plunge.r.size < whirl.r.size
    # an unbound orbit at larger L escapes instead of failing (the azimuth integrator cannot)
    high = labs.potential_curves(5.0)
    escape = labs.strong_field_orbit(5.0, high.v_top - 1e-2, 30.0, tau_end=1500.0)
    assert not escape.captured and escape.r[-1] > 30.0 and escape.r_min > high.r_unstable
    # starting below V_eff(r0): r0 is used as an apsis, the orbit is still produced
    apsis = labs.strong_field_orbit(4.2, -0.5, 25.0, tau_end=500.0)
    assert apsis.r.max() == pytest.approx(25.0, rel=1e-6)


# --------------------------------------------------------------------------- the app itself
@pytest.mark.parametrize("lab_index", [0, 1, 2, 3])
def test_streamlit_app_runs_each_lab_headless(lab_index):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    labs_radio = at.sidebar.radio[0]
    labs_radio.set_value(labs_radio.options[lab_index]).run()
    assert not at.exception, at.exception
    assert at.header[0].value  # each lab draws its header
    assert len(at.metric) >= 3  # and its readouts
    # a quick widget interaction per lab
    if lab_index == 0:
        at.selectbox[0].set_value("yoshida4").run()
    elif lab_index == 1:
        at.toggle[0].set_value(False).run()
    elif lab_index == 2:
        at.selectbox[0].set_value("Venus").run()
    else:
        at.slider[0].set_value(5.0).run()
    assert not at.exception, at.exception
