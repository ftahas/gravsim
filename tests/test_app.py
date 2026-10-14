"""Tests for the Streamlit labs: the pure back end (``app/labs.py``) and a headless smoke run.

``labs.py`` is imported from its path (``app/`` is not a package).  The back-end tests need
only the gravsim package; the ``AppTest`` smoke tests run the actual script headlessly with
default widget values, once per lab (all seven), and are skipped when streamlit or plotly is
not installed.  Labs 5, 6 and 7 live in ``app/lab_ppn.py``, ``app/lab_light.py`` and
``app/lab_extras.py``; their back ends are tested in depth in ``tests/test_lab_ppn.py``,
``tests/test_lab_light.py`` and ``tests/test_lab_extras.py`` and only smoke-tested here
(import, one call each, the wiring of the script).  No network.
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


def _load(name: str):
    """Import ``app/<name>.py`` from its path under a unique module name."""
    spec = importlib.util.spec_from_file_location(f"gravsim_app_{name}", APP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


labs = _load("labs")
lab_ppn = _load("lab_ppn")
lab_light = _load("lab_light")
lab_extras = _load("lab_extras")
LAB_NAMES = (
    "1 · Integrator lab",
    "2 · Solar system: Newton vs GR",
    "3 · Precession tracker",
    "4 · Effective-potential explorer",
    "5 · PPN playground",
    "6 · Light-bending lab",
    "7 · Competing effects",
)


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


# --------------------------------------------------------------------------- labs 5 and 6
def test_lab_ppn_backend_smoke():
    """PPN playground back end: the factor and a short GR run recover the 43 arcsec."""
    assert lab_ppn.ppn_factor(1.0, 1.0) == 1.0
    assert lab_ppn.ppn_factor(0.0, 1.0) == pytest.approx(4.0 / 3.0)
    assert callable(lab_ppn.render)
    runs = lab_ppn.ppn_runs(1.0, 1.0, 1.0, 2.0)
    res = lab_ppn.ppn_precession(runs)
    assert runs.newton.names == runs.ppn.names == ("Sun", "Mercury")
    assert res.predicted == res.gr == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.01)
    assert res.rate == pytest.approx(res.predicted, abs=0.5)  # two-year span
    assert all(abs(s) < 1.0 for s in lab_ppn.sigmas_from_gr(1.0, 1.0).values())


def test_lab_light_backend_smoke():
    """Light lab back end: 1.751 arcsec at the limb, Soldner's half, 124 us Earth-Mars."""
    assert callable(lab_light.render)
    curve = lab_light.deflection_curve(C.R_SUN_AU, 10.0 * C.R_SUN_AU, n=8)
    limb = curve.exact[0] * C.ARCSEC_PER_RAD
    assert limb == pytest.approx(C.LIGHT_DEFLECTION_LIMB_ARCSEC, abs=1e-3)
    np.testing.assert_allclose(curve.newton, 0.5 * curve.weak, rtol=1e-15)
    shapiro = lab_light.shapiro_scenario(1.0, 1.5237, C.R_SUN_AU)
    assert shapiro.one_way * C.DAY_S * 1e6 == pytest.approx(123.6, abs=0.5)
    assert shapiro.round_trip == 2.0 * shapiro.one_way and not shapiro.captured
    bundle = lab_light.ray_bundle_scene(
        lab_light.bundle_impact_parameters(C.R_SUN_AU, factors=(1.0, 2.0)),
        body_radius=C.R_SUN_AU,
        r_max=8.0 * C.R_SUN_AU,
        n_points=50,
    )
    assert len(bundle.paths) == 4 and bundle.captured.size == 0


def test_lab_extras_backend_smoke():
    """Competing-effects back end: Kaula and Lense-Thirring formulas, the budget, one run."""
    assert callable(lab_extras.render)
    a, e, inc, gm = 0.3871, 0.2056, math.radians(3.38), C.GM_SUN
    raan_j2, argp_j2 = lab_extras.kaula_rates(a, e, inc, gm)
    varpi_j2 = (raan_j2 + argp_j2) * C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS
    assert varpi_j2 == pytest.approx(C.MERCURY_PRECESSION_J2_ARCSEC_CY, abs=0.002)
    raan_lt, argp_lt = lab_extras.lense_thirring_rates(a, e, inc)
    varpi_lt = (raan_lt + argp_lt) * C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS
    assert varpi_lt == pytest.approx(C.MERCURY_PRECESSION_LT_ARCSEC_CY, abs=0.0005)
    assert raan_lt > 0.0 > argp_lt and raan_j2 < 0.0 < argp_j2
    budget = lab_extras.precession_budget()
    assert [b.name.split(" ")[0] for b in budget][:2] == ["planets", "General"]
    modelled = sum(b.rate for b in budget if b.modelled)
    assert modelled == pytest.approx(574.6, abs=0.1)
    runs = lab_extras.extras_runs(1.0, j2_scale=1000.0, lt_scale=0.0)
    res = lab_extras.node_perigee_rates(runs)
    assert runs.newton.names == ("Sun", "Mercury") and runs.j2 == pytest.approx(2.2e-4)
    assert res.raan_rate == pytest.approx(res.predicted["raan"], rel=0.05)  # one-year span
    assert res.argp_rate == pytest.approx(res.predicted["argp"], rel=0.05)


# --------------------------------------------------------------------------- the app itself
def test_app_script_lists_seven_labs():
    source = (APP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    for name in LAB_NAMES:
        assert f'"{name}"' in source, name
    assert "lab_ppn.render" in source and "lab_light.render" in source
    assert "lab_extras.render" in source


@pytest.mark.parametrize("lab_index", range(len(LAB_NAMES)))
def test_streamlit_app_runs_each_lab_headless(lab_index):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_DIR / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    labs_radio = at.sidebar.radio[0]
    assert tuple(labs_radio.options) == LAB_NAMES
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
    elif lab_index == 3:
        at.slider[0].set_value(5.0).run()
    elif lab_index == 4:
        assert at.header[0].value.startswith("PPN playground")
        at.slider[0].set_value(0.5).run()  # beta = 1/2: factor (2 + 2 - 0.5)/3 = 1.167
        assert at.metric[3].value == "1.167"
    elif lab_index == 5:
        assert at.header[0].value.startswith("Light-bending lab")
        assert at.metric[0].value.startswith("1.751")  # exact limb deflection, arcsec
        at.toggle[0].set_value(False).run()  # one-way Shapiro delay
    else:
        assert at.header[0].value.startswith("Competing effects")
        assert len(at.toggle) == 3 and len(at.slider) == 3
        # defaults J2 x 1000, J x 1e5, 10 yr: the node formula reads +73.03 arcsec/cy
        # (+100.92 frame dragging, -27.89 J2) and the fit agrees with it; the lunar readouts
        # appear with the Moon toggle (test_lab_extras)
        assert at.metric[1].value.startswith("+73.0")
        fit = float(at.metric[0].value.split("±")[0])
        assert fit == pytest.approx(73.03, rel=0.03)
        at.toggle[2].set_value(False).run()  # frame dragging off: the node is J2's alone
        assert at.metric[1].value.startswith("-27.8")
    assert not at.exception, at.exception
