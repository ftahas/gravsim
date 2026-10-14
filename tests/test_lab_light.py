"""Tests for the light-bending lab (``app/lab_light.py``): pure back end and a headless run.

``lab_light.py`` is imported from its path (``app/`` is not a package).  The back-end tests need
only the gravsim package; the ``AppTest`` smoke test renders the lab headlessly through a
three-line script and is skipped when streamlit or plotly is not installed.  No network.  Many
quantities are tiny in AU/day units (deflections ~1e-5 rad, delays ~1e-9 days), so relative
comparisons pass ``abs=0`` to make the stated ``rel`` the tolerance enforced.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from gravsim import constants as C
from gravsim.metric import null

APP_DIR = Path(__file__).resolve().parents[1] / "app"
MU_SUN = C.GM_SUN / C.C_AU_DAY**2
MICROSECOND_DAYS = 1e-6 / C.DAY_S


def _load_lab():
    spec = importlib.util.spec_from_file_location("gravsim_app_lab_light", APP_DIR / "lab_light.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


lab = _load_lab()


# --------------------------------------------------------------------------- helpers
def test_impact_parameter_units():
    assert lab.impact_parameter_au(2.0, "solar radii") == pytest.approx(2.0 * C.R_SUN_AU, rel=1e-15)
    assert lab.impact_parameter_au(10.0, "GM/c²") == pytest.approx(10.0 * MU_SUN, rel=1e-15, abs=0)
    # the GM/c^2 unit scales with the mass, the solar radius does not
    assert lab.impact_parameter_au(10.0, "GM/c²", gm=100.0 * C.GM_SUN) == pytest.approx(
        1000.0 * MU_SUN, rel=1e-15, abs=0
    )
    assert lab.impact_parameter_au(1.0, "solar radii", gm=100.0 * C.GM_SUN) == C.R_SUN_AU
    with pytest.raises(ValueError):
        lab.impact_parameter_au(1.0, "parsec")
    assert lab.UNITS == ("solar radii", "GM/c²")


def test_bundle_impact_parameters_and_amplification():
    b = lab.bundle_impact_parameters(2.0, factors=(1.0, 3.0))
    np.testing.assert_array_equal(b, [-6.0, -2.0, 2.0, 6.0])
    assert lab.bundle_impact_parameters(1.0).size == 2 * len(lab.BUNDLE_FACTORS)
    x = np.linspace(-1.0, 1.0, 5)
    y = 1.0 - 0.1 * x**2
    np.testing.assert_array_equal(lab.amplify_deviation(x, y, 1.0, 1.0), y)
    np.testing.assert_allclose(lab.amplify_deviation(x, y, 1.0, 10.0), 1.0 + 10.0 * (y - 1.0))
    # a mirrored ray (b < 0) amplifies towards the mass as well
    np.testing.assert_allclose(lab.amplify_deviation(x, -y, -1.0, 10.0), -1.0 - 10.0 * (y - 1.0))


# --------------------------------------------------------------------------- deflection curve
def test_deflection_curve_sun():
    curve = lab.deflection_curve(C.R_SUN_AU, 100.0 * C.R_SUN_AU, n=40)
    assert curve.b.shape == curve.exact.shape == curve.weak.shape == curve.newton.shape == (40,)
    assert curve.b[0] == pytest.approx(C.R_SUN_AU, rel=1e-12)
    assert curve.b[-1] == pytest.approx(100.0 * C.R_SUN_AU, rel=1e-12)
    assert np.all(np.diff(curve.b) > 0.0) and np.all(np.diff(curve.exact) < 0.0)
    assert curve.b_crit == pytest.approx(null.critical_impact_parameter(C.GM_SUN), rel=1e-15, abs=0)
    # limb: Einstein's 1.751", exact above the weak-field value by (15 pi/16) mu/b
    limb_arcsec = curve.weak[0] * C.ARCSEC_PER_RAD
    assert limb_arcsec == pytest.approx(C.LIGHT_DEFLECTION_LIMB_ARCSEC, abs=1e-3)
    np.testing.assert_allclose(curve.exact, curve.weak, rtol=1e-5)
    assert curve.exact[0] / curve.weak[0] - 1.0 == pytest.approx(
        15.0 * math.pi / 16.0 * MU_SUN / C.R_SUN_AU, rel=0.05
    )
    # Soldner is exactly half of GR, and the weak-field line follows (1 + gamma)/2
    np.testing.assert_allclose(curve.newton, 0.5 * curve.weak, rtol=1e-15)
    half = lab.deflection_curve(C.R_SUN_AU, 100.0 * C.R_SUN_AU, gamma=0.0, n=40)
    np.testing.assert_array_equal(half.weak, curve.newton)
    np.testing.assert_array_equal(half.exact, curve.exact)  # the exact curve is GR regardless
    assert half.gamma == 0.0


def test_deflection_curve_clips_at_capture_and_validates():
    b_crit = null.critical_impact_parameter(1.0, 1.0)
    curve = lab.deflection_curve(0.0, 100.0, gm=1.0, c=1.0, n=30)
    assert curve.b[0] == pytest.approx(1.001 * b_crit, rel=1e-12)
    assert np.all(np.isfinite(curve.exact)) and curve.exact[0] > 2.0 * math.pi  # a full loop
    with pytest.raises(ValueError):
        lab.deflection_curve(10.0, 5.0, gm=1.0, c=1.0)
    with pytest.raises(ValueError):
        lab.deflection_curve(0.0, b_crit, gm=1.0, c=1.0)


# --------------------------------------------------------------------------- ray bundle
def test_ray_bundle_scene_sun_to_scale():
    b = lab.bundle_impact_parameters(C.R_SUN_AU)
    bundle = lab.ray_bundle_scene(b, body_radius=C.R_SUN_AU, r_max=8.0 * C.R_SUN_AU, n_points=300)
    assert bundle.captured.size == 0 and bundle.b.shape == (10,) and len(bundle.paths) == 10
    assert bundle.r_max == 8.0 * C.R_SUN_AU
    np.testing.assert_array_equal(bundle.b, b)
    for bb, (x, y), delta, r0 in zip(
        bundle.b, bundle.paths, bundle.deflection, bundle.closest_approach, strict=True
    ):
        assert x.shape == y.shape == (300,)
        assert y[0] == pytest.approx(bb, rel=1e-6)  # enters along y = b from the left
        assert x[0] < 0.0 < x[-1]
        assert np.hypot(x, y).min() == pytest.approx(r0, rel=1e-4)  # 300 samples in azimuth
        assert delta == pytest.approx(null.deflection_angle(abs(bb), C.GM_SUN), rel=1e-12, abs=0)
        assert r0 == pytest.approx(abs(bb) - MU_SUN, rel=1e-6)
    # the two limb rays graze below the photosphere by one gravitational radius: flagged
    assert bundle.blocked.sum() == 2 and np.all(np.abs(bundle.b[bundle.blocked]) == C.R_SUN_AU)
    # default window: 4 max|b|; no body: nothing blocked
    free = lab.ray_bundle_scene(b, n_points=100)
    assert free.r_max == pytest.approx(4.0 * 5.0 * C.R_SUN_AU) and not free.blocked.any()


def test_ray_bundle_scene_captures_and_geometric_units():
    gm, c = 1.0, 1.0
    b_crit = null.critical_impact_parameter(gm, c)
    b = np.array([-20.0, -b_crit, -2.0, 0.0, 3.0, 6.0, 20.0])
    bundle = lab.ray_bundle_scene(b, gm, c, r_max=50.0, n_points=200)
    np.testing.assert_array_equal(bundle.captured, [-b_crit, -2.0, 0.0, 3.0])
    np.testing.assert_array_equal(bundle.b, [-20.0, 6.0, 20.0])
    assert bundle.b_crit == pytest.approx(b_crit)
    assert bundle.deflection[0] == bundle.deflection[2] < bundle.deflection[1]
    assert bundle.deflection[1] > math.radians(30.0)  # b = 6 mu is bent by about 34 degrees
    empty = lab.ray_bundle_scene([], gm, c)
    assert empty.b.size == 0 and empty.paths == () and empty.captured.size == 0


def test_ray_bundle_compact_sun_captures_the_limb_ray():
    # 1e5 solar masses inside the solar radius: b_crit = 1.1 R_sun, the limb ray is captured
    gm = 1e5 * C.GM_SUN
    bundle = lab.ray_bundle_scene(
        lab.bundle_impact_parameters(C.R_SUN_AU), gm, body_radius=C.R_SUN_AU,
        r_max=8.0 * C.R_SUN_AU, n_points=200,
    )
    np.testing.assert_array_equal(bundle.captured, [-C.R_SUN_AU, C.R_SUN_AU])
    assert bundle.b.shape == (8,) and not bundle.blocked.any()
    assert np.degrees(bundle.deflection.max()) == pytest.approx(63.0, abs=1.0)  # at 1.5 R_sun


# --------------------------------------------------------------------------- Shapiro delay
def test_shapiro_scenario_earth_mars():
    res = lab.shapiro_scenario(1.0, 1.5237, C.R_SUN_AU)
    assert res.one_way == pytest.approx(123.6 * MICROSECOND_DAYS, rel=1e-3, abs=0)
    assert res.round_trip == 2.0 * res.one_way
    assert res.conjunction_form == pytest.approx(res.one_way, rel=1e-5, abs=0)
    assert res.exact == pytest.approx(res.one_way, rel=1e-3, abs=0)
    assert res.light_time == pytest.approx(2.5237 / C.C_AU_DAY, rel=1e-4)  # chord < r_e + r_r
    assert res.one_way / res.light_time == pytest.approx(1e-7, rel=0.05)
    assert not res.captured and res.gamma == 1.0 and res.b == C.R_SUN_AU
    assert res.one_way == pytest.approx(
        null.shapiro_delay(1.0, 1.5237, C.R_SUN_AU, C.GM_SUN), rel=1e-15, abs=0
    )


def test_shapiro_scenario_gamma_and_capture():
    gr = lab.shapiro_scenario(1.0, 8.43, 1.6 * C.R_SUN_AU)
    half = lab.shapiro_scenario(1.0, 8.43, 1.6 * C.R_SUN_AU, gamma=0.0)
    assert half.one_way == pytest.approx(0.5 * gr.one_way, rel=1e-15, abs=0)
    assert half.conjunction_form == pytest.approx(0.5 * gr.conjunction_form, rel=1e-15, abs=0)
    assert half.exact == gr.exact  # the exact geodesic is GR whatever gamma says
    # a compact Sun whose b_crit exceeds the chord: weak-field numbers still, exact is NaN
    captured = lab.shapiro_scenario(1.0, 1.5237, C.R_SUN_AU, gm=1e5 * C.GM_SUN)
    assert captured.captured and math.isnan(captured.exact)
    assert captured.one_way == pytest.approx(
        null.shapiro_delay(1.0, 1.5237, C.R_SUN_AU, 1e5 * C.GM_SUN), rel=1e-15, abs=0
    )
    with pytest.raises(ValueError):
        lab.shapiro_scenario(1.0, 1.5237, 1.2)


def test_conjunction_series_shape_symmetry_and_peak():
    days = np.linspace(-60.0, 60.0, 241)
    series = lab.conjunction_series(1.0, 1.5237, 2.0 * C.R_SUN_AU, days)
    assert series.days.shape == series.b.shape == series.delay.shape == (241,)
    assert series.occulted.dtype == bool and series.valid.dtype == bool
    assert not series.occulted.any() and series.valid.all()
    assert np.all(np.isfinite(series.delay))
    # closest approach exactly b_min at conjunction, symmetric, delay peaks there
    assert series.b.min() == pytest.approx(2.0 * C.R_SUN_AU, rel=1e-8)
    assert series.b[120] == series.b.min() and series.b_min == 2.0 * C.R_SUN_AU
    np.testing.assert_allclose(series.b, series.b[::-1], rtol=1e-10)
    np.testing.assert_allclose(series.delay, series.delay[::-1], rtol=1e-10)
    assert np.argmax(series.delay) == 120
    assert np.all(np.diff(series.delay[120:]) < 0.0) and np.all(np.diff(series.b[120:]) > 0.0)
    assert series.delay[120] == pytest.approx(
        lab.shapiro_scenario(1.0, 1.5237, 2.0 * C.R_SUN_AU).one_way, rel=1e-8, abs=0
    )
    # synodic rate of Earth and Mars from Kepler's third law: 2 pi (1/365 - 1/687) rad/day
    assert series.synodic_rate == pytest.approx(2 * math.pi * (1 / 365.25 - 1 / 687.0), rel=2e-3)
    # the chord moves away from the Sun at about Earth's speed times r_r/(r_e + r_r)
    expected_b = 60.0 * series.synodic_rate * 1.5237 / 2.5237  # AU after 60 days
    assert series.b[-1] == pytest.approx(expected_b, rel=0.1)


def test_conjunction_series_occultation_validity_and_gamma():
    days = np.linspace(-120.0, 120.0, 481)
    grazing = lab.conjunction_series(1.0, 1.5237, C.R_SUN_AU, days)
    assert not grazing.occulted.any()  # a chord exactly at the limb counts as visible
    assert np.isfinite(grazing.delay[240])
    inside = lab.conjunction_series(1.0, 1.5237, 0.5 * C.R_SUN_AU, days)
    assert inside.occulted[240] and np.isnan(inside.delay[240])
    assert inside.occulted.sum() < 10 and np.isfinite(inside.delay[~inside.occulted]).all()
    # Mercury at 0.387 AU: the chord passes the Sun between the two bodies while the Earth's
    # angle from conjunction satisfies cos(theta) > -r_r/r_e, i.e. theta < 113 deg = 36 days at
    # the synodic rate (the window stays inside half a synodic period, so one conjunction)
    days_m = np.linspace(-50.0, 50.0, 201)
    mercury = lab.conjunction_series(1.0, 0.3871, 2.0 * C.R_SUN_AU, days_m)
    assert mercury.valid[100] and not mercury.valid[0] and not mercury.valid[-1]
    assert mercury.synodic_rate == pytest.approx(2 * math.pi / 115.9, rel=5e-3)
    edge = np.abs(days_m[mercury.valid]).max()
    assert edge == pytest.approx(math.acos(-0.3871) / mercury.synodic_rate, abs=0.6)
    assert np.isnan(mercury.delay[~mercury.valid]).all()
    assert np.isfinite(mercury.delay[mercury.valid]).all()
    # gamma scales the whole curve
    half = lab.conjunction_series(1.0, 1.5237, 2.0 * C.R_SUN_AU, days, gamma=0.0)
    full = lab.conjunction_series(1.0, 1.5237, 2.0 * C.R_SUN_AU, days)
    np.testing.assert_allclose(half.delay, 0.5 * full.delay, rtol=1e-12)
    for bad in (0.0, -1.0, 0.5, 2.0):
        with pytest.raises(ValueError):
            lab.conjunction_series(1.0, 0.3871, bad, days)


def test_measurement_table_brackets_einstein():
    labels = [m[0] for m in lab.DEFLECTION_MEASUREMENTS]
    assert labels[0].startswith("1919") and len(labels) >= 4
    for _label, value, err in lab.DEFLECTION_MEASUREMENTS:
        assert err > 0.0
        assert abs(value - C.LIGHT_DEFLECTION_LIMB_ARCSEC) < 3.0 * err  # all consistent with GR
        assert abs(value - 0.5 * C.LIGHT_DEFLECTION_LIMB_ARCSEC) > 2.0 * err  # none with Soldner
    assert set(lab.GEOMETRIES) >= {"Earth – Mars (Viking 1976)", "Earth – Saturn (Cassini 2002)"}
    assert all(r_e > 0.0 and r_r > 0.0 for r_e, r_r in lab.GEOMETRIES.values())


# --------------------------------------------------------------------------- the lab itself
_SCRIPT = f"""
import sys
sys.path.insert(0, {str(APP_DIR)!r})
import lab_light
lab_light.render()
"""


def test_render_headless_and_widgets():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(_SCRIPT, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.header[0].value.startswith("Light-bending lab")
    assert len(at.metric) >= 8 and not at.warning  # deflection and Shapiro readouts, no capture
    assert at.metric[0].value.startswith("1.751")  # exact limb deflection in arcsec
    assert at.metric[4].value.endswith("μs")
    # a compact Sun captures the limb ray and the limb chord: warnings, no exception
    at.select_slider[0].set_value(1e5).run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "captured" and len(at.warning) == 2
    # moving the chord out to 5 R_sun restores the Shapiro section
    at.slider[3].set_value(5.0).run()
    assert not at.exception, at.exception
    assert len(at.warning) == 1 and len(at.metric) >= 8
    # geometric units: the mass drops out, the readout switches to degrees near b_crit
    at.select_slider[0].set_value(1.0).run()
    at.radio[0].set_value("GM/c²").run()
    assert not at.exception, at.exception
    assert at.slider[0].label.endswith("[GM/c²]")
    at.slider[0].set_value(5.3).run()
    assert not at.exception, at.exception
    assert at.metric[0].value.endswith("°")
    # Shapiro geometry, one-way toggle and gamma = 0 halve the readouts without errors
    at.selectbox[0].set_value("Earth – Mercury (radar 1967)").run()
    at.toggle[0].set_value(False).run()
    assert not at.exception, at.exception
    one_way = float(at.metric[4].value.split()[0])
    at.slider[1].set_value(0.0).run()
    assert not at.exception, at.exception
    assert float(at.metric[4].value.split()[0]) == pytest.approx(0.5 * one_way, abs=0.1)
