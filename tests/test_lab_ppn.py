"""Tests for the PPN playground lab (``app/lab_ppn.py``): the pure back end and a headless run.

``lab_ppn.py`` is imported from its path (``app/`` is not a package); its back end needs only
the gravsim package.  The smoke test runs ``render()`` headlessly through
``streamlit.testing.v1.AppTest`` with the default widget values and is skipped when streamlit
or plotly is not installed.  No network.

Physics being checked
---------------------
* ``ppn_factor`` is ``(2 + 2 gamma - beta)/3``: 1 in GR, 2/3 at gamma = 1/2, 4/3 at beta = 0,
  1/3 at gamma = 0 and 0 at (beta, gamma) = (2, 0).
* Sun + Mercury at the real ``c`` in GR: the differenced rate is 42.98 arcsec/century, the PPN
  prediction equals the GR value, and the Newtonian run alone does not precess (noise floor).
* At ``c/100`` (2 years, half-day samples) the measured rate follows the factor for (0, 1),
  (1, 0) and (2, 0) to the 1 % second-order level of Lesson 04.
* The experimental bounds (Cassini, lunar laser ranging, Mercury) contain GR within one sigma
  and exclude a visibly non-GR theory by thousands of sigma; the Mercury band is the line
  ``2 gamma - beta = 1`` on which the factor is 1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from gravsim import constants as C
from gravsim.bodies import Trajectory

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _load_lab():
    spec = importlib.util.spec_from_file_location("gravsim_app_lab_ppn", APP_DIR / "lab_ppn.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


lab = _load_lab()


# --------------------------------------------------------------------------- the factor
def test_ppn_factor_values_and_arrays():
    assert lab.ppn_factor(1.0, 1.0) == 1.0
    assert lab.ppn_factor(1.0, 0.5) == pytest.approx(2.0 / 3.0)
    assert lab.ppn_factor(0.0, 1.0) == pytest.approx(4.0 / 3.0)
    assert lab.ppn_factor(1.0, 0.0) == pytest.approx(1.0 / 3.0)
    assert lab.ppn_factor(2.0, 0.0) == 0.0
    assert isinstance(lab.ppn_factor(1, 1), float)
    betas = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(lab.ppn_factor(betas, 1.0), (4.0 - betas) / 3.0)
    # the Mercury degeneracy: every theory with 2 gamma - beta = 1 has factor 1
    gammas = np.linspace(0.0, 2.0, 7)
    np.testing.assert_allclose(lab.ppn_factor(2.0 * gammas - 1.0, gammas), 1.0)


# --------------------------------------------------------------------------- runs
@pytest.fixture(scope="module")
def gr_runs():
    return lab.ppn_runs(1.0, 1.0, 1.0, 3.0)


def test_ppn_runs_shapes_grid_and_validation(gr_runs):
    assert isinstance(gr_runs.newton, Trajectory) and isinstance(gr_runs.ppn, Trajectory)
    assert gr_runs.newton.names == gr_runs.ppn.names == ("Sun", "Mercury")
    np.testing.assert_array_equal(gr_runs.newton.t, gr_runs.ppn.t)
    assert gr_runs.newton.t[0] == 0.0
    assert gr_runs.newton.t[-1] == pytest.approx(3.0 * C.JULIAN_YEAR_DAYS, abs=1.0)
    assert gr_runs.newton.meta["method"] == "DOP853" and gr_runs.ppn.meta["method"] == "DOP853"
    assert (gr_runs.beta, gr_runs.gamma, gr_runs.c) == (1.0, 1.0, C.C_AU_DAY)
    # the two laws really differ: GR moves Mercury by ~500 km in three years
    r_ppn, r_newton = gr_runs.ppn.body("Mercury")[0], gr_runs.newton.body("Mercury")[0]
    sep_km = np.linalg.norm(r_ppn - r_newton, axis=1) * C.AU_KM
    assert 10.0 < sep_km[-1] < 2000.0
    for bad in ({"c_scale": 0.0}, {"c_scale": -5.0}, {"years": 0.0}, {"step": -1.0}):
        with pytest.raises(ValueError):
            lab.ppn_runs(1.0, 1.0, **({"c_scale": 1.0, "years": 1.0, "step": 1.0} | bad))


def test_ppn_precession_recovers_gr(gr_runs):
    res = lab.ppn_precession(gr_runs)
    assert res.t.shape == res.difference.shape == gr_runs.newton.t.shape
    assert np.all(np.abs(res.difference) <= np.pi)
    assert res.factor == 1.0
    assert res.predicted == res.gr
    assert res.gr == pytest.approx(C.MERCURY_PRECESSION_GR_ARCSEC_CY, abs=0.01)
    assert res.rate == pytest.approx(res.predicted, abs=0.3)  # three-year span
    assert 0.0 < res.rate_err < 0.3
    assert res.v_over_c == pytest.approx(1.3e-4, rel=0.05)  # 0.013 at c/100 (Lesson 04)
    assert res.elements["a"] == pytest.approx(0.3871, abs=5e-4)
    assert res.elements["e"] == pytest.approx(0.2056, abs=5e-4)


@pytest.mark.parametrize("beta,gamma", [(0.0, 1.0), (1.0, 0.0), (2.0, 0.0)])
def test_ppn_scaling_at_reduced_c(beta, gamma):
    runs = lab.ppn_runs(beta, gamma, 100.0, 3.0, step=0.5)  # 12 orbits
    res = lab.ppn_precession(runs)
    assert runs.c == pytest.approx(C.C_AU_DAY / 100.0)
    assert res.gr == pytest.approx(1e4 * C.MERCURY_PRECESSION_GR_ARCSEC_CY, rel=1e-3)
    assert res.predicted == pytest.approx(lab.ppn_factor(beta, gamma) * res.gr, rel=1e-12)
    # second order in GM/(c^2 p) at c/100 (~0.3 % of the GR rate) plus the periodic terms of
    # the differenced series, which twelve orbits do not average out below ~1 %
    assert abs(res.rate - res.predicted) < 0.02 * res.gr, (beta, gamma, res.rate, res.predicted)
    assert res.v_over_c == pytest.approx(0.013, rel=0.05)


# --------------------------------------------------------------------------- the plane
def test_factor_map_grid_and_values():
    betas, gammas, factor = lab.factor_map(n=41)
    assert betas.shape == gammas.shape == (41,) and factor.shape == (41, 41)
    assert betas[0] == lab.BETA_RANGE[0] and betas[-1] == lab.BETA_RANGE[1]
    i = np.searchsorted(betas, 1.0)
    j = np.searchsorted(gammas, 1.0)
    assert factor[j, i] == pytest.approx(1.0)
    # factor[j, i] belongs to (betas[i], gammas[j]): rises with gamma (rows), falls with beta
    assert np.all(np.diff(factor, axis=0) > 0.0) and np.all(np.diff(factor, axis=1) < 0.0)
    np.testing.assert_allclose(factor, lab.ppn_factor(betas[None, :], gammas[:, None]))


def test_experimental_bounds_bands_and_sigmas():
    bounds = lab.experimental_bounds()
    assert [b.name.split(" ")[0] for b in bounds] == ["Cassini", "Lunar", "Mercury's"]
    for b in bounds:
        assert b.sigma > 0.0
        assert abs(b.sigmas(1.0, 1.0)) < 1.0  # GR is inside every one-sigma band
        x, lo, hi = lab.bound_band(b, half_width=1e-3, n=5)
        assert x.shape == lo.shape == hi.shape == (5,) and np.all(hi > lo)
        # both edges satisfy a x + b y = value -/+ sigma
        np.testing.assert_allclose(np.sort([b.a * x[0] + b.b * lo[0], b.a * x[0] + b.b * hi[0]]),
                                   np.sort([b.value - b.sigma, b.value + b.sigma]), atol=1e-15)
    mercury = bounds[2]
    assert mercury.residual(1.4, 1.2) == pytest.approx(0.0, abs=1e-12)  # 2 gamma - beta = 1
    sig = lab.sigmas_from_gr(1.0, 0.5)
    assert set(sig) == {b.name for b in bounds}
    assert abs(sig[bounds[0].name]) > 1e3  # Cassini excludes gamma = 1/2 by far
    assert sig == lab.sigmas_from_gr(1.0, 0.5)
    with pytest.raises(ValueError):
        lab.bound_band(lab.Bound("vertical", 1.0, 0.0, 0.0, 1e-4))


# --------------------------------------------------------------------------- the front end
def test_cached_wrapper_is_streamlit_cache_when_available():
    if lab.st is None:
        pytest.skip("streamlit not installed")
    assert hasattr(lab._ppn_lab, "clear")  # st.cache_data wraps the back-end call


def test_render_headless():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(APP_DIR)!r})\n"
        "import lab_ppn\n"
        "lab_ppn.render()\n"
    )
    at = AppTest.from_string(script, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert at.header[0].value.startswith("PPN playground")
    assert len(at.metric) >= 4
    assert len(at.slider) == 3 and len(at.selectbox) == 1
    # the default (GR, real c, 10 yr) readout quotes the 43 arcsec
    assert "42.9" in at.metric[1].value or "43" in at.metric[1].value
    at.slider[0].set_value(0.5).run()  # beta = 1/2, gamma = 1: factor (2 + 2 - 0.5)/3
    assert not at.exception, at.exception
    assert at.metric[3].value == "1.167"
