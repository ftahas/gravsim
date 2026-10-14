"""Smoke tests for gravsim.viz on a synthetic two-body circular orbit (Agg backend, no network).

plotly tests run only when plotly is importable; the ImportError path is tested regardless.
"""

from __future__ import annotations

import math
import sys

import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D

from gravsim.bodies import Trajectory
from gravsim.constants import (
    ARCSEC_PER_RAD,
    AU_KM,
    C_AU_DAY,
    DEG_PER_RAD,
    GM,
    GM_SUN,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
    PLANETS,
)
from gravsim.viz import interactive, plots
from gravsim.viz.plots import (
    BODY_COLORS,
    INK,
    MUTED,
    PALETTE,
    body_colors,
    cube_limits,
    figure_grid,
    plot_effective_potential,
    plot_energy_error,
    plot_orbits_2d,
    plot_orbits_3d,
    plot_precession,
    plot_separation,
    precession_series,
    resolve_bodies,
)

matplotlib.use("Agg")


# --------------------------------------------------------------------------- fixtures
def _circular(a: float, inc_deg: float, gm: float, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relative position and velocity (T, 3) of a circular orbit of radius a, inclination inc."""
    w = 2.0 * math.pi / (2.0 * math.pi * math.sqrt(a**3 / gm))
    inc = math.radians(inc_deg)
    ph = w * t
    ci, si = math.cos(inc), math.sin(inc)
    r = a * np.column_stack([np.cos(ph), np.sin(ph) * ci, np.sin(ph) * si])
    v = a * w * np.column_stack([-np.sin(ph), np.cos(ph) * ci, np.cos(ph) * si])
    return r, v


@pytest.fixture
def circular_traj() -> Trajectory:
    """Sun + Mercury on a circular 0.387 AU orbit, barycentric, 3 periods, 601 samples."""
    a = 0.387
    gm_s, gm_m = GM_SUN, GM["Mercury"]
    mu = gm_s + gm_m
    period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    t = np.linspace(0.0, 3.0 * period, 601)
    r, v = _circular(a, 7.0, mu, t)
    f_s, f_m = -gm_m / mu, gm_s / mu
    pos = np.stack([f_s * r, f_m * r], axis=1)
    vel = np.stack([f_s * v, f_m * v], axis=1)
    return Trajectory(t, pos, vel, np.array([gm_s, gm_m]), ("Sun", "Mercury"), {"period": period})


@pytest.fixture
def three_body_traj() -> Trajectory:
    """Sun fixed at the origin with Mercury and Venus on circular orbits."""
    t = np.linspace(0.0, 400.0, 401)
    r1, v1 = _circular(0.387, 7.0, GM_SUN, t)
    r2, v2 = _circular(0.723, 3.4, GM_SUN, t)
    zeros = np.zeros_like(r1)
    pos = np.stack([zeros, r1, r2], axis=1)
    vel = np.stack([zeros, v1, v2], axis=1)
    gm = np.array([GM_SUN, GM["Mercury"], GM["Venus"]])
    return Trajectory(t, pos, vel, gm, ("Sun", "Mercury", "Venus"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _span(lim: tuple[float, float]) -> float:
    return float(lim[1] - lim[0])


# --------------------------------------------------------------------------- palette
def test_palette_is_fixed_and_covers_planets():
    assert len(PALETTE) == 8 and len(set(PALETTE)) == 8
    assert all(BODY_COLORS[p] == PALETTE[i] for i, p in enumerate(PLANETS))
    assert BODY_COLORS["Earth"] == BODY_COLORS["EMB"]


def test_body_colors_follow_entity_not_rank():
    assert body_colors(["Venus"]) == [BODY_COLORS["Venus"]]
    assert body_colors(["Venus", "Mercury"]) == [BODY_COLORS["Venus"], BODY_COLORS["Mercury"]]
    # unknown bodies take slots the planets did not use, never a planet's colour
    cols = body_colors(["Mercury", "probe", "Venus", "other"])
    assert cols[1] not in (BODY_COLORS["Mercury"], BODY_COLORS["Venus"])
    assert cols[1] != cols[3] and cols[3] in PALETTE
    # the palette is never cycled: the ninth distinct series is muted
    many = body_colors([f"b{i}" for i in range(9)])
    assert many[:8] == list(PALETTE) and many[8] == MUTED


def test_resolve_bodies(circular_traj, three_body_traj):
    assert resolve_bodies(circular_traj, None, "Sun") == ("Sun", ["Mercury"])
    assert resolve_bodies(three_body_traj, None, 0) == ("Sun", ["Mercury", "Venus"])
    assert resolve_bodies(three_body_traj, "Venus", "Sun") == ("Sun", ["Venus"])
    assert resolve_bodies(three_body_traj, [2, "Mercury"], "Sun") == ("Sun", ["Venus", "Mercury"])
    with pytest.raises(KeyError):
        resolve_bodies(circular_traj, ["Pluto"], "Sun")
    with pytest.raises(KeyError):
        resolve_bodies(circular_traj, None, "Vulcan")


def test_cube_limits_equal_extent():
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
    xl, yl, zl = cube_limits(pts, pad=0.0)
    assert _span(xl) == _span(yl) == _span(zl) == pytest.approx(2.0)
    assert xl == pytest.approx((0.0, 2.0)) and yl == pytest.approx((-0.5, 1.5))


# --------------------------------------------------------------------------- 3-D orbits
def test_plot_orbits_3d_relative(circular_traj):
    ax = plot_orbits_3d(circular_traj)
    assert isinstance(ax, Axes3D)
    assert len(ax.lines) == 1 and ax.lines[0].get_label() == "Mercury"
    assert ax.lines[0].get_color() == BODY_COLORS["Mercury"]
    rel = circular_traj.relative("Mercury", "Sun")[0]
    x, y, z = ax.lines[0].get_data_3d()
    np.testing.assert_allclose(np.column_stack([x, y, z]), rel)
    assert len(ax.collections) == 1  # the central-body marker
    assert ax.get_legend() is None and "Mercury" in ax.get_title()
    assert _span(ax.get_xlim()) == pytest.approx(_span(ax.get_ylim()))
    assert _span(ax.get_xlim()) == pytest.approx(_span(ax.get_zlim()))
    assert _span(ax.get_xlim()) > 2 * 0.387  # encloses the orbit


def test_plot_orbits_3d_barycentric_draws_central_path(circular_traj):
    ax = plot_orbits_3d(circular_traj, relative=False)
    assert len(ax.lines) == 2
    assert [ln.get_label() for ln in ax.lines] == ["Mercury", "Sun"]
    assert ax.get_legend() is not None and ax.get_title() == ""
    x, _, _ = ax.lines[0].get_data_3d()
    np.testing.assert_allclose(x, circular_traj.body("Mercury")[0][:, 0])


def test_plot_orbits_3d_subset_and_given_ax(three_body_traj):
    import matplotlib.pyplot as plt

    ax = plot_orbits_3d(three_body_traj)
    assert len(ax.lines) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Mercury", "Venus"]

    fig = plt.figure()
    ax2 = fig.add_subplot(projection="3d")
    out = plot_orbits_3d(three_body_traj, bodies=["Venus"], central=0, ax=ax2, equal_aspect=False)
    assert out is ax2 and len(ax2.lines) == 1
    assert ax2.lines[0].get_color() == BODY_COLORS["Venus"]  # Venus stays orange when alone

    with pytest.raises(TypeError):
        plot_orbits_3d(three_body_traj, ax=fig.add_subplot())


def test_plot_orbits_3d_repeated_calls_keep_cube(three_body_traj):
    ax = plot_orbits_3d(three_body_traj, bodies=["Mercury"])
    plot_orbits_3d(three_body_traj, bodies=["Venus"], ax=ax)
    assert len(ax.lines) == 2
    assert _span(ax.get_xlim()) == pytest.approx(_span(ax.get_zlim()))
    assert _span(ax.get_xlim()) > 2 * 0.723
    assert ax.get_legend() is not None and ax.get_title() == ""


# --------------------------------------------------------------------------- 2-D orbits
def test_plot_orbits_2d(circular_traj, three_body_traj):
    ax = plot_orbits_2d(circular_traj)
    assert isinstance(ax, Axes) and not isinstance(ax, Axes3D)
    assert ax.get_aspect() == 1.0
    assert len(ax.lines) == 1 and len(ax.collections) == 1
    rel = circular_traj.relative("Mercury", "Sun")[0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), rel[:, 0])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), rel[:, 1])
    assert ax.get_xlabel() == "x [AU]" and ax.get_ylabel() == "y [AU]"

    ax3 = plot_orbits_2d(three_body_traj, relative=False)
    assert len(ax3.lines) == 3 and ax3.get_legend() is not None

    with pytest.raises(TypeError):
        plot_orbits_2d(circular_traj, ax=plot_orbits_3d(circular_traj))


# --------------------------------------------------------------------------- energy error
def test_plot_energy_error_overlays_series():
    t = np.linspace(0.0, 1000.0, 200)
    err = 1e-12 * np.sin(t / 50.0) - 1e-13 * t / 1000.0
    err[0] = 0.0  # the initial sample is exactly zero in practice
    ax = plot_energy_error(t, err, label="leapfrog")
    assert isinstance(ax, Axes) and ax.get_yscale() == "log"
    assert len(ax.lines) == 1
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), np.abs(err))
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), t)

    out = plot_energy_error(t, 10 * err, label="rk4", ax=ax)
    assert out is ax and len(ax.lines) == 2
    assert [ln.get_color() for ln in ax.lines] == [PALETTE[0], PALETTE[1]]
    assert [tx.get_text() for tx in ax.get_legend().get_texts()] == ["leapfrog", "rk4"]
    ax.figure.canvas.draw()  # log scale with a zero sample must render


def test_plot_energy_error_linear_and_units():
    t = np.linspace(0.0, 2 * JULIAN_YEAR_DAYS, 50)
    err = np.linspace(-1e-9, 1e-9, 50)
    ax = plot_energy_error(t, err, logy=False, t_units="years")
    assert ax.get_yscale() == "linear" and ax.get_legend() is None
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), err)  # signed, not abs
    np.testing.assert_allclose(ax.lines[0].get_xdata()[-1], 2.0)
    assert ax.get_xlabel() == "t [years]"
    with pytest.raises(ValueError):
        plot_energy_error(t, err[:-1])
    with pytest.raises(ValueError):
        plot_energy_error(t, err, t_units="fortnights")


# --------------------------------------------------------------------------- precession
def test_precession_series_unwraps_and_converts():
    rate = 1.0e-3  # rad/day: wraps several times across the span
    t = np.linspace(0.0, 2.0e4, 400)
    varpi = (3.0 + rate * t + math.pi) % (2.0 * math.pi) - math.pi  # wrapped to (-pi, pi]
    s = precession_series(t, varpi, rate)
    np.testing.assert_allclose(s.t, t / JULIAN_YEAR_DAYS)
    np.testing.assert_allclose(s.dvarpi, rate * t * ARCSEC_PER_RAD, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(s.fit, s.dvarpi, rtol=1e-12, atol=1e-6)  # exact line, zero offset
    rate_cy = rate * JULIAN_CENTURY_DAYS * ARCSEC_PER_RAD
    assert s.rate_label == f"fit: {rate_cy:.4g} arcsec/century"
    assert s.t_label == "t [years]" and s.y_label == "Δϖ [arcsec]"

    s2 = precession_series(t, varpi, None, units="deg", t_units="days")
    assert s2.fit is None and s2.rate_label is None
    np.testing.assert_allclose(s2.t, t)
    np.testing.assert_allclose(s2.dvarpi, rate * t * DEG_PER_RAD, rtol=1e-12, atol=1e-9)


def test_plot_precession_mercury_like():
    rate = 42.98 / ARCSEC_PER_RAD / JULIAN_CENTURY_DAYS  # rad/day
    t = np.linspace(0.0, JULIAN_CENTURY_DAYS, 500)
    varpi = 1.35 + rate * t + 2e-7 * np.sin(t / 88.0)
    varpi_copy = varpi.copy()
    ax = plot_precession(t, varpi, rate, label="1PN")
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 2
    np.testing.assert_array_equal(varpi, varpi_copy)  # input untouched
    series, fit = ax.lines
    np.testing.assert_allclose(series.get_xdata()[-1], 100.0)
    np.testing.assert_allclose(series.get_ydata()[-1], 42.98, atol=0.1)
    assert fit.get_linestyle() == "--" and fit.get_label().startswith("fit: 42.98")
    assert [tx.get_text() for tx in ax.get_legend().get_texts()] == ["1PN", fit.get_label()]
    assert ax.get_xlabel() == "t [years]" and ax.get_ylabel() == "Δϖ [arcsec]"

    # a second series on the same axes gets the next colour, the fit line consumes no slot
    plot_precession(t, varpi * 0.5, ax=ax, label="Newton")
    assert ax.lines[2].get_color() == PALETTE[1]

    ax2 = plot_precession(t, varpi, units="deg", t_units="days")
    assert ax2.get_ylabel() == "Δϖ [deg]" and ax2.get_xlabel() == "t [days]"
    assert len(ax2.lines) == 1 and ax2.get_legend() is None


# --------------------------------------------------------------------------- effective potential
def _potentials(gm: float, c: float, ell: float, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v_newton = -gm / r + ell**2 / (2.0 * r**2)
    v_gr = v_newton - gm * ell**2 / (c**2 * r**3)
    return v_gr, v_newton


def test_plot_effective_potential():
    gm, c = GM_SUN, C_AU_DAY / 1000.0  # exaggerated field so the barrier is visible
    rg = gm / c**2
    ell = 4.0 * gm / c  # above sqrt(12) gm/c: a barrier at r = 4 rg and a well at 12 rg
    r = np.linspace(2.2 * rg, 60.0 * rg, 500)
    v_gr, v_newton = _potentials(gm, c, ell, r)
    marks = {"photon sphere": 3.0 * rg, "ISCO": 6.0 * rg}
    ax = plot_effective_potential(r, v_gr, v_newton, marks=marks)
    assert isinstance(ax, Axes)
    assert len(ax.lines) == 4  # two curves + two vertical marks
    assert [ln.get_label() for ln in ax.lines[:2]] == ["Schwarzschild", "Newton"]
    assert sorted(tx.get_text() for tx in ax.texts) == ["ISCO", "photon sphere"]
    lo, hi = ax.get_ylim()
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo > v_gr.min()  # the r -> 0 dive is clipped ...
    inner = (r > 3.0 * rg) & (r < 6.0 * rg)  # the barrier sits at r = 4 rg (4.25 rg below)
    top, well = v_gr[inner].max(), v_gr[r > 4.0 * rg].min()
    assert lo < well < top < hi  # ... but the barrier top and the well are in view
    # The GR features fill the panel (15 % pad each side -> 1/1.3 of the axis); Newton's
    # steep L^2/(2 r^2) rise at the barrier radius must not squash them into a sliver.
    assert (top - well) / (hi - lo) > 0.7
    assert ax.get_legend() is not None

    # the default range is set by the Schwarzschild curve alone
    ax_gr = plot_effective_potential(r, v_gr)
    assert ax_gr.get_ylim() == pytest.approx((lo, hi))

    # barrier top below the asymptote (sqrt(12) < L c / GM < 4) is still found and framed
    v_low, v_low_n = _potentials(gm, c, 3.8 * gm / c, r)
    ax_low = plot_effective_potential(r, v_low, v_low_n)
    lo_l, hi_l = ax_low.get_ylim()
    top_l, well_l = v_low[inner].max(), v_low[r > 4.5 * rg].min()
    assert top_l < v_low[-1] < 0.0  # the premise: the barrier is lower than the tail
    assert lo_l > v_low.min() and lo_l < well_l < top_l < hi_l
    assert (v_low[-1] - well_l) / (hi_l - lo_l) > 0.7

    ax1 = plot_effective_potential(r, v_gr, ylim=(-1.0, 1.0), xlabel="r [GM/c²]")
    assert len(ax1.lines) == 1 and ax1.get_ylim() == (-1.0, 1.0)
    assert ax1.get_legend() is None and ax1.get_xlabel() == "r [GM/c²]"

    # no barrier (small L): the heuristic steps aside and autoscale applies
    v_gr_nb, _ = _potentials(gm, c, 2.0 * gm / c, r)
    ax2 = plot_effective_potential(r, v_gr_nb)
    assert ax2.get_ylim()[0] <= v_gr_nb.min()

    with pytest.raises(ValueError):
        plot_effective_potential(r, v_gr, v_newton[:-1])


# --------------------------------------------------------------------------- separation
def test_plot_separation_units():
    t = np.linspace(0.0, 10 * JULIAN_YEAR_DAYS, 60)
    sep = np.linspace(0.0, 2.0e-6, 60)
    ax = plot_separation(t, sep, label="Newton vs 1PN")
    assert isinstance(ax, Axes) and len(ax.lines) == 1
    np.testing.assert_allclose(ax.lines[0].get_ydata(), sep * AU_KM)
    np.testing.assert_allclose(ax.lines[0].get_xdata()[-1], 10.0)
    assert ax.get_ylabel() == "separation [km]" and ax.get_xlabel() == "t [years]"
    assert [tx.get_text() for tx in ax.get_legend().get_texts()] == ["Newton vs 1PN"]

    ax2 = plot_separation(t, sep, units="AU", t_units="days")
    np.testing.assert_allclose(ax2.lines[0].get_ydata(), sep)
    np.testing.assert_allclose(ax2.lines[0].get_xdata(), t)
    assert ax2.get_ylabel() == "separation [AU]" and ax2.get_legend() is None
    with pytest.raises(ValueError):
        plot_separation(t, sep, units="parsec")


# --------------------------------------------------------------------------- grids
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7])
def test_figure_grid(n):
    fig, axes = figure_grid(n)
    assert isinstance(fig, Figure)
    assert isinstance(axes, np.ndarray) and axes.shape == (n,)
    assert all(isinstance(ax, Axes) for ax in axes)
    assert len(fig.axes) == n


def test_figure_grid_3d_and_errors():
    fig, axes = figure_grid(2, projection="3d", figsize=(8.0, 4.0))
    assert all(isinstance(ax, Axes3D) for ax in axes)
    assert fig.get_size_inches() == pytest.approx((8.0, 4.0))
    with pytest.raises(ValueError):
        figure_grid(0)


# --------------------------------------------------------------------------- global behaviour
def test_functions_never_call_show(monkeypatch, circular_traj):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: pytest.fail("plt.show() was called"))
    t = circular_traj.t
    plot_orbits_3d(circular_traj)
    plot_orbits_2d(circular_traj)
    plot_energy_error(t, 1e-12 * np.ones_like(t))
    plot_precession(t, 0.1 + 1e-9 * t, 1e-9)
    r = np.linspace(1.0, 10.0, 50)
    plot_effective_potential(r, -1.0 / r, -0.9 / r, marks={"x": 3.0})
    plot_separation(t, 1e-9 * t)
    figure_grid(3)


def test_inputs_are_not_mutated(circular_traj):
    t = circular_traj.t.copy()
    pos = circular_traj.pos.copy()
    plot_orbits_3d(circular_traj)
    plot_orbits_2d(circular_traj, relative=False)
    np.testing.assert_array_equal(circular_traj.t, t)
    np.testing.assert_array_equal(circular_traj.pos, pos)
    err = np.linspace(-1e-9, 1e-9, t.size)
    err_copy = err.copy()
    plot_energy_error(t, err)
    np.testing.assert_array_equal(err, err_copy)
    sep = np.abs(err)
    sep_copy = sep.copy()
    plot_separation(t, sep)
    np.testing.assert_array_equal(sep, sep_copy)


def test_axes_facecolor_and_chrome(circular_traj):
    ax = plot_orbits_2d(circular_traj)
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == plots.SURFACE
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    ax3 = plot_orbits_3d(circular_traj)
    assert ax3.collections[0].get_facecolor().tolist()[0][:3] == pytest.approx(
        matplotlib.colors.to_rgb(INK)
    )


# --------------------------------------------------------------------------- plotly
def test_interactive_import_error_mentions_pip(monkeypatch, circular_traj):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    with pytest.raises(ImportError, match="pip install plotly"):
        interactive.orbits_3d_figure(circular_traj)
    with pytest.raises(ImportError, match="pip install plotly"):
        interactive.precession_figure(circular_traj.t, 0.1 + 1e-9 * circular_traj.t)


def test_orbits_3d_figure(three_body_traj):
    go = pytest.importorskip("plotly.graph_objects")
    fig = interactive.orbits_3d_figure(three_body_traj)
    assert isinstance(fig, go.Figure)
    assert [tr.name for tr in fig.data] == ["Mercury", "Venus", "Sun"]
    assert fig.data[0].mode == "lines" and fig.data[-1].mode == "markers"
    assert fig.data[0].line.color == BODY_COLORS["Mercury"]
    scene = fig.layout.scene
    assert scene.aspectmode == "cube"
    xr, yr, zr = scene.xaxis.range, scene.yaxis.range, scene.zaxis.range
    assert _span(xr) == pytest.approx(_span(yr)) == pytest.approx(_span(zr))
    assert _span(xr) > 2 * 0.723
    np.testing.assert_allclose(np.asarray(fig.data[1].x), three_body_traj.body("Venus")[0][:, 0])

    fig2 = interactive.orbits_3d_figure(three_body_traj, bodies=["Venus"], relative=False)
    assert [tr.name for tr in fig2.data] == ["Venus", "Sun (path)", "Sun"]


def test_precession_figure():
    go = pytest.importorskip("plotly.graph_objects")
    rate = 42.98 / ARCSEC_PER_RAD / JULIAN_CENTURY_DAYS
    t = np.linspace(0.0, JULIAN_CENTURY_DAYS, 300)
    varpi = 1.35 + rate * t
    fig = interactive.precession_figure(t, varpi, rate)
    assert isinstance(fig, go.Figure) and len(fig.data) == 2
    assert fig.data[1].name.startswith("fit: 42.98") and fig.data[1].line.dash == "dash"
    np.testing.assert_allclose(np.asarray(fig.data[0].y)[-1], 42.98, atol=1e-6)
    assert fig.layout.xaxis.title.text == "t [years]"
    assert fig.layout.yaxis.title.text == "Δϖ [arcsec]"
    assert len(interactive.precession_figure(t, varpi).data) == 1
