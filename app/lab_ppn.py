"""PPN playground (Streamlit lab 5): Sun + Mercury under the PPN-parametrised EIH equations.

Physics
-------
In the parametrised post-Newtonian framework every metric theory of gravity has, around a
static mass and to the order that matters in the solar system, the metric
``g_00 = -(1 - 2U + 2 beta U^2)``, ``g_ij = (1 + 2 gamma U) delta_ij`` with ``U = GM/(c^2 r)``:
``gamma`` is the space curvature per unit mass and ``beta`` the non-linearity of the
superposition of gravity (General Relativity: ``beta = gamma = 1``; Brans-Dicke:
``gamma = (1 + omega)/(2 + omega)``, ``beta = 1``).  The equations of motion are the EIH
equations with PPN coefficients (``gravsim.forces.eih.make_eih(c, beta, gamma)``) and the
secular advance of the perihelion is

    dvarpi = (2 + 2 gamma - beta)/3 * 6 pi GM / (c^2 a (1 - e^2))   per orbit

(Will 2014, Living Rev. Relativ. 17:4, eq. 68; derivation in ``docs/theory/ppn.md``), so
Mercury measures the single combination ``2 gamma - beta``.  The lab integrates Sun + Mercury
with DOP853 twice on one grid -- Newton, and the PPN-EIH law with the chosen ``beta``,
``gamma`` and speed of light ``c = C_AU_DAY / c_scale`` -- and fits the secular rate of the
differenced perihelion longitude (``analysis.compare``), which cancels the integrator error
and leaves the PPN advance; the result is compared with the formula for the same ``c`` and
with the GR value.

Experimental bounds drawn in the lab (1 sigma): Cassini 2003 Shapiro delay,
``gamma - 1 = (2.1 +- 2.3)e-5`` (Bertotti, Iess & Tortora 2003); lunar laser ranging, Nordtvedt
parameter ``eta = 4 beta - gamma - 3 = (4.4 +- 4.5)e-4`` (Williams, Turyshev & Boggs 2004);
Mercury's perihelion, ``2 gamma - beta - 1 = 0 +- 8e-5`` with the solar J2 from helioseismology
(Will 2014, Table 4).

Structure
---------
The back end (``ppn_factor``, ``ppn_runs``, ``ppn_precession``, ``factor_map``,
``experimental_bounds``, ``bound_band``, ``sigmas_from_gr``) is pure: plain numbers in, numpy
arrays and frozen dataclasses out, no Streamlit (``tests/test_lab_ppn.py``).  ``render()``
adds the widgets, ``st.cache_data`` memoisation and the plotly figures in the idiom of
``streamlit_app.py``, which calls it for this lab.  streamlit and plotly are optional (the
``app`` extra): the back end imports without them.

Units: AU, day, GM in AU^3/day^2, ``c`` in AU/day, rates in arcsec per Julian century.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gravsim import constants as C
from gravsim.analysis import (
    fit_secular_rate,
    gr_precession_arcsec_per_century,
    perihelion_longitude_series,
    state_to_elements,
    wrap_angle,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.data.horizons import load_j2000
from gravsim.forces import accel_newton, make_eih
from gravsim.integrators import integrate_scipy

try:  # front-end dependencies (the ``app`` extra); the back end does not need them
    import plotly.graph_objects as go
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only without the app extra
    go = st = None

C_SCALES: dict[str, float] = {
    "c (reality)": 1.0,
    "c / 10": 10.0,
    "c / 100": 100.0,
    "c / 1000": 1000.0,
}
BETA_RANGE: tuple[float, float] = (0.0, 2.0)
GAMMA_RANGE: tuple[float, float] = (0.0, 2.0)
BOUND_HALF_WIDTH = 6.0e-4  # extent of the bounds chart in (beta - 1), (gamma - 1)


# --------------------------------------------------------------------------- helpers
def ppn_factor(beta, gamma):
    """The PPN perihelion factor ``(2 + 2 gamma - beta)/3`` (1 in GR); accepts arrays."""
    out = (2.0 + 2.0 * np.asarray(gamma, dtype=np.float64) - np.asarray(beta, dtype=np.float64))
    out = out / 3.0
    return float(out) if out.ndim == 0 else out


def two_body_state() -> SystemState:
    """Sun + Mercury from the J2000 snapshot, in their own barycentric frame."""
    return load_j2000().subset(("Sun", "Mercury")).to_barycentric()


def daily_grid(t0: float, years: float, step: float = 1.0) -> np.ndarray:
    """Output times ``t0, t0 + step, ...`` spanning ``years`` Julian years."""
    n = int(round(years * C.JULIAN_YEAR_DAYS / step))
    return t0 + step * np.arange(n + 1)


def osculating(state: SystemState, body: str, central: str = "Sun") -> dict[str, float]:
    """Osculating ``a, e, inc, period, gm`` of ``body`` about ``central``."""
    r, v = state.relative_to(central)
    i = state.index(body)
    gm = float(state.gm[state.index(central)] + state.gm[i])
    el = state_to_elements(r[i], v[i], gm)
    return {k: float(el[k]) for k in ("a", "e", "inc", "period")} | {"gm": gm}


# --------------------------------------------------------------------------- runs
@dataclass(frozen=True)
class PPNRuns:
    """Newton and PPN-EIH runs of Sun + Mercury on one grid, with the theory that made them."""

    newton: Trajectory
    ppn: Trajectory
    beta: float
    gamma: float
    c: float  # AU/day


def ppn_runs(
    beta: float, gamma: float, c_scale: float = 1.0, years: float = 10.0, step: float = 1.0
) -> PPNRuns:
    """Integrate Sun + Mercury with Newton and with the PPN-EIH law (DOP853, common grid).

    Parameters
    ----------
    beta, gamma : float
        PPN parameters (GR: 1, 1).
    c_scale : float
        The speed of light used is ``C_AU_DAY / c_scale`` (1 = reality, 100 = visible rosette).
    years : float
        Span in Julian years.
    step : float
        Sample spacing [day].

    Returns
    -------
    PPNRuns

    Raises
    ------
    ValueError
        If ``c_scale``, ``years`` or ``step`` is not positive.
    """
    if c_scale <= 0.0:
        raise ValueError(f"c_scale must be positive, got {c_scale}")
    if years <= 0.0 or step <= 0.0:
        raise ValueError(f"years and step must be positive, got {years} and {step}")
    state = two_body_state()
    t_eval = daily_grid(state.t, years, step)
    c = C.C_AU_DAY / c_scale
    newton = integrate_scipy(state, accel_newton, t_eval)
    ppn = integrate_scipy(state, make_eih(c=c, beta=beta, gamma=gamma), t_eval)
    return PPNRuns(newton=newton, ppn=ppn, beta=float(beta), gamma=float(gamma), c=c)


@dataclass(frozen=True)
class PPNPrecession:
    """Differenced perihelion longitude of Mercury, its fitted rate and the predictions."""

    t: np.ndarray
    difference: np.ndarray  # wrapped varpi_ppn - varpi_newton [rad]
    rate: float  # arcsec per Julian century
    rate_err: float
    predicted: float  # PPN formula with the run's c, beta, gamma
    gr: float  # the same with beta = gamma = 1
    factor: float  # (2 + 2 gamma - beta)/3
    v_over_c: float  # Mercury at J2000, with the run's c
    elements: dict[str, float]


def ppn_precession(runs: PPNRuns) -> PPNPrecession:
    """Fit the PPN advance of Mercury in ``runs`` and compare it with the analytic rates."""
    t, w_n = perihelion_longitude_series(runs.newton, "Mercury")
    _, w_p = perihelion_longitude_series(runs.ppn, "Mercury")
    difference = wrap_angle(w_p - w_n)
    fit = fit_secular_rate(t, difference)
    state = runs.newton.state_at(0)
    el = osculating(state, "Mercury")
    predicted = gr_precession_arcsec_per_century(
        el["a"], el["e"], el["gm"], c=runs.c, beta=runs.beta, gamma=runs.gamma
    )
    gr = gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"], c=runs.c)
    _, v = state.relative_to("Sun")
    return PPNPrecession(
        t=t,
        difference=difference,
        rate=float(fit.rate_arcsec_per_century),
        rate_err=float(fit.rate_err_arcsec_per_century),
        predicted=float(predicted),
        gr=float(gr),
        factor=ppn_factor(runs.beta, runs.gamma),
        v_over_c=float(np.linalg.norm(v[state.index("Mercury")]) / runs.c),
        elements=el,
    )


# --------------------------------------------------------------------------- the plane
def factor_map(
    beta_range: tuple[float, float] = BETA_RANGE,
    gamma_range: tuple[float, float] = GAMMA_RANGE,
    n: int = 81,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the factor ``(2 + 2 gamma - beta)/3`` on an ``n x n`` grid of the plane.

    Returns
    -------
    betas, gammas : ndarray, shape (n,)
    factor : ndarray, shape (n, n)
        ``factor[j, i]`` belongs to ``(betas[i], gammas[j])`` (the meshgrid convention).
    """
    betas = np.linspace(beta_range[0], beta_range[1], n)
    gammas = np.linspace(gamma_range[0], gamma_range[1], n)
    bb, gg = np.meshgrid(betas, gammas)
    return betas, gammas, ppn_factor(bb, gg)


@dataclass(frozen=True)
class Bound:
    """A solar-system constraint ``a (beta - 1) + b (gamma - 1) = value +- sigma``."""

    name: str
    a: float
    b: float
    value: float
    sigma: float

    def residual(self, beta: float, gamma: float) -> float:
        """``a (beta - 1) + b (gamma - 1) - value``."""
        return self.a * (beta - 1.0) + self.b * (gamma - 1.0) - self.value

    def sigmas(self, beta: float, gamma: float) -> float:
        """The residual in units of ``sigma`` (how many sigma the theory is from the bound)."""
        return self.residual(beta, gamma) / self.sigma


def experimental_bounds() -> tuple[Bound, ...]:
    """The three bounds of the module docstring, in the order Cassini, LLR, Mercury."""
    return (
        Bound("Cassini 2003 (Shapiro delay): γ − 1", 0.0, 1.0, 2.1e-5, 2.3e-5),
        Bound("Lunar laser ranging: η = 4β − γ − 3", 4.0, -1.0, 4.4e-4, 4.5e-4),
        Bound("Mercury's perihelion: 2γ − β − 1", -1.0, 2.0, 0.0, 8.0e-5),
    )


def bound_band(
    bound: Bound, half_width: float = BOUND_HALF_WIDTH, n: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ``+- sigma`` band of a bound as ``x, y_low, y_high`` in the (beta - 1, gamma - 1) plane.

    Raises
    ------
    ValueError
        If ``bound.b == 0`` (a vertical band cannot be written as ``y(x)``).
    """
    if bound.b == 0.0:
        raise ValueError(f"{bound.name}: b = 0, the band is vertical")
    x = np.linspace(-half_width, half_width, n)
    y_low = (bound.value - bound.sigma - bound.a * x) / bound.b
    y_high = (bound.value + bound.sigma - bound.a * x) / bound.b
    return x, np.minimum(y_low, y_high), np.maximum(y_low, y_high)


def sigmas_from_gr(beta: float, gamma: float) -> dict[str, float]:
    """How many sigma a theory ``(beta, gamma)`` is from each experimental bound."""
    return {b.name: b.sigmas(beta, gamma) for b in experimental_bounds()}


# --------------------------------------------------------------------------- cached runs
def _cached(message: str) -> Callable:
    """``st.cache_data(show_spinner=message)``, or the identity when streamlit is absent."""
    if st is None:
        return lambda fn: fn
    return st.cache_data(show_spinner=message)


@_cached("integrating Sun + Mercury with the PPN-EIH law and with Newton ...")
def _ppn_lab(beta: float, gamma: float, c_scale: float, years: float):
    runs = ppn_runs(beta, gamma, c_scale, years)
    return runs, ppn_precession(runs)


# --------------------------------------------------------------------------- front end
def _factor_figure(beta: float, gamma: float):
    betas, gammas, factor = factor_map()
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=betas,
            y=gammas,
            z=factor,
            colorscale="Viridis",
            contours={"start": 0.0, "end": 2.0, "size": 0.1},
            colorbar={"title": "(2 + 2γ − β)/3"},
            hovertemplate="β = %{x:.2f}<br>γ = %{y:.2f}<br>factor = %{z:.3f}<extra></extra>",
        )
    )
    line_b = np.array(BETA_RANGE)
    fig.add_trace(
        go.Scatter(
            x=line_b,
            y=(1.0 + line_b) / 2.0,
            mode="lines",
            name="2γ − β = 1: same advance as GR",
            line={"color": "white", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1.0],
            y=[1.0],
            mode="markers",
            name="General Relativity",
            marker={"size": 12, "color": "white", "symbol": "star", "line": {"color": "black"}},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[beta],
            y=[gamma],
            mode="markers",
            name=f"this theory: ×{ppn_factor(beta, gamma):.3f}",
            marker={"size": 12, "color": "#d62728", "symbol": "x", "line": {"color": "white"}},
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="β (non-linearity of superposition)",
        yaxis_title="γ (space curvature per unit mass)",
        height=440,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
    )
    return fig


def _bounds_figure(beta: float, gamma: float, palette: tuple[str, ...], ink: str):
    scale = 1.0e4
    fig = go.Figure()
    for k, bound in enumerate(experimental_bounds()):
        x, lo, hi = bound_band(bound)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]) * scale,
                y=np.concatenate([lo, hi[::-1]]) * scale,
                fill="toself",
                mode="lines",
                name=f"{bound.name} = ({bound.value * 1e4:+.2f} ± {bound.sigma * 1e4:.2f}) × 10⁻⁴",
                line={"color": palette[k], "width": 1},
                opacity=0.35,
                hoverinfo="name",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0], y=[0.0], mode="markers", name="General Relativity",
            marker={"size": 14, "color": ink, "symbol": "star"},
        )
    )
    if abs(beta - 1.0) <= BOUND_HALF_WIDTH and abs(gamma - 1.0) <= BOUND_HALF_WIDTH:
        fig.add_trace(
            go.Scatter(
                x=[(beta - 1.0) * scale], y=[(gamma - 1.0) * scale], mode="markers",
                name="this theory", marker={"size": 12, "color": "#d62728", "symbol": "x"},
            )
        )
    ext = BOUND_HALF_WIDTH * scale
    fig.update_layout(
        template="plotly_white",
        xaxis={"title": "(β − 1) × 10⁴", "range": [-ext, ext]},
        yaxis={"title": "(γ − 1) × 10⁴", "range": [-ext, ext], "scaleanchor": "x"},
        height=440,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
    )
    return fig


def render() -> None:
    """Draw the PPN playground (needs streamlit and plotly; called by ``streamlit_app.py``)."""
    if st is None or go is None:
        raise ImportError("the PPN lab needs streamlit and plotly: uv pip install -e '.[app]'")
    from gravsim.viz import PALETTE, orbits_3d_figure, precession_figure
    from gravsim.viz.plots import INK

    st.header("PPN playground: what if gravity were a different theory?")
    st.markdown(
        "Every metric theory of gravity has the same Newtonian limit; at the next order it is "
        "described by $\\gamma$ (space curvature per unit mass) and $\\beta$ (non-linearity of "
        "superposition), with $\\beta = \\gamma = 1$ in General Relativity. Sun + Mercury is "
        "integrated with the PPN-parametrised Einstein–Infeld–Hoffmann equations and with "
        "Newton on one grid; the differenced perihelion longitude gives the advance, which the "
        "theory predicts as $\\tfrac{2+2\\gamma-\\beta}{3}\\,6\\pi GM/(c^2a(1-e^2))$ per orbit "
        "(Lessons 06–07)."
    )
    c1, c2, c3, c4 = st.columns(4)
    beta = c1.slider("β (non-linearity)", BETA_RANGE[0], BETA_RANGE[1], 1.0, step=0.05)
    gamma = c2.slider("γ (space curvature)", GAMMA_RANGE[0], GAMMA_RANGE[1], 1.0, step=0.05)
    c_label = c3.selectbox("speed of light", list(C_SCALES), index=0, key="c_ppn")
    c_scale = C_SCALES[c_label]
    years = c4.slider("span [years]", 2.0, 30.0, 10.0, step=1.0)
    runs, res = _ppn_lab(beta, gamma, c_scale, years)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("fit, PPN − Newton", f"{res.rate:.4g} ± {res.rate_err:.2g} ″/cy")
    m2.metric("PPN prediction (same c)", f"{res.predicted:.4g} ″/cy")
    m3.metric("General Relativity (same c)", f"{res.gr:.4g} ″/cy")
    m4.metric("(2 + 2γ − β)/3", f"{res.factor:.3f}")

    units = "deg" if c_scale >= 100.0 else "arcsec"
    to_units = C.DEG_PER_RAD if units == "deg" else C.ARCSEC_PER_RAD
    left, right = st.columns(2)
    with left:
        fig = orbits_3d_figure(runs.ppn, ["Mercury"])
        fig.data[0].update(name=f"Mercury, β = {beta:g}, γ = {gamma:g}")
        for tr in orbits_3d_figure(runs.newton, ["Mercury"]).data[:-1]:
            tr.update(name="Mercury (Newton)", line={"dash": "dot", "width": 2}, opacity=0.8)
            fig.add_trace(tr)
        fig.update_layout(height=480, title=f"Sun + Mercury, {years:g} yr, {c_label}")
        st.plotly_chart(fig, width="stretch")
    with right:
        fit_rate = res.rate / (C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS)
        fig = precession_figure(res.t, res.difference, fit_rate, units=units)
        fig.data[0].update(name="ϖ(PPN) − ϖ(Newton)")
        pred_rate = res.predicted / (C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS)
        t_yr = (res.t - res.t[0]) / C.JULIAN_YEAR_DAYS
        fig.add_trace(
            go.Scatter(
                x=t_yr,
                y=pred_rate * (res.t - res.t[0]) * to_units,
                mode="lines",
                name=f"prediction: {res.predicted:.4g} {units}/century",
                line={"color": PALETTE[1], "width": 1, "dash": "dot"},
            )
        )
        fig.update_layout(title="the PPN advance, measured and predicted", height=480)
        st.plotly_chart(fig, width="stretch")
    el = res.elements
    st.caption(
        f"Osculating elements at J2000: a = {el['a']:.4f} AU, e = {el['e']:.4f}, "
        f"P = {el['period']:.2f} d; v/c of Mercury = {res.v_over_c:.3g}. The fit falls short of "
        "the formula by a few tenths of a percent at c/100 (second order in GM/(c²p), Lesson 04); "
        "the 1PN expansion stops being trustworthy at c/1000."
    )

    st.subheader("Where the theory sits")
    sig = sigmas_from_gr(beta, gamma)
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(_factor_figure(beta, gamma), width="stretch")
        st.caption(
            "Mercury measures only 2γ − β: every theory on the white line gives the same 43″ "
            "per century. γ counts twice (velocity-dependent force and the GM/r term) and β "
            "once, negatively."
        )
    with bottom_right:
        st.plotly_chart(_bounds_figure(beta, gamma, PALETTE, INK), width="stretch")
        verdict = ", ".join(f"{name.split(':')[0]} {s:+.3g} σ" for name, s in sig.items())
        st.caption(
            f"This theory versus experiment (1σ bands, chart ±{BOUND_HALF_WIDTH * 1e4:g} × 10⁻⁴): "
            f"{verdict}. Cassini fixes γ alone, lunar laser ranging 4β − γ, Mercury 2γ − β; "
            "any two of them pin both parameters."
        )
