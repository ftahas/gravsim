"""gravsim interactive lab (Streamlit): integrators, Newton vs GR in 3-D, precession, potentials.

Run from the repository root with the project environment::

    .venv/bin/streamlit run app/streamlit_app.py

Needs the ``app`` extra (``uv pip install -e ".[app]"``: streamlit + plotly).  All physics is in
``labs.py``, ``lab_ppn.py``, ``lab_light.py`` and ``lab_extras.py`` next to this file and in
the ``gravsim`` package; this file only adds widgets, ``st.cache_data`` memoisation of the runs
and plotly figures (labs 5, 6 and 7 bring their own ``render()`` in the same idiom).  Orbits
are rendered in 3-D (``gravsim.viz.interactive``); measurement plots (precession, energy
error, potentials, deflection, delay) are 2-D.

The seven labs correspond to lessons 01-09 of ``notebooks/``:

1. Integrator lab -- choose a fixed-step method and a step, watch the energy error and the
   spurious perihelion precession against the 43 arcsec/century GR signal.
2. Solar system -- Sun + 8 planets, Newton/GR toggle, speed-of-light slider, and the
   GR-minus-Newton displacement amplified so that it is visible.
3. Precession tracker -- live varpi(t) of a planet in both runs with the fitted rates.
4. Effective-potential explorer -- drag L and E, see the Schwarzschild orbit (precession,
   zoom-whirl, capture) in geometric units.
5. PPN playground (lessons 06-07) -- Sun + Mercury under the PPN-parametrised EIH equations,
   sliders for beta and gamma, the (2 + 2 gamma - beta)/3 factor and the experimental bounds.
6. Light-bending lab (lesson 08) -- ray bundles past the Sun, deflection versus impact
   parameter against the 1919-2004 measurements, capture, and the Shapiro delay at conjunction.
7. Competing effects (lesson 09) -- solar oblateness J2 and Lense-Thirring frame dragging,
   amplified, measured on the node and the argument of perihelion in the solar-equator frame;
   the Moon as a body (sidereal month, perigee and node rates); Mercury's precession budget.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gravsim import constants as C
from gravsim.viz import BODY_COLORS, PALETTE, orbits_3d_figure, precession_figure
from gravsim.viz.plots import INK

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import lab_extras  # noqa: E402  (sibling modules; the path above makes them importable)
import lab_light  # noqa: E402
import lab_ppn  # noqa: E402
import labs  # noqa: E402

LABS = (
    "1 · Integrator lab",
    "2 · Solar system: Newton vs GR",
    "3 · Precession tracker",
    "4 · Effective-potential explorer",
    "5 · PPN playground",
    "6 · Light-bending lab",
    "7 · Competing effects",
)
C_SCALES = {"c (reality)": 1.0, "c / 10": 10.0, "c / 100": 100.0, "c / 1000": 1000.0}


# --------------------------------------------------------------------------- cached runs
@st.cache_data(show_spinner="integrating Sun + Mercury ...")
def _integrator_lab(name: str, dt: float, years: float):
    return labs.integrator_lab(name, dt, years)


@st.cache_data(show_spinner="integrating the solar system (Newton and Newton + 1PN) ...")
def _solar_system_runs(years: float, c_scale: float, bodies: tuple[str, ...], step: float):
    return labs.solar_system_runs(years, c_scale, bodies, step)


@st.cache_data(show_spinner="tracking the perihelion ...")
def _precession_track(years: float, c_scale: float, body: str, bodies: tuple[str, ...]):
    newton, gr = labs.solar_system_runs(years, c_scale, bodies)
    return labs.precession_track(newton, gr, body, c_scale)


@st.cache_data(show_spinner="integrating the geodesic ...")
def _strong_field_orbit(L: float, energy: float, r0: float, tau_end: float, inward: bool):
    return labs.strong_field_orbit(L, energy, r0, tau_end, inward)


# --------------------------------------------------------------------------- figure helpers
def _line_figure(x, ys: dict[str, np.ndarray], x_title: str, y_title: str, log_y=False):
    fig = go.Figure()
    for k, (name, y) in enumerate(ys.items()):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line={"color": PALETTE[k % 8]}))
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        yaxis_type="log" if log_y else "linear",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        height=360,
    )
    return fig


def _ring(radius: float, n: int = 120):
    th = np.linspace(0.0, 2.0 * math.pi, n)
    return radius * np.cos(th), radius * np.sin(th), np.zeros(n)


# --------------------------------------------------------------------------- lab 1
def lab_integrators() -> None:
    st.header("Integrator lab: Sun + Mercury")
    st.markdown(
        "Every scheme integrates $\\ddot{\\mathbf r} = -GM\\,\\mathbf r/r^3$ for the Sun–Mercury "
        "pair. Two error meters: the relative energy error $(E-E_0)/|E_0|$ and the *spurious* "
        "precession of the perihelion (a Kepler orbit must not precess). Compare the latter with "
        f"the GR signal, **{C.MERCURY_PRECESSION_GR_ARCSEC_CY} arcsec/century** (Lesson 01)."
    )
    col1, col2, col3 = st.columns(3)
    name = col1.selectbox("integrator", labs.FIXED_STEP_NAMES, index=2)
    dt = col2.select_slider("step dt [days]", options=[4.0, 2.0, 1.0, 0.5, 0.25, 0.125], value=1.0)
    years = col3.slider("span [years]", 1.0, 30.0, 5.0, step=1.0)
    res = _integrator_lab(name, dt, years)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("steps per orbit", f"{res.steps_per_orbit:.0f}")
    m2.metric("max |ΔE/E₀|", f"{np.abs(res.energy_error).max():.1e}")
    m3.metric("spurious precession", f"{res.spurious_rate:+.1f} ″/cy")
    m4.metric("GR signal", f"{C.MERCURY_PRECESSION_GR_ARCSEC_CY:.2f} ″/cy")

    left, right = st.columns([1.1, 1.0])
    with left:
        fig = orbits_3d_figure(res.traj, ["Mercury"])
        fig.update_layout(
            height=480, title=f"{name}, dt = {res.traj.meta['dt']:.3f} d, {years:g} yr"
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        t_yr = (res.traj.t - res.traj.t[0]) / C.JULIAN_YEAR_DAYS
        err = np.abs(res.energy_error)
        err[err == 0.0] = np.nan
        st.plotly_chart(
            _line_figure(t_yr, {f"{name}": err}, "t [years]", "|ΔE/E₀|", log_y=True),
            width="stretch",
        )
        st.caption(
            "Symplectic schemes (symplectic_euler, leapfrog, yoshida4/6) keep the energy error "
            "bounded; euler and rk4 drift. The spurious precession scales as (2π/N)^order with "
            "N steps per orbit: halve dt and watch it fall by 2^order."
        )


# --------------------------------------------------------------------------- lab 2
def lab_solar_system() -> None:
    st.header("Solar system: Newton versus General Relativity")
    st.markdown(
        "Sun + 8 planets from the J2000 Horizons snapshot, integrated with DOP853 twice: Newton, "
        "and Newton + the 1PN correction of the Sun's field "
        "$\\frac{GM}{c^2 r^3}\\big[(4GM/r - v^2)\\,\\mathbf r "
        "+ 4(\\mathbf r\\cdot\\mathbf v)\\mathbf v\\big]$. "
        "At the real $c$ the two are indistinguishable by eye ($\\sim10^{-8}$), so the "
        "GR − Newton displacement of one body is drawn **amplified**, or $c$ is reduced."
    )
    c1, c2, c3, c4 = st.columns(4)
    years = c1.slider("span [years]", 1.0, 50.0, 5.0, step=1.0)
    c_label = c2.selectbox("speed of light", list(C_SCALES), index=0)
    c_scale = C_SCALES[c_label]
    show_gr = c3.toggle("draw the GR run", value=True)
    bodies = c4.multiselect("planets", list(C.PLANETS), default=list(labs.INNER_PLANETS))
    if not bodies:
        st.info("select at least one planet")
        return
    names = ("Sun",) + tuple(b for b in C.PLANETS if b in bodies)
    newton, gr = _solar_system_runs(years, c_scale, names, 1.0)

    d1, d2 = st.columns(2)
    body = d1.selectbox("body for the difference view", [b for b in names if b != "Sun"], index=0)
    default_amp = 5.0 if c_scale == 1.0 else 0.0
    amp_log = d2.slider("amplification of (GR − Newton), log10", 0.0, 8.0, default_amp, 0.5)
    factor = 10.0**amp_log
    r_n, r_g, r_amp = labs.amplified_difference(newton, gr, body, factor)
    sep = labs.separation_km(newton, gr, body)

    fig = orbits_3d_figure(newton, list(names[1:]))
    if show_gr:
        for tr in orbits_3d_figure(gr, list(names[1:])).data[:-1]:
            tr.update(name=f"{tr.name} (GR)", line={"dash": "dot", "width": 2}, opacity=0.8)
            fig.add_trace(tr)
    fig.add_trace(
        go.Scatter3d(
            x=r_amp[:, 0],
            y=r_amp[:, 1],
            z=r_amp[:, 2],
            mode="lines",
            name=f"{body}: Newton + {factor:g}×(GR − Newton)",
            line={"color": INK, "width": 3},
            hovertemplate="amplified GR displacement<extra></extra>",
        )
    )
    fig.update_layout(height=620, title=f"Orbits relative to the Sun, {years:g} yr, {c_label}")
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"|r_GR − r_Newton| of {body} after {years:g} yr", f"{sep[-1]:.3g} km")
    m2.metric("max over the span", f"{sep.max():.3g} km")
    m3.metric("v/c of Mercury at J2000", f"{0.0137 * c_scale:.3g}")
    st.plotly_chart(
        _line_figure(
            (newton.t - newton.t[0]) / C.JULIAN_YEAR_DAYS,
            {body: sep},
            "t [years]",
            "|r_GR − r_Newton| [km]",
        ),
        width="stretch",
    )
    st.caption(
        "The displacement grows along the orbit because the GR perihelion runs ahead of the "
        "Newtonian one (Lesson 04). The 1PN expansion is valid while v/c stays small: c/100 is "
        "fine for Mercury (v/c = 0.013), c/1000 is at the edge."
    )


# --------------------------------------------------------------------------- lab 3
def lab_precession() -> None:
    st.header("Precession tracker")
    st.markdown(
        "The longitude of perihelion $\\varpi = \\Omega + \\omega$ from the eccentricity vector "
        "$\\mathbf e = \\mathbf v\\times\\mathbf h/\\mu - \\mathbf r/r$ at every sample, in the "
        "Newton and Newton + 1PN runs, with least-squares secular fits. The *difference* of the "
        "two series cancels the planetary wobble and the integrator error and leaves the GR "
        "advance $6\\pi GM/(c^2 a(1-e^2))$ per orbit (Lessons 03–04)."
    )
    c1, c2, c3, c4 = st.columns(4)
    body = c1.selectbox("planet", list(labs.INNER_PLANETS), index=0)
    years = c2.slider("span [years]", 2.0, 100.0, 10.0, step=1.0)
    c_label = c3.selectbox("speed of light", list(C_SCALES), index=0, key="c_prec")
    c_scale = C_SCALES[c_label]
    system = c4.radio("system", ("Sun + planet", "Sun + 8 planets"), index=0)
    names = ("Sun", body) if system == "Sun + planet" else C.DEFAULT_BODIES
    track = _precession_track(years, c_scale, body, names)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "fit, GR − Newton", f"{track.rate_difference:.3f} ± {track.rate_difference_err:.3f} ″/cy"
    )
    m2.metric("analytic 1PN (same c)", f"{track.analytic:.3f} ″/cy")
    rate_n = track.rate_newton * C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS
    rate_g = track.rate_gr * C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS
    m3.metric("dense fit, Newton", f"{rate_n:.2f} ″/cy")
    m4.metric("dense fit, GR", f"{rate_g:.2f} ″/cy")

    units = "deg" if c_scale >= 100.0 else "arcsec"
    left, right = st.columns(2)
    with left:
        fig = precession_figure(track.t, track.varpi_newton, track.rate_newton, units=units)
        fig.data[0].update(name="Newton", line={"color": BODY_COLORS[body]})
        fig_g = precession_figure(track.t, track.varpi_gr, track.rate_gr, units=units)
        fig_g.data[0].update(name="Newton + 1PN", line={"color": INK})
        fig.add_trace(fig_g.data[0])
        fig.update_layout(title=f"{body}: ϖ(t) − ϖ(0)", height=400)
        st.plotly_chart(fig, width="stretch")
    with right:
        fit_rate = track.rate_difference / (C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS)
        fig = precession_figure(track.t, track.difference, fit_rate, units=units)
        fig.data[0].update(name="(Newton + 1PN) − Newton")
        fig.update_layout(title="run-vs-run difference", height=400)
        st.plotly_chart(fig, width="stretch")
    el = track.elements
    st.caption(
        f"osculating elements at J2000: a = {el['a']:.4f} AU, e = {el['e']:.4f}, "
        f"P = {el['period']:.2f} d. With the planets present the dense fits wobble by tens of "
        "arcsec over a decade and their slopes are biased by long-period terms; the difference "
        "is not."
    )


# --------------------------------------------------------------------------- lab 4
def lab_potential() -> None:
    st.header("Effective-potential explorer (Schwarzschild)")
    st.markdown(
        "Geometric units $GM = c = 1$ (horizon at $r = 2$, photon sphere $3$, ISCO $6$). "
        "$V_{\\rm eff}(r) = -1/r + L^2/2r^2 - L^2/r^3$; the radial motion obeys "
        "$\\tfrac12\\dot r^2 + V_{\\rm eff} = \\mathcal E$. Choose $L$ and the energy "
        "$\\mathcal E$ "
        "(relative to the barrier top when there is one) and start the particle at $r_0$ "
        "(Lesson 05)."
    )
    c1, c2, c3, c4 = st.columns(4)
    L = c1.slider("L [GM/c]", 3.0, 6.0, 4.0, step=0.01)
    curves = labs.potential_curves(L)
    has_barrier = np.isfinite(curves.r_unstable)
    if has_barrier:
        d_energy = c2.slider("ℰ − V_top (×10⁻³)", -40.0, 5.0, -0.02, step=0.01)
        energy = curves.v_top + 1e-3 * d_energy
    else:
        energy = c2.slider("ℰ", -0.2, 0.05, -0.02, step=0.005)
    r0 = c3.slider("start radius r₀", 2.5, 60.0, 30.0, step=0.5)
    tau_end = c4.slider("proper time span τ [GM/c³]", 200.0, 10000.0, 2000.0, step=100.0)
    v0 = float(np.interp(r0, curves.r, curves.v_gr))
    if energy < v0:
        st.warning(
            f"ℰ = {energy:.4f} is below V_eff(r₀) = {v0:.4f}: r₀ is not reachable; "
            "r₀ is used as an apsis."
        )
    try:
        orbit = _strong_field_orbit(L, energy, r0, tau_end, True)
    except RuntimeError as exc:  # pragma: no cover - defensive: the integrator gave up
        st.error(f"the geodesic integration failed: {exc}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("stable circular orbit", f"{curves.r_stable:.2f}" if has_barrier else "none")
    m2.metric("unstable circular orbit", f"{curves.r_unstable:.2f}" if has_barrier else "none")
    m3.metric("closest approach", f"{orbit.r_min:.2f}")
    if orbit.captured:
        outcome = f"captured after {orbit.turns:.1f} turns"
    elif orbit.r[-1] > 2.0 * r0:
        outcome = f"escaping, {orbit.turns:.1f} turns"
    else:
        outcome = f"bound, {orbit.turns:.1f} turns"
    m4.metric("outcome", outcome)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=curves.r, y=curves.v_gr, name="Schwarzschild", line={"color": PALETTE[0]})
        )
        fig.add_trace(
            go.Scatter(x=curves.r, y=curves.v_newton, name="Newton", line={"color": PALETTE[1]})
        )
        fig.add_hline(y=energy, line={"color": INK, "dash": "dash"}, annotation_text="ℰ")
        for x_mark, lab in ((3.0, "photon sphere"), (6.0, "ISCO")):
            fig.add_vline(x=x_mark, line={"color": "#898781", "width": 0.8}, annotation_text=lab)
        lo = min(curves.v_well if has_barrier else -0.1, energy) - 0.02
        hi = max(curves.v_top if has_barrier else 0.05, energy) + 0.02
        fig.update_layout(
            template="plotly_white",
            xaxis_title="r [GM/c²]",
            yaxis_title="V_eff [c²]",
            yaxis_range=[lo, hi],
            xaxis_range=[2.0, 60.0],
            height=460,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
            margin={"l": 60, "r": 20, "t": 40, "b": 50},
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=orbit.x, y=orbit.y, z=np.zeros_like(orbit.x), mode="lines",
                name="orbit", line={"color": PALETTE[0], "width": 3},
            )
        )
        rings = ((2.0, "horizon", INK), (3.0, "photon sphere", "#52514e"), (6.0, "ISCO", "#898781"))
        for radius, lab, col in rings:
            x, y, z = _ring(radius)
            fig.add_trace(
                go.Scatter3d(x=x, y=y, z=z, mode="lines", name=lab, line={"color": col, "width": 2})
            )
        lim = float(max(np.abs(orbit.x).max(), np.abs(orbit.y).max(), 8.0)) * 1.05
        fig.update_layout(
            template="plotly_white",
            scene={
                "xaxis": {"title": "x [GM/c²]", "range": [-lim, lim]},
                "yaxis": {"title": "y [GM/c²]", "range": [-lim, lim]},
                "zaxis": {"title": "z", "range": [-lim, lim]},
                "aspectmode": "cube",
            },
            height=460,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
        )
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Just below the barrier top the particle whirls near the unstable circular orbit before "
        "zooming out (zoom-whirl); above it, it plunges through the horizon. Below "
        "L = √12 = 3.46 there is no barrier at all and every inward orbit is captured. "
        "ℰ > 0 orbits are unbound: they come in once, bounce off the barrier and leave."
    )


# --------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="gravsim lab", page_icon="☉", layout="wide")
    st.sidebar.title("gravsim lab")
    st.sidebar.markdown(
        "Newton vs. Einstein in the solar system. Units AU, day, GM; frame: barycentric J2000 "
        "ecliptic. Each lab mirrors a lesson in `notebooks/`: labs 1–4 lessons 01–05 (Newton, "
        "measurement, 1PN, Schwarzschild), lab 5 lessons 06–07 (EIH and PPN), lab 6 lesson 08 "
        "(light), lab 7 lesson 09 (solar J₂, frame dragging, the Moon)."
    )
    lab = st.sidebar.radio("lab", LABS, index=0)
    st.sidebar.caption(
        f"Reference numbers: Mercury 1PN {C.MERCURY_PRECESSION_GR_ARCSEC_CY} ″/cy, "
        f"planets {C.MERCURY_PRECESSION_NEWTON_ARCSEC_CY} ″/cy, "
        f"observed {C.MERCURY_PRECESSION_OBSERVED_ARCSEC_CY} ″/cy; light at the solar limb "
        f"{C.LIGHT_DEFLECTION_LIMB_ARCSEC} ″."
    )
    pages = {
        LABS[0]: lab_integrators,
        LABS[1]: lab_solar_system,
        LABS[2]: lab_precession,
        LABS[3]: lab_potential,
        LABS[4]: lab_ppn.render,
        LABS[5]: lab_light.render,
        LABS[6]: lab_extras.render,
    }
    pages[lab]()


main()  # streamlit executes the script top to bottom on every rerun
