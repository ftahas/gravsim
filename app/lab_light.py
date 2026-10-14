"""Light-bending lab (Phase 6): deflection, ray bundles and the Shapiro delay.

A self-contained Streamlit lab.  The top of the module is a pure back end (plain numbers in,
numpy arrays and frozen dataclasses out, no Streamlit) built on ``gravsim.metric.null``; it is
what ``tests/test_lab_light.py`` exercises.  ``render()`` at the bottom adds the widgets,
``st.cache_data`` memoisation and plotly figures, in the style of ``streamlit_app.py``, which
wires it into its lab list.  Streamlit and plotly are imported only if present (the ``app``
extra), so the back end also imports without them.

Physics (details in the module docstring of ``gravsim.metric.null`` and docs/theory/light.md)
-----------------------------------------------------------------------------------------
* A photon obeys the null Binet equation ``u'' + u = 3 (gm/c^2) u^2``; everything depends on the
  geometry only through ``mu/b`` with ``mu = gm/c^2`` the gravitational radius and ``b`` the
  impact parameter.  ``null.deflection_angle`` integrates it exactly (GR); the weak-field PPN
  value is ``(1 + gamma)/2 * 4 gm/(c^2 b)``: 1.7512" at the solar limb for ``gamma = 1``,
  Soldner's corpuscle value 0.8756" for ``gamma = 0``.
* Rays with ``b <= b_crit = 3 sqrt(3) mu`` are captured (the black-hole shadow); the deflection
  diverges logarithmically as ``b -> b_crit``.
* The ray bundle is drawn with the *deviation* from the incoming straight line amplified,
  ``y_amp = b + k (y - b)``, because the solar bending (8.5e-6 rad) is invisible to scale.
* Shapiro delay of a signal whose chord passes the mass at distance ``b`` between emitter and
  receiver: ``(1 + gamma) gm/c^3 ln[(r_e + r_r + R)/(r_e + r_r - R)]``, ``R`` the chord;
  124 microseconds one way for Earth--Mars at the limb (Viking 1979: ``gamma = 1.000 +- 0.002``;
  Cassini 2003: ``gamma - 1 = (2.1 +- 2.3)e-5``).  Around superior conjunction the delay
  rises logarithmically as the chord's closest approach shrinks: the Shapiro "spike".

Units: AU, day, GM in AU^3/day^2, ``c`` explicit (default ``constants.C_AU_DAY``).  The
sliders work in solar radii or in units of ``GM/c^2``; :func:`impact_parameter_au` converts.
The "mass scale" multiplies ``GM_sun`` while the occulting disc keeps the solar radius, i.e.
"compress ``m`` Suns inside the Sun's radius": at ``m = 1e5`` the limb ray is captured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from gravsim import constants as C
from gravsim.metric import null

try:  # the front end needs the ``app`` extra; the back end does not
    import plotly.graph_objects as go
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only without the extra installed
    go = st = None

GM_SUN = C.GM_SUN
C_AU_DAY = C.C_AU_DAY
R_SUN_AU = C.R_SUN_AU
MICROSECOND_DAYS = 1e-6 / C.DAY_S

UNITS: tuple[str, ...] = ("solar radii", "GM/c²")
MASS_SCALES: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
BUNDLE_FACTORS: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0)
GEOMETRIES: dict[str, tuple[float, float]] = {  # (r_emit, r_recv) in AU
    "Earth – Mars (Viking 1976)": (1.0, 1.5237),
    "Earth – Venus (radar 1967)": (1.0, 0.7233),
    "Earth – Mercury (radar 1967)": (1.0, 0.3871),
    "Earth – Saturn (Cassini 2002)": (1.0, 8.43),
}
DEFLECTION_MEASUREMENTS: tuple[tuple[str, float, float], ...] = (
    # (label, deflection scaled to the solar limb [arcsec], uncertainty)
    ("1919 Sobral", 1.98, 0.12),
    ("1919 Príncipe", 1.61, 0.30),
    ("1922 Wallal", 1.72, 0.11),
    ("1973 Mauritania", 1.66, 0.18),
    ("1976 radio", 1.007 * C.LIGHT_DEFLECTION_LIMB_ARCSEC, 0.009 * C.LIGHT_DEFLECTION_LIMB_ARCSEC),
    ("2004 VLBI", 0.9999 * C.LIGHT_DEFLECTION_LIMB_ARCSEC, 0.0002 * C.LIGHT_DEFLECTION_LIMB_ARCSEC),
)
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
_LIMB_TOLERANCE = 1e-9  # relative: the root finder places b_min this close to the limb


# --------------------------------------------------------------------------- helpers
def impact_parameter_au(value: float, unit: str, gm: float = GM_SUN, c: float = C_AU_DAY) -> float:
    """Convert a slider value in ``"solar radii"`` or ``"GM/c²"`` to an impact parameter in AU."""
    if unit == UNITS[0]:
        return float(value) * R_SUN_AU
    if unit == UNITS[1]:
        return float(value) * gm / c**2
    raise ValueError(f"unknown unit {unit!r}; expected one of {UNITS}")


def bundle_impact_parameters(b: float, factors=BUNDLE_FACTORS) -> np.ndarray:
    """Signed impact parameters ``+-b * factors`` of a symmetric bundle, in increasing order."""
    fac = np.asarray(factors, dtype=np.float64).reshape(-1)
    return np.sort(np.concatenate([-b * fac, b * fac]))


def amplify_deviation(x, y, b: float, factor: float) -> np.ndarray:
    """``b + factor * (y - b)``: the ray's deviation from its incoming line ``y = b`` amplified."""
    y = np.asarray(y, dtype=np.float64)
    return b + factor * (y - b)


# --------------------------------------------------------------------------- deflection curve
@dataclass(frozen=True)
class DeflectionCurve:
    """Exact (GR), weak-field PPN and Newtonian-corpuscle deflections on a grid of ``b``."""

    b: np.ndarray  # impact parameters (AU), log-spaced, all above b_crit
    exact: np.ndarray  # rad, null geodesic (GR)
    weak: np.ndarray  # rad, (1 + gamma)/2 * 4 gm/(c^2 b)
    newton: np.ndarray  # rad, 2 gm/(c^2 b)
    b_crit: float  # AU
    gamma: float


def deflection_curve(
    b_min: float,
    b_max: float,
    gm: float = GM_SUN,
    c: float = C_AU_DAY,
    gamma: float = 1.0,
    n: int = 60,
) -> DeflectionCurve:
    """Sample the three deflection laws between ``b_min`` and ``b_max`` (AU, log-spaced).

    The grid starts at ``max(b_min, 1.001 b_crit)`` so that every ray escapes; ``b_max`` must
    lie above that.  Only the weak-field curve depends on ``gamma``.
    """
    b_crit = null.critical_impact_parameter(gm, c)
    lo = max(float(b_min), 1.001 * b_crit)
    if not float(b_max) > lo:
        raise ValueError(f"b_max = {b_max} must exceed {lo} (the larger of b_min and 1.001 b_crit)")
    b = np.geomspace(lo, float(b_max), int(n))
    exact = np.array([null.deflection_angle(float(bb), gm, c) for bb in b])
    return DeflectionCurve(
        b=b,
        exact=exact,
        weak=null.deflection_angle_weak(b, gm, c, gamma),
        newton=null.deflection_newtonian_corpuscle(b, gm, c),
        b_crit=b_crit,
        gamma=gamma,
    )


# --------------------------------------------------------------------------- ray bundle
@dataclass(frozen=True)
class RayBundle:
    """Traces of the rays of a bundle that escape, plus the ones that do not."""

    b: np.ndarray  # signed impact parameters (AU) of the drawn rays
    paths: tuple[tuple[np.ndarray, np.ndarray], ...]  # (x, y) per drawn ray, mass at the origin
    deflection: np.ndarray  # rad per drawn ray (GR)
    closest_approach: np.ndarray  # AU per drawn ray
    blocked: np.ndarray  # bool per drawn ray: closest approach inside ``body_radius``
    captured: np.ndarray  # signed impact parameters with |b| <= b_crit (not drawn)
    b_crit: float
    r_max: float  # radius at which the arms are cut


def ray_bundle_scene(
    b_values,
    gm: float = GM_SUN,
    c: float = C_AU_DAY,
    body_radius: float = 0.0,
    r_max: float | None = None,
    n_points: int = 800,
) -> RayBundle:
    """Trace the rays with signed impact parameters ``b_values`` past a mass at the origin.

    Rays with ``|b| <= b_crit`` are captured and reported instead of traced; rays whose closest
    approach lies inside ``body_radius`` (they would hit the surface of an ordinary body) are
    traced but flagged ``blocked``.  ``r_max`` defaults to ``4 max|b|``, a window that shows the
    bending; ``null.ray_path`` requires it to exceed the impact parameter of every ray.
    """
    b_values = np.asarray(b_values, dtype=np.float64).reshape(-1)
    b_crit = null.critical_impact_parameter(gm, c)
    keep = np.abs(b_values) > b_crit
    b_ok, captured = b_values[keep], b_values[~keep]
    if r_max is None:
        r_max = 4.0 * float(np.abs(b_values).max()) if b_values.size else 1.0
    paths = tuple(null.ray_bundle(b_ok, gm, c, n_points=n_points, r_max=r_max))
    deflection = np.array([null.deflection_angle(abs(float(b)), gm, c) for b in b_ok])
    r0 = np.asarray(null.closest_approach_radius(np.abs(b_ok), gm, c), dtype=np.float64).reshape(-1)
    return RayBundle(
        b=b_ok,
        paths=paths,
        deflection=deflection,
        closest_approach=r0,
        blocked=r0 < float(body_radius),
        captured=captured,
        b_crit=b_crit,
        r_max=float(r_max),
    )


# --------------------------------------------------------------------------- Shapiro delay
@dataclass(frozen=True)
class ShapiroResult:
    """One-way Shapiro delay of one emitter/receiver/closest-approach configuration (days)."""

    r_emit: float  # AU
    r_recv: float
    b: float  # closest approach of the chord to the mass (AU)
    one_way: float  # weak-field PPN, days
    round_trip: float  # 2 * one_way
    conjunction_form: float  # (1 + gamma) gm/c^3 ln(4 r_e r_r / b^2)
    exact: float  # GR null geodesic with impact parameter b (gamma = 1), days; NaN if captured
    light_time: float  # chord / c, days
    gamma: float
    captured: bool  # b <= 3 sqrt(3) gm/c^2: no ray with this impact parameter gets through


def shapiro_scenario(
    r_emit: float,
    r_recv: float,
    b: float,
    gm: float = GM_SUN,
    c: float = C_AU_DAY,
    gamma: float = 1.0,
) -> ShapiroResult:
    """Weak-field (PPN ``gamma``), conjunction-form and exact-geodesic delays for one geometry.

    The weak-field formulas are evaluated for any ``b`` in ``(0, min(r_emit, r_recv))``; the
    exact geodesic exists only for ``b > b_crit`` and is NaN otherwise (``captured`` True).
    """
    one_way = null.shapiro_delay(r_emit, r_recv, b, gm, c, gamma)
    chord = math.sqrt(r_emit * r_emit - b * b) + math.sqrt(r_recv * r_recv - b * b)
    captured = not b > null.critical_impact_parameter(gm, c)
    return ShapiroResult(
        r_emit=float(r_emit),
        r_recv=float(r_recv),
        b=float(b),
        one_way=one_way,
        round_trip=2.0 * one_way,
        conjunction_form=null.shapiro_delay_conjunction(r_emit, r_recv, b, gm, c, gamma),
        exact=math.nan if captured else null.shapiro_delay_numeric(r_emit, r_recv, b, gm, c),
        light_time=chord / c,
        gamma=gamma,
        captured=captured,
    )


@dataclass(frozen=True)
class ConjunctionSeries:
    """The Shapiro spike: closest approach and delay against days from superior conjunction."""

    days: np.ndarray
    b: np.ndarray  # closest approach of the chord to the mass (AU)
    delay: np.ndarray  # one-way delay (days); NaN where occulted or where the chord misses
    occulted: np.ndarray  # bool: b < body_radius
    valid: np.ndarray  # bool: the chord passes the mass between the two bodies
    synodic_rate: float  # rad/day
    b_min: float  # AU, the closest approach at exact conjunction


def _conjunction_height(r_emit: float, r_recv: float, b_min: float) -> float:
    """Height ``z`` of the receiver above the emitter's orbital plane giving ``b_min``.

    At exact conjunction the emitter is at ``(r_e, 0, 0)`` and the receiver at
    ``(-sqrt(r_r^2 - z^2), 0, z)``; the distance of the origin from the chord is
    ``r_e z / |chord|``, solved for ``z`` with ``brentq``.
    """

    def gap(z: float) -> float:
        x_r = math.sqrt(r_recv * r_recv - z * z)
        return r_emit * z / math.hypot(r_emit + x_r, z) - b_min

    return brentq(gap, 0.0, r_recv * (1.0 - 1e-12))


def conjunction_series(
    r_emit: float,
    r_recv: float,
    b_min: float,
    days,
    gm: float = GM_SUN,
    c: float = C_AU_DAY,
    gamma: float = 1.0,
    body_radius: float = R_SUN_AU,
) -> ConjunctionSeries:
    """Shapiro delay around superior conjunction for two bodies on circular orbits.

    Both orbit the mass with Kepler periods ``2 pi sqrt(r^3/gm)``; the emitter moves in the
    reference plane, the receiver sits on the far side of the mass, lifted out of the plane so
    that the chord passes the mass at ``b_min`` at exact conjunction (``days = 0``), as the real
    conjunctions do because of the orbital inclinations.  The delay is the weak-field
    ``null.shapiro_delay`` of the chord's closest approach; it is NaN where the chord's closest
    approach lies inside ``body_radius`` (occulted) and where the foot of the perpendicular from
    the mass no longer falls between the two bodies (far from conjunction, ``valid`` False).

    Parameters
    ----------
    r_emit, r_recv : float
        Orbital radii (AU), both larger than ``b_min``.
    b_min : float
        Closest approach of the chord at conjunction (AU), positive.
    days : array_like
        Times relative to conjunction (days).
    gm, c, gamma : float
        Central GM, speed of light, PPN parameter.
    body_radius : float
        Radius of the occulting body (AU).
    """
    if not 0.0 < b_min < min(r_emit, r_recv):
        raise ValueError("b_min must be positive and smaller than both orbital radii")
    days = np.asarray(days, dtype=np.float64).reshape(-1)
    n_e = math.sqrt(gm / r_emit**3)
    n_r = math.sqrt(gm / r_recv**3)
    rate = abs(n_e - n_r)
    theta = rate * days
    z = _conjunction_height(r_emit, r_recv, b_min)
    x_e = np.stack([r_emit * np.cos(theta), r_emit * np.sin(theta), np.zeros_like(theta)], axis=1)
    x_r = np.array([-math.sqrt(r_recv * r_recv - z * z), 0.0, z])
    chord = x_r - x_e
    length = np.linalg.norm(chord, axis=1)
    b = np.linalg.norm(np.cross(x_e, x_r), axis=1) / length
    foot = -np.einsum("ij,ij->i", x_e, chord) / length**2  # position of the foot along the chord
    valid = (foot > 0.0) & (foot < 1.0)
    occulted = b < body_radius * (1.0 - _LIMB_TOLERANCE)  # a grazing chord counts as visible
    delay = np.full(days.shape, np.nan)
    for i in np.flatnonzero(valid & ~occulted):
        delay[i] = null.shapiro_delay(r_emit, r_recv, float(b[i]), gm, c, gamma)
    return ConjunctionSeries(
        days=days,
        b=b,
        delay=delay,
        occulted=occulted,
        valid=valid,
        synodic_rate=rate,
        b_min=float(b_min),
    )


# =========================================================================== Streamlit front end
def _cache(message: str):
    """``st.cache_data`` with a spinner message, or the identity without streamlit."""
    if st is None:
        return lambda fn: fn
    return st.cache_data(show_spinner=message)


@_cache("integrating the null geodesics ...")
def _deflection_curve(b_min: float, b_max: float, gm: float, gamma: float):
    return deflection_curve(b_min, b_max, gm, C_AU_DAY, gamma)


@_cache("tracing the ray bundle ...")
def _ray_bundle_scene(b: float, gm: float, body_radius: float, r_max: float):
    return ray_bundle_scene(bundle_impact_parameters(b), gm, C_AU_DAY, body_radius, r_max)


@_cache("computing the Shapiro delay ...")
def _shapiro(r_emit: float, r_recv: float, b: float, gm: float, gamma: float):
    return shapiro_scenario(r_emit, r_recv, b, gm, C_AU_DAY, gamma)


@_cache("sweeping the conjunction ...")
def _conjunction(
    r_emit: float, r_recv: float, b_min: float, window: float, gm: float, gamma: float
):
    days = np.linspace(-window, window, 801)
    return conjunction_series(r_emit, r_recv, b_min, days, gm, C_AU_DAY, gamma)


def _angle_text(rad: float) -> str:
    """Arcseconds below 0.1 degree, degrees above."""
    if rad < 0.1 * math.pi / 180.0:
        return f"{rad * C.ARCSEC_PER_RAD:.4g} ″"
    return f"{math.degrees(rad):.4g} °"


def _base_layout(fig, x_title: str, y_title: str, height: int = 420):
    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0.0},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        height=height,
    )
    return fig


def _bundle_figure(bundle: RayBundle, scale: float, unit_label: str, body_radius: float,
                   amplify: float, gm: float):
    fig = go.Figure()
    for k, (b, (x, y)) in enumerate(zip(bundle.b, bundle.paths, strict=True)):
        y_draw = amplify_deviation(x, y, float(b), amplify)
        fig.add_trace(
            go.Scatter(
                x=x / scale, y=y_draw / scale, mode="lines",
                name=f"b = {b / scale:+.3g}", showlegend=k < 2 or k >= len(bundle.b) - 2,
                line={"color": PALETTE[1] if bundle.blocked[k] else PALETTE[0], "width": 1.5},
                hovertemplate=f"b = {b / scale:+.4g} {unit_label}: "
                f"δ = {_angle_text(float(bundle.deflection[k]))}<extra></extra>",
            )
        )
    mu = gm / C_AU_DAY**2
    for radius, name, color, filled in (
        (body_radius, "occulting body", "#eda100", True),
        (2.0 * mu, "horizon 2GM/c²", INK, True),
        (3.0 * mu, "photon sphere 3GM/c²", INK_SECONDARY, False),
    ):
        if radius <= 0.0 or radius / scale > 0.5 * bundle.r_max / scale:
            continue
        r = radius / scale
        fig.add_shape(
            type="circle", x0=-r, y0=-r, x1=r, y1=r, line={"color": color, "width": 1},
            fillcolor=color if filled else "rgba(0,0,0,0)", opacity=0.8 if filled else 1.0,
        )
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name=name, line={"color": color})
        )
    lim = bundle.r_max / scale * 0.72
    fig.update_layout(
        xaxis_range=[-lim, lim], yaxis_range=[-lim, lim],
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    return _base_layout(fig, f"x [{unit_label}]", f"y [{unit_label}]", height=460)


def _deflection_figure(curve: DeflectionCurve, b_chosen: float, delta_chosen: float | None,
                       scale: float, unit_label: str, show_measurements: bool):
    fig = go.Figure()
    bb = curve.b / scale
    arcsec = C.ARCSEC_PER_RAD
    fig.add_trace(go.Scatter(x=bb, y=curve.exact * arcsec, name="exact null geodesic (GR)",
                             line={"color": PALETTE[0], "width": 2.5}))
    fig.add_trace(go.Scatter(x=bb, y=curve.weak * arcsec,
                             name=f"(1+γ)/2 · 4GM/(c²b), γ = {curve.gamma:g}",
                             line={"color": INK, "width": 1, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=bb, y=curve.newton * arcsec, name="2GM/(c²b)  (Soldner 1801)",
                             line={"color": PALETTE[1], "width": 1, "dash": "dash"}))
    if delta_chosen is not None:
        fig.add_trace(go.Scatter(x=[b_chosen / scale], y=[delta_chosen * arcsec], mode="markers",
                                 name="chosen b", marker={"color": PALETTE[0], "size": 10}))
    if show_measurements:
        for k, (label, value, err) in enumerate(DEFLECTION_MEASUREMENTS):
            fig.add_trace(go.Scatter(
                x=[1.0 + 0.02 * k], y=[value], mode="markers", name=label,
                marker={"color": PALETTE[2 + k % 6], "size": 7},
                error_y={"type": "data", "array": [err], "visible": True},
            ))
    fig.update_layout(xaxis_type="log", yaxis_type="log")
    return _base_layout(fig, f"impact parameter b [{unit_label}]", "deflection [arcsec]", 460)


def _conjunction_figure(series: ConjunctionSeries, round_trip: bool):
    factor = 2.0 if round_trip else 1.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.days, y=factor * series.delay / MICROSECOND_DAYS,
                             mode="lines", name="round trip" if round_trip else "one way",
                             line={"color": PALETTE[0], "width": 2}))
    if series.occulted.any():
        occ = series.days[series.occulted]
        fig.add_vrect(x0=float(occ.min()), x1=float(occ.max()), fillcolor="#898781", opacity=0.3,
                      line_width=0, annotation_text="occulted", annotation_position="top left")
    return _base_layout(fig, "days from superior conjunction", "Shapiro delay [μs]", height=380)


def render() -> None:
    """Draw the light-bending lab (called by ``streamlit_app.py``; no page config here)."""
    if st is None or go is None:  # pragma: no cover - the app extra is installed with the app
        raise ImportError("the light-bending lab needs streamlit and plotly (the 'app' extra)")
    st.header("Light-bending lab: Eddington 1919 and the Shapiro delay")
    st.markdown(
        "A photon obeys $u'' + u = 3\\,(GM/c^2)\\,u^2$, the Binet equation without its Newtonian "
        "term, so everything depends on $GM/(c^2 b)$ alone. The weak-field deflection is "
        "$\\delta = \\frac{1+\\gamma}{2}\\,\\frac{4GM}{c^2 b}$: $1.75''$ at the solar limb in GR "
        "($\\gamma = 1$), half of it for Soldner's Newtonian corpuscle ($\\gamma = 0$). Rays with "
        "$b \\le 3\\sqrt3\\,GM/c^2$ are captured. A signal passing the Sun at distance $b$ is late "
        "by $(1+\\gamma)\\frac{GM}{c^3}\\ln\\frac{4 r_e r_r}{b^2}$ (Shapiro). Lesson 08."
    )
    c1, c2, c3, c4 = st.columns(4)
    unit = c1.radio("impact parameter in", UNITS, index=0, horizontal=True)
    if unit == UNITS[0]:
        b_value = c2.slider("impact parameter b [R☉]", 1.0, 20.0, 1.0, step=0.1)
    else:
        b_value = c2.slider("impact parameter b [GM/c²]", 5.3, 100.0, 10.0, step=0.1)
    mass_scale = c3.select_slider(
        "mass inside the Sun's radius [M☉]", options=MASS_SCALES, value=1.0,
        format_func=lambda m: f"{m:g}",
    )
    gamma = c4.slider("PPN γ (weak-field lines, Shapiro)", 0.0, 2.0, 1.0, step=0.05)
    gm = mass_scale * GM_SUN
    mu = gm / C_AU_DAY**2
    b = impact_parameter_au(b_value, unit, gm, C_AU_DAY)
    b_crit = null.critical_impact_parameter(gm, C_AU_DAY)
    scale = R_SUN_AU if unit == UNITS[0] else mu
    unit_label = "R☉" if unit == UNITS[0] else "GM/c²"
    body_radius = R_SUN_AU if unit == UNITS[0] else 0.0
    captured = not b > b_crit
    delta = None if captured else null.deflection_angle(b, gm, C_AU_DAY)
    weak = null.deflection_angle_weak(b, gm, C_AU_DAY, gamma)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("exact deflection (GR)", "captured" if captured else _angle_text(delta))
    m2.metric(f"weak field, γ = {gamma:g}", _angle_text(weak))
    m3.metric("Soldner (γ = 0)", _angle_text(null.deflection_newtonian_corpuscle(b, gm, C_AU_DAY)))
    m4.metric("b / b_crit", f"{b / b_crit:.3g}  (b_crit = {b_crit / scale:.3g} {unit_label})")
    if captured:
        st.warning(
            f"b = {b_value:g} {unit_label} is inside the critical impact parameter "
            f"3√3 GM/c² = {b_crit / scale:.3g} {unit_label}: the ray spirals into the photon "
            "sphere. Increase b or reduce the mass."
        )

    left, right = st.columns(2)
    with left:
        default_amp = 5.0 if (unit == UNITS[0] and mass_scale <= 10.0) else 0.0
        amp_log = st.slider("amplification of the deviation from the straight line, log10",
                            0.0, 6.0, default_amp, 0.5)
        bundle = _ray_bundle_scene(b, gm, body_radius, 8.0 * b)
        st.plotly_chart(
            _bundle_figure(bundle, scale, unit_label, body_radius, 10.0**amp_log, gm),
            width="stretch",
        )
        notes = [f"{len(bundle.b)} rays at ±{', ±'.join(f'{f:g}' for f in BUNDLE_FACTORS)} × b"]
        if bundle.captured.size:
            notes.append(f"{bundle.captured.size} captured (not drawn)")
        if bundle.blocked.any():
            notes.append(f"{int(bundle.blocked.sum())} hit the surface (orange)")
        st.caption("; ".join(notes) + ". Drawn as y = b + k (y − b): k = 1 is the honest picture.")
    with right:
        curve = _deflection_curve(max(b / 10.0, 1.001 * b_crit), 50.0 * b, gm, gamma)
        st.plotly_chart(
            _deflection_figure(curve, b, delta, scale, unit_label,
                               show_measurements=(unit == UNITS[0] and mass_scale == 1.0)),
            width="stretch",
        )
        st.caption(
            "The exact curve exceeds 4GM/(c²b) by (15π/16) GM/(c²b) and diverges "
            "logarithmically at b_crit. Measurements (shown for the real Sun) are scaled to the "
            "limb by the 1/b law; the 1919 stars sat at 2–6 R☉."
        )

    st.subheader("Shapiro delay at superior conjunction")
    s1, s2, s3, s4 = st.columns(4)
    geometry = s1.selectbox("emitter – receiver", list(GEOMETRIES), index=0)
    r_emit, r_recv = GEOMETRIES[geometry]
    b_min = s2.slider("closest approach at conjunction [R☉]", 1.0, 10.0, 1.0, step=0.1)
    window = s3.slider("window [days]", 10.0, 120.0, 60.0, step=5.0)
    round_trip = s4.toggle("round trip (radar / transponder)", value=True)
    result = _shapiro(r_emit, r_recv, b_min * R_SUN_AU, gm, gamma)
    if result.captured:
        st.warning(
            f"with {mass_scale:g} M☉ inside the Sun's radius the critical impact parameter is "
            f"{b_crit / R_SUN_AU:.2f} R☉: a chord at {b_min:g} R☉ is captured and no signal "
            "gets through. Increase the closest approach or reduce the mass."
        )
        return
    series = _conjunction(r_emit, r_recv, b_min * R_SUN_AU, window, gm, gamma)
    factor = 2.0 if round_trip else 1.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"delay at b = {b_min:g} R☉, γ = {gamma:g}",
              f"{factor * result.one_way / MICROSECOND_DAYS:.1f} μs")
    k2.metric("exact geodesic (GR)", f"{factor * result.exact / MICROSECOND_DAYS:.1f} μs")
    k3.metric("light time", f"{factor * result.light_time * C.DAY_S / 60.0:.1f} min")
    k4.metric("delay / light time", f"{result.one_way / result.light_time:.1e}")
    st.plotly_chart(_conjunction_figure(series, round_trip), width="stretch")
    st.caption(
        "Circular orbits with Kepler periods; the receiver passes behind the Sun at the chosen "
        "closest approach. Viking (1979) measured the Earth–Mars round trip to 0.1 %; Cassini "
        "(2003) measured the time derivative of this curve as a Doppler shift and found "
        "γ − 1 = (2.1 ± 2.3)×10⁻⁵. The curve stops where the chord no longer passes the Sun "
        "between the two bodies."
    )
