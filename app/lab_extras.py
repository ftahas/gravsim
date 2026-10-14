"""Competing effects (Streamlit lab 7): solar oblateness, frame dragging and the Moon.

Physics
-------
In an inertial frame Mercury's perihelion advances by 574.1"/century.  The budget: the planets
(531.6", Newtonian, Lesson 02), General Relativity (42.98", Lessons 04-07), the quadrupole
field of the rotating Sun (+0.028" for ``J2 = 2.2e-7``), the gravitomagnetic field of its spin
(Lense-Thirring, -0.002"), the Earth-Moon pair treated as a single body at its barycentre
(-0.0025", measured in Lesson 09 and ``tests/test_moon.py``) and the unmodelled asteroid belt
(~+0.0015" for a ring of ``1.2e-9 M_sun`` at 2.7 AU, ``ring_precession_rate``).  The lab adds
the two solar corrections of ``gravsim.forces.extras`` to Newton, amplified by sliders so that
they show in ten years, and measures the secular rates of the node ``Omega`` and the argument
of perihelion ``omega`` -- the elements that tell the effects apart, since GR moves ``omega``
only -- against the first-order formulas (Kaula 1966; Lense & Thirring 1918; derivations in
``docs/theory/extras.md``):

    J2:  dOmega/dt = -(3/2) n J2 (R/p)^2 cos i
         domega/dt =  (3/4) n J2 (R/p)^2 (5 cos^2 i - 1)
    LT:  dOmega/dt =  2 GJ / (c^2 a^3 (1 - e^2)^1.5)
         domega/dt = -6 GJ cos i / (c^2 a^3 (1 - e^2)^1.5)

with ``n`` the mean motion, ``p = a (1 - e^2)`` and ``i`` the inclination to the solar equator
(Mercury: 3.38 deg).  The elements are taken after rotating the trajectory into the frame whose
``z`` axis is ``constants.SUN_POLE_ECLIPTIC`` (``equator_frame``), and every rate is the
run-vs-run difference against a Newtonian run on the same grid (``analysis.compare``), which
cancels the planetary wobble when the full system is integrated.

With the Moon toggle the system is Sun + planets + Earth + Moon (``load_j2000(with_moon=True)``)
and the lab also fits the lunar orbit about the Earth: the sidereal month (27.32 d), the advance
of the perigee (+40.69 deg/yr) and the regression of the node (-19.34 deg/yr) -- the largest
precessions in the solar system, and purely Newtonian: the Sun's tide on the lunar orbit.  To
first order in ``m = n_sun / n_moon`` both rates are ``(3/4) m^2 n = 20.1 deg/yr`` (Newton 1687);
the higher orders (Clairaut 1749; Delaunay's series, Brouwer & Clemence 1961, ch. 12) double the
perigee rate and barely touch the node:

    dvarpi/dt = n [ (3/4) m^2 + (225/32) m^3 + (4071/128) m^4 + (265493/2048) m^5 + ... ]
    dOmega/dt = n [-(3/4) m^2 +   (9/32) m^3 +  (273/128) m^4 +   (9797/2048) m^5 + ... ]

(``moon_tidal_rates``; the series converges slowly: the partial sums to ``m^5`` fall about
1 % short, 40.3 and -19.2 deg/yr for ``m = 0.0748``).

Structure
---------
The back end (``kaula_rates``, ``lense_thirring_rates``, ``moon_tidal_rates``,
``ring_precession_rate``, ``equator_frame``, ``extras_runs``, ``node_perigee_rates``,
``moon_rates``, ``precession_budget``) is pure: plain numbers in, numpy arrays and frozen
dataclasses out, no Streamlit (``tests/test_lab_extras.py``).  ``render()`` adds the widgets,
``st.cache_data`` memoisation and the plotly figures in the idiom of ``streamlit_app.py``
(``lab_ppn.py``).  streamlit and plotly are optional (the ``app`` extra): the back end imports
without them.

Units: AU, day, GM in AU^3/day^2, GJ in AU^5/day^3, ``c`` in AU/day; Mercury rates in arcsec
per Julian century, lunar rates in degrees per Julian year.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gravsim import constants as C
from gravsim.analysis import (
    element_series,
    fit_secular_rate,
    inclination_to_axis,
    rate_arcsec_per_century,
    state_to_elements,
    wrap_angle,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.data.horizons import load_j2000
from gravsim.forces import accel_newton, combine, make_j2, make_lense_thirring
from gravsim.integrators import integrate_scipy

try:  # front-end dependencies (the ``app`` extra); the back end does not need them
    import plotly.graph_objects as go
    import streamlit as st
except ImportError:  # pragma: no cover - exercised only without the app extra
    go = st = None

J2_LOG_RANGE: tuple[float, float] = (0.0, 4.0)  # amplification sliders, log10
LT_LOG_RANGE: tuple[float, float] = (0.0, 6.0)
MOON_EMB_SHIFT_ARCSEC_CY = -0.0025
"""Change of Mercury's Newtonian rate when Earth and Moon replace the EMB (Lesson 09, 10 yr)."""
ASTEROID_RING_GM = 1.2e-9 * C.GM_SUN  # main belt, ~2.4e21 kg (Pitjeva & Pitjev 2018)
ASTEROID_RING_AU = 2.7
DEG_PER_YEAR = C.DEG_PER_RAD * C.JULIAN_YEAR_DAYS  # rad/day -> deg per Julian year
_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- formulas
def kaula_rates(
    a: ArrayLike,
    e: ArrayLike,
    inc: ArrayLike,
    gm: ArrayLike,
    j2: float = C.SUN_J2,
    r_eq: float = C.R_SUN_AU,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order secular rates of the node and the argument of perihelion from ``J2``.

    ``dOmega/dt = -(3/2) n J2 (r_eq/p)^2 cos i`` and
    ``domega/dt = (3/4) n J2 (r_eq/p)^2 (5 cos^2 i - 1)`` with ``n = sqrt(gm/a^3)``,
    ``p = a (1 - e^2)`` and ``i`` measured from the equator of the oblate body.

    Parameters
    ----------
    a, e, inc, gm : array_like
        Semi-major axis [AU], eccentricity, inclination to the equator [rad] and ``G (M + m)``
        [AU^3/day^2]; broadcast together.
    j2, r_eq : float, optional
        Quadrupole coefficient and the radius it refers to [AU] (defaults: the Sun).

    Returns
    -------
    d_raan, d_argp : ndarray
        Rates in rad/day (scalars for scalar inputs).
    """
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    ci = np.cos(np.asarray(inc, dtype=np.float64))
    n = np.sqrt(np.asarray(gm, dtype=np.float64) / a**3)
    k = n * j2 * (r_eq / (a * (1.0 - e * e))) ** 2
    d_raan = -1.5 * k * ci
    d_argp = 0.75 * k * (5.0 * ci * ci - 1.0)
    return d_raan[()], d_argp[()]


def lense_thirring_rates(
    a: ArrayLike,
    e: ArrayLike,
    inc: ArrayLike,
    gj: float = C.GJ_SUN,
    c: float = C.C_AU_DAY,
) -> tuple[np.ndarray, np.ndarray]:
    """Secular node and argument rates from frame dragging by a spin ``GJ`` [AU^5/day^3].

    ``dOmega/dt = 2 GJ / (c^2 a^3 (1 - e^2)^1.5)`` (independent of ``i``) and
    ``domega/dt = -6 GJ cos i / (c^2 a^3 (1 - e^2)^1.5)``, ``i`` from the spin equator.

    Returns
    -------
    d_raan, d_argp : ndarray
        Rates in rad/day (scalars for scalar inputs).
    """
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    ci = np.cos(np.asarray(inc, dtype=np.float64))
    k = gj / (c * c * a**3 * (1.0 - e * e) ** 1.5)
    d_raan = 2.0 * k * np.ones_like(ci)
    d_argp = -6.0 * k * ci
    return d_raan[()], d_argp[()]


@dataclass(frozen=True)
class TidalRates:
    """Solar-tide rates of the lunar perigee and node [rad/day] with the expansion ratio ``m``."""

    m: float  # n_sun / n_moon
    leading: float  # (3/4) m^2 n: the first-order size of both rates
    perigee: float  # Delaunay series to m^5
    node: float


def moon_tidal_rates(n_moon: float, n_sun: float) -> TidalRates:
    """Perigee and node rates of a satellite perturbed by the Sun, to fifth order in ``m``.

    Parameters
    ----------
    n_moon, n_sun : float
        Mean motions of the satellite about its planet and of the planet about the Sun
        [rad/day]; ``m = n_sun / n_moon`` (0.0748 for the Moon).

    Returns
    -------
    TidalRates
        ``leading = (3/4) m^2 n_moon`` (the perigee advances and the node regresses by this
        much to first order), ``perigee`` and ``node`` from the series of the module docstring.

    Raises
    ------
    ValueError
        Unless both mean motions are positive.
    """
    if n_moon <= 0.0 or n_sun <= 0.0:
        raise ValueError(f"mean motions must be positive, got {n_moon} and {n_sun}")
    m = n_sun / n_moon
    leading = 0.75 * m * m * n_moon
    perigee = n_moon * (
        0.75 * m**2 + 225.0 / 32.0 * m**3 + 4071.0 / 128.0 * m**4 + 265493.0 / 2048.0 * m**5
    )
    node = n_moon * (
        -0.75 * m**2 + 9.0 / 32.0 * m**3 + 273.0 / 128.0 * m**4 + 9797.0 / 2048.0 * m**5
    )
    return TidalRates(m=float(m), leading=float(leading), perigee=float(perigee), node=float(node))


def ring_precession_rate(
    gm_ring: float, a_ring: float, a: float, gm: float = C.GM_SUN
) -> float:
    """Secular perihelion advance of an inner orbit due to an exterior circular ring.

    To lowest order in ``a / a_ring`` the ring's secular potential ``(gm_ring / a_ring)
    [1 + (a/a_ring)^2 (1 + 3e^2/2)/4]`` gives ``dvarpi/dt = (3/4) n (gm_ring / gm) (a / a_ring)^3``.

    Returns
    -------
    float
        Rate in arcsec per Julian century.
    """
    n = math.sqrt(gm / a**3)
    return float(rate_arcsec_per_century(0.75 * n * (gm_ring / gm) * (a / a_ring) ** 3))


# --------------------------------------------------------------------------- helpers
def two_body_state() -> SystemState:
    """Sun + Mercury from the J2000 snapshot, in their own barycentric frame."""
    return load_j2000().subset(("Sun", "Mercury")).to_barycentric()


def daily_grid(t0: float, years: float, step: float = 1.0) -> np.ndarray:
    """Output times ``t0, t0 + step, ...`` spanning ``years`` Julian years."""
    n = int(round(years * C.JULIAN_YEAR_DAYS / step))
    return t0 + step * np.arange(n + 1)


def equator_frame(pole: ArrayLike = C.SUN_POLE_ECLIPTIC) -> np.ndarray:
    """Rotation matrix (rows ``x, y, z``) of a frame whose ``z`` axis is ``pole``.

    ``x`` points to the ascending node of the equator (the plane perpendicular to ``pole``) on
    the reference ``xy`` plane, ``z_ref x pole`` normalised, and ``y = z x x``; coordinates
    transform as ``r_eq = R @ r``.

    Raises
    ------
    ValueError
        If ``pole`` is parallel to the reference ``z`` axis (the node is undefined) or zero.
    """
    z = np.asarray(pole, dtype=np.float64)
    norm = np.linalg.norm(z)
    if z.shape != (3,) or norm == 0.0:
        raise ValueError("pole must be a non-zero 3-vector")
    z = z / norm
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-12:
        raise ValueError("pole is parallel to z: the equator's node is undefined")
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def rotate_trajectory(traj: Trajectory, rotation: np.ndarray) -> Trajectory:
    """The same trajectory with every position and velocity rotated by ``rotation``."""
    return Trajectory(
        t=traj.t,
        pos=traj.pos @ rotation.T,
        vel=traj.vel @ rotation.T,
        gm=traj.gm,
        names=traj.names,
        meta=dict(traj.meta),
    )


def osculating(state: SystemState, body: str, central: str = "Sun") -> dict[str, float]:
    """Osculating ``a, e, inc, period, gm`` of ``body`` about ``central`` plus ``inc_equator``.

    ``inc`` is the classical inclination to the frame's ``xy`` plane (the ecliptic);
    ``inc_equator`` the inclination to the solar equator (``SUN_POLE_ECLIPTIC``), in radians.
    """
    r, v = state.relative_to(central)
    i = state.index(body)
    gm = float(state.gm[state.index(central)] + state.gm[i])
    el = state_to_elements(r[i], v[i], gm)
    out = {k: float(el[k]) for k in ("a", "e", "inc", "period")} | {"gm": gm}
    out["inc_equator"] = float(inclination_to_axis(r[i], v[i], C.SUN_POLE_ECLIPTIC))
    return out


# --------------------------------------------------------------------------- runs
@dataclass(frozen=True)
class ExtrasRuns:
    """Newton and Newton + J2 + Lense-Thirring runs on one grid, with the sizes used."""

    newton: Trajectory
    extras: Trajectory
    j2: float  # the J2 actually used (SUN_J2 * j2_scale)
    gj: float  # G J actually used [AU^5/day^3] (GJ_SUN * lt_scale)
    with_moon: bool
    years: float


def extras_runs(
    years: float,
    j2_scale: float = 1.0,
    lt_scale: float = 1.0,
    with_moon: bool = False,
    step: float = 1.0,
) -> ExtrasRuns:
    """Integrate Newton and Newton + J2 (x ``j2_scale``) + Lense-Thirring (x ``lt_scale``).

    Parameters
    ----------
    years : float
        Span in Julian years.
    j2_scale, lt_scale : float
        Multipliers on ``constants.SUN_J2`` and on the solar spin ``constants.GJ_SUN``
        (0 switches a term off exactly, 1 is reality).
    with_moon : bool
        ``False``: Sun + Mercury in their own barycentric frame.  ``True``: the full system
        with Earth and Moon as separate bodies (``load_j2000(with_moon=True)``).
    step : float
        Sample spacing [day].

    Returns
    -------
    ExtrasRuns

    Raises
    ------
    ValueError
        If ``years`` or ``step`` is not positive or a scale is negative.
    """
    if years <= 0.0 or step <= 0.0:
        raise ValueError(f"years and step must be positive, got {years} and {step}")
    if j2_scale < 0.0 or lt_scale < 0.0:
        raise ValueError(f"scales must be non-negative, got {j2_scale} and {lt_scale}")
    state = load_j2000(with_moon=True) if with_moon else two_body_state()
    t_eval = daily_grid(state.t, years, step)
    central = state.index("Sun")
    j2 = C.SUN_J2 * float(j2_scale)
    gj = C.GJ_SUN * float(lt_scale)
    law = combine(
        accel_newton,
        make_j2(j2=j2, central=central),
        make_lense_thirring(gj=gj, central=central),
    )
    newton = integrate_scipy(state, accel_newton, t_eval)
    extras = integrate_scipy(state, law, t_eval)
    return ExtrasRuns(
        newton=newton, extras=extras, j2=j2, gj=gj, with_moon=bool(with_moon), years=float(years)
    )


@dataclass(frozen=True)
class NodePerigee:
    """Differenced node and argument of ``body`` in the solar-equator frame with the fits."""

    t: np.ndarray
    d_raan: np.ndarray  # wrapped Omega(extras) - Omega(newton) [rad]
    d_argp: np.ndarray  # wrapped omega(extras) - omega(newton) [rad]
    raan_rate: float  # arcsec per Julian century
    raan_err: float
    argp_rate: float
    argp_err: float
    varpi_rate: float  # fit of the differenced varpi (not the sum of the two fits)
    varpi_err: float
    predicted: dict[str, float]  # raan_j2, argp_j2, raan_lt, argp_lt, raan, argp, varpi
    elements: dict[str, float]


def node_perigee_rates(runs: ExtrasRuns, body: str = "Mercury") -> NodePerigee:
    """Fit the node and argument drifts of ``body`` in ``runs`` and compare with the formulas.

    The two trajectories are rotated into the solar-equator frame (``equator_frame``), the
    osculating ``raan``, ``argp`` and ``varpi`` are differenced sample by sample (wrapped) and
    fitted with ``fit_secular_rate``; the predictions use the osculating ``a``, ``e`` and the
    inclination to the solar equator at the first sample.
    """
    rotation = equator_frame()
    extras = rotate_trajectory(runs.extras, rotation)
    newton = rotate_trajectory(runs.newton, rotation)
    series = {}
    fits = {}
    for key in ("raan", "argp", "varpi"):
        t, x_extras = element_series(extras, body, key)
        _, x_newton = element_series(newton, body, key)
        series[key] = wrap_angle(x_extras - x_newton)
        fits[key] = fit_secular_rate(t, series[key])
    el = osculating(runs.newton.state_at(0), body)
    raan_j2, argp_j2 = kaula_rates(el["a"], el["e"], el["inc_equator"], el["gm"], j2=runs.j2)
    raan_lt, argp_lt = lense_thirring_rates(el["a"], el["e"], el["inc_equator"], gj=runs.gj)
    predicted = {
        "raan_j2": float(rate_arcsec_per_century(raan_j2)),
        "argp_j2": float(rate_arcsec_per_century(argp_j2)),
        "raan_lt": float(rate_arcsec_per_century(raan_lt)),
        "argp_lt": float(rate_arcsec_per_century(argp_lt)),
    }
    predicted["raan"] = predicted["raan_j2"] + predicted["raan_lt"]
    predicted["argp"] = predicted["argp_j2"] + predicted["argp_lt"]
    predicted["varpi"] = predicted["raan"] + predicted["argp"]
    return NodePerigee(
        t=t,
        d_raan=series["raan"],
        d_argp=series["argp"],
        raan_rate=float(fits["raan"].rate_arcsec_per_century),
        raan_err=float(fits["raan"].rate_err_arcsec_per_century),
        argp_rate=float(fits["argp"].rate_arcsec_per_century),
        argp_err=float(fits["argp"].rate_err_arcsec_per_century),
        varpi_rate=float(fits["varpi"].rate_arcsec_per_century),
        varpi_err=float(fits["varpi"].rate_err_arcsec_per_century),
        predicted=predicted,
        elements=el,
    )


@dataclass(frozen=True)
class MoonRates:
    """The lunar orbit about the Earth in a run: months, perigee and node rates, references."""

    t: np.ndarray
    varpi: np.ndarray  # osculating longitude of perigee [rad], ecliptic frame
    raan: np.ndarray  # osculating node [rad]
    sidereal_days: float  # 2 pi over the fitted mean motion of the mean longitude
    perigee_rate: float  # deg per Julian year
    perigee_err: float
    node_rate: float
    node_err: float
    perigee_reference: float  # constants.MOON_PERIGEE_RATE_DEG_YR
    node_reference: float
    tidal: TidalRates  # the analytic series, in rad/day
    mean_distance_km: float
    e_mean: float
    inc_mean_deg: float  # to the ecliptic


def moon_rates(runs: ExtrasRuns) -> MoonRates:
    """Fit the lunar months and the perigee and node rates of the Newtonian run of ``runs``.

    Raises
    ------
    ValueError
        If the runs do not contain the Moon (``with_moon=False``).
    """
    traj = runs.newton
    if "Moon" not in traj.names or "Earth" not in traj.names:
        raise ValueError("moon_rates needs runs made with with_moon=True")
    r_rel, v_rel = traj.relative("Moon", "Earth")
    gm_em = float(traj.gm[traj.index("Earth")] + traj.gm[traj.index("Moon")])
    el = state_to_elements(r_rel, v_rel, gm_em)
    n_sidereal = fit_secular_rate(traj.t, el["varpi"] + el["M"]).rate
    fit_varpi = fit_secular_rate(traj.t, el["varpi"])
    fit_raan = fit_secular_rate(traj.t, el["raan"])
    r_sun, v_sun = traj.relative("Earth", "Sun")
    el_sun = state_to_elements(r_sun, v_sun, C.GM_SUN + gm_em)
    n_moon = math.sqrt(gm_em / float(np.mean(el["a"])) ** 3)
    n_sun = math.sqrt((C.GM_SUN + gm_em) / float(np.mean(el_sun["a"])) ** 3)
    return MoonRates(
        t=traj.t.copy(),
        varpi=np.asarray(el["varpi"]),
        raan=np.asarray(el["raan"]),
        sidereal_days=_TWO_PI / n_sidereal if n_sidereal > 0.0 else math.nan,
        perigee_rate=float(fit_varpi.rate * DEG_PER_YEAR),
        perigee_err=float(fit_varpi.rate_err * DEG_PER_YEAR),
        node_rate=float(fit_raan.rate * DEG_PER_YEAR),
        node_err=float(fit_raan.rate_err * DEG_PER_YEAR),
        perigee_reference=C.MOON_PERIGEE_RATE_DEG_YR,
        node_reference=C.MOON_NODE_RATE_DEG_YR,
        tidal=moon_tidal_rates(n_moon, n_sun),
        mean_distance_km=float(np.linalg.norm(r_rel, axis=1).mean() * C.AU_KM),
        e_mean=float(np.mean(el["e"])),
        inc_mean_deg=float(np.mean(el["inc"]) * C.DEG_PER_RAD),
    )


# --------------------------------------------------------------------------- the budget
@dataclass(frozen=True)
class BudgetEntry:
    """One contribution to Mercury's perihelion rate [arcsec per Julian century, signed]."""

    name: str
    rate: float
    modelled: bool  # present in the gravsim runs of the lab
    note: str


def precession_budget(j2_scale: float = 1.0, lt_scale: float = 1.0) -> tuple[BudgetEntry, ...]:
    """The precession budget of Mercury, with the solar terms multiplied by the lab's scales.

    Six entries in decreasing size: planets, GR, solar J2 (x ``j2_scale``), the Moon as a body
    versus the EMB, Lense-Thirring (x ``lt_scale``) and the asteroid belt (not modelled).
    """
    asteroid = ring_precession_rate(ASTEROID_RING_GM, ASTEROID_RING_AU, 0.3871)
    return (
        BudgetEntry(
            "planets (Newtonian)",
            C.MERCURY_PRECESSION_NEWTON_ARCSEC_CY,
            True,
            "Clemence 1947; Lesson 02",
        ),
        BudgetEntry(
            "General Relativity", C.MERCURY_PRECESSION_GR_ARCSEC_CY, True, "6πGM/(c²a(1−e²))"
        ),
        BudgetEntry(
            f"solar J₂ × {j2_scale:g}",
            C.MERCURY_PRECESSION_J2_ARCSEC_CY * j2_scale,
            True,
            "Kaula: (3/2) n J₂ (R/p)², J₂ = 2.2e-7",
        ),
        BudgetEntry(
            "Moon as a body vs EMB",
            MOON_EMB_SHIFT_ARCSEC_CY,
            True,
            "run-vs-run difference, Lesson 09",
        ),
        BudgetEntry(
            f"Lense–Thirring × {lt_scale:g}",
            C.MERCURY_PRECESSION_LT_ARCSEC_CY * lt_scale,
            True,
            "frame dragging by the solar spin",
        ),
        BudgetEntry(
            "asteroid belt",
            asteroid,
            False,
            "ring of 1.2e-9 M☉ at 2.7 AU, not modelled",
        ),
    )


# --------------------------------------------------------------------------- cached runs
def _cached(message: str) -> Callable:
    """``st.cache_data(show_spinner=message)``, or the identity when streamlit is absent."""
    if st is None:
        return lambda fn: fn
    return st.cache_data(show_spinner=message)


@_cached("integrating with Newton and with Newton + J2 + Lense-Thirring ...")
def _extras_lab(years: float, j2_scale: float, lt_scale: float, with_moon: bool):
    runs = extras_runs(years, j2_scale, lt_scale, with_moon)
    return runs, node_perigee_rates(runs), (moon_rates(runs) if with_moon else None)


# --------------------------------------------------------------------------- front end
def _budget_figure(j2_scale: float, lt_scale: float, palette: tuple[str, ...], ink: str):
    entries = precession_budget(j2_scale, lt_scale)
    names = [e.name for e in entries][::-1]
    rates = np.array([e.rate for e in entries])[::-1]
    colors = [
        (palette[0] if e.rate >= 0.0 else palette[3]) if e.modelled else "#898781"
        for e in entries
    ][::-1]
    fig = go.Figure(
        go.Bar(
            x=np.abs(rates),
            y=names,
            orientation="h",
            marker={"color": colors},
            text=[f"{r:+.4g} ″/cy" for r in rates],
            textposition="outside",
            hovertemplate="%{y}: %{text}<extra></extra>",
        )
    )
    fig.add_vline(
        x=0.010, line={"color": ink, "dash": "dot", "width": 1},
        annotation_text="ranging precision ~0.01 ″/cy", annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis={"type": "log", "title": "|rate| [arcsec per century]", "range": [-3.3, 3.3]},
        height=360,
        margin={"l": 10, "r": 20, "t": 40, "b": 50},
        title="Mercury's perihelion budget (blue advances, red regresses, grey unmodelled)",
    )
    return fig


def _axis_trace(length: float, ink: str):
    n = np.asarray(C.SUN_POLE_ECLIPTIC)
    seg = np.array([-length * n, length * n])
    return go.Scatter3d(
        x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", name="solar spin axis",
        line={"color": ink, "width": 4, "dash": "dash"}, hoverinfo="name",
    )


def render() -> None:
    """Draw the competing-effects lab (needs streamlit and plotly; called by the app)."""
    if st is None or go is None:
        raise ImportError("the extras lab needs streamlit and plotly: uv pip install -e '.[app]'")
    from gravsim.viz import PALETTE, orbits_3d_figure, precession_figure
    from gravsim.viz.plots import INK

    st.header("Competing effects: what else moves Mercury's perihelion?")
    st.markdown(
        "Besides the planets and General Relativity, Mercury's perihelion feels the quadrupole "
        "field of the rotating Sun ($J_2$, $+0.028''$/cy), the gravitomagnetic field of its spin "
        "(Lense–Thirring frame dragging, $-0.002''$/cy) and the Earth–Moon pair's departure "
        "from a point mass ($-0.0025''$/cy). The lab adds the solar terms to Newton, amplified, "
        "and fits the **node** $\\Omega$ and the **argument of perihelion** $\\omega$ in the "
        "solar-equator frame against the first-order formulas — GR moves only $\\omega$, "
        "$J_2$ and frame dragging move both (Lesson 09)."
    )
    c1, c2, c3 = st.columns(3)
    with_moon = c1.toggle("Earth and Moon as separate bodies (full system)", value=False)
    use_j2 = c2.toggle("solar oblateness J₂", value=True)
    use_lt = c3.toggle("Lense–Thirring frame dragging", value=True)
    s1, s2, s3 = st.columns(3)
    j2_log = s1.slider("J₂ amplification, log10", J2_LOG_RANGE[0], J2_LOG_RANGE[1], 3.0, 0.5)
    lt_log = s2.slider(
        "J amplification (frame dragging), log10", LT_LOG_RANGE[0], LT_LOG_RANGE[1], 5.0, 0.5
    )
    years = s3.slider("span [years]", 2.0, 30.0, 10.0, step=1.0)
    j2_scale = 10.0**j2_log if use_j2 else 0.0
    lt_scale = 10.0**lt_log if use_lt else 0.0
    runs, res, moon = _extras_lab(years, j2_scale, lt_scale, with_moon)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("node Ω, fit", f"{res.raan_rate:+.4g} ± {res.raan_err:.2g} ″/cy")
    m2.metric("node Ω, formula", f"{res.predicted['raan']:+.4g} ″/cy")
    m3.metric("argument ω, fit", f"{res.argp_rate:+.4g} ± {res.argp_err:.2g} ″/cy")
    m4.metric("argument ω, formula", f"{res.predicted['argp']:+.4g} ″/cy")

    left, right = st.columns(2)
    with left:
        if with_moon:
            fig = orbits_3d_figure(runs.newton, ["Moon"], central="Earth")
            fig.update_layout(
                height=480, title=f"the Moon about the Earth, {years:g} yr (perigee +40.7°/yr)"
            )
        else:
            fig = orbits_3d_figure(runs.extras, ["Mercury"])
            fig.add_trace(_axis_trace(0.5, INK))
            fig.update_layout(height=480, title=f"Sun + Mercury, {years:g} yr, with the spin axis")
        st.plotly_chart(fig, width="stretch")
    with right:
        rate_raan = res.raan_rate / (C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS)
        fig = precession_figure(res.t, res.d_raan, rate_raan, units="arcsec")
        fig.data[0].update(name="ΔΩ (node)", line={"color": PALETTE[0]})
        rate_argp = res.argp_rate / (C.ARCSEC_PER_RAD * C.JULIAN_CENTURY_DAYS)
        fig_w = precession_figure(res.t, res.d_argp, rate_argp, units="arcsec")
        fig_w.data[0].update(name="Δω (argument of perihelion)", line={"color": PALETTE[1]})
        fig.add_trace(fig_w.data[0])
        if len(fig_w.data) > 1:
            fig_w.data[1].update(name=f"fit ω: {res.argp_rate:+.4g} ″/cy", line={"color": INK})
            fig.add_trace(fig_w.data[1])
        fig.update_layout(
            title="(Newton + extras) − Newton, solar-equator frame", height=480,
            yaxis_title="Δ angle [arcsec]",
        )
        st.plotly_chart(fig, width="stretch")
    el = res.elements
    pred = res.predicted
    st.caption(
        f"J₂ × {j2_scale:g}: Ω {pred['raan_j2']:+.4g}, ω {pred['argp_j2']:+.4g} ″/cy; "
        f"frame dragging × {lt_scale:g}: Ω {pred['raan_lt']:+.4g}, ω {pred['argp_lt']:+.4g} ″/cy; "
        f"ϖ = Ω + ω fitted {res.varpi_rate:+.4g} ± {res.varpi_err:.2g} against "
        f"{pred['varpi']:+.4g}. Mercury: a = {el['a']:.4f} AU, e = {el['e']:.4f}, inclination to "
        f"the solar equator {math.degrees(el['inc_equator']):.2f}°. "
        "The oblate Sun regresses the node and advances the argument; frame dragging does the "
        "opposite: the node is dragged in the sense of the spin, the argument driven back three "
        "times faster. Rates are run-vs-run differences on one grid."
    )

    if moon is not None:
        st.subheader("The Moon: the largest precession in the solar system")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("sidereal month", f"{moon.sidereal_days:.4f} d", "27.3217 d")
        n2.metric(
            "perigee rate", f"{moon.perigee_rate:+.2f} ± {moon.perigee_err:.2f} °/yr",
            f"{moon.perigee_reference:+.2f} °/yr secular",
        )
        n3.metric(
            "node rate", f"{moon.node_rate:+.2f} ± {moon.node_err:.2f} °/yr",
            f"{moon.node_reference:+.2f} °/yr secular",
        )
        leading = moon.tidal.leading * DEG_PER_YEAR
        n4.metric("(3/4) m² n, first order", f"±{leading:.1f} °/yr")
        st.caption(
            f"Purely Newtonian: the Sun's tide on the lunar orbit (m = n☉/n☾ = {moon.tidal.m:.4f})."
            f" To first order perigee and node move by ±{leading:.1f}°/yr; the higher orders "
            f"(Clairaut 1749) double the perigee rate: the series to m⁵ gives "
            f"{moon.tidal.perigee * DEG_PER_YEAR:+.1f} and {moon.tidal.node * DEG_PER_YEAR:+.1f}"
            f"°/yr. Mean distance {moon.mean_distance_km:,.0f} km, e = {moon.e_mean:.4f}, "
            f"inclination to "
            f"the ecliptic {moon.inc_mean_deg:.2f}°. Short spans carry the evection and variation "
            "as noise in the fitted rates."
        )

    st.subheader("The budget")
    st.plotly_chart(_budget_figure(j2_scale, lt_scale, PALETTE, INK), width="stretch")
    st.caption(
        "With Mercury's perihelion alone, J₂ and the PPN combination 2γ − β are degenerate: both "
        "advance ϖ. What separates them: J₂ also moves the node (GR does not), the J₂/GR ratio "
        "falls as 1/(a(1−e²)) from Mercury to Mars, and helioseismology measures J₂ directly. "
        "Frame dragging is a hundred times below today's ranging precision for Mercury; it has "
        "been measured around the Earth (Gravity Probe B, LAGEOS)."
    )
