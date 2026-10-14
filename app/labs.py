"""Computational back end of the gravsim Streamlit labs (no Streamlit import, pure functions).

Every function here takes plain numbers and strings, runs the package's own force laws,
integrators and measurement toolkit, and returns numpy arrays or ``Trajectory`` objects.
The Streamlit front end (``streamlit_app.py``) only adds widgets, caching and plotly figures,
so this module can be tested without Streamlit (``tests/test_app.py``).

Physics (details in the module docstrings of the package)
---------------------------------------------------------
* Integrator lab: Sun + Mercury with the fixed-step schemes of ``gravsim.integrators``;
  the relative energy error ``(E - E0)/|E0|`` and the spurious precession of the longitude of
  perihelion ``varpi`` (least-squares slope, arcsec per Julian century) measure the method.
* Solar system lab: Sun + 8 planets, DOP853, Newton versus Newton + 1PN with the speed of
  light ``c`` as a parameter (``c_scale`` divides ``constants.C_AU_DAY``).  The GR-minus-Newton
  displacement of a body is drawn amplified, ``r_N + k (r_GR - r_N)``, because at the real ``c``
  it is ~1e-8 relative.
* Precession tracker: ``varpi(t)`` of a planet in both runs, the fitted secular rates and the
  run-vs-run difference (``analysis.compare.perihelion_rate_difference``), against the analytic
  1PN rate ``6 pi GM / (c^2 a (1 - e^2))`` per orbit.
* Effective-potential explorer: Schwarzschild ``V_eff(r) = -GM/r + L^2/2r^2 - GM L^2/c^2 r^3``
  in geometric units ``GM = c = 1``; the orbit for a given ``(L, E)`` from the Binet equation.

Units: AU, day, GM in AU^3/day^2 everywhere except the explorer (geometric units).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gravsim import constants as C
from gravsim.analysis import (
    fit_secular_rate,
    gr_precession_arcsec_per_century,
    perihelion_longitude_series,
    perihelion_rate_difference,
    relative_energy_error,
    state_to_elements,
    wrap_angle,
)
from gravsim.bodies import SystemState, Trajectory
from gravsim.data.horizons import load_j2000
from gravsim.forces import accel_newton, make_newton_pn1
from gravsim.integrators import INTEGRATORS, integrate, integrate_scipy
from gravsim.metric import schwarzschild as sch

INNER_PLANETS: tuple[str, ...] = C.PLANETS[:4]
FIXED_STEP_NAMES: tuple[str, ...] = tuple(INTEGRATORS)


# --------------------------------------------------------------------------- helpers
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


# --------------------------------------------------------------------------- lab 1
@dataclass(frozen=True)
class IntegratorResult:
    """One fixed-step run of Sun + Mercury and its two error meters."""

    traj: Trajectory
    energy_error: np.ndarray  # (T,) relative
    spurious_rate: float  # arcsec per Julian century
    spurious_rate_err: float
    steps_per_orbit: float
    n_steps: int
    period: float  # days


def integrator_lab(name: str, dt: float, years: float) -> IntegratorResult:
    """Run Sun + Mercury with a fixed-step integrator and measure its errors.

    Parameters
    ----------
    name : str
        Key of ``gravsim.integrators.INTEGRATORS`` (``"euler"``, ``"leapfrog"``, ...).
    dt : float
        Requested step [day]; the driver shrinks it minimally to land on ``t_end``.
    years : float
        Span in Julian years.

    Returns
    -------
    IntegratorResult
    """
    state = two_body_state()
    period = osculating(state, "Mercury")["period"]
    t_end = state.t + years * C.JULIAN_YEAR_DAYS
    save_every = max(1, int(round(1.0 / dt)))  # about one sample per day
    traj = integrate(state, accel_newton, t_end, dt, name, save_every=save_every)
    t, varpi = perihelion_longitude_series(traj, "Mercury")
    fit = fit_secular_rate(t, varpi)
    return IntegratorResult(
        traj=traj,
        energy_error=relative_energy_error(traj),
        spurious_rate=float(fit.rate_arcsec_per_century),
        spurious_rate_err=float(fit.rate_err_arcsec_per_century),
        steps_per_orbit=period / traj.meta["dt"],
        n_steps=int(traj.meta["n_steps"]),
        period=period,
    )


# --------------------------------------------------------------------------- lab 2
def solar_system_runs(
    years: float,
    c_scale: float = 1.0,
    bodies: tuple[str, ...] = C.DEFAULT_BODIES,
    step: float = 1.0,
) -> tuple[Trajectory, Trajectory]:
    """Newton and Newton + 1PN runs of ``bodies`` (DOP853) on a common grid.

    Parameters
    ----------
    years : float
        Span in Julian years.
    c_scale : float
        The speed of light used is ``C_AU_DAY / c_scale`` (1 = reality, 100 = visible rosette).
    bodies : tuple of str
        Subset of ``constants.DEFAULT_BODIES``; must contain ``"Sun"``.
    step : float
        Sample spacing [day].

    Returns
    -------
    (newton, gr) : tuple of Trajectory
    """
    if "Sun" not in bodies:
        raise ValueError("the Sun must be among the bodies")
    state = load_j2000().subset(tuple(bodies))
    t_eval = daily_grid(state.t, years, step)
    newton = integrate_scipy(state, accel_newton, t_eval)
    law = make_newton_pn1(c=C.C_AU_DAY / c_scale, central=state.index("Sun"))
    gr = integrate_scipy(state, law, t_eval)
    return newton, gr


def amplified_difference(
    newton: Trajectory, gr: Trajectory, body: str, factor: float, central: str = "Sun"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions of ``body`` relative to ``central``: Newton, GR and the amplified GR path.

    Returns
    -------
    r_newton, r_gr, r_amplified : ndarray, shape (T, 3)
        ``r_amplified = r_newton + factor * (r_gr - r_newton)``.
    """
    r_n = newton.relative(body, central)[0]
    r_g = gr.relative(body, central)[0]
    return r_n, r_g, r_n + factor * (r_g - r_n)


def separation_km(newton: Trajectory, gr: Trajectory, body: str) -> np.ndarray:
    """``|r_GR - r_Newton|`` of ``body`` in km (same grid assumed)."""
    r_n = newton.body(body)[0]
    r_g = gr.body(body)[0]
    return np.linalg.norm(r_g - r_n, axis=1) * C.AU_KM


# --------------------------------------------------------------------------- lab 3
@dataclass(frozen=True)
class PrecessionTrack:
    """``varpi(t)`` of one body in the Newton and GR runs with the fitted rates."""

    t: np.ndarray
    varpi_newton: np.ndarray
    varpi_gr: np.ndarray
    rate_newton: float  # rad/day
    rate_gr: float
    difference: np.ndarray  # wrapped varpi_gr - varpi_newton [rad]
    rate_difference: float  # arcsec per Julian century
    rate_difference_err: float
    analytic: float  # arcsec per Julian century (1PN, same c)
    elements: dict[str, float]


def precession_track(
    newton: Trajectory, gr: Trajectory, body: str, c_scale: float = 1.0
) -> PrecessionTrack:
    """Track the perihelion of ``body`` in two runs and fit the rates."""
    t, w_n = perihelion_longitude_series(newton, body)
    _, w_g = perihelion_longitude_series(gr, body)
    fit_n = fit_secular_rate(t, w_n)
    fit_g = fit_secular_rate(t, w_g)
    rate, err = perihelion_rate_difference(gr, newton, body)
    el = osculating(newton.state_at(0), body)
    analytic = float(
        gr_precession_arcsec_per_century(el["a"], el["e"], el["gm"], c=C.C_AU_DAY / c_scale)
    )
    return PrecessionTrack(
        t=t,
        varpi_newton=w_n,
        varpi_gr=w_g,
        rate_newton=float(fit_n.rate),
        rate_gr=float(fit_g.rate),
        difference=wrap_angle(w_g - w_n),
        rate_difference=rate,
        rate_difference_err=err,
        analytic=analytic,
        elements=el,
    )


# --------------------------------------------------------------------------- lab 4
@dataclass(frozen=True)
class PotentialCurves:
    """Schwarzschild and Newtonian effective potentials for one ``L`` (geometric units)."""

    r: np.ndarray
    v_gr: np.ndarray
    v_newton: np.ndarray
    r_stable: float  # nan when L^2 < 12
    r_unstable: float
    v_top: float  # V_eff at the barrier (nan when absent)
    v_well: float  # V_eff at the stable circular orbit (nan when absent)


def potential_curves(
    L: float, r_min: float = 2.05, r_max: float = 60.0, n: int = 1500
) -> PotentialCurves:
    """Sample ``V_eff`` for angular momentum ``L`` (units of ``GM/c``) with ``GM = c = 1``."""
    r = np.linspace(r_min, r_max, n)
    v_gr = sch.effective_potential(r, L, 1.0, 1.0)
    v_n = sch.newton_effective_potential(r, L, 1.0)
    r_st, r_un = sch.circular_orbit_radii(L, 1.0, 1.0)
    v_top = float(sch.effective_potential(r_un, L, 1.0, 1.0)) if np.isfinite(r_un) else math.nan
    v_well = float(sch.effective_potential(r_st, L, 1.0, 1.0)) if np.isfinite(r_st) else math.nan
    return PotentialCurves(r, v_gr, v_n, float(r_st), float(r_un), v_top, v_well)


@dataclass(frozen=True)
class StrongFieldOrbit:
    """A timelike geodesic in geometric units, ready to draw in the equatorial plane."""

    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    tau: np.ndarray  # proper time
    t: np.ndarray  # coordinate time
    captured: bool
    r_min: float
    turns: float  # azimuthal turns completed


def strong_field_orbit(
    L: float,
    energy: float,
    r0: float,
    tau_end: float = 2000.0,
    inward: bool = True,
    n_samples: int = 6000,
) -> StrongFieldOrbit:
    """Integrate the geodesic with angular momentum ``L`` and effective energy ``energy``.

    ``energy`` is ``(E^2/c^2 - c^2)/2 = v_r^2/2 + V_eff(r0)``; it must be ``>= V_eff(r0)`` so
    that the radial velocity is real (clipped to zero otherwise, i.e. ``r0`` is taken as an
    apsis).  The sign of the initial radial velocity is set by ``inward``.  ``GM = c = 1``.  The
    proper-time integration (``metric.schwarzschild.geodesic_orbit``) follows bound, escaping
    and plunging orbits alike; a plunge stops at ``1.001 r_s`` and sets ``captured``.
    """
    v_phi = L / r0
    v_r2 = 2.0 * (energy - float(sch.effective_potential(r0, L, 1.0, 1.0)))
    v_r = math.sqrt(max(v_r2, 0.0)) * (-1.0 if inward else 1.0)
    tau, t, r, phi = sch.geodesic_orbit(
        [r0, 0.0], [v_r, v_phi], 1.0, 1.0, tau_end=tau_end, n_samples=n_samples
    )
    captured = bool(r.size < n_samples)  # the capture event truncates the output
    return StrongFieldOrbit(
        x=r * np.cos(phi),
        y=r * np.sin(phi),
        r=r,
        phi=phi,
        tau=tau,
        t=t,
        captured=captured,
        r_min=float(r.min()),
        turns=float(abs(phi[-1] - phi[0]) / (2.0 * math.pi)),
    )
