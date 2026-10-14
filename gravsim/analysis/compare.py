"""Run-versus-run comparisons: separations and differential perihelion rates.

Physics
-------
The GR signal in the solar system is ~1e-8 relative, far below the periodic planetary wobble of
any osculating element.  Comparing two runs that share the initial state, time grid and integrator
and differ only in the force law (Newton vs Newton + 1PN, or beta/gamma values) cancels the wobble
*and* most of the numerical error, leaving the clean secular difference (ROADMAP sec. 1.2).

* ``separation``                  ``|r_A(t) - r_B(t)|`` of one body, same time grid.
* ``perihelion_rate_difference``  fits ``varpi_A(t) - varpi_B(t)`` (differenced sample by sample,
  wrapped to ``(-pi, pi]``, then unwrapped) with ``fit_secular_rate`` and returns the slope in
  arcsec per Julian century with its OLS standard error.

Time grids must be identical (same length and the same values to within ``t_atol`` days,
default exactly equal); a ``ValueError`` is raised otherwise.
"""

from __future__ import annotations

import numpy as np

from gravsim.analysis.elements import _norm, wrap_angle
from gravsim.analysis.precession import (
    fit_secular_rate,
    perihelion_longitude_series,
    rate_arcsec_per_century,
)
from gravsim.bodies import Trajectory

__all__ = ["separation", "perihelion_rate_difference", "check_same_grid"]


def check_same_grid(trajA: Trajectory, trajB: Trajectory, t_atol: float = 0.0) -> None:
    """Raise ``ValueError`` unless the two trajectories share the same time grid.

    Parameters
    ----------
    trajA, trajB : Trajectory
    t_atol : float, optional
        Absolute tolerance on the times [day]; ``0`` (default) requires exact equality.
    """
    if trajA.t.shape != trajB.t.shape:
        raise ValueError(f"time grids differ in length: {trajA.t.size} vs {trajB.t.size}")
    if not np.allclose(trajA.t, trajB.t, rtol=0.0, atol=t_atol):
        raise ValueError("time grids differ; comparisons need identical sampling")


def separation(trajA: Trajectory, trajB: Trajectory, body: str, t_atol: float = 0.0) -> np.ndarray:
    """``|r_A - r_B|`` of ``body`` between two runs on the same time grid.

    Parameters
    ----------
    trajA, trajB : Trajectory
        Runs sharing the time grid.
    body : str
        Body name present in both.
    t_atol : float, optional
        Tolerance for the grid check [day] (default exact).

    Returns
    -------
    ndarray, shape (T,)
        Separation [AU].
    """
    check_same_grid(trajA, trajB, t_atol)
    rA, _ = trajA.body(body)
    rB, _ = trajB.body(body)
    return _norm(rA - rB)


def perihelion_rate_difference(
    trajA: Trajectory,
    trajB: Trajectory,
    body: str,
    central: str = "Sun",
    t_atol: float = 0.0,
) -> tuple[float, float]:
    """Secular rate of ``varpi_A - varpi_B`` for ``body``, in arcsec per Julian century.

    Parameters
    ----------
    trajA, trajB : Trajectory
        Runs sharing the time grid (e.g. GR and Newton).
    body, central : str
        Body names.
    t_atol : float, optional
        Tolerance for the grid check [day] (default exact).

    Returns
    -------
    rate : float
        Slope of the differenced, unwrapped series [arcsec/century] (A minus B).
    err : float
        OLS standard error of the slope [arcsec/century].
    """
    check_same_grid(trajA, trajB, t_atol)
    t, varpi_a = perihelion_longitude_series(trajA, body, central)
    _, varpi_b = perihelion_longitude_series(trajB, body, central)
    fit = fit_secular_rate(t, wrap_angle(varpi_a - varpi_b))
    return float(rate_arcsec_per_century(fit.rate)), float(rate_arcsec_per_century(fit.rate_err))
