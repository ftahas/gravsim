"""Perihelion tracking and secular precession rates.

Physics
-------
Perihelion longitude.  For a body orbiting ``central`` we take the relative state
``r = r_body - r_central``, ``v = v_body - v_central`` with ``gm = gm_central + gm_body`` and
compute the osculating longitude of perihelion ``varpi = raan + argp`` from the eccentricity
(LRL) vector (``gravsim.analysis.elements``).  Along an unperturbed Kepler orbit ``varpi`` is
constant; perturbations make it drift.  The drift has a periodic part (planetary wobble,
numerical noise) and a secular part ``varpi(t) ~ varpi_0 + rate * t``; the secular rate is the
"precession of the perihelion".

Perihelion passages.  A passage is a strict local minimum of ``|r(t)|`` on the sampled series,
refined by the vertex of the parabola through the three samples around the minimum.  Note that
these minima are spaced by the *anomalistic* period, which differs from the sidereal period by the
precession itself; ``varpi`` at each passage gives a lower-noise but sparser series.

Secular fit.  Ordinary least squares ``angle = intercept + rate * t`` after ``numpy.unwrap``.
The standard error of the slope is the textbook OLS estimate
``rate_err = sqrt( sum(res^2) / (n - 2) / sum((t - <t>)^2) )``, which treats the residuals as
white noise; for strongly periodic residuals it is a nominal, not a rigorous, uncertainty.

GR precession (1PN, PPN-parametrised).  Per orbit, in the harmonic/isotropic gauge
(Will 2014, *Living Rev. Relativ.* 17:4, eq. 68; Weinberg 1972, sec. 8.6):

    dvarpi = (2 + 2 gamma - beta) / 3 * 6 pi gm / (c^2 a (1 - e^2))   [rad/orbit]

with beta = gamma = 1 in GR.  Multiplying by the number of orbits per Julian century,
``36525 / P`` with ``P = 2 pi sqrt(a^3 / gm)``, and by ``206264.8 arcsec/rad`` gives the familiar
42.98 arcsec/century for Mercury.

Units: AU, day, rad; rates in rad/day unless converted with ``rate_arcsec_per_century``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from gravsim.analysis.elements import _norm, perihelion_longitude
from gravsim.bodies import Trajectory
from gravsim.constants import ARCSEC_PER_RAD, C_AU_DAY, JULIAN_CENTURY_DAYS

__all__ = [
    "SecularFit",
    "perihelion_longitude_series",
    "perihelion_passages",
    "fit_secular_rate",
    "rate_arcsec_per_century",
    "gr_precession_arcsec_per_century",
]


def _relative_with_gm(
    traj: Trajectory, body: str, central: str
) -> tuple[np.ndarray, np.ndarray, float]:
    r, v = traj.relative(body, central)
    gm = float(traj.gm[traj.index(central)] + traj.gm[traj.index(body)])
    return r, v, gm


# --------------------------------------------------------------------------- series
def perihelion_longitude_series(
    traj: Trajectory, body: str, central: str = "Sun"
) -> tuple[np.ndarray, np.ndarray]:
    """Osculating longitude of perihelion of ``body`` about ``central`` along a trajectory.

    Parameters
    ----------
    traj : Trajectory
        Barycentric trajectory containing both bodies.
    body, central : str
        Body names; ``gm = gm[central] + gm[body]`` is used for the osculating elements.

    Returns
    -------
    t : ndarray, shape (T,)
        Copy of the trajectory times [day].
    varpi : ndarray, shape (T,)
        ``raan + argp`` [rad], wrapped to ``(-pi, pi]``.
    """
    r, v, gm = _relative_with_gm(traj, body, central)
    return traj.t.copy(), perihelion_longitude(r, v, gm)


def perihelion_passages(
    traj: Trajectory, body: str, central: str = "Sun"
) -> tuple[np.ndarray, np.ndarray]:
    """Perihelion passage times and the longitude of perihelion at each passage.

    Passages are local minima of ``|r_body - r_central|`` at interior samples (strictly below
    the previous sample, not above the next one, so a rounding-level tie counts once); each time
    is refined to the vertex of the parabola through the three samples around the minimum, and
    ``varpi`` is taken at the sample nearest to the refined time.  The first and last samples
    are never passages, so start a run away from perihelion if the first passage matters.

    Parameters
    ----------
    traj : Trajectory
        Trajectory with strictly increasing ``t``.
    body, central : str
        Body names.

    Returns
    -------
    t_peri : ndarray, shape (K,)
        Refined passage times [day] (empty if none found).
    varpi_peri : ndarray, shape (K,)
        Longitude of perihelion [rad] at the nearest sample to each passage.
    """
    t = traj.t
    if t.size >= 2 and not np.all(np.diff(t) > 0.0):
        raise ValueError("perihelion_passages needs strictly increasing times")
    r, v, gm = _relative_with_gm(traj, body, central)
    rmag = _norm(r)
    if t.size < 3:
        return np.empty(0), np.empty(0)

    # Strict on the left, non-strict on the right: a pair of exactly tied samples straddling
    # the minimum (it happens at rounding level when the passage sits on a half step) counts
    # once, at its left member; the parabola through the three points still finds the vertex.
    interior = (rmag[1:-1] < rmag[:-2]) & (rmag[1:-1] <= rmag[2:])
    idx = np.nonzero(interior)[0] + 1
    if idx.size == 0:
        return np.empty(0), np.empty(0)

    t0, t1, t2 = t[idx - 1], t[idx], t[idx + 1]
    r0, r1, r2 = rmag[idx - 1], rmag[idx], rmag[idx + 1]
    tau0, tau2 = t0 - t1, t2 - t1
    d0 = (r0 - r1) / tau0
    d2 = (r2 - r1) / tau2
    A = (d2 - d0) / (tau2 - tau0)  # > 0 for a strict minimum
    B = d0 - A * tau0
    t_peri = t1 - 0.5 * B / A

    candidates = np.stack([t0, t1, t2], axis=-1)
    nearest = idx - 1 + np.argmin(np.abs(candidates - t_peri[:, None]), axis=-1)
    varpi_peri = perihelion_longitude(r[nearest], v[nearest], gm)
    return t_peri, varpi_peri


# --------------------------------------------------------------------------- secular fit
@dataclass(frozen=True)
class SecularFit:
    """Result of a straight-line fit ``angle = intercept + rate * t``.

    Attributes
    ----------
    rate : float
        Slope [rad/day].
    rate_err : float
        Standard error of the slope [rad/day] (``nan`` with fewer than three points).
    intercept : float
        Fitted angle at ``t = 0`` [rad] (unwrapped, not re-wrapped).
    residuals : ndarray
        ``angle_unwrapped - (intercept + rate * t)`` [rad].
    """

    rate: float
    rate_err: float
    intercept: float
    residuals: np.ndarray

    @property
    def rate_arcsec_per_century(self) -> float:
        """Slope converted to arcsec per Julian century."""
        return rate_arcsec_per_century(self.rate)

    @property
    def rate_err_arcsec_per_century(self) -> float:
        """Slope uncertainty converted to arcsec per Julian century."""
        return rate_arcsec_per_century(self.rate_err)


def fit_secular_rate(t: ArrayLike, angle: ArrayLike) -> SecularFit:
    """Least-squares secular rate of an angle series (unwrapped first).

    Parameters
    ----------
    t : array_like, shape (T,)
        Times [day]; at least two distinct values.
    angle : array_like, shape (T,)
        Angle [rad], possibly wrapped; ``numpy.unwrap`` is applied before fitting.

    Returns
    -------
    SecularFit
        ``rate`` and ``rate_err`` in rad/day.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    angle = np.unwrap(np.asarray(angle, dtype=np.float64).ravel())
    if t.shape != angle.shape:
        raise ValueError(f"t and angle must have the same length, got {t.size} and {angle.size}")
    n = t.size
    if n < 2:
        raise ValueError("need at least two samples to fit a rate")
    t_mean = t.mean()
    a_mean = angle.mean()
    dt = t - t_mean
    da = angle - a_mean
    sxx = float(np.dot(dt, dt))
    if sxx == 0.0:
        raise ValueError("need at least two distinct times to fit a rate")
    rate = float(np.dot(dt, da) / sxx)
    intercept = float(a_mean - rate * t_mean)
    residuals = da - rate * dt
    if n > 2:
        rate_err = float(np.sqrt(np.dot(residuals, residuals) / (n - 2) / sxx))
    else:
        rate_err = float("nan")
    return SecularFit(rate=rate, rate_err=rate_err, intercept=intercept, residuals=residuals)


# --------------------------------------------------------------------------- conversions
def rate_arcsec_per_century(rate_rad_per_day: ArrayLike) -> np.ndarray:
    """Convert a rate from rad/day to arcsec per Julian century (36525 days)."""
    rate = np.asarray(rate_rad_per_day, dtype=np.float64)
    out = rate * ARCSEC_PER_RAD * JULIAN_CENTURY_DAYS
    return out[()] if out.ndim == 0 else out


def gr_precession_arcsec_per_century(
    a: ArrayLike,
    e: ArrayLike,
    gm: ArrayLike,
    c: float = C_AU_DAY,
    beta: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Analytic 1PN (PPN) perihelion advance in arcsec per Julian century.

    ``dvarpi = (2 + 2 gamma - beta)/3 * 6 pi gm / (c^2 a (1 - e^2))`` per orbit, times
    ``36525 / P`` orbits per century with ``P = 2 pi sqrt(a^3 / gm)``.

    Parameters
    ----------
    a : array_like
        Semi-major axis [AU].
    e : array_like
        Eccentricity.
    gm : array_like
        Gravitational parameter of the two-body problem [AU^3/day^2].
    c : float, optional
        Speed of light [AU/day] (default ``constants.C_AU_DAY``).
    beta, gamma : float, optional
        PPN parameters (GR: 1, 1).

    Returns
    -------
    ndarray
        Precession rate [arcsec/century].
    """
    a = np.asarray(a, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)
    per_orbit = (2.0 + 2.0 * gamma - beta) / 3.0 * 6.0 * np.pi * gm / (c * c * a * (1.0 - e * e))
    period = 2.0 * np.pi * np.sqrt(a**3 / gm)
    out = per_orbit * (JULIAN_CENTURY_DAYS / period) * ARCSEC_PER_RAD
    return out[()] if out.ndim == 0 else out
