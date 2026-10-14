"""Osculating-element primitives: angular momentum, node and eccentricity (LRL) vectors.

Physics
-------
For the relative two-body problem with gravitational parameter ``gm`` (= G (M + m), AU^3/day^2),
``r`` the relative position (AU) and ``v`` the relative velocity (AU/day):

* specific angular momentum   ``h = r x v``                          [AU^2/day]
* node vector                 ``n = z_hat x h = (-h_y, h_x, 0)``     (toward the ascending node)
* eccentricity vector         ``e = (v x h)/gm - r/|r|``             [dimensionless]

``e`` is the Laplace-Runge-Lenz vector divided by ``gm``: its magnitude is the eccentricity and it
points from the focus to perihelion.  It is an exact constant of the Kepler problem, so a secular
drift of its direction measures a perturbation (planetary pulls, the 1PN term, ...).

Angles follow the classical convention (Bate, Mueller & White 1971, *Fundamentals of
Astrodynamics*, sec. 2.4; Vallado 2013, *Fundamentals of Astrodynamics and Applications*, alg. 9):

* longitude of the ascending node   ``raan = atan2(n_y, n_x)``
* argument of perihelion            ``argp`` = angle from ``n`` to ``e`` measured in the orbital
  plane in the direction of motion, i.e. ``atan2(h_hat . (n_hat x e_hat), n_hat . e_hat)``
* longitude of perihelion           ``varpi = raan + argp``  (a "dog-leg" angle, but continuous
  and well conditioned when ``inc -> 0``, where ``raan`` and ``argp`` separately are not)

Degenerate cases: for an equatorial orbit (``|n| = 0``, i.e. ``inc = 0`` or ``pi``) the node is
undefined and is taken along +x, so that ``varpi = atan2(e_y, e_x)`` in the prograde case.  For a
circular orbit (``e = 0``) the perihelion is undefined and ``argp = 0`` (``varpi = raan``).
Rectilinear orbits (``h = 0``) are not supported.

All functions broadcast over leading axes: inputs ``(..., 3)`` give outputs ``(...)`` or
``(..., 3)``.  Angles are radians, wrapped to ``(-pi, pi]``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "wrap_angle",
    "specific_h",
    "node_vector",
    "lrl_vector",
    "perihelion_longitude",
]

NODE_TOL = 1e-14
"""Relative threshold ``|n| / |h|`` below which an orbit is treated as equatorial."""

_X_HAT = np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------- helpers
def _norm(x: np.ndarray) -> np.ndarray:
    """Euclidean norm along the last axis."""
    return np.sqrt(np.sum(x * x, axis=-1))


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Dot product along the last axis."""
    return np.sum(x * y, axis=-1)


def _unit(x: np.ndarray) -> np.ndarray:
    """Unit vector along the last axis (a zero vector stays zero)."""
    m = _norm(x)
    safe = np.where(m > 0.0, m, 1.0)
    return x / safe[..., None]


def wrap_angle(x: ArrayLike) -> np.ndarray:
    """Wrap an angle (radians) to the interval ``(-pi, pi]``.

    Parameters
    ----------
    x : array_like
        Angle(s) in radians.

    Returns
    -------
    ndarray
        Same shape as ``x``; ``pi`` maps to ``pi`` and ``-pi`` maps to ``pi``.
    """
    x = np.asarray(x, dtype=np.float64)
    two_pi = 2.0 * np.pi
    return x - two_pi * np.ceil((x - np.pi) / two_pi)


# --------------------------------------------------------------------------- vectors
def specific_h(r: ArrayLike, v: ArrayLike) -> np.ndarray:
    """Specific angular momentum ``h = r x v``.

    Parameters
    ----------
    r, v : array_like, shape (..., 3)
        Relative position [AU] and velocity [AU/day].

    Returns
    -------
    ndarray, shape (..., 3)
        ``h`` in AU^2/day.
    """
    return np.cross(np.asarray(r, dtype=np.float64), np.asarray(v, dtype=np.float64))


def node_vector(h: ArrayLike, tol: float = NODE_TOL) -> np.ndarray:
    """Node vector ``n = z_hat x h = (-h_y, h_x, 0)``, pointing to the ascending node.

    Parameters
    ----------
    h : array_like, shape (..., 3)
        Specific angular momentum.
    tol : float, optional
        Orbits with ``|n| <= tol * |h|`` are equatorial; their node is undefined and the unit
        vector +x is returned instead (so ``raan = 0`` there).

    Returns
    -------
    ndarray, shape (..., 3)
        ``n`` (magnitude ``|h| sin(inc)``), or ``(1, 0, 0)`` for equatorial orbits.
    """
    h = np.asarray(h, dtype=np.float64)
    n = np.stack([-h[..., 1], h[..., 0], np.zeros_like(h[..., 2])], axis=-1)
    degenerate = _norm(n) <= tol * _norm(h)
    return np.where(degenerate[..., None], _X_HAT, n)


def lrl_vector(r: ArrayLike, v: ArrayLike, gm: ArrayLike) -> np.ndarray:
    """Eccentricity vector ``e = (v x h)/gm - r/|r|`` (Laplace-Runge-Lenz vector over ``gm``).

    Parameters
    ----------
    r, v : array_like, shape (..., 3)
        Relative position [AU] and velocity [AU/day].
    gm : array_like, scalar or shape (...)
        Gravitational parameter G (M + m) [AU^3/day^2].

    Returns
    -------
    ndarray, shape (..., 3)
        Dimensionless vector of magnitude ``e`` pointing from the focus to perihelion.
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)
    h = specific_h(r, v)
    return np.cross(v, h) / gm[..., None] - r / _norm(r)[..., None]


def perihelion_longitude(r: ArrayLike, v: ArrayLike, gm: ArrayLike) -> np.ndarray:
    """Longitude of perihelion ``varpi = raan + argp`` from a relative state.

    ``raan`` is taken from the node vector and ``argp`` is the angle from the node to the
    eccentricity vector, measured in the orbital plane in the direction of motion (sign from
    ``h``).  For an equatorial prograde orbit this reduces to ``atan2(e_y, e_x)``; for a circular
    orbit it reduces to ``raan``.

    Parameters
    ----------
    r, v : array_like, shape (..., 3)
        Relative position [AU] and velocity [AU/day].
    gm : array_like, scalar or shape (...)
        Gravitational parameter G (M + m) [AU^3/day^2].

    Returns
    -------
    ndarray, shape (...)
        ``varpi`` in radians, wrapped to ``(-pi, pi]``.
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    h = specific_h(r, v)
    h_hat = _unit(h)
    n_hat = _unit(node_vector(h))
    e_hat = _unit(lrl_vector(r, v, gm))
    raan = np.arctan2(n_hat[..., 1], n_hat[..., 0])
    argp = np.arctan2(_dot(np.cross(n_hat, e_hat), h_hat), _dot(n_hat, e_hat))
    return wrap_angle(raan + argp)
