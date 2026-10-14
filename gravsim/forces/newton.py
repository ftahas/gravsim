"""Pairwise Newtonian gravitational acceleration.

Physics
-------
For N point masses with gravitational parameters ``GM_j`` at positions ``r_j`` the
acceleration of body ``i`` is

    a_i = - sum_{j != i} GM_j (r_i - r_j) / |r_i - r_j|^3 .

Only the gravitational parameter of the *attracting* body ``j`` enters, so a body with
``GM = 0`` (a test particle) feels the field of every massive body but exerts no force.
Newton's third law holds pair by pair, which gives momentum conservation

    sum_i GM_i a_i = 0

to roundoff. No softening is applied: coincident bodies produce ``inf``/``nan``.

Units: AU, day, ``GM`` in AU^3/day^2 (``gravsim.constants``). The velocity and time
arguments are accepted so that the function satisfies the ``AccelFn`` protocol but are
ignored, since Newtonian gravity is velocity independent and autonomous.

Reference: Newton, *Principia* (1687); Murray & Dermott, *Solar System Dynamics* (1999),
Sec. 2.2; Danby, *Fundamentals of Celestial Mechanics* (1988), Ch. 6.

Implementation
--------------
Two interchangeable implementations are provided and tested against each other:

* ``accel_newton_numba`` -- a ``numba.njit(cache=True)`` kernel looping over the N(N-1)/2
  unordered pairs once and using the antisymmetry of the pair force.
* ``accel_newton_numpy`` -- pure-numpy broadcasting over the full (N, N, 3) separation
  tensor with the diagonal masked.

``accel_newton`` dispatches to the numba kernel when numba imports and to numpy otherwise.

Both public wrappers validate the input shapes with ``check_inputs`` (``pos`` and ``vel``
(N, 3), ``gm`` (N,)) and raise ``ValueError`` otherwise. This matters for the jitted kernel,
which runs without bounds checking: a ``gm`` shorter than N would silently read past the end
of the array and return plausible-looking garbage instead of failing.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

AccelFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
"""Force-law protocol: ``accel(pos, vel, gm, t) -> acc`` with shapes
``(N, 3), (N, 3), (N,), float -> (N, 3)``."""

try:  # numba is a hard dependency of the package but the fallback keeps the module importable
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only when numba is missing
    HAVE_NUMBA = False

__all__ = [
    "AccelFn",
    "HAVE_NUMBA",
    "accel_newton",
    "accel_newton_numba",
    "accel_newton_numpy",
    "check_inputs",
]


def _as_f64(a: np.ndarray) -> np.ndarray:
    """C-contiguous float64 view/copy of ``a`` (never aliases a non-conforming input)."""
    return np.ascontiguousarray(a, dtype=np.float64)


def check_inputs(
    pos: np.ndarray, vel: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate ``AccelFn`` inputs and return them as C-contiguous float64 arrays.

    Parameters
    ----------
    pos, vel : array_like, shape (N, 3)
        Positions [AU] and velocities [AU/day].
    gm : array_like, shape (N,)
        Gravitational parameters [AU^3/day^2].

    Returns
    -------
    pos, vel, gm : ndarray
        The inputs coerced with ``np.ascontiguousarray(..., dtype=float64)`` (no copy when
        they already conform). The inputs themselves are never modified.

    Raises
    ------
    ValueError
        If ``pos`` is not (N, 3), ``vel`` is not the same shape as ``pos`` or ``gm`` is not
        (N,). Checked up front because the numba kernel has no bounds checking and a short
        ``gm`` would otherwise be read past its end without any error.
    """
    pos = _as_f64(pos)
    vel = _as_f64(vel)
    gm = _as_f64(gm)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {pos.shape}")
    if vel.shape != pos.shape:
        raise ValueError(f"vel shape {vel.shape} != pos shape {pos.shape}")
    if gm.shape != (pos.shape[0],):
        raise ValueError(f"gm must have shape (N,) = ({pos.shape[0]},), got {gm.shape}")
    return pos, vel, gm


# --------------------------------------------------------------------------- numba kernel
def _newton_kernel_py(pos: np.ndarray, gm: np.ndarray) -> np.ndarray:
    """Pair loop used by both the jitted kernel and (if numba is missing) the fallback."""
    n = pos.shape[0]
    acc = np.zeros((n, 3))
    for i in range(n):
        xi = pos[i, 0]
        yi = pos[i, 1]
        zi = pos[i, 2]
        gmi = gm[i]
        for j in range(i + 1, n):
            dx = xi - pos[j, 0]
            dy = yi - pos[j, 1]
            dz = zi - pos[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            # pull of j on i (i feels gm_j) and, by antisymmetry, pull of i on j (j feels gm_i)
            wj = gm[j] * inv_r3
            wi = gmi * inv_r3
            acc[i, 0] -= wj * dx
            acc[i, 1] -= wj * dy
            acc[i, 2] -= wj * dz
            acc[j, 0] += wi * dx
            acc[j, 1] += wi * dy
            acc[j, 2] += wi * dz
    return acc


if HAVE_NUMBA:
    _newton_kernel = njit(cache=True)(_newton_kernel_py)
else:  # pragma: no cover
    _newton_kernel = _newton_kernel_py


def accel_newton_numba(
    pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0
) -> np.ndarray:
    """Newtonian acceleration from the numba pair-loop kernel.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Barycentric positions [AU].
    vel : ndarray, shape (N, 3)
        Velocities [AU/day]; ignored (kept for the ``AccelFn`` protocol) but shape-checked.
    gm : ndarray, shape (N,)
        Gravitational parameters [AU^3/day^2]; ``0`` marks a test particle.
    t : float, optional
        Time [day]; ignored.

    Returns
    -------
    ndarray, shape (N, 3)
        Accelerations [AU/day^2]. A new array; inputs are not modified.

    Raises
    ------
    ValueError
        If the shapes are not (N, 3), (N, 3), (N,) (see ``check_inputs``).
    """
    pos, _, gm = check_inputs(pos, vel, gm)
    return _newton_kernel(pos, gm)


# --------------------------------------------------------------------------- numpy fallback
def accel_newton_numpy(
    pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0
) -> np.ndarray:
    """Newtonian acceleration by numpy broadcasting (pure-python fallback).

    Builds the (N, N, 3) tensor ``d[i, j] = r_i - r_j``, masks the diagonal and contracts
    ``a_i = -sum_j gm_j d_ij / |d_ij|^3``. Memory is O(N^2), fine for N <~ 1000.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Barycentric positions [AU].
    vel : ndarray, shape (N, 3)
        Velocities [AU/day]; ignored (kept for the ``AccelFn`` protocol) but shape-checked.
    gm : ndarray, shape (N,)
        Gravitational parameters [AU^3/day^2]; ``0`` marks a test particle.
    t : float, optional
        Time [day]; ignored.

    Returns
    -------
    ndarray, shape (N, 3)
        Accelerations [AU/day^2]. A new array; inputs are not modified.

    Raises
    ------
    ValueError
        If the shapes are not (N, 3), (N, 3), (N,) (see ``check_inputs``).
    """
    pos, _, gm = check_inputs(pos, vel, gm)
    d = pos[:, None, :] - pos[None, :, :]  # d[i, j] = r_i - r_j
    r2 = np.einsum("ijk,ijk->ij", d, d)
    np.fill_diagonal(r2, 1.0)  # placeholder: the diagonal is masked out below
    inv_r3 = 1.0 / (r2 * np.sqrt(r2))
    np.fill_diagonal(inv_r3, 0.0)
    w = gm[None, :] * inv_r3  # w[i, j] = gm_j / |r_i - r_j|^3
    return -np.einsum("ij,ijk->ik", w, d)


# --------------------------------------------------------------------------- dispatcher
def accel_newton(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0) -> np.ndarray:
    """Pairwise Newtonian acceleration ``a_i = -sum_{j != i} gm_j (r_i - r_j)/|r_i - r_j|^3``.

    Dispatches to the numba kernel (``accel_newton_numba``) when numba is importable and to
    ``accel_newton_numpy`` otherwise; both give the same result to roundoff.

    Parameters
    ----------
    pos : ndarray, shape (N, 3)
        Barycentric positions [AU].
    vel : ndarray, shape (N, 3)
        Velocities [AU/day]; ignored (kept for the ``AccelFn`` protocol) but shape-checked.
    gm : ndarray, shape (N,)
        Gravitational parameters [AU^3/day^2]; ``0`` marks a test particle.
    t : float, optional
        Time [day]; ignored.

    Returns
    -------
    ndarray, shape (N, 3)
        Accelerations [AU/day^2]. A new array; inputs are not modified.

    Raises
    ------
    ValueError
        If the shapes are not (N, 3), (N, 3), (N,) (see ``check_inputs``).
    """
    if HAVE_NUMBA:
        return accel_newton_numba(pos, vel, gm, t)
    return accel_newton_numpy(pos, vel, gm, t)  # pragma: no cover
