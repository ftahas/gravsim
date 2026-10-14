"""Conserved quantities of the Newtonian N-body problem: energy and angular momentum.

Physics
-------
Masses are carried as ``gm_i = G m_i`` (AU^3/day^2) and ``G`` is never used on its own, so the
quantities returned here are the physical ones multiplied by ``G``:

* ``energy``            returns  ``G E = 1/2 sum_i gm_i |v_i|^2 - sum_{i<j} gm_i gm_j / r_ij``
                                           (units AU^5/day^4)
* ``angular_momentum``  returns  ``G L = sum_i gm_i r_i x v_i``        (units AU^5/day^3)

Both are exactly conserved by Newtonian point-mass gravity, so their drift measures integrator
error (symplectic schemes keep the energy error bounded; Runge-Kutta schemes let it grow).
Test particles (``gm = 0``) contribute nothing, consistent with their zero mass.

For a bound two-body system in its barycentric frame the energy is
``G E = - gm_1 gm_2 / (2 a)`` (from ``E = -G m_1 m_2 / (2a)``), and
``G L = gm_1 gm_2 / (gm_1 + gm_2) * (r_rel x v_rel)``.  A centre-of-mass velocity ``V`` adds
``1/2 (gm_1 + gm_2) |V|^2`` to ``G E``.

The 1PN equations conserve a different (post-Newtonian) energy, so for GR runs
``relative_energy_error`` shows a small periodic O(v^2/c^2) oscillation on top of the drift.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from gravsim.bodies import Trajectory

__all__ = ["energy", "angular_momentum", "relative_energy_error"]

_CHUNK = 4096  # samples per block when forming the (T, pairs, 3) separation array


def _as_series(
    pos: ArrayLike, vel: ArrayLike, gm: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)
    single = pos.ndim == 2
    if single:
        pos, vel = pos[None], vel[None]
    if pos.ndim != 3 or pos.shape[-1] != 3 or vel.shape != pos.shape:
        raise ValueError(
            f"pos/vel must have shape (N, 3) or (T, N, 3); got {pos.shape}, {vel.shape}"
        )
    if gm.shape != (pos.shape[1],):
        raise ValueError(f"gm must have shape (N,) = ({pos.shape[1]},), got {gm.shape}")
    return pos, vel, gm, single


def energy(pos: ArrayLike, vel: ArrayLike, gm: ArrayLike) -> np.ndarray:
    """Total mechanical energy times ``G``: ``1/2 sum gm_i v_i^2 - sum_{i<j} gm_i gm_j / r_ij``.

    Parameters
    ----------
    pos, vel : array_like, shape (N, 3) or (T, N, 3)
        Barycentric positions [AU] and velocities [AU/day].
    gm : array_like, shape (N,)
        ``G m_i`` [AU^3/day^2].

    Returns
    -------
    float or ndarray, shape (T,)
        ``G E`` in AU^5/day^4 (a scalar for a single state).
    """
    pos, vel, gm, single = _as_series(pos, vel, gm)
    n_t, n = pos.shape[:2]
    kinetic = 0.5 * np.einsum("tn,n->t", np.sum(vel * vel, axis=-1), gm)
    i, j = np.triu_indices(n, k=1)
    gmgm = gm[i] * gm[j]
    potential = np.zeros(n_t)
    for start in range(0, n_t, _CHUNK):
        block = pos[start : start + _CHUNK]
        d = np.sqrt(np.sum((block[:, i] - block[:, j]) ** 2, axis=-1))
        potential[start : start + _CHUNK] = -np.sum(gmgm / d, axis=-1)
    out = kinetic + potential
    return float(out[0]) if single else out


def angular_momentum(pos: ArrayLike, vel: ArrayLike, gm: ArrayLike) -> np.ndarray:
    """Total angular momentum times ``G``: ``sum_i gm_i r_i x v_i``.

    Parameters
    ----------
    pos, vel : array_like, shape (N, 3) or (T, N, 3)
        Barycentric positions [AU] and velocities [AU/day].
    gm : array_like, shape (N,)
        ``G m_i`` [AU^3/day^2].

    Returns
    -------
    ndarray, shape (3,) or (T, 3)
        ``G L`` in AU^5/day^3.
    """
    pos, vel, gm, single = _as_series(pos, vel, gm)
    out = np.einsum("tnk,n->tk", np.cross(pos, vel), gm)
    return out[0] if single else out


def relative_energy_error(traj: Trajectory) -> np.ndarray:
    """``(E(t) - E(0)) / |E(0)|`` along a trajectory.

    Parameters
    ----------
    traj : Trajectory

    Returns
    -------
    ndarray, shape (T,)
        Relative energy error; exactly zero at the first sample.

    Raises
    ------
    ValueError
        If the initial energy is zero (no scale to measure the error against).
    """
    E = energy(traj.pos, traj.vel, traj.gm)
    E0 = E[0]
    if E0 == 0.0:
        raise ValueError("initial energy is zero; relative error undefined")
    return (E - E0) / abs(E0)
