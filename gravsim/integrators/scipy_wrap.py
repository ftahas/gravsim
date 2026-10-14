"""Adaptive integration with ``scipy.integrate.solve_ivp`` (DOP853 by default).

Physics
-------
The N-body system is flattened to ``y = [pos.ravel(), vel.ravel()]`` (length 6N) with

    dy/dt = [vel.ravel(), accel(pos, vel, gm, t).ravel()],

and handed to ``solve_ivp``.  The default method is DOP853, the Dormand-Prince explicit
Runge-Kutta pair of order 8 with 5th/3rd-order error estimation (Hairer, Nørsett & Wanner
1993, *Solving ODEs I*, §II.5).  It is the production integrator for the GR runs
(ROADMAP §1.3): being a general Runge-Kutta scheme it handles velocity-dependent forces
without any special treatment.  It is not symplectic: its energy error drifts, but at the
level of the tolerance.

Measured accuracy (Newtonian 2-body, Mercury-like test particle, a = 0.387 AU, e = 0.2056,
against the analytic Kepler solution; see ``tests/test_integrators.py``)
-----------------------------------------------------------------------
With the defaults ``rtol=1e-12, atol=1e-15``:

* 100 orbits (~24 yr): max position error ~3e-8 AU, |dE/E| ~1e-10, ~7e4 force evaluations;
* one Julian century: max position error ~5e-7 AU (3e-7 at the end), |dE/E| ~5e-10;
* numerical perihelion drift (longitude of the Laplace-Runge-Lenz vector, least-squares
  slope over a century): ~+1e-3 arcsec/century -- three orders of magnitude below the
  0.5 arcsec/century noise-floor target of ROADMAP §4 and four below the 43 arcsec/century
  GR signal, which is what the project actually needs.

Tightening ``rtol`` helps only to a point: ``rtol=1e-13`` gives ~2e-9 AU per 100 orbits,
``rtol=1e-14`` ~4e-10 AU (and ~5e-9 AU per century, drift 0.0000 arcsec/cy); scipy clamps
``rtol`` at ``2.2e-14`` and warns below that, so ~4e-10 AU per 100 orbits is the floor of
this formulation (absolute coordinates of order 0.3 AU in double precision, error
accumulating over ~10^4 steps).  The ROADMAP §4 row "2-body vs. Kepler analytic, 100 orbits,
< 1e-10 AU (DOP853)" is therefore not attainable as written: it should be loosened to
~1e-9 AU (at ``rtol=1e-14``) / ~1e-7 AU (defaults), or restated as the precession noise
floor above.  The validation suite must not assert 1e-10 AU.

Units: those of ``state`` and ``accel`` (AU, day, AU^3/day^2 in gravsim).  ``atol=1e-15``
is appropriate for positions of order 1 AU and velocities of order 1e-2 AU/day.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from gravsim.bodies import SystemState, Trajectory
from gravsim.integrators.base import AccelFn

__all__ = ["integrate_scipy"]


def integrate_scipy(
    state: SystemState,
    accel: AccelFn,
    t_eval: np.ndarray,
    method: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-15,
    **solve_ivp_kwargs,
) -> Trajectory:
    """Integrate ``state`` with ``scipy.integrate.solve_ivp`` and sample it at ``t_eval``.

    Parameters
    ----------
    state : SystemState
        Initial conditions at ``state.t``; not modified.
    accel : AccelFn
        Force law ``accel(pos, vel, gm, t) -> (N, 3)``.
    t_eval : array_like, shape (T,)
        Absolute output times [day], monotonic, all on the same side of ``state.t``
        (``t_eval[0]`` may equal ``state.t``; decreasing times integrate backwards).
    method : str
        Any ``solve_ivp`` method (``"DOP853"``, ``"RK45"``, ``"Radau"``, ...).
    rtol, atol : float
        Relative and absolute tolerances.
    **solve_ivp_kwargs
        Forwarded to ``solve_ivp`` (e.g. ``max_step``, ``first_step``).

    Returns
    -------
    Trajectory
        Samples at exactly ``t_eval``; ``meta`` records ``integrator="solve_ivp"``,
        ``method``, ``rtol``, ``atol``, ``nfev`` (force evaluations) and ``n_eval_steps``.

    Raises
    ------
    ValueError
        If ``t_eval`` is empty, non-monotonic or not on one side of ``state.t``.
    RuntimeError
        If ``solve_ivp`` does not succeed (``sol.status != 0``), with scipy's message.
    """
    t_eval = np.atleast_1d(np.asarray(t_eval, dtype=np.float64))
    if t_eval.ndim != 1 or t_eval.size == 0:
        raise ValueError("t_eval must be a non-empty 1-D array of times")
    t0 = state.t
    t_last = float(t_eval[-1])
    diffs = np.diff(t_eval)
    if t_last >= t0:
        monotonic = bool(np.all(diffs >= 0.0)) and bool(t_eval[0] >= t0)
    else:
        monotonic = bool(np.all(diffs <= 0.0)) and bool(t_eval[0] <= t0)
    if not monotonic:
        raise ValueError(
            "t_eval must be monotonic and lie entirely on one side of state.t "
            f"(state.t={t0}, t_eval[0]={t_eval[0]}, t_eval[-1]={t_last})"
        )

    n = state.n
    gm = state.gm
    y0 = np.concatenate([state.pos.ravel(), state.vel.ravel()])

    if t_last == t0:
        # Nothing to integrate: every requested time is the initial time.
        pos = np.repeat(state.pos[None], t_eval.size, axis=0)
        vel = np.repeat(state.vel[None], t_eval.size, axis=0)
        meta = _meta(method, rtol, atol, nfev=0, n_eval_steps=0)
        return Trajectory(t=t_eval, pos=pos, vel=vel, gm=gm.copy(), names=state.names, meta=meta)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        pos = y[: 3 * n].reshape(n, 3)
        vel = y[3 * n :].reshape(n, 3)
        acc = accel(pos, vel, gm, t)
        return np.concatenate([vel.ravel(), np.asarray(acc, dtype=np.float64).ravel()])

    sol = solve_ivp(
        rhs,
        (t0, t_last),
        y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **solve_ivp_kwargs,
    )
    if sol.status != 0 or not sol.success:
        last_output = sol.t[-1] if sol.t.size else t0
        raise RuntimeError(
            f"solve_ivp ({method}) failed with status {sol.status}: {sol.message} "
            f"(last output time reached: t={last_output}, nfev={sol.nfev})"
        )

    n_t = sol.t.size
    pos = np.ascontiguousarray(sol.y[: 3 * n].T.reshape(n_t, n, 3))
    vel = np.ascontiguousarray(sol.y[3 * n :].T.reshape(n_t, n, 3))
    meta = _meta(method, rtol, atol, nfev=int(sol.nfev), n_eval_steps=n_t)
    return Trajectory(t=sol.t, pos=pos, vel=vel, gm=gm.copy(), names=state.names, meta=meta)


def _meta(method: str, rtol: float, atol: float, nfev: int, n_eval_steps: int) -> dict:
    return {
        "integrator": "solve_ivp",
        "method": method,
        "rtol": float(rtol),
        "atol": float(atol),
        "nfev": nfev,
        "n_eval_steps": n_eval_steps,
    }
