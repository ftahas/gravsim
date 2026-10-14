"""Fixed-step driver: run an integrator from a ``SystemState`` to ``t_end``.

Physics
-------
Repeated application of ``integrator.step`` on the uniform grid

    t_i = t0 + i * dt,     i = 0 .. n_steps,     t_{n_steps} = t_end exactly,

where ``t0 = state.t``.  The requested step is shrunk *minimally* so that an integer number
of steps lands on ``t_end``: ``n_steps = ceil(|t_end - t0| / |dt|)`` (rounded up to a
multiple of ``save_every`` so the saved samples are uniformly spaced and the last one is
``t_end``), and the step actually used is ``dt = (t_end - t0) / n_steps``.  Its sign follows
``t_end - t0``, so ``t_end < t0`` integrates backwards in time.  Time is formed as
``t0 + i * dt`` rather than accumulated, to avoid summation roundoff in ``t``.

Sample 0 of the returned ``Trajectory`` is the initial state; subsequent samples are every
``save_every``-th step.  ``meta`` records ``integrator`` (name), ``order``, ``dt`` (actual),
``dt_requested``, ``n_steps``, ``save_every``, ``t_end`` and any integrator options.

Units: those of ``state`` and ``accel`` (AU, day, AU^3/day^2 in gravsim).
"""

from __future__ import annotations

import math

import numpy as np

from gravsim.bodies import SystemState, Trajectory
from gravsim.integrators.base import AccelFn, Integrator
from gravsim.integrators.fixed_step import get_integrator

__all__ = ["integrate", "step_grid"]


def step_grid(t0: float, t_end: float, dt: float, save_every: int = 1) -> tuple[int, float]:
    """Number of steps and the signed step size that reach ``t_end`` from ``t0`` exactly.

    Parameters
    ----------
    t0, t_end : float
        Start and end times [day].  ``t_end < t0`` gives a negative step.
    dt : float
        Requested step magnitude [day] (its sign is ignored); must be finite and non-zero.
    save_every : int
        ``n_steps`` is rounded up to a multiple of this, so that every saved sample,
        including the last, sits on the uniform output grid.

    Returns
    -------
    n_steps : int
        Zero when ``t_end == t0``.
    dt_actual : float
        ``(t_end - t0) / n_steps`` (``0.0`` when ``n_steps == 0``); ``|dt_actual| <= |dt|``.
    """
    if not math.isfinite(dt) or dt == 0.0:
        raise ValueError(f"dt must be finite and non-zero, got {dt}")
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")
    span = float(t_end) - float(t0)
    if span == 0.0:
        return 0, 0.0
    q = abs(span) / abs(dt)
    # ceil with a relative guard so that spans that are an exact multiple of dt up to
    # roundoff (e.g. 0.1 * 3 / 0.1) do not get a spurious extra step
    n = max(1, math.ceil(q * (1.0 - 1e-12)))
    n = math.ceil(n / save_every) * save_every
    return n, span / n


def integrate(
    state: SystemState,
    accel: AccelFn,
    t_end: float,
    dt: float,
    integrator: Integrator | str,
    save_every: int = 1,
) -> Trajectory:
    """Integrate ``state`` to ``t_end`` with a fixed-step integrator.

    Parameters
    ----------
    state : SystemState
        Initial conditions; not modified.
    accel : AccelFn
        Force law ``accel(pos, vel, gm, t) -> (N, 3)``.
    t_end : float
        Final time [day]; may be smaller than ``state.t``.
    dt : float
        Requested step magnitude [day]; reduced minimally so that an integer number of
        steps (a multiple of ``save_every``) ends exactly at ``t_end``.
    integrator : Integrator or str
        An integrator instance, or a key of ``gravsim.integrators.INTEGRATORS``.
    save_every : int
        Store every ``save_every``-th step (sample 0 is always the initial state).

    Returns
    -------
    Trajectory
        ``t`` of shape (T,), ``pos``/``vel`` of shape (T, N, 3), with ``T = n_steps //
        save_every + 1`` and ``t[-1] == t_end``.  ``meta`` holds the run parameters.
    """
    integ = get_integrator(integrator)
    n_steps, dt_actual = step_grid(state.t, t_end, dt, save_every)
    n_samples = n_steps // save_every + 1

    pos = state.pos
    vel = state.vel
    gm = state.gm
    t0 = state.t

    out_t = np.empty(n_samples, dtype=np.float64)
    out_pos = np.empty((n_samples, state.n, 3), dtype=np.float64)
    out_vel = np.empty((n_samples, state.n, 3), dtype=np.float64)
    out_t[0] = t0
    out_pos[0] = pos
    out_vel[0] = vel

    k = 1
    for i in range(n_steps):
        t_i = t0 + i * dt_actual
        pos, vel = integ.step(pos, vel, gm, t_i, dt_actual, accel)
        if (i + 1) % save_every == 0:
            out_t[k] = t0 + (i + 1) * dt_actual
            out_pos[k] = pos
            out_vel[k] = vel
            k += 1
    if n_steps > 0:
        out_t[-1] = t_end  # land exactly, independent of roundoff in n * dt

    meta = {
        "integrator": integ.name,
        "order": integ.order,
        "symplectic": integ.symplectic,
        "dt": dt_actual,
        "dt_requested": float(dt),
        "n_steps": n_steps,
        "save_every": int(save_every),
        "t_end": float(t_end),
    }
    if hasattr(integ, "vel_dependent"):
        meta["vel_dependent"] = integ.vel_dependent
        meta["n_iter"] = integ.n_iter
    return Trajectory(
        t=out_t, pos=out_pos, vel=out_vel, gm=gm.copy(), names=state.names, meta=meta
    )
