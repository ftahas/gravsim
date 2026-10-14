"""Fixed-step integrators: Euler, symplectic Euler, leapfrog, RK4, Yoshida 4 and 6.

Physics
-------
All schemes advance ``pos' = vel``, ``vel' = a(pos, vel, t)`` by one step ``dt``.  The
geometric ones are compositions of two exact flows of the split Hamiltonian
``H = T(v) + V(x)``:

    kick  K(h):  vel -> vel + h * a(pos, vel, t)        (exact flow of V)
    drift D(h):  pos -> pos + h * vel                   (exact flow of T)

* ``Euler`` (explicit, order 1): ``pos' = pos + dt vel``, ``vel' = vel + dt a(pos, vel)``,
  both from the old state.  Not symplectic: the energy of an orbit grows secularly.
* ``SymplecticEuler`` (order 1): ``K(dt)`` then ``D(dt)`` -- the drift uses the *new*
  velocity.  Symplectic (Hairer, Lubich & Wanner 2006, *Geometric Numerical Integration*,
  §I.1.2).
* ``Leapfrog`` (order 2): kick-drift-kick ``K(dt/2) D(dt) K(dt/2)`` = velocity Verlet
  (Störmer-Verlet).  Symplectic and time-reversible; energy error bounded (HLW §I.1.4).
* ``RK4`` (order 4): the classical Runge-Kutta scheme applied to ``y = (pos, vel)``,
  ``y' = f(t, y) = (vel, a)``:  ``k1 = f(t, y)``, ``k2 = f(t + dt/2, y + dt k1/2)``,
  ``k3 = f(t + dt/2, y + dt k2/2)``, ``k4 = f(t + dt, y + dt k3)``,
  ``y' = y + dt (k1 + 2 k2 + 2 k3 + k4) / 6``.  Not symplectic: energy drifts
  monotonically, but at O(dt^4) per orbit.
* ``Yoshida4`` (order 4): three leapfrog substeps of sizes ``w1 dt, w0 dt, w1 dt`` with
  ``w1 = 1 / (2 - 2^(1/3))``, ``w0 = -2^(1/3) / (2 - 2^(1/3))`` (Yoshida 1990, Phys. Lett. A
  150, 262; also Forest & Ruth 1990).  Symplectic.
* ``Yoshida6`` (order 6): seven leapfrog substeps ``w3, w2, w1, w0, w1, w2, w3`` with
  Yoshida's (1990) "solution A" ``w1 = -1.17767998417887``, ``w2 = 0.235573213359357``,
  ``w3 = 0.784513610477560``, ``w0 = 1 - 2 (w1 + w2 + w3)``.  Symplectic.

Adjacent half-kicks of consecutive leapfrog substeps act at the same ``(pos, t)``, so the
composition evaluates the force once there and applies both: ``Leapfrog`` costs 2
evaluations per step, ``Yoshida4`` 4 and ``Yoshida6`` 8 (instead of 2, 6 and 14).

Velocity-dependent forces (``vel_dependent=True``)
--------------------------------------------------
The kick is the exact flow of ``V(x)`` only if ``a`` does not depend on ``vel``.  The 1PN
correction does (``a ~ (4 GM/r - v^2) r + 4 (r.v) v``), so the plain KDK scheme loses its
time symmetry and its favourable energy behaviour (ROADMAP §1.3).  The standard remedy
(HLW §VIII.1; the same idea underlies REBOUNDx's implicit-midpoint GR operator, Tamayo et
al. 2020) is to keep the first half-kick explicit, ``vel -> vel + (dt/2) a(pos, vel, t)``,
and to make the *final* half-kick implicit,

    vel' = vel + (dt/2) a(pos', vel', t + dt),

which restores time-reversibility.  The implicit equation is solved by fixed-point
iteration starting from the pre-kick velocity: ``v(0) = vel``,
``v(k+1) = vel + (dt/2) a(pos', v(k), t + dt)`` for ``k < n_iter``.  Each iteration gains one
power of ``dt`` in the kick error, so ``n_iter >= order`` guarantees the nominal order for
strongly velocity-dependent forces; with fewer iterations the global order is at least
``n_iter`` (``Leapfrog`` keeps order 2 with the default ``n_iter=2``; ``Yoshida4`` drops to 3,
not 2, because the triple-jump condition ``2 w1^3 + w0^3 = 0`` cancels the leading
iteration error).  For the 1PN force the velocity dependence is ~1e-8 of the Newtonian term
and the default ``n_iter=2`` is ample.  ``SymplecticEuler`` has a single kick, which is the
final one and is made implicit.

Units: whatever the force law uses (AU, day, AU^3/day^2 in gravsim); the schemes are
dimensionless.  All functions are pure: inputs are never modified.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from gravsim.integrators.base import AccelFn, Integrator

__all__ = [
    "INTEGRATORS",
    "Euler",
    "Leapfrog",
    "RK4",
    "SymplecticEuler",
    "Yoshida4",
    "Yoshida6",
    "drift",
    "get_integrator",
    "implicit_kick",
    "kick",
]


# --------------------------------------------------------------------------- building blocks
def drift(pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
    """Drift: ``pos + dt * vel`` (exact flow of the kinetic term). Returns a new array."""
    return pos + dt * vel


def kick(vel: np.ndarray, acc: np.ndarray, dt: float) -> np.ndarray:
    """Explicit kick: ``vel + dt * acc`` for a precomputed acceleration. Returns a new array."""
    return vel + dt * acc


def implicit_kick(
    pos: np.ndarray,
    vel: np.ndarray,
    gm: np.ndarray,
    t: float,
    dt: float,
    accel: AccelFn,
    n_iter: int = 2,
) -> np.ndarray:
    """Implicit kick ``v' = vel + dt * accel(pos, v', gm, t)`` by fixed-point iteration.

    Starts from ``v' = vel`` and iterates ``n_iter`` times (``n_iter`` force evaluations).
    For a velocity-independent force one iteration already gives the explicit kick exactly.

    Parameters
    ----------
    pos, vel : ndarray, shape (N, 3)
        Position at which the force is evaluated and the pre-kick velocity.
    gm : ndarray, shape (N,)
    t : float
        Time at which the force is evaluated.
    dt : float
        Kick length (a half step in leapfrog).
    accel : AccelFn
    n_iter : int
        Number of fixed-point iterations (>= 1).

    Returns
    -------
    ndarray, shape (N, 3)
        The post-kick velocity (a new array).
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")
    v_new = vel
    for _ in range(n_iter):
        v_new = kick(vel, accel(pos, v_new, gm, t), dt)
    return v_new


def _kdk_composition(
    pos: np.ndarray,
    vel: np.ndarray,
    gm: np.ndarray,
    t: float,
    dt: float,
    accel: AccelFn,
    weights: tuple[float, ...],
    vel_dependent: bool,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose leapfrog (KDK) substeps of sizes ``w * dt`` for ``w`` in ``weights``.

    The last half-kick of one substep and the first half-kick of the next act at the same
    position and time; with a velocity-independent force both use one evaluation.  With
    ``vel_dependent`` the closing half-kick of every substep is implicit (see module
    docstring) and the force is re-evaluated at the converged velocity for the opening
    half-kick of the following substep.
    """
    n_sub = len(weights)
    tau = t
    acc = accel(pos, vel, gm, tau)  # explicit: evaluated at the known start state
    for i, w in enumerate(weights):
        h = w * dt
        vel = kick(vel, acc, 0.5 * h)
        pos = drift(pos, vel, h)
        tau = tau + h
        if vel_dependent:
            vel = implicit_kick(pos, vel, gm, tau, 0.5 * h, accel, n_iter)
            if i + 1 < n_sub:
                acc = accel(pos, vel, gm, tau)
        else:
            acc = accel(pos, vel, gm, tau)
            vel = kick(vel, acc, 0.5 * h)
    return pos, vel


# --------------------------------------------------------------------------- non-symplectic
class Euler(Integrator):
    """Explicit (forward) Euler, order 1. Pedagogical only: orbits spiral outwards."""

    name: ClassVar[str] = "euler"
    order: ClassVar[int] = 1
    symplectic: ClassVar[bool] = False

    def step(self, pos, vel, gm, t, dt, accel):
        acc = accel(pos, vel, gm, t)
        return drift(pos, vel, dt), kick(vel, acc, dt)


class RK4(Integrator):
    """Classical fourth-order Runge-Kutta on the combined state ``(pos, vel)``."""

    name: ClassVar[str] = "rk4"
    order: ClassVar[int] = 4
    symplectic: ClassVar[bool] = False

    def step(self, pos, vel, gm, t, dt, accel):
        half = 0.5 * dt
        # k = (dpos/dt, dvel/dt) = (vel, acc)
        k1_x, k1_v = vel, accel(pos, vel, gm, t)
        x2, v2 = pos + half * k1_x, vel + half * k1_v
        k2_x, k2_v = v2, accel(x2, v2, gm, t + half)
        x3, v3 = pos + half * k2_x, vel + half * k2_v
        k3_x, k3_v = v3, accel(x3, v3, gm, t + half)
        x4, v4 = pos + dt * k3_x, vel + dt * k3_v
        k4_x, k4_v = v4, accel(x4, v4, gm, t + dt)
        sixth = dt / 6.0
        pos_new = pos + sixth * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
        vel_new = vel + sixth * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return pos_new, vel_new


# --------------------------------------------------------------------------- symplectic
class _Symplectic(Integrator):
    """Common constructor for the kick/drift schemes.

    Parameters
    ----------
    vel_dependent : bool
        Make the closing kick(s) implicit so velocity-dependent forces (1PN) keep the
        scheme time-reversible.
    n_iter : int
        Fixed-point iterations per implicit kick (only used when ``vel_dependent``).
    """

    symplectic: ClassVar[bool] = True

    def __init__(self, vel_dependent: bool = False, n_iter: int = 2) -> None:
        if n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {n_iter}")
        self.vel_dependent = bool(vel_dependent)
        self.n_iter = int(n_iter)

    def __repr__(self) -> str:
        if self.vel_dependent:
            return f"{type(self).__name__}(vel_dependent=True, n_iter={self.n_iter})"
        return f"{type(self).__name__}()"


class SymplecticEuler(_Symplectic):
    """Symplectic Euler, order 1: kick with the old position, then drift with the new velocity."""

    name: ClassVar[str] = "symplectic_euler"
    order: ClassVar[int] = 1

    def step(self, pos, vel, gm, t, dt, accel):
        if self.vel_dependent:
            vel_new = implicit_kick(pos, vel, gm, t, dt, accel, self.n_iter)
        else:
            vel_new = kick(vel, accel(pos, vel, gm, t), dt)
        return drift(pos, vel_new, dt), vel_new


class _Composition(_Symplectic):
    """Leapfrog substeps with relative sizes ``weights`` (must sum to 1)."""

    weights: ClassVar[tuple[float, ...]] = (1.0,)

    def step(self, pos, vel, gm, t, dt, accel):
        return _kdk_composition(
            pos, vel, gm, t, dt, accel, self.weights, self.vel_dependent, self.n_iter
        )


class Leapfrog(_Composition):
    """Kick-drift-kick leapfrog (velocity Verlet), order 2."""

    name: ClassVar[str] = "leapfrog"
    order: ClassVar[int] = 2
    weights: ClassVar[tuple[float, ...]] = (1.0,)


_CBRT2 = 2.0 ** (1.0 / 3.0)
_Y4_W1 = 1.0 / (2.0 - _CBRT2)
_Y4_W0 = -_CBRT2 / (2.0 - _CBRT2)

_Y6_W1 = -1.17767998417887
_Y6_W2 = 0.235573213359357
_Y6_W3 = 0.784513610477560
_Y6_W0 = 1.0 - 2.0 * (_Y6_W1 + _Y6_W2 + _Y6_W3)


class Yoshida4(_Composition):
    """Yoshida (1990) fourth-order symplectic: leapfrog substeps ``w1, w0, w1``."""

    name: ClassVar[str] = "yoshida4"
    order: ClassVar[int] = 4
    weights: ClassVar[tuple[float, ...]] = (_Y4_W1, _Y4_W0, _Y4_W1)


class Yoshida6(_Composition):
    """Yoshida (1990) sixth-order symplectic, solution A: substeps ``w3 w2 w1 w0 w1 w2 w3``."""

    name: ClassVar[str] = "yoshida6"
    order: ClassVar[int] = 6
    weights: ClassVar[tuple[float, ...]] = (
        _Y6_W3,
        _Y6_W2,
        _Y6_W1,
        _Y6_W0,
        _Y6_W1,
        _Y6_W2,
        _Y6_W3,
    )


# --------------------------------------------------------------------------- registry
INTEGRATORS: dict[str, type[Integrator]] = {
    cls.name: cls for cls in (Euler, SymplecticEuler, Leapfrog, RK4, Yoshida4, Yoshida6)
}
"""Lower-case name -> integrator class."""


def get_integrator(spec: str | Integrator, **kwargs) -> Integrator:
    """Return an integrator instance from a name (case-insensitive) or pass one through.

    ``kwargs`` (e.g. ``vel_dependent=True``) are forwarded to the constructor when ``spec``
    is a name; they are rejected for an instance.
    """
    if isinstance(spec, Integrator):
        if kwargs:
            raise TypeError("constructor kwargs are only allowed with an integrator name")
        return spec
    key = str(spec).lower()
    try:
        cls = INTEGRATORS[key]
    except KeyError as exc:
        raise KeyError(f"unknown integrator {spec!r}; known: {sorted(INTEGRATORS)}") from exc
    return cls(**kwargs)
