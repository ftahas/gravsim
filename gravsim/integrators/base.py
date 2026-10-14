"""Integrator protocol: one time step of the N-body equations of motion.

Physics
-------
Every force law in ``gravsim.forces`` is an acceleration function

    accel(pos, vel, gm, t) -> acc,      shapes (N, 3), (N, 3), (N,), float -> (N, 3)

and the equations of motion are the first-order system

    d pos / dt = vel,
    d vel / dt = accel(pos, vel, gm, t).

An integrator is a map ``step(pos, vel, gm, t, dt, accel) -> (pos', vel')`` that
advances the system from time ``t`` to ``t + dt``.  Its *order* ``p`` means that the
local error of one step is O(dt^(p+1)) and the global error after a fixed interval is
O(dt^p)  (Hairer, Nørsett & Wanner 1993, *Solving ODEs I*, ch. II).

Units are those of the rest of the package: AU, day and GM in AU^3/day^2.  Nothing in this
module depends on them, so the same integrators serve dimensionless lessons (e.g. the
harmonic oscillator with ``accel = -pos``).

Conventions
-----------
* ``step`` is pure: it never writes into ``pos``/``vel`` and returns fresh arrays.
* ``name`` is a lower-case identifier (the key used by ``gravsim.integrators.INTEGRATORS``
  and recorded in ``Trajectory.meta["integrator"]``); ``order`` is the nominal order.
* ``symplectic`` flags the geometric integrators (bounded energy error on Hamiltonian
  problems); it is informational only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar

import numpy as np

AccelFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]
"""Signature of every force law: ``accel(pos, vel, gm, t) -> acc`` with ``acc`` of shape (N, 3)."""


class Integrator(ABC):
    """Base class for one-step integrators of ``pos' = vel, vel' = accel(pos, vel, gm, t)``.

    Attributes
    ----------
    name : str
        Lower-case identifier (e.g. ``"leapfrog"``).
    order : int
        Nominal order of accuracy: global error O(dt**order).
    symplectic : bool
        True for the geometric (kick/drift) integrators.
    """

    name: ClassVar[str] = "integrator"
    order: ClassVar[int] = 0
    symplectic: ClassVar[bool] = False

    @abstractmethod
    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        gm: np.ndarray,
        t: float,
        dt: float,
        accel: AccelFn,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the system by one step from ``t`` to ``t + dt``.

        Parameters
        ----------
        pos, vel : ndarray, shape (N, 3)
            Positions [AU] and velocities [AU/day] at time ``t``.
        gm : ndarray, shape (N,)
            GM of each body [AU^3/day^2]; passed through to ``accel`` unchanged.
        t : float
            Current time [day].
        dt : float
            Step size [day]; may be negative to integrate backwards.
        accel : AccelFn
            Force law ``accel(pos, vel, gm, t) -> (N, 3)``.

        Returns
        -------
        pos_new, vel_new : ndarray, shape (N, 3)
            New arrays; the inputs are never modified.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
