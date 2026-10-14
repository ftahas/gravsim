"""Time integrators for ``pos' = vel, vel' = accel(pos, vel, gm, t)``.

Fixed-step schemes (``Euler``, ``SymplecticEuler``, ``Leapfrog``, ``RK4``, ``Yoshida4``,
``Yoshida6``) are driven by ``integrate``; ``integrate_scipy`` wraps ``solve_ivp`` (DOP853).
``INTEGRATORS`` maps lower-case names to the fixed-step classes.  See the module docstrings
for the formulas and references.
"""

from gravsim.integrators.base import AccelFn, Integrator
from gravsim.integrators.driver import integrate, step_grid
from gravsim.integrators.fixed_step import (
    INTEGRATORS,
    RK4,
    Euler,
    Leapfrog,
    SymplecticEuler,
    Yoshida4,
    Yoshida6,
    drift,
    get_integrator,
    implicit_kick,
    kick,
)
from gravsim.integrators.scipy_wrap import integrate_scipy

__all__ = [
    "INTEGRATORS",
    "RK4",
    "AccelFn",
    "Euler",
    "Integrator",
    "Leapfrog",
    "SymplecticEuler",
    "Yoshida4",
    "Yoshida6",
    "drift",
    "get_integrator",
    "implicit_kick",
    "integrate",
    "integrate_scipy",
    "kick",
    "step_grid",
]
