"""Extra central-body forces that compete with GR: oblateness (J2) and frame dragging.

Both laws are *corrections only*: add them to Newton, 1PN or EIH with ``forces.combine``.
They describe the field of one spinning central body (row ``central``) with unit spin axis
``n`` (default ``constants.SUN_POLE_ECLIPTIC``), felt by every other body ``i`` through the
relative state ``r = pos_i - pos_c``, ``v = vel_i - vel_c``.  The central body receives the
reaction ``a_c = -sum_{i != c} gm_i a_i / gm_c`` so that the total momentum derivative
``sum_i gm_i a_i`` vanishes exactly (a massless central body, ``gm_c = 0``, gets no reaction;
for J2 the planets' term is then zero anyway).  Test particles (``gm_i = 0``) feel the forces
but contribute nothing to the reaction.

Physics: quadrupole field (J2)
------------------------------
The potential per unit mass of an axisymmetric oblate body, truncated at the second zonal
harmonic, is ``Phi = -gm/r + U`` with (Kaula 1966, *Theory of Satellite Geodesy*;
Murray & Dermott 1999, *Solar System Dynamics*, ch. 6)

    U = (gm j2 r_eq^2 / (2 r^3)) (3 (r_hat . n)^2 - 1) = gm j2 (r_eq/r)^2 P2(cos theta) / r ,

where ``theta`` is the colatitude from the spin axis and ``P2`` the Legendre polynomial.  With
``s = r . n`` and ``k = gm j2 r_eq^2`` this is ``U = (k/2) (3 s^2 / r^5 - 1 / r^3)``, whose
gradient (``grad s = n``, ``grad r = r/r``) is

    grad U = (3 k / (2 r^5)) [ 2 s n - 5 s^2 r / r^2 + r ] ,

so the acceleration ``a = -grad U`` has the closed form used by ``make_j2``:

    a = (3 j2 gm r_eq^2 / (2 r^5)) [ (5 (r . n)^2 / r^2 - 1) r - 2 (r . n) n ] .

Checks: on the spin axis (``r . n = r``) ``a = +3 k n / r^4`` (the pull is weakened: there is
less mass under the poles), on the equator (``r . n = 0``) ``a = -3 k r / (2 r^5)`` (the
equatorial bulge pulls harder).  First-order secular rates of the osculating elements
(Kaula 1966; Murray & Dermott 1999, ch. 6; Vallado 2013, *Fundamentals of Astrodynamics and
Applications*, ch. 9), with
``n_mean = sqrt(gm/a^3)``, ``p = a (1 - e^2)`` and ``i`` the inclination to the equator of the
central body (the plane perpendicular to ``n``):

    dOmega/dt = -(3/2) n_mean j2 (r_eq/p)^2 cos i
    domega/dt =  (3/4) n_mean j2 (r_eq/p)^2 (5 cos^2 i - 1)

(``a``, ``e``, ``i`` have no secular change).  For Mercury (``i`` = 3.38 deg to the solar
equator) ``dvarpi/dt = dOmega/dt + domega/dt ~ (3/2) n_mean j2 (r_eq/p)^2`` = +0.028"/century
for ``j2 = 2.2e-7``, about 1500 times smaller than the 1PN advance.

Physics: Lense-Thirring (gravitomagnetic) field
-----------------------------------------------
A body with angular momentum ``J`` drags inertial frames.  To 1PN order its field acts on a
test body like a magnetic dipole on a charge, with the sign reversed because like masses
attract (parallel mass currents repel).  Writing ``GJ = G J`` (AU^5/day^3; for the Sun
``constants.GJ_SUN`` ~ 1.1e-10) the acceleration is (Lense & Thirring 1918, Phys. Z. 19, 156;
Soffel 1989, *Relativity in Astrometry, Celestial Mechanics and Geodesy*; Iorio 2005,
A&A 431, 385)

    a = (2 / (c^2 r^3)) [ (3 / r^2) (r . GJ) (r x v) + (v x GJ) ]
      = v x B,     B = (2 / (c^2 r^3)) [ GJ - 3 (r_hat . GJ) r_hat ] ,

a work-free force (``a . v = 0``) in a dipolar field.  ``make_lense_thirring`` carries
``GJ = gj * scale * n``; ``scale`` exaggerates ``J`` for lessons (the real Mercury effect is
-0.002"/century, invisible to the eye).  For a circular equatorial orbit (``r . GJ = 0``,
``v`` perpendicular to ``r``) the acceleration is purely radial, ``a = 2 GJ v r_hat/(c^2 r^3)``
outward for prograde motion.

Secular rates from the Gauss equations.  Take the spin along ``z`` and the node along ``x``,
so ``r = r (cos u, sin u cos i, sin u sin i)`` with ``u = omega + nu`` the argument of latitude
and orbit normal ``h_hat = (0, -sin i, cos i)``.  Then ``r . GJ = GJ r sin i sin u``,
``r x v = h h_hat`` and ``h_hat . (v x GJ) = GJ v_x sin i`` with
``v_x = rdot cos u - (h/r) sin u``, so the normal (out-of-plane) component of ``a`` is

    W = a . h_hat = (2 GJ sin i / (c^2 r^3)) [ 2 (h/r) sin u + rdot cos u ] .

The Gauss equation ``dOmega/dt = r sin u W / (h sin i)`` gives
``(2 GJ / (c^2 r^2 h)) [ 2 (h/r) sin^2 u + rdot sin u cos u ]``; averaging over one orbit with
``dt = r^2 dnu / h``, ``r = p / (1 + e cos nu)`` and ``rdot = (h e / p) sin nu``, the
``rdot`` term integrates to zero and ``<sin^2 u / r^3> = pi / (h p T)``, so

    <dOmega/dt> = 4 pi GJ / (c^2 h p T) = 2 GJ / (c^2 a^3 (1 - e^2)^1.5)

(``h = sqrt(gm p)``, ``T = 2 pi sqrt(a^3/gm)``): the node is dragged in the sense of the spin,
independently of ``i``.  The same averages of the radial and transverse components ``R``, ``S``
in ``domega/dt = (1/(e h)) [-p cos nu R + (p + r) sin nu S] - cos i dOmega/dt`` give

    <domega/dt> = -6 GJ cos i / (c^2 a^3 (1 - e^2)^1.5) ,

while ``<di/dt> = <r cos u W / h> = 0``.  For Mercury: ``dOmega/dt`` = +0.001"/century,
``domega/dt`` = -0.003"/century, ``dvarpi/dt`` = -0.002"/century.

Conventions
-----------
* ``central`` is an integer row index (resolve names with ``SystemState.index("Sun")``).
* ``axis`` need not be normalised; it is normalised once when the law is built.
* Units: AU, day, ``GM`` in AU^3/day^2, ``GJ`` in AU^5/day^3; ``c`` in AU/day (default
  ``constants.C_AU_DAY``).  ``t`` is ignored (autonomous forces).
* Each returned ``AccelFn`` carries a ``params`` dict (``law``, parameters) like ``pn1``.
"""

from __future__ import annotations

import operator

import numpy as np
from numpy.typing import ArrayLike

from gravsim.constants import C_AU_DAY, GJ_SUN, R_SUN_AU, SUN_J2, SUN_POLE_ECLIPTIC
from gravsim.forces.newton import AccelFn, check_inputs

__all__ = ["make_j2", "make_lense_thirring"]


# --------------------------------------------------------------------------- helpers
def _check_central(central: int) -> int:
    """Validate that ``central`` is an integer index (names cannot be resolved here)."""
    if isinstance(central, str):
        raise TypeError(
            f"central must be an integer row index, got {central!r}; "
            "resolve names with SystemState.index(name)"
        )
    return operator.index(central)


def _resolve_central(central: int, n: int) -> int:
    """Non-negative row index of the central body, or ``IndexError``."""
    ic = central if central >= 0 else central + n
    if not 0 <= ic < n:
        raise IndexError(f"central index {central} out of range for {n} bodies")
    return ic


def _unit_axis(axis: ArrayLike) -> np.ndarray:
    """Unit spin vector as a (3,) float64 array (``ValueError`` for a zero/non-finite axis)."""
    n = np.array(axis, dtype=np.float64).reshape(-1)
    if n.shape != (3,):
        raise ValueError(f"axis must have 3 components, got shape {n.shape}")
    norm = float(np.sqrt(n @ n))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("axis must be a finite, non-zero vector")
    return n / norm


def _with_reaction(acc: np.ndarray, gm: np.ndarray, ic: int) -> np.ndarray:
    """Zero row ``ic`` of ``acc`` (in place) and replace it by the momentum-conserving reaction.

    ``acc[ic] = -sum_{i != ic} gm_i acc_i / gm_ic``; a massless central body gets zero.
    """
    acc[ic, :] = 0.0
    gm_c = gm[ic]
    if gm_c > 0.0:
        acc[ic, :] = -np.einsum("i,ij->j", gm, acc) / gm_c
    return np.ascontiguousarray(acc)


# --------------------------------------------------------------------------- J2
def make_j2(
    j2: float = SUN_J2,
    r_eq: float = R_SUN_AU,
    central: int = 0,
    axis: ArrayLike = SUN_POLE_ECLIPTIC,
) -> AccelFn:
    """Build the quadrupole (J2) correction of one oblate central body (correction only).

    Parameters
    ----------
    j2 : float, optional
        Dimensionless quadrupole coefficient (default ``constants.SUN_J2`` = 2.2e-7); scale it
        up to make the effect visible.
    r_eq : float, optional
        Equatorial radius the coefficient refers to [AU] (default ``constants.R_SUN_AU``).
    central : int, optional
        Row index of the oblate body (default 0).
    axis : array_like, shape (3,), optional
        Spin axis of the central body in the frame of ``pos`` (normalised internally; default
        ``constants.SUN_POLE_ECLIPTIC``, the solar pole in the J2000 ecliptic frame).

    Returns
    -------
    AccelFn
        ``accel(pos, vel, gm, t) -> acc`` of shape (N, 3) with, for ``i != central``,
        ``acc[i] = (3 j2 gm_c r_eq^2 / (2 r^5)) [ (5 (r.n)^2/r^2 - 1) r - 2 (r.n) n ]`` and
        ``acc[central] = -sum_i gm_i acc[i] / gm_c``.  ``vel`` and ``t`` are ignored (shape
        checked).  Raises ``ValueError`` unless the shapes are (N, 3), (N, 3), (N,) and
        ``IndexError`` if ``central`` is out of range for N.
    """
    central = _check_central(central)
    j2 = float(j2)
    r_eq = float(r_eq)
    if r_eq < 0.0:
        raise ValueError(f"r_eq must be non-negative, got {r_eq}")
    n_axis = _unit_axis(axis)
    k_half = 1.5 * j2 * r_eq * r_eq  # (3/2) j2 r_eq^2; multiplied by gm_c / r^5 per body

    def accel_j2(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0) -> np.ndarray:
        pos, vel, gm = check_inputs(pos, vel, gm)
        n = pos.shape[0]
        ic = _resolve_central(central, n)
        gm_c = gm[ic]

        r = pos - pos[ic]  # (N, 3) relative position
        r2 = np.einsum("ij,ij->i", r, r)
        r2[ic] = 1.0  # placeholder; the central row is replaced by the reaction below
        s = r @ n_axis  # (N,) r . n
        pref = k_half * gm_c / (r2 * r2 * np.sqrt(r2))  # 3 j2 gm_c r_eq^2 / (2 r^5)
        coef_r = 5.0 * s * s / r2 - 1.0
        acc = pref[:, None] * (coef_r[:, None] * r - (2.0 * s)[:, None] * n_axis)
        return _with_reaction(acc, gm, ic)

    accel_j2.__name__ = "accel_j2"
    accel_j2.__qualname__ = "make_j2.<locals>.accel_j2"
    accel_j2.params = {  # type: ignore[attr-defined]
        "law": "j2",
        "j2": j2,
        "r_eq": r_eq,
        "central": central,
        "axis": tuple(float(x) for x in n_axis),
    }
    return accel_j2


# --------------------------------------------------------------------------- Lense-Thirring
def make_lense_thirring(
    gj: float = GJ_SUN,
    c: float = C_AU_DAY,
    central: int = 0,
    axis: ArrayLike = SUN_POLE_ECLIPTIC,
    scale: float = 1.0,
) -> AccelFn:
    """Build the Lense-Thirring (frame-dragging) correction of one spinning central body.

    Parameters
    ----------
    gj : float, optional
        ``G J`` of the central body [AU^5/day^3] (default ``constants.GJ_SUN`` ~ 1.1e-10).
    c : float, optional
        Speed of light [AU/day] (default ``constants.C_AU_DAY``).
    central : int, optional
        Row index of the spinning body (default 0).
    axis : array_like, shape (3,), optional
        Spin axis in the frame of ``pos`` (normalised internally; default
        ``constants.SUN_POLE_ECLIPTIC``).
    scale : float, optional
        Multiplier on ``J`` (default 1); ``scale = 0`` switches the force off exactly.

    Returns
    -------
    AccelFn
        ``accel(pos, vel, gm, t) -> acc`` of shape (N, 3) with, for ``i != central``,
        ``acc[i] = (2/(c^2 r^3)) [ (3/r^2) (r . GJ) (r x v) + (v x GJ) ]`` where
        ``GJ = gj * scale * axis``, ``r = pos_i - pos_c``, ``v = vel_i - vel_c``, and
        ``acc[central] = -sum_i gm_i acc[i] / gm_c``.  ``t`` is ignored.  Raises ``ValueError``
        unless the shapes are (N, 3), (N, 3), (N,) and ``IndexError`` if ``central`` is out of
        range for N.
    """
    central = _check_central(central)
    c = float(c)
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")
    gj = float(gj)
    scale = float(scale)
    n_axis = _unit_axis(axis)
    GJ = gj * scale * n_axis  # (3,) G J vector [AU^5/day^3]
    two_inv_c2 = 2.0 / (c * c)

    def accel_lense_thirring(
        pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        pos, vel, gm = check_inputs(pos, vel, gm)
        n = pos.shape[0]
        ic = _resolve_central(central, n)

        r = pos - pos[ic]  # (N, 3) relative position
        v = vel - vel[ic]  # (N, 3) relative velocity
        r2 = np.einsum("ij,ij->i", r, r)
        r2[ic] = 1.0  # placeholder; the central row is replaced by the reaction below
        rJ = r @ GJ  # (N,) r . GJ
        pref = two_inv_c2 / (r2 * np.sqrt(r2))  # 2 / (c^2 r^3)
        acc = pref[:, None] * ((3.0 * rJ / r2)[:, None] * np.cross(r, v) + np.cross(v, GJ))
        return _with_reaction(acc, gm, ic)

    accel_lense_thirring.__name__ = "accel_lense_thirring"
    accel_lense_thirring.__qualname__ = "make_lense_thirring.<locals>.accel_lense_thirring"
    accel_lense_thirring.params = {  # type: ignore[attr-defined]
        "law": "lense_thirring",
        "gj": gj,
        "scale": scale,
        "c": c,
        "central": central,
        "axis": tuple(float(x) for x in n_axis),
    }
    return accel_lense_thirring
