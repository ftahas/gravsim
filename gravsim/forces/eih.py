"""Einstein-Infeld-Hoffmann (EIH) 1PN N-body equations of motion, PPN-parametrised.

Physics
-------
At first post-Newtonian order every body curves spacetime: the acceleration of body ``i``
depends on the potentials, velocities and (Newtonian) accelerations of all the others.  In the
form used by the JPL DE ephemerides (Moyer 2003, eq. 4-26; Newhall, Standish & Williams 1983,
eq. 1; Will 2014, Sec. 3.4), with ``mu_j = GM_j``, ``r_ij = |r_i - r_j|``, ``v_i`` the
barycentric velocity of body ``i`` and ``a_j^N`` the Newtonian acceleration of body ``j``,

    a_i = sum_{j != i} mu_j (r_j - r_i)/r_ij^3 * { 1
            - (2 (beta + gamma)/c^2) sum_{l != i} mu_l/r_il
            - ((2 beta - 1)/c^2) sum_{k != j} mu_k/r_jk
            + gamma |v_i|^2/c^2 + (1 + gamma) |v_j|^2/c^2 - (2 (1 + gamma)/c^2) (v_i . v_j)
            - (3/(2 c^2)) ((r_i - r_j) . v_j / r_ij)^2
            + (1/(2 c^2)) (r_j - r_i) . a_j^N }
        + (1/c^2) sum_{j != i} mu_j/r_ij^3 * { (r_i - r_j) . ((2 + 2 gamma) v_i
                                                              - (1 + 2 gamma) v_j) } (v_i - v_j)
        + ((3 + 4 gamma)/(2 c^2)) sum_{j != i} mu_j a_j^N / r_ij .

``beta`` measures the nonlinearity of the superposition of gravity and ``gamma`` the space
curvature per unit rest mass; ``beta = gamma = 1`` is General Relativity, for which the
coefficients become ``4, 1, 1, 2, 4, 3/2, 1/2`` and ``4 v_i - 3 v_j``, ``7/2`` (Einstein,
Infeld & Hoffmann 1938; Soffel 1989, Sec. 4.2).  The first line is the Newtonian attraction
multiplied by a bracket of ``1 + O(1/c^2)``; the Newtonian term is therefore *included* in the
returned acceleration (EIH does not separate cleanly into "Newton + correction" because every
term carries ``mu_j``).  Using ``a_j^N`` rather than the full ``a_j`` inside the ``1/c^2``
terms changes the result only at ``O(1/c^4)``, which is the accuracy of the expansion.

Test-particle limit.  For a single massive body ``j = 0`` at rest at the origin and a test
particle ``i`` (``mu_i = 0``) the sums over ``l`` and ``k`` reduce to ``mu_0/r`` and ``0``,
``v_0 = 0`` kills the ``v_j`` terms, ``a_0^N = 0`` because nothing attracts body 0, and what
survives is

    a_i = -mu_0 r/r^3 + (mu_0/(c^2 r^3)) [ (2 (beta + gamma) mu_0/r - gamma v^2) r
                                             + 2 (1 + gamma) (r . v) v ] ,

the Newtonian term plus exactly the correction of ``gravsim.forces.pn1.make_pn1``.  In the
real solar system the two laws differ for Mercury by ~1e-3 of the 1PN correction itself,
dominated by the Sun's barycentric motion (``v_Sun ~ 1e-5 AU/day``) and by the planets'
potentials at Mercury and at the Sun; the direct 1PN terms of the planets are ~1e-6.

Conventions
-----------
* The velocity-dependent terms are *not* Galilean invariant (the equations are Lorentz
  invariant at this order), so positions and velocities must be given in the solar-system
  barycentric frame (``SystemState.to_barycentric()``); the Newtonian part does not care.
* A body with ``gm == 0`` is a test particle: it feels every term but contributes nothing
  (its ``mu_j = 0``), so adding one leaves the other bodies' accelerations unchanged.
* No softening: coincident bodies give ``inf``/``nan``, as in ``accel_newton``.
* Units: AU, day, ``GM`` in AU^3/day^2; ``c`` in AU/day (default ``constants.C_AU_DAY``).
  Reduce ``c`` to exaggerate relativity; the expansion stays valid while ``v/c << 1``.

Implementation
--------------
Pure numpy broadcasting over the (N, N, 3) separation tensor ``d[i, j] = r_i - r_j`` with the
diagonal masked (``1/r_ii`` set to 0, which removes every ``i == j`` term at once); the
Newtonian accelerations ``a_j^N`` come from ``accel_newton``.  The returned value is
``a^N + (1/c^2) * (correction)`` so that ``c -> inf`` reproduces ``accel_newton`` to
roundoff.  No Python loop over bodies; memory O(N^2), intended for N <~ 10.

References
----------
* Einstein, Infeld & Hoffmann, Ann. Math. 39, 65 (1938).
* Moyer, *Formulation for Observed and Computed Values of Deep Space Network Data Types for
  Navigation* (JPL Deep-Space Communications and Navigation Series, 2003), eq. (4-26).
* Newhall, Standish & Williams, A&A 125, 150 (1983), eq. (1).
* Will, "The Confrontation between General Relativity and Experiment", Living Rev. Relativ.
  17, 4 (2014), Sec. 3.4 (PPN n-body equations of motion).
* Soffel, *Relativity in Astrometry, Celestial Mechanics and Geodesy* (1989), Sec. 4.2.
* Tamayo, Rein, Shi & Hernandez, MNRAS 491, 2885 (2020): REBOUNDx ``gr_full`` implements the
  same equations with ``beta = gamma = 1`` (iterating ``a_j`` to the full acceleration, an
  ``O(1/c^4)`` difference), and serves as the independent oracle in ``tests/test_eih.py``.
"""

from __future__ import annotations

import numpy as np

from gravsim.constants import C_AU_DAY
from gravsim.forces.newton import AccelFn, accel_newton, check_inputs

__all__ = ["make_eih"]


def make_eih(c: float = C_AU_DAY, beta: float = 1.0, gamma: float = 1.0) -> AccelFn:
    """Build the full EIH 1PN N-body acceleration (Newtonian term included).

    Parameters
    ----------
    c : float, optional
        Speed of light [AU/day]. Default ``constants.C_AU_DAY``; lower it to exaggerate GR.
    beta, gamma : float, optional
        PPN parameters; ``beta = gamma = 1`` is General Relativity.

    Returns
    -------
    AccelFn
        ``accel(pos, vel, gm, t) -> acc`` of shape (N, 3): the Newtonian acceleration plus
        the 1PN terms of the module docstring, evaluated with the barycentric velocities
        given.  ``t`` is ignored.  Raises ``ValueError`` unless the shapes are (N, 3), (N, 3),
        (N,) (``newton.check_inputs``).  The function carries ``params = {"law": "eih", "c",
        "beta", "gamma"}`` for ``Trajectory.meta``.

    Raises
    ------
    ValueError
        If ``c`` is not positive.
    """
    c = float(c)
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")
    beta = float(beta)
    gamma = float(gamma)
    inv_c2 = 1.0 / (c * c)
    k_phi_i = 2.0 * (beta + gamma)  # coefficient of sum_{l != i} mu_l / r_il
    k_phi_j = 2.0 * beta - 1.0  # coefficient of sum_{k != j} mu_k / r_jk
    k_vi = gamma  # coefficient of |v_i|^2
    k_vj = 1.0 + gamma  # coefficient of |v_j|^2
    k_vivj = 2.0 * (1.0 + gamma)  # coefficient of v_i . v_j
    k_dot_i = 2.0 + 2.0 * gamma  # (r_i - r_j) . v_i in the second line
    k_dot_j = 1.0 + 2.0 * gamma  # (r_i - r_j) . v_j in the second line
    k_acc = 0.5 * (3.0 + 4.0 * gamma)  # third line

    def accel_eih(pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0) -> np.ndarray:
        pos, vel, gm = check_inputs(pos, vel, gm)
        a_newt = accel_newton(pos, vel, gm, t)  # a_j^N for every body, (N, 3)

        d = pos[:, None, :] - pos[None, :, :]  # d[i, j] = r_i - r_j
        r2 = np.einsum("ijk,ijk->ij", d, d)
        np.fill_diagonal(r2, 1.0)  # placeholder: the diagonal is masked out below
        inv_r = 1.0 / np.sqrt(r2)
        np.fill_diagonal(inv_r, 0.0)  # every i == j term vanishes through this factor
        mu_over_r = gm[None, :] * inv_r  # mu_j / r_ij
        w = mu_over_r * inv_r * inv_r  # mu_j / r_ij^3
        phi = mu_over_r.sum(axis=1)  # phi[i] = sum_{l != i} mu_l / r_il

        v2 = np.einsum("ik,ik->i", vel, vel)  # |v_i|^2
        vi_vj = vel @ vel.T  # v_i . v_j
        d_vi = np.einsum("ijk,ik->ij", d, vel)  # (r_i - r_j) . v_i
        d_vj = np.einsum("ijk,jk->ij", d, vel)  # (r_i - r_j) . v_j
        d_aj = np.einsum("ijk,jk->ij", d, a_newt)  # (r_i - r_j) . a_j^N

        # c^2 * (bracket - 1) of the first line
        bracket = (
            -k_phi_i * phi[:, None]
            - k_phi_j * phi[None, :]
            + k_vi * v2[:, None]
            + k_vj * v2[None, :]
            - k_vivj * vi_vj
            - 1.5 * (d_vj * inv_r) ** 2
            - 0.5 * d_aj
        )
        line1 = -np.einsum("ij,ijk->ik", w * bracket, d)  # sum_j mu_j (r_j - r_i)/r^3 (...)
        # second line: sum_j mu_j/r^3 [d . ((2+2g) v_i - (1+2g) v_j)] (v_i - v_j)
        coef = w * (k_dot_i * d_vi - k_dot_j * d_vj)
        line2 = coef.sum(axis=1)[:, None] * vel - coef @ vel
        # third line: (3+4g)/2 sum_j mu_j a_j^N / r_ij
        line3 = k_acc * (mu_over_r @ a_newt)

        return np.ascontiguousarray(a_newt + inv_c2 * (line1 + line2 + line3))

    accel_eih.__name__ = "accel_eih"
    accel_eih.__qualname__ = "make_eih.<locals>.accel_eih"
    accel_eih.params = {  # type: ignore[attr-defined]
        "law": "eih",
        "c": c,
        "beta": beta,
        "gamma": gamma,
    }
    return accel_eih
