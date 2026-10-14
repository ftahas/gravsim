# The Newtonian N-body problem

*Modules: `gravsim/forces/newton.py`, `gravsim/analysis/kepler.py`, `gravsim/analysis/elements.py`,
`gravsim/analysis/conserved.py`. Lessons 01–02.*

## Equations of motion

For $N$ point masses with gravitational parameters $GM_j$ (we never separate $G$ from the masses:
only the products are known to many digits) at barycentric positions $\mathbf r_j$,

$$
\ddot{\mathbf r}_i = -\sum_{j\neq i} GM_j\,\frac{\mathbf r_i-\mathbf r_j}{|\mathbf r_i-\mathbf r_j|^3} .
$$

Only the *attracting* mass enters each term. A body with $GM = 0$ is a test particle: it feels
the field but exerts none, and `accel_newton` handles it without a special case. The force is
pairwise antisymmetric, so `accel_newton` loops over the $N(N-1)/2$ unordered pairs once
(numba-compiled; a numpy broadcasting fallback is tested against it to $10^{-14}$).

## Conserved quantities

Three exact invariants, all used as checks on the numerics (`analysis/conserved.py` returns
them multiplied by $G$ because masses are carried as $GM$):

$$
\mathbf P = \sum_i GM_i\,\mathbf v_i, \qquad
\mathbf L = \sum_i GM_i\,\mathbf r_i\times\mathbf v_i, \qquad
E = \tfrac12\sum_i GM_i v_i^2 - \sum_{i<j}\frac{GM_i\,GM_j}{r_{ij}} .
$$

Momentum conservation is equivalent to the uniform motion of the **barycentre**
$\mathbf R = \sum_i GM_i \mathbf r_i / \sum_i GM_i$; in the barycentric frame it stays at the
origin. Because the pair force is exactly antisymmetric, every integrator conserves $\mathbf P$
to roundoff — including explicit Euler. Energy and angular momentum are the discriminating
checks.

**The Sun moves.** Jupiter has $GM_J/GM_\odot = 1/1047$ at $a_J = 5.2$ AU, so the Sun orbits
the Sun–Jupiter barycentre at $a_J\,GM_J/(GM_\odot+GM_J) = 0.0050$ AU $\approx 1.07\,R_\odot$ with
Jupiter's period; Saturn adds a second loop and the path over 60 years reaches $2 R_\odot$
(Lesson 02). All measurements in gravsim are therefore made *relative to the Sun* or in the
barycentric frame, never relative to a fixed origin.

## The two-body problem and Kepler's solution

For two bodies the relative vector $\mathbf r = \mathbf r_2-\mathbf r_1$ obeys
$\ddot{\mathbf r} = -\mu\,\mathbf r/r^3$ with $\mu = GM_1+GM_2$. The bound solutions are ellipses with
the focus at the origin:

* semi-latus rectum $p = a(1-e^2)$, radius $r = p/(1+e\cos\nu)$;
* vis-viva $v^2 = \mu(2/r - 1/a)$, hence $a = 1/(2/r - v^2/\mu)$;
* mean motion $n = \sqrt{\mu/a^3}$, period $P = 2\pi/n$ (Kepler III);
* Kepler's equation $M = E - e\sin E$ with $M = n(t-t_p)$, solved by Halley iteration from
  Danby's starting guess $E_0 = M + 0.85\,e\,\mathrm{sign}(\sin M)$ (`solve_kepler`);
* $\tan(\nu/2) = \sqrt{(1+e)/(1-e)}\,\tan(E/2)$.

The perifocal axes $\mathbf P$ (toward perihelion) and $\mathbf Q = \mathbf W\times\mathbf P$ are rotated
into the reference frame with the 3-1-3 sequence $R_z(\Omega)R_x(i)R_z(\omega)$
(`elements_to_state`), and `propagate` advances an elliptic state analytically. These are the
ground truth of the integrator tests.

## Osculating elements and the eccentricity vector

Given a relative state $(\mathbf r,\mathbf v)$ and $\mu$:

$$
\mathbf h = \mathbf r\times\mathbf v, \qquad
\mathbf n = \hat{\mathbf z}\times\mathbf h, \qquad
\mathbf e = \frac{\mathbf v\times\mathbf h}{\mu} - \frac{\mathbf r}{r} .
$$

$\mathbf e$ is the Laplace–Runge–Lenz vector divided by $\mu$: its magnitude is the eccentricity
and it points from the focus to the perihelion. It is an exact constant of the Kepler problem,
so any secular drift of its direction is a perturbation (planets, $J_2$, GR) or an integrator
artefact. The classical angles are

$$
i = \arccos(h_z/h), \qquad
\Omega = \operatorname{atan2}(n_y, n_x), \qquad
\omega = \operatorname{atan2}\big(\hat{\mathbf h}\cdot(\hat{\mathbf n}\times\hat{\mathbf e}),\;\hat{\mathbf n}\cdot\hat{\mathbf e}\big),
$$

and the quantity gravsim tracks is the **longitude of perihelion** $\varpi = \Omega+\omega$, a
"dog-leg" angle that stays continuous and well conditioned as $i\to0$, where $\Omega$ and $\omega$
separately do not (an equatorial orbit takes its node along $+x$; a circular one its perihelion
at the node). See [Measuring a perihelion drift](measurement.md).

## Reference numbers

| quantity | value |
|---|---|
| $GM_\odot$ | $2.9591220828\times10^{-4}$ AU³/day² ($= k^2$, Gauss's constant $k = 0.01720209895$) |
| Mercury at J2000 | $a = 0.38710$ AU, $e = 0.20563$, $i = 7.01°$ (ecliptic), $P = 87.969$ d |
| planetary perturbations on Mercury's perihelion | 531.6″/century (Clemence 1947), of which Venus 277.9, Earth 90.0, Jupiter 153.6, Saturn 7.3, Mars 2.5 |
| Sun's wobble | $\sim 1$–$2\,R_\odot$ |

References: Murray & Dermott, *Solar System Dynamics* (1999), ch. 2 and 6; Danby,
*Fundamentals of Celestial Mechanics* (1988), ch. 6; Vallado, *Fundamentals of Astrodynamics
and Applications* (2013), alg. 9–10.
