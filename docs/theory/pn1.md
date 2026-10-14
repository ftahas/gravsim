# The 1PN correction and Mercury's 43 arcseconds

*Modules: `gravsim/forces/pn1.py`, `gravsim/analysis/precession.py`
(`gr_precession_arcsec_per_century`). Lesson 04, app labs 2–3.*

## History in three lines

1859: Le Verrier finds that, after all planetary perturbations, Mercury's perihelion still
advances by $38''$/century (Newcomb 1882: $43''$). An intra-Mercurial planet, Vulcan, is
proposed, searched for and never found. November 1915: Einstein computes the perihelion shift
of his new field equations and obtains $43''$ — "for a few days I was beside myself with
joyous excitement".

## The post-Newtonian equation of motion

In General Relativity a test body in the field of a static mass $GM$ follows a geodesic of
the Schwarzschild metric. Expanding in the small parameter $\epsilon \sim GM/(c^2r) \sim v^2/c^2$
and writing the geodesic equation in **harmonic coordinates** (the coordinates in which the
metric is $g_{00} = -(1 - 2U/c^2 + 2U^2/c^4)$, $g_{ij} = \delta_{ij}(1+2U/c^2)$ to the needed
order, with $U = GM/r$) gives, to first post-Newtonian order,

$$
\mathbf a = -\frac{GM}{r^3}\mathbf r
+ \frac{GM}{c^2 r^3}\Big[\Big(4\frac{GM}{r} - v^2\Big)\mathbf r + 4(\mathbf r\cdot\mathbf v)\mathbf v\Big]
+ O(c^{-4}).
$$

This is the test-particle limit of the Einstein–Infeld–Hoffmann equations used by the JPL
ephemerides (Moyer 2003, eq. 4-26; Soffel 1989, §4.2). `make_pn1` returns the bracket alone;
`make_newton_pn1` adds the Newtonian term. The correction is velocity dependent (the
$(\mathbf r\cdot\mathbf v)\mathbf v$ term), which is why the production runs use DOP853 (see
[Integrators](integrators.md)).

**PPN form.** In the parametrised post-Newtonian framework (Will 2014, *Living Rev. Relativ.*
17:4, eq. 56) the bracket becomes

$$
\Big(2(\beta+\gamma)\frac{GM}{r} - \gamma v^2\Big)\mathbf r + 2(1+\gamma)(\mathbf r\cdot\mathbf v)\mathbf v ,
$$

where $\gamma$ measures the space curvature per unit mass and $\beta$ the non-linearity of the
superposition of gravity; GR has $\beta = \gamma = 1$. `make_pn1(c, central, beta, gamma)`
takes both, so the Phase 5 PPN playground is already available at the test-particle level.

## Orders of magnitude

For Mercury at J2000: $v^2/c^2 = 1.7\times10^{-8}$, $GM/(c^2r) = 2.1\times10^{-8}$,
$|\mathbf a_{1PN}|/|\mathbf a_N| = 6.8\times10^{-8}$. An animation with GR "on" is indistinguishable
from one with it "off". The effect is seen only in a **secular** quantity, which is why the
project's primary outputs are measurement plots, not animations.

## Derivation of the perihelion advance

The cleanest route is the Binet equation of the exact Schwarzschild orbit (see
[Schwarzschild geodesics](schwarzschild.md)). With $u = 1/r$, $L = r^2\,d\phi/d\tau$ and
$\mu = GM/c^2$,

$$
u'' + u = \frac{GM}{L^2} + 3\mu u^2 .
$$

Without the last term the solution is the Kepler conic $u_0 = (1+e\cos\phi)/p$, $p = L^2/GM$.
Treat $3\mu u^2$ as a perturbation and substitute $u_0$: the right-hand side contains a term
$3\mu(1/p^2)\,2e\cos\phi$ resonant with the homogeneous solution, whose particular solution is
the secular piece $\delta u = (3\mu e/p^2)\,\phi\sin\phi$. To first order

$$
u \approx \frac1p\Big[1 + e\cos\big((1-3\mu/p)\,\phi\big)\Big],
$$

so the perihelion recurs after $\phi = 2\pi/(1-3\mu/p) \approx 2\pi + 6\pi\mu/p$: the advance per
orbit is

$$
\boxed{\;\Delta\varpi = \frac{6\pi GM}{c^2\,a(1-e^2)}\;}
$$

(Einstein 1915; Weinberg 1972, §8.6). In the PPN framework the coefficient becomes
$(2+2\gamma-\beta)/3$ times this. Per century, multiply by $36525/P$ orbits and by $206264.8''$/rad:

| planet | $a$ [AU] | $e$ | $\Delta\varpi$ per orbit | per century |
|---|---|---|---|---|
| Mercury | 0.3871 | 0.2056 | $5.02\times10^{-7}$ rad $= 0.1035''$ | **42.98″** |
| Venus | 0.7233 | 0.0068 | $0.053''$ | 8.62″ |
| Earth (EMB) | 1.0000 | 0.0167 | $0.038''$ | 3.84″ |
| Mars | 1.5237 | 0.0934 | $0.025''$ | 1.35″ |
| Jupiter | 5.204 | 0.049 | $0.0074''$ | 0.062″ |

Carrying the expansion one order further (for an orbit started at perihelion with the
Newtonian velocity of the osculating ellipse, the convention of
`precession_per_orbit_numeric`) gives
$\Delta\phi = 6\pi(\mu/p)\,[1 + \tfrac{30+5e^2}{4}(\mu/p) + \ldots]$: the exact value exceeds
1PN by $\sim7.5\,\mu/p$, i.e. $2\times10^{-7}$ relative for Mercury and $2\times10^{-3}$ at $c/100$.

## What the simulation gives

All numbers are run-vs-run differences (GR minus Newton on the same grid, DOP853), see
[Validation](../validation.md):

* Sun + Mercury, 10 yr: $42.987\pm0.028$″/cy; 100 yr: $42.9805$.
* Sun + 8 planets, 100 yr: Mercury $42.977$, Venus $8.637$, EMB $3.851$, Mars $1.351$.
* Horizons (DE441, which contains GR) versus our runs over a century: the Newtonian run
  drifts away by $5700$ km along the orbit, the 1PN run stays within $180$ km — Le Verrier's
  anomaly seen "from data".
* $c/100$: $0.288°$ per orbit, fitted/analytic $= 0.998$, with $v/c = 0.013$.
* PPN: $(\beta,\gamma) = (1,\tfrac12), (0,1), (1,0)$ give $0.667, 1.337, 0.333$ of the GR rate,
  against $(2+2\gamma-\beta)/3 = 0.667, 1.333, 0.333$.

## Validity of the expansion

The expansion parameter is $v^2/c^2$. Dividing $c$ by 100 keeps $v/c = 0.013$ for Mercury and
the next order at $2\times10^{-4}$ of the 1PN term; at $c/1000$ ($v/c = 0.13$) the exact advance
exceeds 1PN by 27 %; at $c/1900$ Mercury's $L$ drops below $\sqrt{12}\,GM/c$ and the orbit is
captured (Lesson 05).

## What is still missing (Phase 5)

The 1PN term above is the Sun's field acting on a test body. The full Einstein–Infeld–
Hoffmann equations add the mutual 1PN terms between planets and the back-reaction on the Sun;
for Mercury's secular rate they change nothing at the $0.01''$ level, but they are what the
ephemerides integrate and they are needed for the all-planet, all-effects comparison with
Horizons.

What else moves the perihelion at the $0.01''$ level — the oblate Sun, frame dragging, the Moon as a body — is the subject of [What competes with GR](extras.md).
