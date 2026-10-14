# The Einstein–Infeld–Hoffmann equations

*Module: `gravsim/forces/eih.py` (`make_eih(c, beta, gamma)`). Lesson 06, app lab 5.*

## From one body to $N$

The [1PN correction](pn1.md) of Lesson 04 is the motion of a *test* body in the *static* field
of one mass: Einstein's 1915 calculation. In 1938 Einstein, Infeld and Hoffmann derived from
the field equations the equations of motion of $N$ bodies of comparable masses to first
post-Newtonian order, treating the bodies as singularities of the field (Einstein, Infeld &
Hoffmann 1938; the same result was obtained by Lorentz & Droste in 1917 and, from a
Lagrangian, by Fichtenholz 1950). They are the equations that the JPL *DE*, the Paris *INPOP*
and the Russian *EPM* ephemerides integrate, and — with the PPN coefficients below — the
equations the [PPN playground](ppn.md) explores.

The physical content is simple to state. At Newtonian order gravity is a potential
$U = \sum_j \mu_j/r_j$ ($\mu_j = GM_j$) and bodies move on $\ddot{\mathbf r} = \nabla U$. At the
next order, $O(v^2/c^2)\sim O(U/c^2)$,

* every form of energy gravitates, so kinetic and potential energies of the sources and of
  the test body enter the force;
* the superposition of fields is non-linear ($g_{00}$ contains $U^2$);
* moving masses produce a *gravitomagnetic* field that acts on moving bodies, giving forces
  proportional to velocities;
* the field propagates, so the acceleration of a source appears in the force on another body
  (retardation).

## The 1PN Lagrangian

The cleanest derivation starts from the $N$-body Lagrangian of General Relativity to
$O(c^{-2})$ (Landau & Lifshitz, *Classical Theory of Fields*, §106, eq. 106.17; Damour &
Deruelle 1985). With $\mathbf n_{ab} = (\mathbf r_a-\mathbf r_b)/r_{ab}$,

$$
\begin{aligned}
L ={}& \sum_a \frac{m_a v_a^2}{2} + \sum_a \frac{m_a v_a^4}{8c^2}
 + \sum_{a<b}\frac{G m_a m_b}{r_{ab}}\Big[1 + \frac{3(v_a^2+v_b^2)}{2c^2}
   - \frac{7\,\mathbf v_a\!\cdot\!\mathbf v_b}{2c^2}
   - \frac{(\mathbf v_a\!\cdot\!\mathbf n_{ab})(\mathbf v_b\!\cdot\!\mathbf n_{ab})}{2c^2}\Big] \\
 &- \frac{G^2}{2c^2}\sum_a\sum_{b\neq a}\sum_{c\neq a}\frac{m_a m_b m_c}{r_{ab}\,r_{ac}} .
\end{aligned}
$$

The $v^4$ term is special-relativistic kinetic energy; the bracket is the interaction of
moving masses (it comes from $g_{0i}$ and from the velocity dependence of $g_{00}$, $g_{ij}$);
the triple sum, which for two bodies reduces to $-G^2m_1m_2(m_1+m_2)/(2c^2r^2)$, is the
non-linearity of $g_{00}$. The Euler–Lagrange equations $d(\partial L/\partial\mathbf v_a)/dt
= \partial L/\partial\mathbf r_a$, with the Newtonian equations substituted in every
$O(c^{-2})$ term, give the EIH equations of motion.

## The equations of motion (PPN form)

In the form used by the ephemerides (Moyer 2003, eq. 4-26; Newhall, Standish & Williams
1983, eq. 1; Will 2014, §3.4), parametrised by $\beta$ (non-linearity) and $\gamma$ (space
curvature per unit mass), with $\mathbf r_{ij} = \mathbf r_i-\mathbf r_j$, barycentric
velocities $\mathbf v_i$ and the Newtonian accelerations $\mathbf a^N_j$:

$$
\begin{aligned}
\mathbf a_i = {}& \sum_{j\neq i}\frac{\mu_j\,\mathbf r_{ji}}{r_{ij}^3}\Big\{1
 - \frac{2(\beta+\gamma)}{c^2}\sum_{l\neq i}\frac{\mu_l}{r_{il}}
 - \frac{2\beta-1}{c^2}\sum_{k\neq j}\frac{\mu_k}{r_{jk}}
 + \gamma\frac{v_i^2}{c^2} + (1+\gamma)\frac{v_j^2}{c^2}
 - \frac{2(1+\gamma)}{c^2}\,\mathbf v_i\!\cdot\!\mathbf v_j
 - \frac{3}{2c^2}\Big(\frac{\mathbf r_{ij}\!\cdot\!\mathbf v_j}{r_{ij}}\Big)^2
 + \frac{1}{2c^2}\,\mathbf r_{ji}\!\cdot\!\mathbf a^N_j\Big\} \\
 &+ \frac{1}{c^2}\sum_{j\neq i}\frac{\mu_j}{r_{ij}^3}\Big[\mathbf r_{ij}\cdot\big((2+2\gamma)\mathbf v_i-(1+2\gamma)\mathbf v_j\big)\Big](\mathbf v_i-\mathbf v_j)
 + \frac{3+4\gamma}{2c^2}\sum_{j\neq i}\frac{\mu_j\,\mathbf a^N_j}{r_{ij}} .
\end{aligned}
$$

| coefficient | PPN | GR ($\beta=\gamma=1$) | origin |
|---|---|---|---|
| potential at the field point $i$ | $2(\beta+\gamma)$ | 4 | $g_{00}$ non-linearity and $g_{ij}$ |
| potential at the source $j$ | $2\beta-1$ | 1 | $g_{00}$ non-linearity |
| $v_i^2/c^2$ | $\gamma$ | 1 | $g_{ij}$ |
| $v_j^2/c^2$ | $1+\gamma$ | 2 | $g_{0i}$, $g_{ij}$ |
| $\mathbf v_i\cdot\mathbf v_j/c^2$ | $2(1+\gamma)$ | 4 | $g_{0i}$ |
| $(\hat{\mathbf r}_{ij}\cdot\mathbf v_j)^2/c^2$ | $3/2$ | $3/2$ | retardation |
| $\mathbf r_{ji}\cdot\mathbf a^N_j/c^2$ | $1/2$ | $1/2$ | retardation |
| velocity force, $\mathbf v_i$ / $\mathbf v_j$ | $2+2\gamma$ / $1+2\gamma$ | 4 / 3 | gravitomagnetism |
| source acceleration | $(3+4\gamma)/2$ | $7/2$ | retardation |

The first line is the Newtonian attraction times a bracket $1+O(c^{-2})$, so the Newtonian
term is *included* in the returned acceleration: EIH does not separate into "Newton plus a
correction" because every term carries $\mu_j$. Using $\mathbf a^N_j$ rather than the full
$\mathbf a_j$ inside the $1/c^2$ terms changes the result at $O(c^{-4})$, the accuracy of the
expansion (REBOUNDx's `gr_full` iterates to the full $\mathbf a_j$; the two agree to
$10^{-10}$ of the acceleration). The velocity terms are Lorentz-, not Galilei-invariant, so
the equations must be written in one inertial frame — the solar-system barycentre
(`SystemState.to_barycentric()`). A body with $\mu_j=0$ feels every term and sources none.

## The test-particle limit

Take body 0 at rest at the origin with $\mu_0$ and a test particle $i$ with $\mu_i=0$ (any
number of other test particles may be present). Term by term:

* $\sum_{l\neq i}\mu_l/r_{il} = \mu_0/r$: the other particles are massless;
* $\sum_{k\neq 0}\mu_k/r_{0k} = 0$ for the same reason;
* $\mathbf v_0 = 0$ kills $v_j^2$, $\mathbf v_i\cdot\mathbf v_j$ and $(\mathbf r_{ij}\cdot\mathbf v_j)^2$;
* $\mathbf a^N_0 = 0$ because nothing attracts body 0, which kills the $\mathbf r_{ji}\cdot\mathbf a^N_j$
  term and the whole third line;
* the second line keeps only $\mathbf r\cdot(2+2\gamma)\mathbf v$.

What survives is

$$
\mathbf a_i = -\frac{\mu_0}{r^3}\mathbf r + \frac{\mu_0}{c^2r^3}\Big[\Big(2(\beta+\gamma)\frac{\mu_0}{r} - \gamma v^2\Big)\mathbf r + 2(1+\gamma)(\mathbf r\cdot\mathbf v)\,\mathbf v\Big],
$$

the Newtonian term plus exactly the bracket of `make_pn1` (derived from the PPN metric in
[the PPN page](ppn.md)). The identity is algebraic and `tests/test_eih.py` checks it to
$10^{-12}$ for arbitrary $\beta,\gamma,c$. With the Sun *moving*, the two laws differ: `pn1`
uses Galilean relative velocities, EIH barycentric ones.

## Two bodies: the advance does not depend on the mass ratio

For two bodies the EIH equations reduce (Damour & Deruelle 1985; Blanchet 2014, *Living Rev.
Relativ.* 17:2, §9) to a relative acceleration with $M = m_1+m_2$, $\nu = m_1m_2/M^2$,
$\mathbf n = \mathbf r/r$, $\dot r = \mathbf n\cdot\mathbf v$:

$$
\mathbf a = -\frac{GM}{r^2}\mathbf n + \frac{GM}{c^2r^2}\Big\{\mathbf n\Big[2(2+\nu)\frac{GM}{r} - (1+3\nu)v^2 + \tfrac32\nu\,\dot r^2\Big] + 2(2-\nu)\,\dot r\,\mathbf v\Big\} .
$$

At $\nu\to0$ this is the test-particle form (since $(\mathbf r\cdot\mathbf v)\mathbf v/r^3 =
\dot r\mathbf v/r^2$). Averaging the $\nu$-dependent pieces with Gauss's equation as on the
[PPN page](ppn.md): the radial part $\nu[2GM/r - 3v^2 - \tfrac12\dot r^2]$ contributes
$+4\pi\nu\,GM/(c^2p)$ per orbit and the transverse part $-2\nu\dot r\mathbf v$ contributes
$-4\pi\nu\,GM/(c^2p)$; they cancel, and the periastron advance is

$$
\Delta\varpi = \frac{6\pi G(m_1+m_2)}{c^2 a(1-e^2)}
$$

for *any* mass ratio (Robertson 1938). This is why `gr_precession_arcsec_per_century` is
evaluated with $gm = \mu_\odot+\mu_{\rm planet}$, and why the same equations serve binary
pulsars. Lesson 06 §6 checks it on a "binary star" on Mercury's orbit (companion masses
$1.7\times10^{-7}$, $0.1$ and $1\,M_\odot$ at $c/100$): the EIH rate stays at $0.996$–$0.997$ of the
formula, the Sun-only term falls to $0.962$ and $0.667$.

## Orders of magnitude for Mercury

At J2000 the EIH correction to Mercury's acceleration is $6.8\times10^{-8}$ of the Newtonian
one — the same as the Sun-only 1PN term to $3\times10^{-4}$ of the correction itself,
i.e. $2\times10^{-11}$ of the force. The extra ingredients, in order:

| ingredient | size relative to the 1PN term |
|---|---|
| the Sun's barycentric velocity in the velocity terms, $v_\odot/v_{\rm Mercury}$ | $4\times10^{-4}$ |
| the planets' potentials at Mercury and at the Sun, $\sum\mu_p/r_p$ over $\mu_\odot/r$ | $1\times10^{-4}$ |
| the Sun's Newtonian acceleration, $\sim\mu_{\rm Jup}/r_{\rm Jup}^2$ | $10^{-5}$ |
| the planets as direct 1PN sources, $(\mu_p/\mu_\odot)(r/r_p)^2$ | $10^{-6}$ |

The largest of these follows the Sun's 12-year wobble around Jupiter and is periodic, so its
secular effect on $\varpi$ is smaller still. For the outer planets the relative difference
reaches $10^{-2}$ (Neptune): the Sun sits in Jupiter's potential, which is $0.6\%$ of the Sun's
own potential at Neptune, and the $2\beta-1$ sum sees it.

## What the simulation gives

All runs DOP853 at `rtol=1e-12`, `atol=1e-15`, daily samples, Sun + 8 planets from the J2000
snapshot; rates are run-vs-run differences on the common grid (see [Validation](../validation.md)
for the method and the bias of absolute fits). Ten-year numbers are Lesson 06's, century
numbers come from `scripts/validate.py` and the `slow` tests of `tests/test_eih.py`.

| quantity | 10 yr | 100 yr |
|---|---|---|
| EIH − Newton, Mercury | $42.986\pm0.028$″/cy | $42.977\pm0.001$ |
| EIH − Newton, Venus / EMB / Mars / Jupiter | $8.53$ / $4.55$ / $1.374$ / — | $8.637$ / $3.851$ / $1.351$ / $0.062$ |
| EIH − (Newton + Sun-only 1PN), Mercury | $-1.9\times10^{-5}\pm0.6\times10^{-5}$ | $1.5\times10^{-5}$ |
| EIH vs Sun-only 1PN, Mercury position | $0.5$ km | $3.3$ km (max $5.1$) |
| Horizons − Newton, Mercury $\varpi$ rate | $43.02\pm0.13$ | $43.001\pm0.004$ |
| Horizons − EIH, Mercury $\varpi$ rate | $0.024$ | $0.025$ |
| Newton vs Horizons, Mercury, final separation | $1457$ km | $5743$ km |
| EIH vs Horizons, Mercury, final separation | $5.1$ km | $176$ km |
| EIH vs REBOUNDx `gr_full`, all bodies, 2 yr | $<1$ km | — |

The Horizons rows are Le Verrier's anomaly "from data": the Newtonian residual is a straight
line of slope $43''$/cy (the ephemeris contains GR), the EIH residual is the $0.025''$/cy of the
solar $J_2$, asteroids, Moon and Pluto that Horizons has and the nine-body system does not.
The analytic values are $42.98$, $8.62$, $3.84$, $1.35$, $0.062$.

## Numerics

The EIH force depends on positions, velocities *and* on the Newtonian accelerations of the
sources, so it is not a function of positions alone and the symplectic kick-drift-kick
schemes lose their justification (their `vel_dependent=True` iterated kick works but is not
symplectic). The production integrator is DOP853 ([Integrators](integrators.md)), which
takes the same number of steps as for Newton because the relativistic terms are $10^{-8}$ of
the force. The implementation is pure numpy over the $(N,N,3)$ separation tensor with the
diagonal masked by $1/r_{ii}\to0$, $O(N^2)$ memory, no Python loop over bodies: a ten-year
nine-body run costs $2.3$ s against $0.5$ s for the numba Newtonian kernel.

## References

* Einstein, Infeld & Hoffmann, *Ann. Math.* 39, 65 (1938).
* Landau & Lifshitz, *The Classical Theory of Fields*, §106.
* Damour & Deruelle, *Ann. Inst. H. Poincaré* 43, 107 (1985).
* Moyer, *Formulation for Observed and Computed Values of Deep Space Network Data Types for
  Navigation* (JPL, 2003), eq. 4-26.
* Newhall, Standish & Williams, *A&A* 125, 150 (1983).
* Will, *Living Rev. Relativ.* 17, 4 (2014), §3.4.
* Blanchet, *Living Rev. Relativ.* 17, 2 (2014), §9.
* Robertson, *Ann. Math.* 39, 101 (1938).
* Tamayo, Rein, Shi & Hernandez, *MNRAS* 491, 2885 (2020): REBOUNDx `gr_full`.

The $+0.025''$/cy residual of the Horizons − EIH comparison is accounted for term by term in [What competes with GR](extras.md).
