# The parametrised post-Newtonian framework

*Modules: `gravsim/forces/pn1.py` and `gravsim/forces/eih.py` (both take `beta`, `gamma`),
`gravsim/analysis/precession.py` (`gr_precession_arcsec_per_century`). Lesson 07, app lab 5.*

## Why parametrise

Mercury's $43''$ confirmed General Relativity in 1915, but by the 1960s there were many
competing metric theories — Brans–Dicke and other scalar–tensor theories, vector–tensor and
bimetric theories — all with the same Newtonian limit and all differing at the next order.
Eddington (1922), Robertson (1962), Schiff, Nordtvedt and Will (1968–1972) built a common
language: in the weak-field, slow-motion limit every metric theory is specified by ten
numbers, the *PPN parameters*, and every solar-system experiment measures a combination of
them. For a theory without preferred frames or locations and with conserved momentum, only
two survive, $\gamma$ and $\beta$. This page derives what gravsim computes from them.

## The PPN metric

Around a static spherical mass, in isotropic coordinates, with $U = GM/r$ the Newtonian
potential (Will 2014, eq. 1; Weinberg 1972, §8.3),

$$
g_{00} = -\Big(1 - \frac{2U}{c^2} + 2\beta\frac{U^2}{c^4}\Big), \qquad
g_{ij} = \Big(1 + 2\gamma\frac{U}{c^2}\Big)\delta_{ij} .
$$

The $-2U/c^2$ term is fixed by the Newtonian limit; the two free numbers are

* $\gamma$: **how much space curvature a unit of rest mass produces** — the coefficient of
  $U$ in the spatial metric. It governs everything light does and the velocity-dependent
  part of the force on a planet.
* $\beta$: **how non-linear the superposition of gravity is** — the coefficient of $U^2$ in
  $g_{00}$: the field of two masses is not the sum of their fields, and gravitational
  energy itself gravitates.

| theory | $\gamma$ | $\beta$ |
|---|---|---|
| General Relativity | 1 | 1 |
| Brans–Dicke, coupling $\omega$ | $(1+\omega)/(2+\omega)$ | 1 |
| general scalar–tensor, $\omega(\phi)$ | $(1+\omega)/(2+\omega)$ | $1 + O(\omega')$, set by $d\omega/d\phi$ (Will 2014, Table 3) |

Two derived combinations recur: the *Nordtvedt parameter* $\eta = 4\beta-\gamma-3$, which
measures violations of the strong equivalence principle (bodies with different gravitational
binding energy falling differently), and $2\gamma-\beta$, which Mercury measures.

## The test-body equation of motion

A test body extremises $\int -mc^2\,d\tau$. With $v^2 = \delta_{ij}v^iv^j$,

$$
\frac{L}{m} = -c^2\frac{d\tau}{dt} = -c^2\sqrt{1 - \frac{2U}{c^2} + \frac{2\beta U^2}{c^4} - \Big(1+\frac{2\gamma U}{c^2}\Big)\frac{v^2}{c^2}} .
$$

Expanding $\sqrt{1-x} = 1 - x/2 - x^2/8 + \ldots$ with $x = 2U/c^2 - 2\beta U^2/c^4 + (1+2\gamma U/c^2)v^2/c^2$
and keeping $O(c^{-2})$ beyond Newton,

$$
\frac{L}{m} = \frac{v^2}{2} + U + \frac{1}{c^2}\Big[\frac{v^4}{8} + \Big(\gamma+\tfrac12\Big)Uv^2 + \Big(\tfrac12-\beta\Big)U^2\Big] + \text{const}
$$

(for GR: $v^4/8 + \tfrac32 Uv^2 - \tfrac12 U^2$, the standard 1PN test-particle Lagrangian in
isotropic coordinates). The Euler–Lagrange equation, with $\mathbf a = \nabla U$ substituted
in every $O(c^{-2})$ term:

$$
\frac{d}{dt}\Big[\mathbf v\Big(1 + \frac{v^2}{2c^2} + (2\gamma+1)\frac{U}{c^2}\Big)\Big]
= \nabla U\Big(1 + \Big(\gamma+\tfrac12\Big)\frac{v^2}{c^2} + (1-2\beta)\frac{U}{c^2}\Big)
$$

$$
\Rightarrow\quad \mathbf a = \nabla U\Big[1 + \gamma\frac{v^2}{c^2} - 2(\beta+\gamma)\frac{U}{c^2}\Big] - 2(1+\gamma)\frac{(\mathbf v\cdot\nabla U)}{c^2}\,\mathbf v .
$$

With $\nabla U = -GM\mathbf r/r^3$ and $\mathbf v\cdot\nabla U = -GM(\mathbf r\cdot\mathbf v)/r^3$,

$$
\boxed{\;\mathbf a = -\frac{GM}{r^3}\mathbf r + \frac{GM}{c^2r^3}\Big[\Big(2(\beta+\gamma)\frac{GM}{r} - \gamma v^2\Big)\mathbf r + 2(1+\gamma)(\mathbf r\cdot\mathbf v)\,\mathbf v\Big]\;}
$$

which is `make_pn1(c, central, beta, gamma)`, and GR's $(4GM/r - v^2)\mathbf r + 4(\mathbf r\cdot\mathbf v)\mathbf v$
at $\beta=\gamma=1$. The $N$-body version — the [EIH equations](eih.md) with the PPN
coefficients $2(\beta+\gamma)$, $2\beta-1$, $\gamma$, $1+\gamma$, $2(1+\gamma)$, $(2+2\gamma)\mathbf v_i-(1+2\gamma)\mathbf v_j$,
$(3+4\gamma)/2$ — is `make_eih(c, beta, gamma)`; its test-particle limit is the boxed
equation (Lesson 06, `tests/test_eih.py`).

## The perihelion advance

Treat the bracket as a perturbing acceleration and use Gauss's planetary equation for the
argument of perihelion. In the orbital plane write the perturbation as a radial component
$R$ and a transverse one $T$ (along $\hat{\boldsymbol\theta}$); with
$\mathbf v = \dot r\,\hat{\mathbf r} + (h/r)\hat{\boldsymbol\theta}$, $h = \sqrt{GMp}$,
$\mathbf r\cdot\mathbf v = r\dot r$:

$$
R = \frac{GM}{c^2r^2}\Big[2(\beta+\gamma)\frac{GM}{r} - \gamma v^2 + 2(1+\gamma)\dot r^2\Big],
\qquad
T = \frac{GM}{c^2r^2}\,2(1+\gamma)\,\dot r\,\frac{h}{r} .
$$

Gauss's equation for a planar orbit, $d\omega/dt = \frac{\sqrt{1-e^2}}{nae}[-R\cos\nu + T(1+r/p)\sin\nu]$,
becomes with $d\nu/dt = h/r^2$ and $h = na^2\sqrt{1-e^2}$ (so that $\sqrt{1-e^2}/(nae\,h) = 1/(GMe)$)

$$
\frac{d\omega}{d\nu} = \frac{r^2}{GMe}\Big[-R\cos\nu + T\Big(1+\frac rp\Big)\sin\nu\Big].
$$

Now insert the Kepler ellipse. With $k = GM/p$, $p = a(1-e^2)$:
$GM/r = k(1+e\cos\nu)$, $v^2 = GM(2/r-1/a) = k(1+2e\cos\nu+e^2)$, $\dot r = \sqrt k\,e\sin\nu$,
$h/r = \sqrt k(1+e\cos\nu)$, $1 + r/p = (2+e\cos\nu)/(1+e\cos\nu)$. Then

$$
r^2R = \frac{GMk}{c^2}\Big[2(\beta+\gamma)(1+e\cos\nu) - \gamma(1+2e\cos\nu+e^2) + 2(1+\gamma)e^2\sin^2\nu\Big],
\qquad
r^2T\Big(1+\frac rp\Big) = \frac{GMk}{c^2}\,2(1+\gamma)\,e\sin\nu\,(2+e\cos\nu),
$$

and

$$
\frac{d\omega}{d\nu} = \frac{k}{c^2e}\Big\{-\cos\nu\big[2(\beta+\gamma)(1+e\cos\nu) - \gamma(1+2e\cos\nu+e^2) + 2(1+\gamma)e^2\sin^2\nu\big] + 2(1+\gamma)\,e\sin^2\nu\,(2+e\cos\nu)\Big\}.
$$

Integrate over one revolution, $\nu\in[0,2\pi]$. Odd powers of $\cos\nu$ and the products
$\cos\nu\sin^2\nu$ integrate to zero; $\int\cos^2\nu = \int\sin^2\nu = \pi$. The surviving
pieces are

| term | integral |
|---|---|
| $-\cos\nu\cdot2(\beta+\gamma)\,e\cos\nu$ | $-2(\beta+\gamma)\,e\pi$ |
| $+\gamma\cos\nu\cdot2e\cos\nu$ | $+2\gamma\,e\pi$ |
| $2(1+\gamma)\,e\sin^2\nu\cdot2$ | $+4(1+\gamma)\,e\pi$ |

whose sum is $e\pi\,(4+4\gamma-2\beta)$, so

$$
\boxed{\;\Delta\varpi = \frac{2\pi k}{c^2}(2+2\gamma-\beta) = \frac{2+2\gamma-\beta}{3}\cdot\frac{6\pi GM}{c^2a(1-e^2)}\;}
$$

per orbit (Robertson 1962; Weinberg 1972, §8.6; Will 2014, eq. 68). The bookkeeping shows
where each parameter enters: $\gamma$ with weight $+2$ (the $\gamma v^2$ term *and* the
velocity-dependent force, each worth one unit after the $-(\beta+\gamma)$ of the $GM/r$ term
is taken into account), $\beta$ with weight $-1$ — more non-linearity *reduces* the advance.
Mercury's perihelion therefore measures the single combination $2\gamma-\beta$: every theory on
the line $2\gamma-\beta = 1$ gives GR's $42.98''$/cy. Per century the conversion is
$36525/P$ orbits and $206264.8''$/rad, as on the [1PN page](pn1.md).

For two comparable masses the $\nu$-dependent terms of the EIH relative acceleration average
to zero by the same calculation (see [EIH](eih.md)), so the formula holds with
$GM\to G(m_1+m_2)$ for any mass ratio.

## The other classical observables

Each is a different combination, which is what makes the framework useful:

| observable | PPN factor | GR value | experiment |
|---|---|---|---|
| light deflection at impact parameter $b$ | $\dfrac{1+\gamma}{2}\cdot\dfrac{4GM}{c^2b}$ | $1.751''$ at the limb | Eddington 1919; VLBI; Gaia |
| Shapiro delay | $(1+\gamma)\dfrac{2GM}{c^3}\ln(\ldots)$ | $\sim250\,\mu$s Earth–Mars at conjunction | Viking; **Cassini 2003** |
| perihelion advance | $\dfrac{2+2\gamma-\beta}{3}\cdot\dfrac{6\pi GM}{c^2p}$ | $42.98''$/cy | radar ranging; MESSENGER |
| Nordtvedt effect | $\eta = 4\beta-\gamma-3$ | 0 | **lunar laser ranging** |
| geodetic precession of a gyroscope on a circular orbit, mean motion $n$ | $\dfrac{1+2\gamma}{3}\cdot\dfrac{3}{2}\dfrac{GM}{c^2r}\,n$ | $6.6''$/yr in low Earth orbit | Gravity Probe B; LLR |

## The bounds

| experiment | combination | result |
|---|---|---|
| Cassini solar conjunction, 2002 (Bertotti, Iess & Tortora 2003) | $\gamma-1$ | $(2.1\pm2.3)\times10^{-5}$ |
| lunar laser ranging (Williams, Turyshev & Boggs 2004) | $\eta = 4\beta-\gamma-3$ | $(4.4\pm4.5)\times10^{-4}$ |
| Mercury's perihelion with $J_2$ from helioseismology (Will 2014, Table 4; Park et al. 2017) | $2\gamma-\beta-1$ | $\lesssim10^{-4}$ |

On the $(\beta-1,\gamma-1)$ plane each is a band: Cassini horizontal, lunar laser ranging of
slope 4, Mercury of slope $1/2$ (Lesson 07 §5 draws them to scale). Mercury and Cassini
together give $\beta-1 = 2(\gamma-1)-(2\gamma-\beta-1)$ with $\sigma\approx9\times10^{-5}$; the
Cassini bound alone requires a Brans–Dicke coupling $\omega > 4\times10^4$. What limits
Mercury's band is not the measurement of the advance — ranging to MESSENGER, like our own
century runs, resolves the rate to $10^{-5}$ — but the *model*: the solar quadrupole $J_2$
also advances the perihelion ($0.03''$/cy, $7\times10^{-4}$ of the GR term) and, from Mercury
alone, is degenerate with $2\gamma-\beta$; helioseismology ($J_2 = (2.2\pm0.1)\times10^{-7}$)
breaks that degeneracy as Cassini breaks the other.

## What the simulation gives

Sun + Mercury, DOP853, run-vs-run differences against Newton on the same grid (Lesson 07;
`tests/test_eih.py`, `tests/test_lab_ppn.py`).

* **The grid.** EIH at $c/100$, 30 orbits, $\beta,\gamma\in\{0,1,2\}$: measured over GR rate
  against $(2+2\gamma-\beta)/3$ —

  | $(\beta,\gamma)$ | (0,0) | (0,1) | (0,2) | (1,0) | (1,1) | (1,2) | (2,0) | (2,1) | (2,2) |
  |---|---|---|---|---|---|---|---|---|---|
  | factor | 0.667 | 1.333 | 2.000 | 0.333 | 1.000 | 1.667 | 0.000 | 0.667 | 1.333 |
  | measured | 0.667 | 1.334 | 2.000 | 0.332 | 0.997 | 1.662 | −0.002 | 0.662 | 1.325 |

  The residuals grow with $\beta$ and $\gamma$: they are second order in $GM/(c^2p)$
  ($2.7\times10^{-4}$ at $c/100$) with theory-dependent coefficients, the price of the
  exaggeration knob (Lesson 04 §6).
* **Degeneracy.** $(\beta,\gamma) = (1.4,1.2)$, on the line $2\gamma-\beta=1$: $0.9985$ of the GR
  rate; the same theory would deflect light by $1.93''$ instead of $1.75''$ and has $\eta = 1.4$.
* **Brans–Dicke $\omega=5$ at the real $c$**, ten years, EIH: predicted $38.887''$/cy, measured
  $38.893\pm0.027$. Le Verrier's original $38''$ corresponds to $\omega\approx4$.
* **Both laws agree.** For $(\beta,\gamma)\in\{(1,1),(0,1),(1,0),(2,1)\}$ at the real $c$ the
  ten-year EIH and `pn1` rates equal $(2+2\gamma-\beta)/3\times42.98$ to $0.15''$/cy
  (`tests/test_eih.py`); the PPN playground of the app (`app/lab_ppn.py`) uses EIH.

## Validity

The PPN metric is the first post-Newtonian order of the theory, so the expansion parameter
is $v^2/c^2\sim GM/(c^2r)$ and the $c$-scaling caveats of the [1PN page](pn1.md) apply:
$c/100$ keeps $v/c = 0.013$ for Mercury and the next order at a few $10^{-4}$; the
strong-field orbits of Lesson 05 have no PPN description at all. Preferred-frame and
preferred-location effects ($\alpha_{1,2,3}$, $\xi$) and the conservation-law parameters
($\zeta_i$) are set to zero in gravsim: they are bounded at $10^{-4}$–$10^{-20}$ by
experiments outside the scope of the project.

## References

* Will, "The Confrontation between General Relativity and Experiment", *Living Rev.
  Relativ.* 17, 4 (2014), §3 and Table 4.
* Will, *Theory and Experiment in Gravitational Physics*, 2nd ed. (2018), ch. 4 and 7.
* Weinberg, *Gravitation and Cosmology* (1972), §8.3, §8.6, §9.5.
* Robertson, in *Space Age Astronomy* (1962).
* Bertotti, Iess & Tortora, *Nature* 425, 374 (2003).
* Williams, Turyshev & Boggs, *Phys. Rev. Lett.* 93, 261101 (2004).
* Park et al., *AJ* 153, 121 (2017): MESSENGER ranging and Mercury's perihelion.
* Nordtvedt, *Phys. Rev.* 169, 1017 (1968); Will & Nordtvedt, *ApJ* 177, 757 (1972).
