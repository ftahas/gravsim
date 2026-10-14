# What competes with GR: the oblate Sun, frame dragging and the Moon

*Module: `gravsim/forces/extras.py` (`make_j2`, `make_lense_thirring`), data
`gravsim/data/j2000_moon.json`, `horizons_moon_ref.npz`. Lesson 09, app lab 7.*

## The budget

In an inertial frame Mercury's perihelion advances by $574.1\pm0.4''$ per century. The
[1PN](pn1.md) and [EIH](eih.md) pages account for $42.98''$ of it and Lesson 02 for the
$531.6''$ of the planets; ranging to Mercury (radar, MESSENGER) now measures the advance to
about $0.01''$/cy — the bound $|2\gamma-\beta-1| < 8\times10^{-5}$ of the [PPN page](ppn.md)
is $3\times8\times10^{-5}\times42.98'' = 0.010''$/cy. At that level four more things move the
perihelion.

| contribution | ″/century | nature |
|---|---|---|
| the planets | $531.6$ | Newtonian |
| General Relativity | $42.98$ | $6\pi GM/(c^2 a(1-e^2))$ per orbit |
| solar quadrupole, $J_2 = 2.2\times10^{-7}$ | $+0.028$ | Newtonian, the Sun's shape |
| Earth and Moon as one body at their barycentre | $-0.0025$ | a modelling choice |
| Lense–Thirring frame dragging | $-0.002$ | relativistic, the Sun's spin |
| the asteroid belt ($1.2\times10^{-9}M_\odot$ at $2.7$ AU) | $\sim+0.0014$ | Newtonian, not modelled |

The $J_2$ entry is three times the precision; everything else is below it. The solar terms are
*corrections only* (added to Newton, 1PN or EIH with `forces.combine`); each applies the
force of the central body to every other body and gives the central body the reaction
$-\sum_i\mu_i\mathbf a_i/\mu_c$ so that momentum is conserved exactly.

## The quadrupole field of an oblate body

### Potential and acceleration

A body flattened by rotation has, to the second zonal harmonic (Kaula 1966; Murray & Dermott
1999, §6.11), the potential per unit mass

$$
\Phi = -\frac{GM}{r} + U, \qquad
U = \frac{GMJ_2R^2}{r^3}\,P_2(\cos\theta) = \frac{k}{2}\Big(\frac{3s^2}{r^5} - \frac{1}{r^3}\Big),
$$

with $\theta$ the colatitude from the unit spin axis $\hat{\mathbf n}$, $s = \mathbf r\cdot\hat{\mathbf n}$
and $k = GMJ_2R^2$. Using $\nabla s = \hat{\mathbf n}$ and $\nabla r = \mathbf r/r$,

$$
\nabla U = \frac{3k}{2r^5}\Big[2s\,\hat{\mathbf n} - 5\frac{s^2}{r^2}\mathbf r + \mathbf r\Big],
\qquad
\mathbf a_{J_2} = -\nabla U = \frac{3k}{2r^5}\Big[\Big(5\frac{s^2}{r^2}-1\Big)\mathbf r - 2s\,\hat{\mathbf n}\Big].
$$

On the axis ($s = r$) this is $+3k\hat{\mathbf n}/r^4$: the pull is weakened, there is less mass
under the poles. On the equator ($s = 0$) it is $-3k\,\mathbf r/(2r^5)$: the bulge pulls harder.
`tests/test_extras.py` checks the closed form against a finite-difference gradient of $U$ for
a tilted axis. With $\hat{\mathbf n}$ the solar pole (inclined $7.25°$ to the ecliptic, node at
longitude $75.76°$; `constants.SUN_POLE_ECLIPTIC`) Mercury's orbit is inclined $i = 3.38°$ to
the solar equator (`analysis.elements.inclination_to_axis`), not the $7.0°$ it makes with the
ecliptic.

### Kaula's secular rates from Gauss's equations

Take the spin along $z$ and write the orbit with the argument of latitude $u = \omega+\nu$, so
that $z = r\sin i\sin u$ and the latitude $\phi$ obeys $\sin\phi = \sin i\sin u$. In spherical
components the acceleration is $a_r = -\tfrac{3k}{2r^4}(1-3\sin^2\phi)$ and
$a_\phi = -\tfrac{3k}{r^4}\sin\phi\cos\phi$ (toward the equator). Resolved along the radial,
transverse and normal directions of the orbit (Vallado 2013, §9.6):

$$
R = -\frac{3k}{2r^4}\big(1 - 3\sin^2 i\sin^2 u\big), \qquad
S = -\frac{3k}{r^4}\sin^2 i\,\sin u\cos u, \qquad
W = -\frac{3k}{r^4}\sin i\cos i\,\sin u .
$$

(At the highest point of the orbit, $u = 90°$, the normal direction *is* the direction of
increasing latitude and $W = a_\phi$ with $\phi = i$, which fixes the sign.) Gauss's equation
for the node,

$$
\frac{d\Omega}{dt} = \frac{r\sin u}{h\sin i}\,W = -\frac{3k\cos i}{h}\,\frac{\sin^2 u}{r^3},
$$

is averaged over one revolution with $dt = r^2\,d\nu/h$ and $r = p/(1+e\cos\nu)$:

$$
\Big\langle\frac{\sin^2 u}{r^3}\Big\rangle = \frac{1}{hT}\int_0^{2\pi}\frac{\sin^2(\omega+\nu)\,(1+e\cos\nu)}{p}\,d\nu = \frac{\pi}{hTp},
$$

because $\sin^2(\omega+\nu)\cos\nu$ integrates to zero. With $h^2 = GMp$ and $T = 2\pi/n$,

$$
\Big\langle\frac{d\Omega}{dt}\Big\rangle = -\frac{3\pi k\cos i}{h^2Tp} = -\frac32\,nJ_2\Big(\frac Rp\Big)^2\cos i .
$$

The argument of perihelion obeys
$d\omega/dt = \tfrac{1}{eh}\big[-p\cos\nu\,R + (p+r)\sin\nu\,S\big] - \cos i\,d\Omega/dt$. The
in-plane terms average to $\tfrac34 nJ_2(R/p)^2(2-3\sin^2 i)$ and the node term adds
$+\tfrac32 nJ_2(R/p)^2\cos^2 i$ (Murray & Dermott 1999, §6.11; Vallado 2013, §9.6), so

$$
\Big\langle\frac{d\omega}{dt}\Big\rangle = \frac34\,nJ_2\Big(\frac Rp\Big)^2\big(5\cos^2 i - 1\big),
$$

which vanishes at the critical inclination $i = 63.4°$. $a$, $e$ and $i$ have no secular change.
For Mercury ($n = 2\pi/87.97$ d, $p = 0.3707$ AU, $R_\odot/p = 1.25\times10^{-2}$,
$i = 3.38°$) and $J_2 = 2.2\times10^{-7}$:

$$
\dot\Omega = -0.0279'',\qquad \dot\omega = +0.0556'',\qquad \dot\varpi = \dot\Omega+\dot\omega = +0.0277''\ \text{per century},
$$

the $0.028''$ of `constants.MERCURY_PRECESSION_J2_ARCSEC_CY`. Lesson 03 used the same force
with $J_2 = 10^{-3}$ and the ecliptic pole to calibrate the measurement toolkit.

### Why $J_2$ and $2\gamma-\beta$ are degenerate, and what breaks it

Both terms advance $\varpi$, so from Mercury's perihelion alone a measured advance
$\Delta = \tfrac{2+2\gamma-\beta}{3}\times42.98'' + 0.028''\times J_2/(2.2\times10^{-7})$ fixes
one combination of $(2\gamma-\beta, J_2)$. Three things separate them. The node: GR moves
$\omega$ only (the ellipse turns in its own plane) while $J_2$ regresses $\Omega$ by as much as
it advances $\varpi$. The other planets: the GR rate goes as $n/(c^2p)\propto a^{-5/2}(1-e^2)^{-1}$,
the $J_2$ rate as $nJ_2(R/p)^2\propto a^{-7/2}(1-e^2)^{-2}$, so the ratio $J_2/{\rm GR}$ falls as
$1/p$ — $6.5\times10^{-4}$ at Mercury, $3.3\times10^{-4}$ at Venus, $2.3\times10^{-4}$ at the Earth,
$1.6\times10^{-4}$ at Mars — and two planets give two equations. And helioseismology measures
$J_2 = (2.2\pm0.1)\times10^{-7}$ directly from the Sun's internal rotation; the MESSENGER
solution of Park et al. (2017) fitted $J_2 = 2.25\times10^{-7}$ together with $\beta$.

## Frame dragging: the Lense–Thirring field

### The gravitomagnetic force

To first post-Newtonian order a body with angular momentum $\mathbf J$ adds to the metric the
off-diagonal terms $g_{0i} = -\tfrac{2G}{c^3r^3}(\mathbf J\times\mathbf r)_i$, the field of a
"gravitomagnetic dipole". A test body moving through it feels (Lense & Thirring 1918; Soffel
1989; Iorio 2005)

$$
\mathbf a_{LT} = \mathbf v\times\mathbf B, \qquad
\mathbf B = \frac{2}{c^2r^3}\Big[G\mathbf J - 3\,(\hat{\mathbf r}\cdot G\mathbf J)\,\hat{\mathbf r}\Big],
$$

a dipole field like a magnet's with the sign reversed (parallel mass currents repel; like
masses attract). Expanded, $\mathbf a_{LT} = \tfrac{2}{c^2r^3}\big[\tfrac{3}{r^2}(\mathbf r\cdot G\mathbf J)(\mathbf r\times\mathbf v) + \mathbf v\times G\mathbf J\big]$,
the form of `make_lense_thirring`; it does no work, $\mathbf a\cdot\mathbf v = 0$. The Sun's
helioseismic angular momentum $J_\odot = 1.9\times10^{41}$ kg m² s⁻¹ gives
$GJ_\odot = 1.1\times10^{-10}$ AU⁵ day⁻³ (`constants.GJ_SUN`); for a prograde circular
equatorial orbit the force is purely radial and outward, $2GJv/(c^2r^3)$.

### Secular rates

With the spin along $z$ and the node along $x$,
$\mathbf r = r(\cos u, \sin u\cos i, \sin u\sin i)$ and the orbit normal is
$\hat{\mathbf h} = (0, -\sin i, \cos i)$. Then $\mathbf r\cdot G\mathbf J = GJ\,r\sin i\sin u$,
$\mathbf r\times\mathbf v = h\hat{\mathbf h}$ and $\hat{\mathbf h}\cdot(\mathbf v\times G\mathbf J) = GJ\,v_x\sin i$
with $v_x = \dot r\cos u - (h/r)\sin u$, so the normal component of the acceleration is

$$
W = \mathbf a\cdot\hat{\mathbf h} = \frac{2GJ\sin i}{c^2r^3}\Big[\frac{2h}{r}\sin u + \dot r\cos u\Big] .
$$

Gauss's equation $d\Omega/dt = r\sin u\,W/(h\sin i)$ becomes
$\tfrac{2GJ}{c^2r^2h}\big[\tfrac{2h}{r}\sin^2u + \dot r\sin u\cos u\big]$. Averaged with
$dt = r^2d\nu/h$, $r = p/(1+e\cos\nu)$ and $\dot r = (he/p)\sin\nu$, the $\dot r$ term integrates
to zero and $\langle\sin^2u/r^3\rangle = \pi/(hpT)$ as above, so

$$
\Big\langle\frac{d\Omega}{dt}\Big\rangle = \frac{4\pi GJ}{c^2hpT} = \frac{2GJ}{c^2a^3(1-e^2)^{3/2}} :
$$

the node is dragged in the sense of the spin whatever the inclination. The same averages of
the radial and transverse components in
$d\omega/dt = \tfrac{1}{eh}[-p\cos\nu\,R + (p+r)\sin\nu\,S] - \cos i\,d\Omega/dt$ give

$$
\Big\langle\frac{d\omega}{dt}\Big\rangle = -\frac{6GJ\cos i}{c^2a^3(1-e^2)^{3/2}},
$$

the argument of perihelion driven back three times faster than the node is dragged forward,
and $\langle di/dt\rangle = 0$. For Mercury: $\dot\Omega = +0.0010''$, $\dot\omega = -0.0030''$,
$\dot\varpi = -0.0020''$ per century (`constants.MERCURY_PRECESSION_LT_ARCSEC_CY`), a hundred
times below the ranging precision; the force itself is $\sim10^{-12}$ of the Newtonian one. Around
the Earth ($J_\oplus = 5.86\times10^{33}$ kg m² s⁻¹) the same node formula gives $31$ mas/yr
for the LAGEOS orbit ($a = 12\,270$ km), measured by laser ranging to $10\%$ (Ciufolini &
Pavlis 2004) and to a few percent with LARES; Gravity Probe B measured the related gyroscope
precession, $37.2\pm7.2$ mas/yr against the predicted $39.2$ (Everitt et al. 2011).

## The Earth–Moon pair

### The EMB approximation and its tidal error

The default system puts one body of mass $m_E+m_M$ at the Earth–Moon barycentre. The Sun's
acceleration of the real pair's barycentre is $w_E\mathbf a(\mathbf r_E) + w_M\mathbf a(\mathbf r_M)$
with $w_E+w_M = 1$; expanding about the barycentre $\mathbf r_B$ with $\mathbf r_E = \mathbf r_B - w_M\boldsymbol\rho$,
$\mathbf r_M = \mathbf r_B + w_E\boldsymbol\rho$ ($\boldsymbol\rho$ the geocentric Moon), the
first-order terms cancel and

$$
\mathbf a_B = \mathbf a(\mathbf r_B) + \tfrac12\,w_Ew_M\,(\boldsymbol\rho\cdot\nabla)^2\mathbf a + O(\rho^3),
\qquad w_Ew_M = \frac{m_Em_M}{(m_E+m_M)^2} = 0.0120 .
$$

For $\mathbf a = -GM\mathbf r/r^3$ the radial part of $(\boldsymbol\rho\cdot\nabla)^2\mathbf a$ is
$-6GM\rho^2/r^4$ when $\boldsymbol\rho\parallel\mathbf r$ and $+3GM\rho^2/r^4$ when
$\boldsymbol\rho\perp\mathbf r$ in the orbital plane; averaged over a month (the Moon's orbit
lies within $5°$ of the ecliptic) this is $-\tfrac32GM\rho^2/r^4$, i.e. an extra *inward*
acceleration

$$
\varepsilon = \frac34\,w_Ew_M\Big(\frac{a_M}{a_E}\Big)^2 = 5.9\times10^{-8}
$$

of the solar one, as if $GM_\odot$ were larger by that fraction. For the same initial state a
stronger centre gives $\delta a/a \simeq -\varepsilon$ and $\delta n/n = \tfrac12\varepsilon - \tfrac32\delta a/a = 2\varepsilon$,
so the real barycentre runs ahead of the point mass along its track by

$$
\Delta s \simeq 2\varepsilon\,n\,a\,t \approx 110\ \text{km per year}.
$$

The split run's barycentre follows Horizons' to $0.9$ km in two years, while the EMB run drifts
from both by $234$ km in two years and $1170$ km in ten ($2\varepsilon nat$: $220$ and $1100$).
The instantaneous relative correction at J2000 is $1.2\times10^{-7}$ (the Moon was near
quadrature). This is why the ephemerides integrate Earth and Moon separately. Mercury barely
notices: the Newtonian rate of its perihelion shifts by $-0.0025''$/cy (ten-year run-vs-run
difference, formally $\pm10^{-5}$) and its GR difference by $10^{-7}\,''$/cy.

### The lunar precession: Newton, Clairaut, and the series in $m$

The Moon's perigee advances by $40.69°$ per year and its node regresses by $19.34°$ per year —
$25\,000$ times Mercury's total advance — through the Sun's tidal force
$\tfrac{GM_\odot}{a_E^3}\big[3(\boldsymbol\rho\cdot\hat{\mathbf r}_E)\hat{\mathbf r}_E - \boldsymbol\rho\big]$
on the geocentric orbit. Averaged over the month and the year the tide reduces to a secular
potential of order $n_\odot^2\rho^2$, and Lagrange's equations give, to first order in
$m = n_\odot/n$,

$$
\dot\varpi = +\tfrac34 m^2 n, \qquad \dot\Omega = -\tfrac34 m^2 n \cos i \approx -\tfrac34 m^2 n,
$$

$20.1°$/yr each for $m = 0.0748$. Newton (1687) obtained the node this way and only half the
observed motion of the perigee; Clairaut (1749) showed that the second-order terms in $m$ —
the perturbation of the eccentricity vector by the tide, which feeds back into the perigee
motion — supply the rest. The expansion (Delaunay 1860; Hill 1877; Brouwer & Clemence 1961,
ch. 12)

$$
\frac{\dot\varpi}{n} = \tfrac34m^2 + \tfrac{225}{32}m^3 + \tfrac{4071}{128}m^4 + \tfrac{265493}{2048}m^5 + \dots,
\qquad
\frac{\dot\Omega}{n} = -\tfrac34m^2 + \tfrac{9}{32}m^3 + \tfrac{273}{128}m^4 + \tfrac{9797}{2048}m^5 + \dots
$$

converges slowly (each perigee term is about a third of the previous one): summed to $m^5$ it
gives $40.3°$ and $-19.2°$/yr. The simulation needs none of it — it integrates Sun, planets,
Earth and Moon and reads the rates off the osculating geocentric elements
($\mu = GM_E + GM_M$). Short spans are biased by the long-period inequalities (evection,
$31.8$ d; variation, $14.8$ d; the annual equation), so the test compares the run with the
Horizons series *over the same span*, where both carry the same bias.

## What the simulation gives

DOP853 at `rtol=1e-12`, `atol=1e-15`, daily samples; the solar terms on Sun + Mercury
(barycentric), the Moon on Sun + planets + Earth + Moon from the Phase 8 snapshot. Rates are
run-vs-run differences on a common grid, node and argument taken in the solar-equator frame
(the trajectory rotated so that $z$ is the solar pole; `tests/test_extras.py`,
`app/lab_extras.py`). Ten-year numbers are Lesson 09's; the century-long solar terms come
from `scripts/validate.py` and the `slow` tests of `tests/test_extras.py`, the ten-year lunar
ones from the `slow` tests of `tests/test_moon.py`.

| quantity | measured | formula |
|---|---|---|
| $J_2\times1000$, 10 yr: $\dot\Omega$ / $\dot\omega$ / $\dot\varpi$ | $-27.892$ / $55.636$ / $27.744$″/cy | $-27.893$ / $55.641$ / $27.748$ |
| $J_2 = 2.2\times10^{-7}$, 10 yr: $\dot\Omega$ / $\dot\omega$ / $\dot\varpi$ | $-0.02789$ / $0.05564$ / $0.02774$″/cy | $-0.02789$ / $0.05564$ / $0.02775$ |
| $J_2 = 2.2\times10^{-7}$, 100 yr: $\dot\varpi$ | $+0.028\pm0.005$″/cy | $0.0277$ |
| $J\times10^5$, 10 yr: $\dot\Omega$ / $\dot\omega$ / $\dot\varpi$ | $100.917$ / $-302.222$ / $-201.305$″/cy | $100.921$ / $-302.237$ / $-201.316$ |
| $J$ real, 10 yr: $\dot\Omega$ / $\dot\omega$ / $\dot\varpi$ | $0.00101$ / $-0.00302$ / $-0.00201$″/cy | $0.00101$ / $-0.00302$ / $-0.00201$ |
| $J$ real, 100 yr: $\dot\varpi$ | $-0.002\pm0.001$″/cy | $-0.0020$ |
| GR (Newton + 1PN) − Newton, 10 yr: $\dot\Omega$ / $\dot\omega$ | $0.0000$ / $42.987$″/cy | $0$ / $42.98$ |
| $J_2/{\rm GR}$ ratio, Mercury / Venus / Earth / Mars | — | $6.5$ / $3.3$ / $2.3$ / $1.6\times10^{-4}$ |
| lunar sidereal month, 10 yr | $27.32165$ d (Horizons $27.32164$) | $27.32166$ |
| lunar perigee / node, 10 yr | $+40.59\pm0.08$ / $-19.37\pm0.01$°/yr (Horizons, same span: $+40.59$ / $-19.37$) | secular $+40.69$ / $-19.34$ |
| lunar perigee / node, 2 yr | $+40.24$ / $-19.43$°/yr (Horizons, same span: $+40.24$ / $-19.43$) | |
| Moon vs Horizons, 2 yr / 10 yr, Newton + 1PN | $38$ / $161$ km | |
| Earth–Moon barycentre: split run vs Horizons, 2 yr | $0.9$ km | |
| EMB run vs split-run barycentre, 2 yr / 10 yr | $234$ / $1170$ km | $2\varepsilon nat$: $220$ / $1100$ |
| Mercury GR − Newton with the Moon, 10 yr | $42.9862$ (EMB: $42.9862$) | $42.98$ |
| Mercury Newton(Earth + Moon) − Newton(EMB), 10 yr | $-0.0025\pm0.00001$″/cy | — |

The residual "Horizons − EIH" of the [EIH page](eih.md), $+0.025''$/cy, is this budget:
$J_2$ ($+0.028$) + the Moon as a body ($-0.0025$) + the asteroids ($+0.0014$) $= +0.027$.

## References

* Kaula, *Theory of Satellite Geodesy* (Blaisdell, 1966).
* Murray & Dermott, *Solar System Dynamics* (Cambridge, 1999), §6.11.
* Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed. (Microcosm, 2013), §9.6.
* Lense & Thirring, *Phys. Z.* 19, 156 (1918); Soffel, *Relativity in Astrometry, Celestial
  Mechanics and Geodesy* (Springer, 1989); Iorio, *A&A* 431, 385 (2005).
* Ciufolini & Pavlis, *Nature* 431, 958 (2004); Everitt et al., *Phys. Rev. Lett.* 106, 221101
  (2011).
* Park et al., *AJ* 153, 121 (2017): MESSENGER ranging, $J_2$ and $\beta$.
* Pijpers, *MNRAS* 297, L76 (1998): the solar angular momentum from helioseismology.
* Brouwer & Clemence, *Methods of Celestial Mechanics* (Academic Press, 1961), ch. 12.
* Pitjeva & Pitjev, *Astron. Lett.* 44, 554 (2018): the mass of the main belt.
