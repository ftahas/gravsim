# Schwarzschild geodesics

*Module: `gravsim/metric/schwarzschild.py`. Lesson 05, app lab 4.*

## Metric and constants of motion

Outside a spherical mass the metric is Schwarzschild's. With the gravitational radius
$\mu = GM/c^2$ and the Schwarzschild radius $r_s = 2\mu$, in the plane of the orbit (every
geodesic is planar),

$$
c^2 d\tau^2 = \Big(1-\frac{r_s}{r}\Big)c^2dt^2 - \frac{dr^2}{1-r_s/r} - r^2d\phi^2 .
$$

The Killing vectors $\partial_t$ and $\partial_\phi$ give, per unit rest mass,

$$
E = \Big(1-\frac{r_s}{r}\Big)c^2\frac{dt}{d\tau}, \qquad L = r^2\frac{d\phi}{d\tau},
$$

and the normalisation $g(u,u) = -c^2$ of the four-velocity reduces the radial motion to

$$
\tfrac12\Big(\frac{dr}{d\tau}\Big)^2 + V_{\rm eff}(r) = \frac{E^2/c^2-c^2}{2} \equiv \mathcal E,
\qquad
V_{\rm eff}(r) = -\frac{GM}{r} + \frac{L^2}{2r^2} - \frac{GM L^2}{c^2r^3}
$$

(MTW §25.5; Carroll §5.4; Hartle ch. 9). The first two terms are Newton's; the third is the
only relativistic one and it overturns the centrifugal barrier at small $r$.

## Circular orbits, ISCO, photon sphere

$dV_{\rm eff}/dr = 0$ is the quadratic $GM r^2 - L^2 r + 3\mu L^2 = 0$:

$$
r_\pm = \frac{L^2}{2GM}\Big[1 \pm \sqrt{1 - 12\,\mu^2c^2/L^2}\,\Big],
$$

the outer root stable (minimum of $V_{\rm eff}$), the inner one unstable (maximum).
They merge at $L^2 = 12\,\mu^2c^2$ at the **innermost stable circular orbit** $r_{\rm isco} = 6\mu$;
for smaller $L$ there is no circular orbit and every inward-moving particle is captured. The
unstable branch ends at the **photon sphere** $r = 3\mu$ as $L\to\infty$. The energy of the ISCO
is $E = \sqrt{8/9}\,c^2$: a particle spiralling down to it has radiated 5.7 % of its rest mass.
The inverse relation is $L^2 = GM r^2/(r-3\mu)$ (`circular_orbit_angular_momentum`).

For the Sun: $\mu = 1.477$ km, $r_s = 2.95$ km, photon sphere 4.4 km, ISCO 8.9 km. Mercury's
$p = a(1-e^2) = 3.8\times10^7\mu$, so $\mu/p = 2.7\times10^{-8}$.

## Shape of the orbit: the Binet equation

With $u = 1/r$ and $' = d/d\phi$, using $dr/d\tau = -L u'$,

$$
u'' + u = \frac{GM}{L^2} + 3\mu\,u^2 ,
$$

whose Newtonian part is solved by the Kepler conic $u = (1+e\cos\phi)/p$, $p = L^2/GM$.
`binet_orbit` integrates this equation in the azimuth (DOP853, `rtol=1e-12`) together with
$dt/d\phi = E r^2/\big(c^2 L(1-r_s/r)\big)$, from 2-D initial data $(\mathbf r_0,\mathbf v_0)$
interpreted as $(dx/d\tau, dy/d\tau)$: $L = x v_y - y v_x$ (signed), $v_r = (\mathbf r_0\cdot\mathbf v_0)/r_0$,
$u'(\phi_0) = -v_r/|L|$ and $E = c\sqrt{(1-r_s/r_0)(c^2 + L^2/r_0^2) + v_r^2}$. A capture event
stops the integration at $1.001\,r_s$. Because the equation is in $\phi$ it cannot follow an
orbit that escapes to infinity ($u\to0$); `geodesic_orbit` integrates the same geodesic in
**proper time** with the exact radial equation

$$
\frac{d^2r}{d\tau^2} = -\frac{dV_{\rm eff}}{dr} = -\frac{GM}{r^2} + \frac{L^2}{r^3} - \frac{3GML^2}{c^2r^4},
\qquad \frac{d\phi}{d\tau} = \frac{L}{r^2}, \qquad \frac{dt}{d\tau} = \frac{E}{c^2(1-r_s/r)},
$$

and handles bound, escaping and plunging orbits alike (it is what the app's explorer uses).

## Perihelion advance: exact versus 1PN

Perturbation theory in $\mu u$ gives $\Delta\phi = 6\pi\mu/p$ per orbit — the 1PN result derived
in [the 1PN page](pn1.md) — with the next term $\tfrac{30+5e^2}{4}(\mu/p)$ for an orbit started
at perihelion with the Newtonian velocity of the osculating ellipse $(a,e)$ (with the apsides
parametrised geometrically instead, $u = (1\pm e)/p$, the textbook coefficient is $(18+e^2)/4$).
`precession_per_orbit_numeric` measures the exact advance from the integrated Binet orbit
(perihelia = minima of $r(\phi)$ refined by a parabola, `periapsis_angles`):

| $c/c_{\rm real}$ | $\mu/p$ | $L/(\mu c)$ | exact/1PN | $1 + 7.5\mu/p$ | advance per orbit |
|---|---|---|---|---|---|
| 1 | $2.7\times10^{-8}$ | $6\times10^3$ | $1 + 2\times10^{-6}$ (numerical resolution) | — | $0.1035''$ |
| 1/100 | $2.7\times10^{-4}$ | 61 | 1.0020 | 1.0020 | $0.29°$ |
| 1/300 | $2.4\times10^{-3}$ | 20 | 1.0185 | 1.0180 | $2.6°$ |
| 1/1000 | $2.7\times10^{-2}$ | 6.1 | 1.265 | 1.200 | $36°$ |
| 1/1500 | $6.0\times10^{-2}$ | 4.1 | 2.08 | 1.45 | $135°$ |
| 1/2000 | $0.11$ | 3.06 | captured ($L < \sqrt{12}$) | — | — |

For Mercury the exact and 1PN values agree to $10^{-4}$″/cy (ROADMAP target $0.01$).

## Strong-field phenomenology (geometric units $GM = c = 1$)

* **Precession of degrees per orbit**: $L = 4.2$ from an aphelion at $r = 25$ gives $126°$ per
  orbit with $r_{\min} = 9.2$.
* **Zoom–whirl**: $L = 4$, energy $2\times10^{-5}$ below the barrier top at $r = 4$: the particle
  whirls twice near the unstable circular orbit, then zooms back out. The number of whirls grows
  as $-\ln\delta$ with the energy deficit $\delta$ (the Lyapunov exponent of the unstable orbit).
* **Plunge**: the same $L$ with energy above the top goes through the horizon; the distant
  observer's clock $t$ diverges there ($dt/d\tau = E/(1-r_s/r)$), the particle's $\tau$ does not.

References: Misner, Thorne & Wheeler, *Gravitation* (1973), ch. 25; Carroll, *Spacetime and
Geometry* (2004), §5.4; Hartle, *Gravity* (2003), ch. 9; Weinberg, *Gravitation and Cosmology*
(1972), §8.6.
