# Light: deflection and the Shapiro delay

*Module: `gravsim/metric/null.py`. Lesson 08, app lab 6 (see [the app page](../app.md)).*

## History in three numbers

1801: Johann von Soldner treats light as Newtonian corpuscles of speed $c$ and finds that a
ray grazing the Sun is bent by $0.87''$. 1911: Einstein, from the equivalence principle alone
(clocks run slow in a potential), obtains the same $0.87''$ and asks astronomers to look.
1915: the full field equations double it, $1.75''$, because space is curved as well as time.
29 May 1919: Dyson, Eddington and Davidson photograph the Hyades around the eclipsed Sun from
Sobral (Brazil) and Príncipe (Gulf of Guinea) and report $1.98'' \pm 0.12''$ and
$1.61'' \pm 0.30''$ (probable errors) at the limb: Einstein, not Soldner. Today the same
number is known to $10^{-4}$ (VLBI) and the related time delay to $2\times10^{-5}$ (Cassini).

## Null geodesics of the Schwarzschild metric

In the plane of the ray the metric is (see [Schwarzschild geodesics](schwarzschild.md))

$$
c^2 d\tau^2 = \Big(1-\frac{r_s}{r}\Big)c^2dt^2 - \frac{dr^2}{1-r_s/r} - r^2d\phi^2,
\qquad \mu = \frac{GM}{c^2},\quad r_s = 2\mu .
$$

A photon has $d\tau = 0$, so the geodesic is parametrised by an affine parameter $\lambda$
instead, with $\dot{} = d/d\lambda$. The two Killing vectors still give two constants,

$$
E = \Big(1-\frac{r_s}{r}\Big)c^2\,\dot t, \qquad L = r^2\dot\phi ,
$$

and the null condition $g(k,k) = 0$ replaces the normalisation of the four-velocity:

$$
\dot r^2 = \frac{E^2}{c^2} - \Big(1-\frac{r_s}{r}\Big)\frac{L^2}{r^2} .
$$

Only the ratio $b = cL/E$ survives: for a photon of momentum $p$ far away, $E = pc$ and $L = pb$
with $b$ the perpendicular distance of its straight-line path from the mass, the **impact
parameter**. With $u = 1/r$ and $' = d/d\phi$, $u' = -\dot r/L$, so

$$
u'^2 = \frac{1}{b^2} - u^2 + r_s u^3
\qquad\Longrightarrow\qquad
u'' + u = 3\mu\,u^2 ,
$$

the **null Binet equation** (Weinberg §8.5; MTW §25.6; Hartle §9.4). Compare the timelike one,
$u'' + u = GM/L^2 + 3\mu u^2$: the Newtonian term is gone (a photon has no $GM/L^2$ scale, its
"energy" is infinite in units of its rest mass) and the only term left is the one that produced
Mercury's precession. Without it, $u = \sin\phi/b$ is the straight line $y = b$ when $\phi$ is
measured from the incoming asymptote. In the dimensionless $w = bu$ the equation reads
$w'' + w = 3\epsilon\,w^2$ with $\epsilon = \mu/b$: the whole problem depends on the geometry
only through $\mu/b$, and the Sun is characterised by $\mu/R_\odot = 2.1\times10^{-6}$.

## Closest approach, photon sphere, capture

The turning point $r_0$ ($u' = 0$) satisfies $b^2 = r_0^2/(1 - r_s/r_0)$, a cubic in $r_0$ with
the closed form $r_0 = (2b/\sqrt3)\cos\big[\tfrac13\arccos(-3\sqrt3\,\mu/b)\big]$
(`closest_approach_radius`). It exists only for

$$
b > b_{\rm crit} = 3\sqrt3\,\mu = 5.196\,\mu ,
$$

the impact parameter of the **photon sphere** $r = 3\mu$, the unstable circular null orbit
(`critical_impact_parameter`). Rays with $b \le b_{\rm crit}$ spiral in and are captured; seen
from far away the black hole is a dark disc of angular radius $b_{\rm crit}/D$, its **shadow**:
$40\,\mu$as across for M87\* ($6.5\times10^9 M_\odot$ at 16.8 Mpc) and $52\,\mu$as for Sgr A\*,
the sizes imaged by the Event Horizon Telescope (2019, 2022). As $b \to b_{\rm crit}^+$ the ray
winds around the photon sphere an arbitrary number of times before escaping and the deflection
diverges logarithmically (Darwin 1959; Bozza 2002),

$$
\delta \simeq -\ln\Big(\frac{b}{b_{\rm crit}} - 1\Big) - 0.4002 .
$$

For the Sun $b_{\rm crit} = 7.7$ km, nine orders of magnitude inside its radius: the solar
deflection is as weak-field as it gets.

## The deflection integral and its expansion

The ray comes in from $u = 0$, turns at $u_0 = 1/r_0$ and goes back out, so the azimuth it
sweeps is twice the integral of $d\phi = du/u'$:

$$
\Delta\phi = 2\int_0^{u_0}\frac{du}{\sqrt{1/b^2 - u^2 + r_s u^3}},
\qquad \delta = \Delta\phi - \pi .
$$

With $r_s = 0$ the integrand is $1/\sqrt{u_0^2 - u^2}$ and $\Delta\phi = \pi$: a straight line.
To first order in $r_s$ one writes $1/b^2 = u_0^2 - r_su_0^3$, factors
$(u_0^2 - u^2)\big[1 - r_s(u_0^2 + u_0u + u^2)/(u_0 + u)\big]$, substitutes $u = u_0\sin\theta$ and
finds $\Delta\phi = \pi + r_su_0\int_0^{\pi/2}\big(\sin\theta + \tfrac{1}{1+\sin\theta}\big)d\theta
= \pi + 2r_su_0$, i.e.

$$
\delta = \frac{4GM}{c^2 r_0} = \frac{4GM}{c^2 b} + O\Big(\frac{\mu^2}{b^2}\Big) .
$$

The same result comes out of the Binet equation by perturbation theory, which is what the code
actually integrates: with $u = \sin\phi/b + u_1$ and $u_1'' + u_1 = 3\mu\sin^2\phi/b^2$, the
solution with $u(0) = 0$, $u'(0) = 1/b$ is

$$
u = \frac{\sin\phi}{b} + \frac{\mu}{b^2}\big(2 - \sin^2\phi - 2\cos\phi\big),
$$

and setting $u = 0$ at $\phi = \pi + \delta$ gives $\delta = 4\mu/b$. Carried to higher order in
$\epsilon = \mu/b$ (Keeton & Petters 2005, eq. 29),

$$
\delta = 4\epsilon + \frac{15\pi}{4}\epsilon^2 + \frac{128}{3}\epsilon^3 + O(\epsilon^4)
$$

(in terms of the closest approach instead, $\delta = 4\mu/r_0 + (15\pi/4 - 4)\mu^2/r_0^2 + \dots$).
`deflection_angle` integrates $w'' + w = 3\epsilon w^2$ from the incoming asymptote with DOP853
(`rtol = 1e-12`) until $w$ returns to zero and returns $\phi_{\rm end} - \pi$; internally it
integrates the *deviation* $w - \sin\phi$ so that a $10^{-5}$ rad deflection is not lost in the
cancellation $\phi_{\rm end} - \pi$. At the solar limb it exceeds $4\mu/R_\odot = 1.7512''$ by
$(15\pi/16)\,\mu/R_\odot = 6\times10^{-6}$ relative; at $b = 50\mu$ the series above is still good
to one percent; at $b = 1.001\,b_{\rm crit}$ the ray makes a full loop ($\delta = 373°$).

## Why a factor two: PPN $\gamma$ and Soldner's corpuscle

In isotropic coordinates the weak-field PPN metric is, to the order needed for light,

$$
ds^2 = -\Big(1 - \frac{2U}{c^2}\Big)c^2dt^2 + \Big(1 + \frac{2\gamma U}{c^2}\Big)d\mathbf x^2,
\qquad U = \frac{GM}{\rho},
$$

where $\gamma$ measures how much space curvature a unit mass produces (GR: $\gamma = 1$). A null
ray has $c\,dt = n\,|d\mathbf x|$ with the effective **refractive index**

$$
n = 1 + (1+\gamma)\frac{GM}{c^2\rho},
$$

and by Fermat's principle it is bent towards the mass by the transverse gradient of $n$
integrated along the straight path:

$$
\delta = \int_{-\infty}^{\infty}\nabla_\perp n\,dx
= (1+\gamma)\frac{GM}{c^2}\int_{-\infty}^{\infty}\frac{b\,dx}{(x^2+b^2)^{3/2}}
= \frac{1+\gamma}{2}\cdot\frac{4GM}{c^2 b}
$$

(Will 2014, eq. 68). The $g_{tt}$ part, the Newtonian potential, contributes exactly half; the
curvature of space, $g_{ij}$, the other half. A theory with $\gamma = 0$, i.e. a calculation that
keeps only the slowing of clocks, gives $2GM/(c^2b) = 0.875''$ at the limb. That is Einstein's
1911 result, and also Soldner's: a Newtonian particle of speed $v$ on a hyperbola of impact
parameter $b$ is turned by $\tan(\delta/2) = GM/(bv^2)$, so with $v = c$ and $GM/(bc^2) \ll 1$,
$\delta = 2GM/(c^2b)$ (`deflection_newtonian_corpuscle`). The 1919 measurement was therefore a
measurement of $\gamma$, as every light test since has been; light cannot see $\beta$, which
only enters at second order in $U$. Combined with Mercury's $(2 + 2\gamma - \beta)/3$
([the 1PN page](pn1.md)) the two experiments separate the two parameters.

## The Shapiro delay

Integrating $c\,dt = n\,dl$ along the straight chord between an emitter at $\mathbf x_e$ and a
receiver at $\mathbf x_r$ (bending changes the path length only at second order, by Fermat's
principle), with the chord passing the mass at distance $b$ and $x$ measured along it,
$\rho = \sqrt{x^2 + b^2}$,

$$
\Delta t = (1+\gamma)\frac{GM}{c^3}\int_{x_e}^{x_r}\frac{dx}{\sqrt{x^2+b^2}}
= (1+\gamma)\frac{GM}{c^3}\,\ln\frac{(r_e + |x_e|)(r_r + x_r)}{b^2}
= (1+\gamma)\frac{GM}{c^3}\,\ln\frac{r_e + r_r + R}{r_e + r_r - R},
$$

with $R = |x_e| + x_r$ the Euclidean length of the chord (Shapiro 1964; Will 2014, eq. 70;
`shapiro_delay`). The last form, in terms of the three sides of the triangle, is the one used in
ephemeris software; the impact parameter enters only through $r_e + r_r - R =
b^2/(r_e + |x_e|) + b^2/(r_r + x_r)$, which the code evaluates in this cancellation-free form.
For $b \ll r_e, r_r$ (superior conjunction),

$$
\Delta t \simeq (1+\gamma)\frac{GM}{c^3}\ln\frac{4r_er_r}{b^2}
$$

(`shapiro_delay_conjunction`). The prefactor for the Sun is $GM_\odot/c^3 = 4.93\,\mu$s, so a
ray from Mars (1.52 AU) to Earth grazing the limb is late by $123.6\,\mu$s one way, $247\,\mu$s
for a radar round trip, on top of a light time of 21 minutes: a part in $10^{7}$. Around
conjunction the delay rises logarithmically as $b$ shrinks, the spike that Shapiro's radar echoes
from Mercury and Venus first showed in 1967. What a spacecraft tracking system actually measures
is often the Doppler shift, i.e. the time derivative $y = -d\Delta t/dt \simeq
-(1+\gamma)(2GM/c^3)\,\dot b/b$, a few parts in $10^{10}$ at Cassini's $b = 1.6R_\odot$.

**Coordinate dependence of the constant term.** "Excess over the Euclidean light time" needs a
definition of Euclidean distance. The formula above holds in *isotropic* coordinates, where the
spatial metric is conformally flat; the Schwarzschild areal radius of each end point exceeds its
isotropic value by $\mu$ (`isotropic_radius`), which would shift the chord by $2\mu$ and the
delay by the constant $2\mu/c = 9.9\,\mu$s, eight percent of the Earth–Mars value. Only the
logarithmic variation with $b$ is observable; the constant is absorbed in the orbit fit
(Will 2014, §3.4.2). `shapiro_delay_numeric` integrates the exact coordinate time
$dt/d\phi = 1/\big(cbu^2(1 - r_su)\big)$ along the geodesic between the two radii and subtracts
the isotropic chord; it agrees with the weak-field formula to $10^{-4}$ for Earth–Mars at the
limb, the residual being the second-order end-point and Fermat terms dissected in the module
docstring (Teyssandier & Le Poncin-Lafitte 2008).

## Measurements

| year | experiment | result |
|---|---|---|
| 1919 | eclipse, Sobral 4-inch / Príncipe (Dyson, Eddington & Davidson 1920) | $1.98 \pm 0.12$ / $1.61 \pm 0.30$ arcsec at the limb (probable errors) |
| 1922 | eclipse, Wallal (Campbell & Trumpler 1923) | $1.72 \pm 0.11$ |
| 1973 | eclipse, Mauritania (Texas, Jones 1976) | $1.66 \pm 0.18$ |
| 1967–71 | radar echoes from Mercury and Venus (Shapiro et al.) | delay confirmed to 20 %, then 5 % |
| 1976 | radio interferometry, 3C279 (Fomalont & Sramek) | $(1+\gamma)/2 = 1.007 \pm 0.009$ |
| 1979 | Viking landers, Mars (Reasenberg et al.) | $\gamma = 1.000 \pm 0.002$ |
| 1997 | Hipparcos, whole sky (Froeschlé et al.) | $\gamma = 0.997 \pm 0.003$ |
| 2003 | Cassini Doppler at conjunction (Bertotti, Iess & Tortora) | $\gamma - 1 = (2.1 \pm 2.3)\times10^{-5}$ |
| 2004 | VLBI 1979–1999 (Shapiro et al.) | $\gamma = 0.9998 \pm 0.0004$ |

The eclipse values are scaled to the limb by the $1/b$ law: the measured stars sat at
$2$–$6\,R_\odot$, where the shifts are $0.3''$–$0.9''$, on plates whose own scale had to be
calibrated to a few hundredths of an arcsecond. The ROADMAP validation target is the limb value
$4GM_\odot/(c^2R_\odot) = 1.751'' \pm 0.001''$ (`solar_limb_deflection_arcsec`).

References: Weinberg, *Gravitation and Cosmology* (1972), §§8.5, 8.7; Misner, Thorne &
Wheeler, *Gravitation* (1973), §§25.6, 40.3–40.4; Hartle, *Gravity* (2003), §9.4; Will,
*Living Rev. Relativ.* 17, 4 (2014), §3.4; Keeton & Petters, *Phys. Rev. D* 72, 104006 (2005);
Bozza, *Phys. Rev. D* 66, 103001 (2002); Shapiro, *Phys. Rev. Lett.* 13, 789 (1964);
Reasenberg et al., *ApJ* 234, L219 (1979); Bertotti, Iess & Tortora, *Nature* 425, 374 (2003);
Dyson, Eddington & Davidson, *Phil. Trans. R. Soc. A* 220, 291 (1920); Soldner, *Berliner
Astronomisches Jahrbuch* (1804); Teyssandier & Le Poncin-Lafitte, *Class. Quantum Grav.* 25,
145020 (2008).
