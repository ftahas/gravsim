# Measuring a perihelion drift of 0.1″ per orbit

*Modules: `gravsim/analysis/precession.py`, `elements.py`, `compare.py`. Lesson 03, app lab 3.*

The relativistic advance of Mercury's perihelion is $5\times10^{-7}$ rad per orbit, on top of a
Newtonian advance ten times larger and a periodic wobble a hundred times larger. The toolkit
has to see it anyway.

## The observable

At every sample of a trajectory the state of a planet relative to the Sun defines an
osculating ellipse, and the eccentricity vector $\mathbf e = \mathbf v\times\mathbf h/\mu - \mathbf r/r$
(with $\mu = GM_\odot + GM_{\rm planet}$) points at its perihelion. The **longitude of
perihelion** $\varpi = \Omega+\omega$ is read off $\mathbf e$ and the node vector
(`perihelion_longitude`), wrapped to $(-\pi,\pi]$ and unwrapped before fitting. It is preferred
to $\omega$ because the node of a nearly equatorial orbit is ill defined while the sum is
not.

## Three estimators of the secular rate

1. **Dense fit.** Ordinary least squares $\varpi(t) = \varpi_0 + \dot\varpi\,t$ on every sample
   (`fit_secular_rate`). The slope error
   $\sigma_{\dot\varpi} = \sqrt{\sum\mathrm{res}^2/(n-2)/\sum(t-\bar t)^2}$ assumes white
   residuals; with periodic residuals it is a *nominal* figure.
2. **Perihelion passages.** Local minima of $|\mathbf r|$ at interior samples, refined to the
   vertex of the parabola through the three samples around the minimum
   (`perihelion_passages`); $\varpi$ at the nearest sample; the same fit on these $\sim P$-spaced
   points. Sparser, less sensitive to short-period terms, same secular rate. The passages are
   spaced by the *anomalistic* period, which differs from the sidereal one by the precession
   itself.
3. **Run-vs-run difference.** Two runs that share the initial state, the time grid and the
   integrator and differ in **one force term**: fit $\varpi_A(t)-\varpi_B(t)$ (wrapped sample by
   sample, then unwrapped; `perihelion_rate_difference`). Everything the runs have in common
   cancels — the planetary wobble *and* the integrator's own spurious precession.

## Calibration on a known signal: the oblate Sun

Before pointing the instrument at GR, Lesson 03 tests it on a perturbation with an analytic
answer. A quadrupole $J_2$ of the central body adds (symmetry axis $z$)

$$
\mathbf a_{J_2} = \frac32 J_2\,\frac{GM R^2}{r^5}\,
\big(x(5z^2/r^2-1),\; y(5z^2/r^2-1),\; z(5z^2/r^2-3)\big),
$$

and the first-order secular theory gives, with $n = 2\pi/P$ and $p = a(1-e^2)$,

$$
\dot\Omega = -\tfrac32 nJ_2\Big(\tfrac Rp\Big)^2\cos i, \qquad
\dot\omega = \tfrac34 nJ_2\Big(\tfrac Rp\Big)^2(5\cos^2 i-1), \qquad
\dot\varpi = \tfrac34 nJ_2\Big(\tfrac Rp\Big)^2(5\cos^2 i - 1 - 2\cos i).
$$

With $J_2 = 10^{-3}$ (exaggerated; the real Sun has $2.2\times10^{-7}$) Mercury should advance by
$123.2$″/cy; the three estimators give $123.22$, $123.17$ and $123.22$. With the real $J_2$ the
analytic $0.0271$″/cy is recovered to four decimals by differencing, while the dense fit
returns $0.0283$ — the signal *plus* the integrator's $0.0012$.

## Short spans versus the century

A least-squares slope over a span shorter than the long-period planetary terms (Jupiter–Saturn
"great inequality" ~900 years; Venus–Earth terms of 8 and 240 years) is biased by whichever
phase of those terms the span covers. The nine-body Newtonian rate of Mercury is $569$″/cy over
ten years, $529.2$″/cy over a century (dense fit; $529.3$ from passages), against Clemence's
$531.6$ (the remaining 2″ are the OLS bias of the long-period terms plus the ~0.1″ of the
omitted Moon, asteroids and Pluto; the Horizons series fitted the same way over the same
century gives 572.2 rather than the observed 574.1). This is why every GR number in gravsim is
a **difference**, which is clean after a decade: $42.98\pm0.03$″/cy from ten years.

## Noise floor

A Kepler orbit must not precess. The analytic solution gives $\varpi$ constant to
$10^{-9}$″ over ten years; DOP853 at `rtol=1e-12` gives $+0.0012$″/cy (dense and passages),
four orders of magnitude below the GR signal and $400\times$ below the ROADMAP target of
$0.5$″/cy. Fixed-step leapfrog and Yoshida4 at half-day steps give $-2\times10^5$ and
$+2.6\times10^3$″/cy: they can only be used in differenced mode, where the spurious part
cancels (Lesson 02's "which planet does what" table does exactly that).

The same toolkit measures the node $\Omega$ and the argument $\omega$ separately (`analysis.precession.element_series`), which is how [What competes with GR](extras.md) tells the oblate Sun and frame dragging apart from GR.
