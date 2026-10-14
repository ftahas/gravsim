# Interactive lab (Streamlit)

```bash
uv pip install -e ".[app]"                 # streamlit + plotly
.venv/bin/streamlit run app/streamlit_app.py
```

`app/streamlit_app.py` adds widgets, caching and plotly figures on top of `app/labs.py`, a
small module of pure functions that call the package (`tests/test_app.py` tests both; the
app is run headlessly with `streamlit.testing.v1.AppTest`). Each lab mirrors a lesson.

## 1 · Integrator lab (Lesson 01)

Sun + Mercury with any fixed-step scheme (`euler`, `symplectic_euler`, `leapfrog`, `rk4`,
`yoshida4`, `yoshida6`), a step from 4 days down to 3 hours and a span of up to 30 years.
Readouts: steps per orbit, maximum relative energy error, and the **spurious perihelion
precession** of the run next to the 42.98″/cy GR signal. The orbit is drawn in 3-D, the energy
error on a log axis. Halve the step and watch the spurious rate fall by $2^{\rm order}$.

## 2 · Solar system: Newton vs GR (Lesson 02, 04)

Sun + 8 planets from the J2000 snapshot, DOP853, two runs on the same grid: Newton and Newton
+ 1PN. Controls: span (1–50 years), **speed of light** ($c$, $c/10$, $c/100$, $c/1000$), which
planets to draw, and an **amplification** of the GR − Newton displacement of one body,
$\mathbf r_N + k\,(\mathbf r_{GR}-\mathbf r_N)$, drawn as a black trace in the 3-D scene. At the
real $c$ the displacement after five years is a few hundred km — invisible at 10⁵ km per pixel
— so $k = 10^5$ makes it a visible loop; at $c/100$ no amplification is needed and the rosette
appears. The separation in km is plotted against time.

## 3 · Precession tracker (Lesson 03, 04)

$\varpi(t)$ of Mercury, Venus, EMB or Mars in the Newton and GR runs (two-body or nine-body),
with the least-squares secular fits, and the **run-vs-run difference** with its fitted rate
next to the analytic $6\pi GM/(c^2 a(1-e^2))$ (converted to arcsec per century with the same
$c$). The span goes up to a century; with the planets present the dense fits wobble while the
difference stays a straight line.

## 4 · Effective-potential explorer (Lesson 05)

Geometric units ($GM = c = 1$). Sliders for $L$ (3–6 $GM/c$), the effective energy
$\mathcal E = (E^2/c^2 - c^2)/2$ relative to the barrier top, the starting radius and the
proper-time span. The left panel shows $V_{\rm eff}$ (Schwarzschild and Newton) with the
energy line, the photon sphere and the ISCO; the right panel the geodesic in the equatorial
plane with the horizon, photon sphere and ISCO as rings. Just below the barrier top the orbit
*whirls*; above it, it plunges; below $L = \sqrt{12}$ there is no barrier. The geodesic is
integrated in proper time (`metric.schwarzschild.geodesic_orbit`), so bound, escaping and
captured orbits are all handled.
