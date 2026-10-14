# Lessons

One notebook per phase, in `notebooks/`. Every lesson states the physics before the code,
runs in well under a minute from the cached ephemerides (no network), ends with a summary and
exercises (★ = harder), and has a matching lab in the Streamlit app. Run them in order with
`.venv/bin/jupyter lab notebooks/`; `Kernel → Restart & Run All` is the intended way to read
them. The notebooks are stored without outputs; `.venv/bin/python -m pytest -q -m slow
tests/test_notebooks.py` executes all of them as a test.

| lesson | phase | runs in | learning goals |
|---|---|---|---|
| **01 Why the integrator matters** | 1 | 7 s | two-body equations and conserved quantities; order of a method from a log–log convergence plot; symplectic (bounded energy error) vs Runge–Kutta (drift); *spurious precession* of fixed-step schemes versus the 43″/cy signal; why DOP853 is the production integrator |
| **02 The clockwork solar system** | 2 | 9 s | N-body equations; the barycentre and the Sun's wobble (~2 R☉, drawn in 3-D); conservation checks; Mercury's osculating elements and the planetary wobble of its perihelion; which planet contributes what; km-level residuals against JPL Horizons and their meaning |
| **03 Measuring a drift of 0.1″ per orbit** | 3 | 12 s | osculating elements, the eccentricity vector, why ϖ = Ω + ω; three estimators (dense fit, perihelion passages, run-vs-run difference); calibration on an oblate Sun (J₂) with an analytic answer; the 10-year vs century bias of the Newtonian rate (569 vs 529/531.6″/cy) |
| **04 Mercury's extra 43 arcseconds** | 4 | 20 s | the 1PN correction and its size (v²/c² ~ 10⁻⁸); Δϖ = 6πGM/(c²a(1−e²)); 42.98″/cy measured in the two-body and nine-body runs and *from the Horizons data*; the 1/(a(1−e²)) law across the planets; the c/100 rosette in 3-D; PPN β, γ |
| **05 Orbits around a black hole** | 4 | 5 s | Schwarzschild effective potential, circular orbits, ISCO, photon sphere; exact vs 1PN precession for Mercury (10⁻⁴″/cy); where 1PN fails and where Mercury would be captured; precessing, zoom-whirl and plunging orbits in geometric units; proper vs coordinate time |

Later phases (EIH N-body, PPN playground, light deflection) will add Lessons 06–08.

## How the lessons connect

1. Lesson 01 establishes the **noise floor**: an integrator whose own precession exceeds
   43″/cy cannot measure GR. DOP853 at `rtol=1e-12` gives 0.001″/cy.
2. Lesson 02 shows the **signal to be removed**: the planets turn Mercury's ellipse by
   ~530″/cy with a wobble of tens of arcseconds on top.
3. Lesson 03 builds the **instrument** and calibrates it on a perturbation of known size.
4. Lesson 04 points the instrument at **GR** and finds 42.98″/cy three ways.
5. Lesson 05 checks the 1PN term against the **exact** Schwarzschild orbit and explores the
   strong field where the expansion breaks down.
