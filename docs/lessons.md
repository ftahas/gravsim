# Lessons

Eight notebooks in `notebooks/`, one or two per phase. Every lesson states the physics before
the code, runs in well under a minute from the cached ephemerides (no network), ends with a
summary and exercises (★ = harder), and has a matching lab in the Streamlit app. Run them in order with
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
| **06 Le Verrier, Vulcan, Einstein** | 5 | 20 s | the history (1859 → Vulcan → November 1915); the Einstein–Infeld–Hoffmann equations in words and in PPN form, and why the ephemerides integrate them; three nine-body runs (Newton, EIH, Newton + 1PN); the anomaly *from data*: ϖ(Horizons) − ϖ(run) drifts at 43.02 ± 0.13″/cy for Newton and is flat for EIH; EIH vs Sun-only 1PN (10⁻⁵″/cy, periodic km); a binary star on Mercury's orbit showing the advance is mass-ratio independent |
| **07 What if gravity were a different theory?** | 5 | 26 s | the PPN metric and the meaning of γ and β; Brans–Dicke; the Gauss-equation derivation of (2 + 2γ − β)/3; a 3×3 grid of (β, γ) runs at c/100 fitted against the prediction (< 1 %); the 2γ − β degeneracy and a theory that mimics GR's 43″ but not its light bending; the Cassini, LLR and Mercury bounds to scale |
| **08 Eddington 1919** | 6 | 8 s | Soldner's 0.875″ vs Einstein's 1.75″ and the (1 + γ)/2 table; deflection vs impact parameter with the 1919–2004 measurements; ray bundles past a compact mass and past the Sun; photon sphere, capture and the EHT shadows; the Shapiro delay (Earth–Mars 124 μs one way, Cassini geometry) and the conjunction spike; why light tests break the 2γ − β degeneracy |

## How the lessons connect

1. Lesson 01 establishes the **noise floor**: an integrator whose own precession exceeds
   43″/cy cannot measure GR. DOP853 at `rtol=1e-12` gives 0.001″/cy.
2. Lesson 02 shows the **signal to be removed**: the planets turn Mercury's ellipse by
   ~530″/cy with a wobble of tens of arcseconds on top.
3. Lesson 03 builds the **instrument** and calibrates it on a perturbation of known size.
4. Lesson 04 points the instrument at **GR** and finds 42.98″/cy three ways.
5. Lesson 05 checks the 1PN term against the **exact** Schwarzschild orbit and explores the
   strong field where the expansion breaks down.
6. Lesson 06 lets **every body** curve spacetime (EIH) and reads the anomaly off the Horizons
   data: Newton drifts at 43″/cy, EIH does not.
7. Lesson 07 asks what **other theories** would predict (β, γ) and finds that Mercury alone
   measures only 2γ − β.
8. Lesson 08 turns to **light**, which measures γ alone (deflection, Shapiro delay) and so
   completes the test.
