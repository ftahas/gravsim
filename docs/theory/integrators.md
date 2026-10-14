# Integrators and their errors

*Modules: `gravsim/integrators/fixed_step.py`, `driver.py`, `scipy_wrap.py`. Lesson 01, app lab 1.*

## One-step methods and order

Every force law is a function $\mathbf a(\mathbf x,\mathbf v,t)$ and the system is
$\dot{\mathbf x} = \mathbf v$, $\dot{\mathbf v} = \mathbf a$. A one-step method maps
$(\mathbf x_n,\mathbf v_n)\mapsto(\mathbf x_{n+1},\mathbf v_{n+1})$ over $\Delta t$. It has **order $p$**
when the local error of one step is $O(\Delta t^{p+1})$ and therefore the global error after a
fixed span is $O(\Delta t^p)$: halving the step divides the error by $2^p$, a straight line of
slope $p$ on a log–log plot of error against step (Lesson 01, §3).

## The schemes in gravsim

Writing the kick $K(h):\ \mathbf v\to\mathbf v + h\,\mathbf a(\mathbf x)$ and the drift
$D(h):\ \mathbf x\to\mathbf x + h\,\mathbf v$ — the exact flows of the potential and kinetic parts of
$H = T(\mathbf v)+V(\mathbf x)$ —

| name | order | composition | force evaluations/step | symplectic |
|---|---|---|---|---|
| `euler` | 1 | both updates from the old state | 1 | no |
| `symplectic_euler` | 1 | $K(\Delta t)\,D(\Delta t)$ | 1 | yes |
| `leapfrog` | 2 | $K(\tfrac{\Delta t}{2})\,D(\Delta t)\,K(\tfrac{\Delta t}{2})$ (velocity Verlet) | 1 (adjacent half-kicks share one evaluation) | yes |
| `rk4` | 4 | classical Runge–Kutta on $(\mathbf x,\mathbf v)$ | 4 | no |
| `yoshida4` | 4 | three leapfrogs of sizes $w_1,w_0,w_1$; $w_1 = 1/(2-2^{1/3})$, $w_0 = -2^{1/3}w_1$ | 4 | yes |
| `yoshida6` | 6 | seven leapfrogs, Yoshida (1990) solution A | 8 | yes |

(Yoshida 1990, *Phys. Lett. A* 150, 262; Hairer, Lubich & Wanner, *Geometric Numerical
Integration*, 2006.)

## Why symplectic matters: the modified Hamiltonian

A symplectic integrator is the *exact* flow (to all orders in $\Delta t$, in the sense of
backward error analysis) of a **modified Hamiltonian** $\tilde H = H + \Delta t^p H_p + \ldots$.
Two consequences, both visible in Lesson 01:

1. The energy error is **bounded**: $H$ oscillates about $\tilde H$ by $O(\Delta t^p)$ and never
   drifts. A Runge–Kutta scheme of the same order lets $H$ drift linearly.
2. The modified problem is **not** a Kepler problem. Its orbit is an ellipse that slowly turns:
   leapfrog on a Kepler orbit precesses by $O\big((2\pi/N)^2\big)$ per orbit with $N$ the number
   of steps per orbit; Yoshida4 by $O\big((2\pi/N)^4\big)$. This *spurious precession* is the
   quantity that decides whether a method can be used for the GR measurement.

Measured on Sun + Mercury at $\Delta t = 0.5$ d ($N = 176$), ten years:

| scheme | max $|\Delta E/E_0|$ | spurious precession |
|---|---|---|
| leapfrog | $3\times10^{-4}$ (bounded) | $-1.97\times10^{5}$″/cy |
| rk4 | drifts | $+85$″/cy |
| yoshida4 | $\sim10^{-9}$ (bounded) | $+2.6\times10^{3}$″/cy ($\propto\Delta t^4$: $+165$ at $0.25$ d) |
| yoshida6 | (bounded) | $-0.7$″/cy |
| DOP853, `rtol=1e-12` | $2\times10^{-11}$ (drifts) | $+0.0012$″/cy |

The GR signal is $+42.98$″/cy. Leapfrog would need $\Delta t \approx 7$ minutes to get below it;
Yoshida4 about 2.5 hours. gravsim's production choice is **DOP853** (Dormand–Prince 8(5,3),
`scipy.integrate.solve_ivp`, `rtol=1e-12`, `atol=1e-15`): adaptive, four orders of magnitude
below the signal, and — decisively — indifferent to velocity-dependent forces.

## Velocity-dependent forces

The kick is the exact flow of $V(\mathbf x)$ only if $\mathbf a$ does not depend on $\mathbf v$.
The 1PN correction does, which breaks the time symmetry of plain kick-drift-kick and with it
the bounded-energy property. The fixed-step classes take `vel_dependent=True`, which keeps the
opening half-kick explicit and makes the closing one **implicit**,
$\mathbf v' = \mathbf v + \tfrac{\Delta t}{2}\mathbf a(\mathbf x',\mathbf v',t+\Delta t)$, solved by
fixed-point iteration (`n_iter=2` by default; each iteration gains one power of $\Delta t$ in
the kick error). This is the idea behind REBOUNDx's implicit-midpoint GR operator. For the
century-long GR runs, however, DOP853 is both simpler and more accurate.

## The driver

`integrate(state, accel, t_end, dt, integrator, save_every)` shrinks the requested step
minimally so that an integer number of steps (a multiple of `save_every`) lands exactly on
`t_end`, forms $t_i = t_0 + i\,\Delta t$ rather than accumulating, stores the initial state as
sample 0, and records the actual step and all options in `Trajectory.meta`.
`integrate_scipy(state, accel, t_eval, method="DOP853", rtol, atol)` flattens the state to
$y = [\mathbf x.\mathrm{ravel}(), \mathbf v.\mathrm{ravel}()]$ and samples exactly at `t_eval`.

## Accuracy floor of DOP853

Against the analytic Kepler solution, with the defaults: $\sim3\times10^{-8}$ AU after 100
Mercury orbits, $\sim5\times10^{-7}$ AU after a century, $|\Delta E/E|\sim10^{-10}$. Tightening
`rtol` reaches $\sim4\times10^{-10}$ AU per 100 orbits, the floor of this formulation in double
precision (absolute coordinates of order 0.3 AU over $10^4$ steps). The ROADMAP's
"$<10^{-10}$ AU" row is therefore replaced by the precession noise floor above, which is the
quantity the project needs.
