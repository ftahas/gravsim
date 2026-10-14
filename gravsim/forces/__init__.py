"""Force laws: every one is an ``AccelFn``, ``accel(pos, vel, gm, t) -> acc``.

Shapes ``(N, 3), (N, 3), (N,), float -> (N, 3)``; units AU, day, ``GM`` in AU^3/day^2.
Force laws are pure functions of their inputs (no mutation, no hidden state) and can be
summed with ``combine``.

* ``newton``: pairwise Newtonian gravity (``accel_newton``; numba kernel + numpy fallback).
* ``pn1``: first post-Newtonian correction of one central mass, PPN-parametrised
  (``make_pn1`` = correction only, ``make_newton_pn1`` = Newton + correction).
* ``eih``: (Phase 5) full Einstein-Infeld-Hoffmann N-body 1PN law, Newtonian term included.
"""

from __future__ import annotations

import numpy as np

from gravsim.forces.newton import (
    HAVE_NUMBA,
    AccelFn,
    accel_newton,
    accel_newton_numba,
    accel_newton_numpy,
)
from gravsim.forces.pn1 import make_newton_pn1, make_pn1

__all__ = [
    "AccelFn",
    "HAVE_NUMBA",
    "accel_newton",
    "accel_newton_numba",
    "accel_newton_numpy",
    "combine",
    "make_newton_pn1",
    "make_pn1",
]


def _law_name(fn: AccelFn) -> str:
    """Short, address-free name of a force law for ``__name__`` and ``params["law"]``.

    Plain functions and closures give their ``__name__``; a ``functools.partial`` gives the
    name of the wrapped function; a callable instance gives its class name.
    """
    name = getattr(fn, "__name__", None)
    if name is None:
        name = getattr(getattr(fn, "func", None), "__name__", None) or type(fn).__name__
    return str(name)


def combine(*fns: AccelFn) -> AccelFn:
    """Return an ``AccelFn`` whose output is the sum of the outputs of ``fns``.

    Parameters
    ----------
    *fns : AccelFn
        One or more force laws with the ``accel(pos, vel, gm, t)`` signature. Any callable
        works (plain functions, closures, ``functools.partial`` objects, callable instances);
        a ``params`` attribute, if present, is merged into the combined law's ``params``.

    Returns
    -------
    AccelFn
        ``accel(pos, vel, gm, t) = sum_k fns[k](pos, vel, gm, t)`` as a new (N, 3) array.

    Raises
    ------
    ValueError
        If no function is given.
    """
    if not fns:
        raise ValueError("combine() needs at least one AccelFn")

    def accel_combined(
        pos: np.ndarray, vel: np.ndarray, gm: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        acc = np.array(fns[0](pos, vel, gm, t), dtype=np.float64)  # own copy, safe to add into
        for fn in fns[1:]:
            acc += fn(pos, vel, gm, t)
        return acc

    names = [_law_name(fn) for fn in fns]
    terms = [
        dict(getattr(fn, "params", None) or {"law": name})
        for fn, name in zip(fns, names, strict=True)
    ]
    accel_combined.__name__ = "combine(" + ", ".join(names) + ")"
    accel_combined.params = {  # type: ignore[attr-defined]
        "law": "+".join(
            str(term.get("law", name)) for term, name in zip(terms, names, strict=True)
        ),
        "terms": terms,
    }
    return accel_combined
