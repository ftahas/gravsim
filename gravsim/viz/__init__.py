"""Plotting helpers for the lessons and the app.

:mod:`gravsim.viz.plots` draws static matplotlib figures (3-D and 2-D orbits, energy error,
perihelion precession, effective potential, run-to-run separation, panel grids) and works
with the non-interactive Agg backend.  :mod:`gravsim.viz.interactive` builds plotly figures
for the app; plotly is optional and only imported when one of its functions is called.

Both share one colour-blind-safe palette (``PALETTE``, with a fixed colour per planet in
``BODY_COLORS``) and the same unit conventions as the rest of the package: AU, day, angles
converted to arcseconds for precession plots, times to Julian years.
"""

from gravsim.viz.interactive import orbits_3d_figure, precession_figure
from gravsim.viz.plots import (
    BODY_COLORS,
    PALETTE,
    PrecessionSeries,
    body_colors,
    body_paths,
    cube_limits,
    figure_grid,
    plot_effective_potential,
    plot_energy_error,
    plot_orbits_2d,
    plot_orbits_3d,
    plot_precession,
    plot_separation,
    precession_series,
    resolve_bodies,
)

__all__ = [
    "BODY_COLORS",
    "PALETTE",
    "PrecessionSeries",
    "body_colors",
    "body_paths",
    "cube_limits",
    "figure_grid",
    "orbits_3d_figure",
    "plot_effective_potential",
    "plot_energy_error",
    "plot_orbits_2d",
    "plot_orbits_3d",
    "plot_precession",
    "plot_separation",
    "precession_figure",
    "precession_series",
    "resolve_bodies",
]
