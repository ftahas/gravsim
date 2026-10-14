"""Static matplotlib figures for the lessons and the app.

Physics and conventions
-----------------------
* Positions are barycentric, J2000-ecliptic coordinates in AU (``Trajectory.pos`` has shape
  ``(T, N, 3)``).  Orbit plots draw every body *relative to the central body*,
  ``r_i(t) = pos_i(t) - pos_c(t)``, so the Sun's barycentric wobble does not smear the
  picture; ``relative=False`` shows the raw barycentric paths (and therefore the wobble).
  The central body is marked with a star at its last sample (the origin when ``relative``).
* Energy error: ``rel_err(t) = (E(t) - E_0) / |E_0|``.  On a log axis the plot shows
  ``|rel_err|``; exactly-zero samples (usually the first) are masked by the log scale.
* Precession: the longitude of perihelion ``varpi = Omega + omega`` [rad] is unwrapped and
  referenced to its initial value, ``Delta varpi(t) = varpi(t) - varpi(t_0)``, then converted
  with ``1 rad = 206264.806 arcsec`` (``constants.ARCSEC_PER_RAD``); time is converted to
  Julian years (365.25 d).  A secular fit with slope ``d varpi / dt`` [rad/day] is drawn as the
  straight line of that slope whose intercept minimises the squared residuals, and labelled
  with the slope in ``units`` per Julian century (36525 d) -- Mercury's GR value is 42.98 "/cy.
* Effective potential (Schwarzschild, per unit mass, ``L`` = specific angular momentum):
  ``V_eff(r) = -GM/r + L^2/(2 r^2) - GM L^2/(c^2 r^3)``; the Newtonian potential lacks the
  last term.  The plot takes already-sampled curves; optional vertical marks name special
  radii (photon sphere ``3GM/c^2``, ISCO ``6GM/c^2``, circular orbits).  Because both curves
  diverge as ``r -> 0`` the default y-range frames the **Schwarzschild curve alone** from
  its centrifugal barrier (innermost interior maximum of ``V_eff``) outward, where the
  barrier top and the well live; the Newtonian curve, whose ``L^2/(2 r^2)`` term is still
  rising steeply at that radius, is clipped there and shown where it rejoins the GR curve.
* Separation: ``|r_A(t) - r_B(t)|`` of the same body in two runs, AU converted to km with
  ``constants.AU_KM`` (1 AU = 149 597 870.700 km).

Styling
-------
Eight-colour colour-blind-safe categorical palette in a fixed order.  Planets own a fixed
slot (Mercury ... Neptune) so a body keeps its colour whichever subset is plotted; other
series take the free slots in order of appearance and anything beyond eight is muted grey.
Thin lines, hairline solid grid, recessive axes.  Orbit plots show a legend for two or more
bodies and name a single body in the title; measurement plots show a legend whenever a
``label`` is given, so identity never rests on colour alone.  Every function returns the
Axes it drew on (``figure_grid`` returns the Figure and its Axes) and never calls
``plt.show()``; pass ``ax`` to compose panels.  ``pyplot`` is imported lazily, so
``matplotlib.use("Agg")`` before or after importing this module both work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gravsim.bodies import Trajectory
from gravsim.constants import (
    ARCSEC_PER_RAD,
    AU_KM,
    DEG_PER_RAD,
    JULIAN_CENTURY_DAYS,
    JULIAN_YEAR_DAYS,
    PLANETS,
)

__all__ = [
    "PALETTE",
    "BODY_COLORS",
    "PrecessionSeries",
    "body_colors",
    "body_paths",
    "cube_limits",
    "figure_grid",
    "plot_effective_potential",
    "plot_energy_error",
    "plot_orbits_2d",
    "plot_orbits_3d",
    "plot_precession",
    "plot_separation",
    "precession_series",
    "resolve_bodies",
]

# ----------------------------------------------------------------------------- palette
PALETTE: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
"""Categorical palette, fixed order; adjacent pairs validated for protan/deutan vision."""

BODY_COLORS: dict[str, str] = dict(zip(PLANETS, PALETTE, strict=True))
BODY_COLORS["Earth"] = BODY_COLORS["EMB"]
"""Fixed colour per planet (Mercury blue, Venus orange, Earth/EMB aqua, ..., Neptune red)."""

INK = "#0b0b0b"  # primary text, central-body marker, fit lines
INK_SECONDARY = "#52514e"  # tick labels, annotations
MUTED = "#898781"  # axis labels, vertical marks, overflow series
GRID = "#e1e0d9"  # hairline grid
AXIS = "#c3c2b7"  # spines
SURFACE = "#fcfcfb"  # axes background and the ring around markers

LINE_WIDTH = 1.5
THIN_WIDTH = 0.8
MARKER_AREA = 90.0  # scatter marker area in points^2
_FIT_GID = "gravsim-fit"  # tag on fit lines so they do not consume a palette slot

_TIME_UNITS: dict[str, tuple[float, str]] = {
    "days": (1.0, "t [days]"),
    "years": (1.0 / JULIAN_YEAR_DAYS, "t [years]"),
    "centuries": (1.0 / JULIAN_CENTURY_DAYS, "t [centuries]"),
}
_TIME_ALIASES: dict[str, str] = {
    "d": "days",
    "day": "days",
    "days": "days",
    "y": "years",
    "yr": "years",
    "year": "years",
    "years": "years",
    "cy": "centuries",
    "century": "centuries",
    "centuries": "centuries",
}
_ANGLE_UNITS: dict[str, tuple[float, str]] = {
    "arcsec": (ARCSEC_PER_RAD, "arcsec"),
    "deg": (DEG_PER_RAD, "deg"),
    "rad": (1.0, "rad"),
}
_ANGLE_ALIASES: dict[str, str] = {
    "arcsec": "arcsec",
    "arcsecond": "arcsec",
    "arcseconds": "arcsec",
    '"': "arcsec",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
}
_LENGTH_UNITS: dict[str, tuple[float, str]] = {
    "km": (AU_KM, "km"),
    "au": (1.0, "AU"),
    "m": (AU_KM * 1e3, "m"),
}
_LENGTH_ALIASES: dict[str, str] = {"km": "km", "au": "au", "m": "m"}


# ----------------------------------------------------------------------------- helpers
def _unit(
    value: str, aliases: Mapping[str, str], table: Mapping[str, tuple[float, str]], kind: str
) -> tuple[float, str]:
    """Return ``(factor, label)`` for a unit name, accepting common aliases."""
    key = aliases.get(str(value).strip().lower())
    if key is None:
        choices = sorted(set(aliases.values()))
        raise ValueError(f"unknown {kind} unit {value!r}; choose from {choices}")
    return table[key]


def _pyplot():
    import matplotlib.pyplot as plt  # lazy: keeps backend selection with the caller

    return plt


def _is_3d(ax: Axes) -> bool:
    return getattr(ax, "name", "") == "3d"


def _new_axes(
    ax: Axes | None, projection: str | None = None, figsize: tuple[float, float] = (6.4, 4.4)
) -> Axes:
    if ax is not None:
        return ax
    fig = _pyplot().figure(figsize=figsize, layout="constrained")
    return fig.add_subplot(projection=projection)


def _style_axes(ax: Axes) -> None:
    """Quiet chrome: hairline solid grid, recessive spines, secondary-ink ticks."""
    ax.set_facecolor(SURFACE)
    if _is_3d(ax):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(True)
        return
    ax.grid(True, color=GRID, linewidth=THIN_WIDTH, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_SECONDARY, width=THIN_WIDTH)


def _legend(ax: Axes, min_entries: int = 2) -> None:
    labels = ax.get_legend_handles_labels()[1]
    if len(labels) >= min_entries:
        # upper left is the empty corner of a 3-D box; upper right collides with the z axis
        ax.legend(frameon=False, fontsize="small", loc="upper left" if _is_3d(ax) else "best")


def _legend_or_title(ax: Axes, frame: str) -> None:
    """Legend for >= 2 labelled series; a single series is named in the title instead."""
    labels = ax.get_legend_handles_labels()[1]
    auto_titles = {f"{lab} ({frame})" for lab in labels}
    if len(labels) >= 2:
        if ax.get_title() in auto_titles:
            ax.set_title("")
        _legend(ax)
    elif len(labels) == 1 and not ax.get_title():
        ax.set_title(f"{labels[0]} ({frame})", fontsize="medium", color=INK)


def _slot_color(ax: Axes) -> str:
    """Next palette slot for an axes that accumulates series over repeated calls."""
    n = sum(1 for line in ax.lines if line.get_gid() != _FIT_GID)
    return PALETTE[n] if n < len(PALETTE) else MUTED


def body_colors(names: Sequence[str]) -> list[str]:
    """Colour per series name: planets keep their fixed slot, others take free slots in order.

    Parameters
    ----------
    names : sequence of str
        Series (body) names in plotting order.

    Returns
    -------
    list of str
        Hex colours, one per name.  Beyond the eight palette slots the colour is a muted grey
        (the palette is never cycled, so two series never share a hue by accident).
    """
    used = {BODY_COLORS[n] for n in names if n in BODY_COLORS}
    free = [c for c in PALETTE if c not in used]
    out: list[str] = []
    for name in names:
        if name in BODY_COLORS:
            out.append(BODY_COLORS[name])
        else:
            out.append(free.pop(0) if free else MUTED)
    return out


def resolve_bodies(
    traj: Trajectory,
    bodies: Sequence[str | int] | str | int | None,
    central: str | int,
) -> tuple[str, list[str]]:
    """Normalise the ``bodies``/``central`` arguments of the orbit plots to names.

    Parameters
    ----------
    traj : Trajectory
    bodies : sequence of str or int, str, int, or None
        Bodies to draw (names or indices).  ``None`` means every body except ``central``.
    central : str or int
        Central body, by name or index.

    Returns
    -------
    (central_name, names) : tuple of str and list of str

    Raises
    ------
    KeyError
        If a name is not in ``traj.names``.
    """
    if isinstance(central, (int, np.integer)):
        c_name = traj.names[int(central)]
    else:
        traj.index(central)
        c_name = str(central)
    if bodies is None:
        return c_name, [n for n in traj.names if n != c_name]
    if isinstance(bodies, (str, int, np.integer)):
        bodies = [bodies]
    names: list[str] = []
    for b in bodies:
        if isinstance(b, (int, np.integer)):
            names.append(traj.names[int(b)])
        else:
            traj.index(b)
            names.append(str(b))
    return c_name, names


def body_paths(
    traj: Trajectory, names: Sequence[str], central: str, relative: bool = True
) -> list[np.ndarray]:
    """Position series ``(T, 3)`` of each body, relative to ``central`` or barycentric."""
    if relative:
        return [traj.relative(n, central)[0] for n in names]
    return [traj.body(n)[0] for n in names]


def cube_limits(
    points: np.ndarray, pad: float = 0.05
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Equal-extent x, y, z ranges enclosing ``points`` (shape ``(..., 3)``).

    The ranges share one half-width (the largest extent, padded by ``pad``) around the
    centre of the bounding box, so a 3-D axes using them has true 1:1:1 scale.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        raise ValueError("cube_limits needs at least one finite point")
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max())
    if half <= 0.0:
        half = 1.0
    half *= 1.0 + pad
    xl, yl, zl = ((float(c - half), float(c + half)) for c in centre)
    return xl, yl, zl


def _set_cube(ax: Axes, points: np.ndarray) -> None:
    xl, yl, zl = cube_limits(points)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.set_zlim(*zl)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _existing_corners_3d(ax: Axes) -> np.ndarray:
    (x0, x1), (y0, y1), (z0, z1) = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    return np.array([[x0, y0, z0], [x1, y1, z1]])


# ----------------------------------------------------------------------------- orbits
def plot_orbits_3d(
    traj: Trajectory,
    bodies: Sequence[str | int] | str | int | None = None,
    central: str | int = "Sun",
    ax: Axes | None = None,
    equal_aspect: bool = True,
    relative: bool = True,
) -> Axes:
    """Draw each body's path in 3-D, relative to the central body, which is marked with a star.

    Parameters
    ----------
    traj : Trajectory
    bodies : sequence of str or int, optional
        Bodies to draw; default every body except ``central``.
    central : str or int, default "Sun"
        Reference body.  Marked at its last sample (the origin when ``relative``).
    ax : Axes3D, optional
        Existing 3-D axes to draw into; a new figure is created when omitted.
    equal_aspect : bool, default True
        Give x, y, z the same range (a cube enclosing all drawn data) and a 1:1:1 box.
    relative : bool, default True
        Subtract the central body's position.  When False the barycentric paths are drawn
        and the central body's own path is added as a thin ink line.

    Returns
    -------
    Axes3D
        One ``Line3D`` per drawn body (plus one for the central path when not ``relative``);
        the central marker is a scatter collection, not a line.
    """
    c_name, names = resolve_bodies(traj, bodies, central)
    if ax is not None and not _is_3d(ax):
        raise TypeError("plot_orbits_3d needs a 3-D axes (projection='3d'); got a 2-D one")
    ax = _new_axes(ax, projection="3d", figsize=(6.5, 6.0))
    had_data = bool(ax.lines) or bool(ax.collections)

    paths = body_paths(traj, names, c_name, relative)
    for name, path, color in zip(names, paths, body_colors(names), strict=True):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=LINE_WIDTH, label=name)

    if relative:
        c_path = np.zeros((1, 3))
        frame = f"relative to {c_name}"
    else:
        c_path = traj.body(c_name)[0]
        ax.plot(
            c_path[:, 0], c_path[:, 1], c_path[:, 2], color=INK, linewidth=THIN_WIDTH, label=c_name
        )
        frame = "barycentric"
    mark = c_path[-1]
    ax.scatter(
        [mark[0]],
        [mark[1]],
        [mark[2]],
        s=MARKER_AREA,
        c=INK,
        marker="*",
        edgecolors=SURFACE,
        linewidths=1.0,
        depthshade=False,
    )

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    _style_axes(ax)
    if equal_aspect:
        pts = [*paths, c_path]
        if had_data:
            pts.append(_existing_corners_3d(ax))
        _set_cube(ax, np.concatenate(pts, axis=0))
    _legend_or_title(ax, frame)
    return ax


def plot_orbits_2d(
    traj: Trajectory,
    bodies: Sequence[str | int] | str | int | None = None,
    central: str | int = "Sun",
    ax: Axes | None = None,
    relative: bool = True,
) -> Axes:
    """Draw each body's path projected on the x-y (ecliptic) plane, central body marked.

    Parameters
    ----------
    traj : Trajectory
    bodies : sequence of str or int, optional
        Bodies to draw; default every body except ``central``.
    central : str or int, default "Sun"
        Reference body, marked with a star at its last sample.
    ax : Axes, optional
    relative : bool, default True
        Subtract the central body's position (see :func:`plot_orbits_3d`).

    Returns
    -------
    Axes
        Equal-aspect axes with one ``Line2D`` per drawn body.
    """
    c_name, names = resolve_bodies(traj, bodies, central)
    if ax is not None and _is_3d(ax):
        raise TypeError("plot_orbits_2d needs a 2-D axes; use plot_orbits_3d for projection='3d'")
    ax = _new_axes(ax, figsize=(6.0, 6.0))

    paths = body_paths(traj, names, c_name, relative)
    for name, path, color in zip(names, paths, body_colors(names), strict=True):
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=LINE_WIDTH, label=name)

    if relative:
        c_path = np.zeros((1, 3))
        frame = f"relative to {c_name}"
    else:
        c_path = traj.body(c_name)[0]
        ax.plot(c_path[:, 0], c_path[:, 1], color=INK, linewidth=THIN_WIDTH, label=c_name)
        frame = "barycentric"
    mark = c_path[-1]
    ax.scatter(
        [mark[0]],
        [mark[1]],
        s=MARKER_AREA,
        c=INK,
        marker="*",
        edgecolors=SURFACE,
        linewidths=1.0,
        zorder=5,
    )

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_aspect("equal", adjustable="datalim")
    _style_axes(ax)
    _legend_or_title(ax, frame)
    return ax


# ----------------------------------------------------------------------------- measurements
def plot_energy_error(
    t: np.ndarray,
    rel_err: np.ndarray,
    label: str | None = None,
    ax: Axes | None = None,
    logy: bool = True,
    t_units: str = "days",
) -> Axes:
    """Relative energy error versus time; call repeatedly on one ``ax`` to overlay series.

    Parameters
    ----------
    t : array_like, shape (T,)
        Time in days.
    rel_err : array_like, shape (T,)
        ``(E(t) - E_0) / |E_0|``.
    label : str, optional
        Legend entry (e.g. the integrator and step).  Each call takes the next palette slot.
    ax : Axes, optional
    logy : bool, default True
        Log y-axis showing ``|rel_err|``; with ``False`` the signed error on a linear axis.
    t_units : {"days", "years", "centuries"}, default "days"
        Unit for the time axis.

    Returns
    -------
    Axes
    """
    t = np.asarray(t, dtype=np.float64)
    err = np.asarray(rel_err, dtype=np.float64)
    if t.ndim != 1 or err.shape != t.shape:
        raise ValueError(f"t and rel_err must both have shape (T,); got {t.shape}, {err.shape}")
    tf, t_label = _unit(t_units, _TIME_ALIASES, _TIME_UNITS, "time")
    ax = _new_axes(ax)
    y = np.abs(err) if logy else err
    ax.plot(t * tf, y, color=_slot_color(ax), linewidth=LINE_WIDTH, label=label)
    ax.set_yscale("log" if logy else "linear")
    ax.set_xlabel(t_label)
    ax.set_ylabel("|ΔE / E₀|" if logy else "ΔE / E₀")
    _style_axes(ax)
    _legend(ax, min_entries=1)
    return ax


class PrecessionSeries(NamedTuple):
    """Converted perihelion-longitude series ready to plot (see :func:`precession_series`)."""

    t: np.ndarray
    dvarpi: np.ndarray
    fit: np.ndarray | None
    rate_label: str | None
    t_label: str
    y_label: str


def precession_series(
    t: np.ndarray,
    varpi_rad: np.ndarray,
    rate_rad_per_day: float | None = None,
    units: str = "arcsec",
    t_units: str = "years",
) -> PrecessionSeries:
    """Unwrap, reference to the initial value and convert a perihelion-longitude series.

    Parameters
    ----------
    t : array_like, shape (T,)
        Time in days.
    varpi_rad : array_like, shape (T,)
        Longitude of perihelion in radians (any branch; it is unwrapped).
    rate_rad_per_day : float, optional
        Fitted secular rate.  When given, ``fit`` is the line of that slope with the
        least-squares intercept and ``rate_label`` quotes the rate per Julian century.
    units : {"arcsec", "deg", "rad"}, default "arcsec"
    t_units : {"days", "years", "centuries"}, default "years"

    Returns
    -------
    PrecessionSeries
        ``t`` and ``dvarpi = varpi - varpi[0]`` in the requested units, ``fit`` (or None),
        ``rate_label`` (or None) and plain-text axis labels.
    """
    t = np.asarray(t, dtype=np.float64)
    varpi = np.asarray(varpi_rad, dtype=np.float64)
    if t.ndim != 1 or varpi.shape != t.shape or t.size == 0:
        raise ValueError(f"t and varpi must have the same shape (T,); got {t.shape}, {varpi.shape}")
    tf, t_label = _unit(t_units, _TIME_ALIASES, _TIME_UNITS, "time")
    af, a_label = _unit(units, _ANGLE_ALIASES, _ANGLE_UNITS, "angle")

    dvarpi = np.unwrap(varpi)
    dvarpi = dvarpi - dvarpi[0]
    fit = None
    rate_label = None
    if rate_rad_per_day is not None:
        rate = float(rate_rad_per_day)
        trend = rate * (t - t[0])
        intercept = float(np.mean(dvarpi - trend))
        fit = (trend + intercept) * af
        rate_per_century = rate * JULIAN_CENTURY_DAYS * af
        rate_label = f"fit: {rate_per_century:.4g} {a_label}/century"
    return PrecessionSeries(t * tf, dvarpi * af, fit, rate_label, t_label, f"Δϖ [{a_label}]")


def plot_precession(
    t: np.ndarray,
    varpi_rad: np.ndarray,
    fit_rate_rad_per_day: float | None = None,
    ax: Axes | None = None,
    units: str = "arcsec",
    t_units: str = "years",
    label: str | None = None,
) -> Axes:
    """Perihelion-longitude drift ``varpi(t) - varpi(t_0)`` with an optional secular fit.

    Parameters
    ----------
    t : array_like, shape (T,)
        Time in days.
    varpi_rad : array_like, shape (T,)
        Longitude of perihelion in radians.
    fit_rate_rad_per_day : float, optional
        Secular rate from a linear fit (rad/day); drawn as a dashed ink line labelled with the
        rate per Julian century (e.g. ``fit: 42.98 arcsec/century``).
    ax : Axes, optional
    units : {"arcsec", "deg", "rad"}, default "arcsec"
    t_units : {"days", "years", "centuries"}, default "years"
    label : str, optional
        Legend entry for the series; repeated calls on one ``ax`` take successive colours.

    Returns
    -------
    Axes
    """
    s = precession_series(t, varpi_rad, fit_rate_rad_per_day, units=units, t_units=t_units)
    ax = _new_axes(ax)
    ax.plot(s.t, s.dvarpi, color=_slot_color(ax), linewidth=LINE_WIDTH, label=label)
    if s.fit is not None:
        ax.plot(
            s.t,
            s.fit,
            color=INK,
            linewidth=1.0,
            linestyle="--",
            label=s.rate_label,
            gid=_FIT_GID,
        )
    ax.set_xlabel(s.t_label)
    ax.set_ylabel(s.y_label)
    _style_axes(ax)
    _legend(ax, min_entries=1)
    return ax


def _potential_ylim(v_gr: np.ndarray) -> tuple[float, float] | None:
    """y-range framing the Schwarzschild curve from its centrifugal barrier outward, or None.

    The barrier is the innermost interior local maximum of the finite samples of ``v_gr``
    (``r`` ascending), so it is found even when its top lies below the large-``r`` asymptote
    (``sqrt(12) < L c / GM < 4``).  Only the GR curve enters the range: the Newtonian
    centrifugal term ``L^2 / (2 r^2)`` is still rising steeply at the GR barrier radius and
    would otherwise squash the barrier and well into a sliver of the panel.  Without an
    interior barrier the function returns None and autoscale applies.
    """
    idx = np.flatnonzero(np.isfinite(v_gr))
    if idx.size < 3:
        return None
    v = v_gr[idx]
    local_max = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])) + 1
    if local_max.size == 0:  # no interior barrier: leave autoscale alone
        return None
    outer = v[int(local_max[0]) :]
    lo, hi = float(outer.min()), float(outer.max())
    if not hi > lo:
        return None
    pad = 0.15 * (hi - lo)
    return lo - pad, hi + pad


def plot_effective_potential(
    r: np.ndarray,
    v_gr: np.ndarray,
    v_newton: np.ndarray | None = None,
    ax: Axes | None = None,
    marks: Mapping[str, float] | None = None,
    xlabel: str = "r [AU]",
    ylabel: str = r"$V_\mathrm{eff}$ [AU²/day²]",
    ylim: tuple[float, float] | None = None,
) -> Axes:
    """Effective potential(s) versus radius with optional named vertical marks.

    Parameters
    ----------
    r : array_like, shape (M,)
        Radii (AU, or any unit named in ``xlabel``).
    v_gr : array_like, shape (M,)
        Schwarzschild ``V_eff(r)`` (per unit mass).
    v_newton : array_like, shape (M,), optional
        Newtonian ``V_eff(r)`` for comparison.
    ax : Axes, optional
    marks : mapping of str to float, optional
        Name -> radius of vertical marks (e.g. ``{"ISCO": 6*gm/c**2}``), labelled in place.
    xlabel, ylabel : str
        Axis labels (override when ``r`` is in units of ``GM/c^2``).
    ylim : (float, float), optional
        y-range; the default frames the Schwarzschild curve from its centrifugal barrier
        outward (barrier top and well), and the Newtonian curve is clipped where it rises
        above that (see module docstring).  Autoscale applies when ``v_gr`` has no barrier.

    Returns
    -------
    Axes
        Lines: the curve(s) followed by one ``axvline`` per mark.
    """
    r = np.asarray(r, dtype=np.float64)
    v_gr = np.asarray(v_gr, dtype=np.float64)
    if r.ndim != 1 or v_gr.shape != r.shape:
        raise ValueError(f"r and v_gr must both have shape (M,); got {r.shape}, {v_gr.shape}")
    if v_newton is not None:
        v_newton = np.asarray(v_newton, dtype=np.float64)
        if v_newton.shape != r.shape:
            raise ValueError(f"v_newton must have shape {r.shape}; got {v_newton.shape}")

    ax = _new_axes(ax)
    ax.plot(r, v_gr, color=PALETTE[0], linewidth=LINE_WIDTH, label="Schwarzschild")
    if v_newton is not None:
        ax.plot(r, v_newton, color=PALETTE[1], linewidth=LINE_WIDTH, label="Newton")
    for name, r_mark in (marks or {}).items():
        ax.axvline(float(r_mark), color=MUTED, linewidth=THIN_WIDTH)
        ax.annotate(
            str(name),
            xy=(float(r_mark), 0.97),
            xycoords=ax.get_xaxis_transform(),
            xytext=(-3.0, 0.0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="top",
            fontsize="small",
            color=INK_SECONDARY,
        )
    if ylim is None:
        ylim = _potential_ylim(v_gr)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    _legend(ax)
    return ax


def plot_separation(
    t: np.ndarray,
    sep_au: np.ndarray,
    ax: Axes | None = None,
    label: str | None = None,
    units: str = "km",
    t_units: str = "years",
) -> Axes:
    """Separation of the same body between two runs versus time.

    Parameters
    ----------
    t : array_like, shape (T,)
        Time in days.
    sep_au : array_like, shape (T,)
        ``|r_A(t) - r_B(t)|`` in AU.
    ax : Axes, optional
    label : str, optional
        Legend entry; repeated calls on one ``ax`` take successive colours.
    units : {"km", "au", "m"}, default "km"
    t_units : {"days", "years", "centuries"}, default "years"

    Returns
    -------
    Axes
    """
    t = np.asarray(t, dtype=np.float64)
    sep = np.asarray(sep_au, dtype=np.float64)
    if t.ndim != 1 or sep.shape != t.shape:
        raise ValueError(f"t and sep_au must both have shape (T,); got {t.shape}, {sep.shape}")
    tf, t_label = _unit(t_units, _TIME_ALIASES, _TIME_UNITS, "time")
    lf, l_label = _unit(units, _LENGTH_ALIASES, _LENGTH_UNITS, "length")
    ax = _new_axes(ax)
    ax.plot(t * tf, sep * lf, color=_slot_color(ax), linewidth=LINE_WIDTH, label=label)
    ax.set_xlabel(t_label)
    ax.set_ylabel(f"separation [{l_label}]")
    _style_axes(ax)
    _legend(ax, min_entries=1)
    return ax


# ----------------------------------------------------------------------------- layout
def figure_grid(
    n: int,
    ncols: int | None = None,
    projection: str | None = None,
    figsize: tuple[float, float] | None = None,
    panel: tuple[float, float] = (4.5, 3.6),
) -> tuple[Figure, np.ndarray]:
    """Figure with ``n`` panels in a near-square grid.

    Parameters
    ----------
    n : int
        Number of panels (>= 1).
    ncols : int, optional
        Columns; default ``n`` for ``n <= 3``, else ``ceil(sqrt(n))``.
    projection : str, optional
        Axes projection for every panel, e.g. ``"3d"``.
    figsize : (float, float), optional
        Figure size in inches; default ``panel`` scaled by the grid shape.
    panel : (float, float), default (4.5, 3.6)
        Size of one panel in inches when ``figsize`` is not given.

    Returns
    -------
    (fig, axes) : Figure and 1-D object array of exactly ``n`` Axes, row-major.
    """
    n = int(n)
    if n < 1:
        raise ValueError("figure_grid needs n >= 1")
    if ncols is None:
        ncols = n if n <= 3 else math.ceil(math.sqrt(n))
    ncols = max(1, int(ncols))
    nrows = math.ceil(n / ncols)
    if figsize is None:
        figsize = (panel[0] * ncols, panel[1] * nrows)
    fig = _pyplot().figure(figsize=figsize, layout="constrained")
    axes = np.empty(n, dtype=object)
    for i in range(n):
        axes[i] = fig.add_subplot(nrows, ncols, i + 1, projection=projection)
    return fig, axes
