"""Tests for the lesson notebooks in ``notebooks/``.

Fast checks (structure): the eight lessons of Phases 1-6 exist, are valid nbformat 4, start
with a markdown title, state the physics before the first code cell, contain no stored
outputs, use no IPython magics, and every code cell parses as Python.

Slow check (execution, ``@pytest.mark.slow``): each notebook is executed from scratch with
``nbclient`` in the ``notebooks/`` directory (so the cached ephemerides are the only data
source; no network) and must run without error.  The whole set takes about two minutes
(lessons 06 and 07 integrate the nine-body EIH equations for ten years).
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
LESSONS = (
    "01_integrators.ipynb",
    "02_clockwork_solar_system.ipynb",
    "03_measuring_precession.ipynb",
    "04_mercury_43_arcsec.ipynb",
    "05_black_hole_orbits.ipynb",
    "06_le_verrier_vulcan_einstein.ipynb",
    "07_ppn_playground.ipynb",
    "08_eddington_1919.ipynb",
)


def _read(name: str) -> nbformat.NotebookNode:
    nb = nbformat.read(NB_DIR / name, as_version=4)
    nbformat.validate(nb)
    return nb


def test_all_lessons_present():
    assert NB_DIR.is_dir()
    present = sorted(p.name for p in NB_DIR.glob("*.ipynb"))
    assert present == sorted(LESSONS)


@pytest.mark.parametrize("name", LESSONS)
def test_notebook_structure(name):
    nb = _read(name)
    number = name[:2]
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[0].source.startswith(f"# Lesson {number}")
    kinds = [c.cell_type for c in nb.cells]
    first_code = kinds.index("code")
    assert first_code >= 2, "the physics must be stated before the first code cell"
    assert "## 1. The physics" in nb.cells[first_code - 1].source
    assert any("## Exercises" in c.source for c in nb.cells if c.cell_type == "markdown")
    markdown = [c.source for c in nb.cells if c.cell_type == "markdown"]
    assert any("Summary" in src for src in markdown)


@pytest.mark.parametrize("name", LESSONS)
def test_code_cells_are_clean_python(name):
    nb = _read(name)
    n_code = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        n_code += 1
        assert cell.outputs == [], "notebooks are stored without outputs"
        assert cell.execution_count is None
        for line in cell.source.splitlines():
            assert not line.lstrip().startswith(("%", "!")), f"magic in {name}: {line}"
        ast.parse(cell.source)  # raises SyntaxError on a broken cell
    assert n_code >= 5


@pytest.mark.parametrize("name", LESSONS)
def test_notebooks_use_the_package_not_the_network(name):
    nb = _read(name)
    source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "from gravsim" in source
    for forbidden in ("astroquery", "fetch_state", "fetch_series", "urllib", "requests"):
        assert forbidden not in source, f"{name} must not touch the network ({forbidden})"


@pytest.mark.slow
@pytest.mark.parametrize("name", LESSONS)
def test_notebook_executes(name):
    from nbclient import NotebookClient

    nb = _read(name)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NB_DIR)}},
    )
    client.execute()
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert errors == []
