"""Browser tests of the Three.js front end (``web/``): FastAPI server and static export.

Marker ``web`` (deselected by default; run with ``-m web``).  Skipped when playwright or its
chromium build is missing.  Two deployments are exercised, both through ``window.__gravsim``
(the hook ``web/app.js`` exposes for these tests):

* ``python -m gravsim.web --port <free>`` started as a subprocess (scenarios are built on
  demand, the default ``solar_system`` takes ~30 s): the page must become ``ready`` with zero
  console errors, every id of ``GET /api/scenarios`` must load through ``loadScenario``,
  render additional frames, accept ``setTime`` / ``setAmplification`` and leave a screenshot
  in ``figures/web_<id>.png``.
* ``scripts/export_web.py`` into a temporary directory (small parameters to keep the test
  fast), served with ``http.server`` next to copies of ``index.html``/``app.js``/``style.css``:
  the page must find the scenarios in ``data/manifest.json`` and load every ``.bin``.

Only localhost is contacted, except that the page itself pulls three.js from cdn.jsdelivr.net
(the pinned import map in ``web/index.html``); nothing else leaves the machine.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = pytest.mark.web

playwright_sync = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
FIGURES_DIR = ROOT / "figures"
EXPORT_SCRIPT = ROOT / "scripts" / "export_web.py"
VIEWPORT = {"width": 1400, "height": 900}
NBODY_IDS = {"solar_system", "earth_moon"}
"""Scenarios with a GR - Newton amplification slider (``setAmplification`` applies).

The rosette is an N-body view too, but at c/100 the GR - Newton difference is already
millions of km, so the page hides the slider there and ``setAmplification`` is a no-op."""
READY_TIMEOUT_MS = 240_000
"""The first scenario (solar_system, 100 yr EIH + Newton) is built on demand by the server."""
SMALL_EXPORT = [
    "solar_system.years=2",
    "rosette.n_orbits=5",
    "black_hole.n_orbits=2",
    "light_rays.n_rays=5",
    "earth_moon.years=0.5",
]

READY_JS = "() => Boolean(window.__gravsim && window.__gravsim.ready === true)"
FRAMES_JS = "() => window.__gravsim.stats.frames"
LAST_ERROR_JS = "() => (window.__gravsim.stats.lastError ?? null)"


# --------------------------------------------------------------------------- helpers
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as res:  # noqa: S310 (localhost only)
            return res.status, res.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def _wait_for_http(url: str, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Poll ``url`` until it answers 200 or ``proc`` dies."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"server exited with {proc.returncode} before answering:\n{out}")
        try:
            status, _ = _http_get(url, timeout=2.0)
            if status == 200:
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.2)
    raise TimeoutError(f"{url} did not answer within {timeout} s")


def _stop(proc: subprocess.Popen) -> str:
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out or ""


class ConsoleLog:
    """Collects console messages of type ``error`` and uncaught page errors."""

    def __init__(self, page) -> None:
        self.errors: list[str] = []
        self.page_errors: list[str] = []
        self.all: list[str] = []
        page.on("console", self._on_console)
        page.on("pageerror", lambda err: self.page_errors.append(str(err)))

    def _on_console(self, msg) -> None:
        self.all.append(f"[{msg.type}] {msg.text}")
        if msg.type == "error":
            self.errors.append(msg.text)

    def assert_clean(self) -> None:
        assert self.page_errors == [], "uncaught page errors:\n" + "\n".join(self.page_errors)
        assert self.errors == [], "console errors:\n" + "\n".join(self.errors)


def _wait_ready(page, timeout_ms: int = READY_TIMEOUT_MS) -> None:
    try:
        page.wait_for_function(READY_JS, timeout=timeout_ms)
    except playwright_sync.TimeoutError:
        status = page.evaluate(
            "() => (document.getElementById('status') || {}).innerText || ''"
        )
        last = page.evaluate("() => (window.__gravsim && window.__gravsim.stats.lastError) || null")
        pytest.fail(
            f"__gravsim.ready never became true; status overlay: {status!r}; lastError: {last!r}"
        )


def _load_and_exercise(page, scenario_id: str, figure: Path | None = None) -> int:
    """``loadScenario`` + frame-count, ``setTime``/``setAmplification`` and a screenshot.

    Returns the number of frames rendered during the exercise.
    """
    frames0 = page.evaluate(FRAMES_JS)
    page.evaluate("(id) => window.__gravsim.loadScenario(id)", scenario_id)
    page.wait_for_function("(f) => window.__gravsim.stats.frames > f", arg=frames0, timeout=20_000)
    assert page.evaluate("() => window.__gravsim.scenario") == scenario_id
    assert page.evaluate(LAST_ERROR_JS) is None, f"{scenario_id}: lastError set"
    assert page.evaluate(READY_JS)

    page.evaluate("() => window.__gravsim.setTime(0.5)")
    # applies to the N-body scenarios; must be a harmless no-op for the others
    page.evaluate("() => window.__gravsim.setAmplification(1000)")
    if scenario_id in NBODY_IDS:
        amp = page.evaluate("() => (document.getElementById('amp-out') || {}).textContent || ''")
        digits = amp.replace(",", "").replace("\u2009", "").replace(" ", "")
        match = re.search(r"[-+]?\d[\d.]*(?:e[-+]?\d+)?", digits, re.IGNORECASE)
        assert match and abs(float(match.group()) - 1000) < 0.5, f"{scenario_id}: k readout {amp!r}"
    if scenario_id == "solar_system":
        # the headline view: both models, k = 1000, and Mercury's perihelion trail, so the GR
        # perihelia visibly run ahead of the Newtonian ones in the saved figure
        page.check('input[name="model"][value="both"]')
        page.check("#peri")
    elif scenario_id == "rosette":
        assert page.is_checked("#peri"), "rosette: perihelion markers should start on"
        assert page.is_hidden("#amp-field"), "rosette: amplification slider should be hidden"
    elif scenario_id in {"light_rays", "earth_moon"}:
        assert page.is_hidden("#peri-wrap"), f"{scenario_id}: no perihelion data, no toggle"
    frames1 = page.evaluate(FRAMES_JS)
    page.wait_for_function(
        "(f) => window.__gravsim.stats.frames > f + 2", arg=frames1, timeout=20_000
    )
    assert page.evaluate(LAST_ERROR_JS) is None, f"{scenario_id}: render error after setTime"
    if figure is not None:
        figure.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(figure), full_page=False)
        assert figure.stat().st_size > 10_000, f"{figure} is suspiciously small"
    return page.evaluate(FRAMES_JS) - frames0


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def browser() -> Iterator:
    with playwright_sync.sync_playwright() as pw:
        exe = Path(pw.chromium.executable_path)
        if not exe.exists():
            pytest.skip(f"chromium is not installed ({exe}); run 'playwright install chromium'")
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception as exc:  # noqa: BLE001 - any launch failure means "no browser here"
            pytest.skip(f"chromium could not be launched: {exc}")
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
    page = context.new_page()
    try:
        yield page
    finally:
        context.close()


@pytest.fixture(scope="module")
def api_server() -> Iterator[str]:
    """``python -m gravsim.web --port <free>`` as a subprocess; yields the base URL."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "gravsim.web", "--port", str(port)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http(f"{base}/api/health", proc, timeout=90.0)
        yield base
    finally:
        _stop(proc)


@pytest.fixture(scope="module")
def static_site(tmp_path_factory) -> Iterator[tuple[str, Path]]:
    """Static export (small parameters) + ``http.server``; yields ``(base_url, site_dir)``."""
    site = tmp_path_factory.mktemp("web_static")
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy(WEB_DIR / name, site / name)
    args = [sys.executable, str(EXPORT_SCRIPT), "--out", str(site / "data")]
    for item in SMALL_EXPORT:
        args += ["--set", item]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((site / "data" / "manifest.json").read_text())
    assert isinstance(manifest, list) and manifest
    for entry in manifest:
        assert entry["url"] == f"data/{entry['id']}.bin"
        assert (site / entry["url"]).is_file()

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "--bind", "127.0.0.1", "-d", str(site), str(port)],
        cwd=site,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http(f"{base}/data/manifest.json", proc, timeout=30.0)
        yield base, site
    finally:
        _stop(proc)


# --------------------------------------------------------------------------- server deployment
def test_api_server_every_scenario(api_server: str, page) -> None:
    status, body = _http_get(f"{api_server}/api/scenarios")
    assert status == 200
    entries = json.loads(body)
    ids = [e["id"] for e in entries]
    assert ids, "no scenarios listed"
    for entry in entries:
        assert entry["url"] == f"/api/scenario/{entry['id']}"

    log = ConsoleLog(page)
    responses: list[tuple[str, int]] = []
    page.on("response", lambda res: responses.append((res.url, res.status)))
    page.goto(f"{api_server}/", wait_until="load")
    _wait_ready(page)
    log.assert_clean()
    assert page.evaluate("() => typeof window.__gravsim.loadScenario") == "function"
    assert page.evaluate("() => typeof window.__gravsim.setTime") == "function"
    assert page.evaluate("() => typeof window.__gravsim.setAmplification") == "function"
    assert page.evaluate(FRAMES_JS) > 0
    assert page.evaluate("() => window.__gravsim.scenario") in ids

    for sid in ids:
        rendered = _load_and_exercise(page, sid, FIGURES_DIR / f"web_{sid}.png")
        assert rendered > 0, f"{sid}: no frame rendered after loadScenario"
        log.assert_clean()

    # every payload came from the API, with the documented media type
    payload_responses = [(u, s) for u, s in responses if "/api/scenario/" in u]
    assert {s for _, s in payload_responses} == {200}, payload_responses
    assert all(any(f"/api/scenario/{sid}" in u for u, _ in payload_responses) for sid in ids)
    assert not [u for u, s in responses if s >= 400], "failed requests: " + str(
        [(u, s) for u, s in responses if s >= 400]
    )


def test_api_server_unknown_scenario_is_reported(api_server: str, page) -> None:
    log = ConsoleLog(page)
    page.goto(f"{api_server}/", wait_until="load")
    _wait_ready(page)
    rejected = page.evaluate(
        "() => window.__gravsim.loadScenario('no_such_scenario').then(() => false, () => true)"
    )
    assert rejected is True
    # the viewer keeps rendering the previous scenario
    frames = page.evaluate(FRAMES_JS)
    page.wait_for_function("(f) => window.__gravsim.stats.frames > f", arg=frames, timeout=10_000)
    log.assert_clean()


# --------------------------------------------------------------------------- static deployment
def test_static_export_loads_from_manifest(static_site: tuple[str, Path], page) -> None:
    base, site = static_site
    manifest = json.loads((site / "data" / "manifest.json").read_text())
    ids = [e["id"] for e in manifest]

    log = ConsoleLog(page)
    requests: list[str] = []
    responses: list[tuple[str, int]] = []
    page.on("request", lambda req: requests.append(req.url))
    page.on("response", lambda res: responses.append((res.url, res.status)))
    page.goto(f"{base}/", wait_until="load")
    _wait_ready(page, timeout_ms=60_000)
    log.assert_clean()

    assert any(u.endswith("/data/manifest.json") for u in requests), requests
    assert not any("/api/" in u for u in requests), "static page must not need the API: " + str(
        [u for u in requests if "/api/" in u]
    )
    for sid in ids:
        rendered = _load_and_exercise(page, sid, None)
        assert rendered > 0, f"{sid}: no frame rendered after loadScenario"
        log.assert_clean()
        assert any(u.endswith(f"/data/{sid}.bin") for u in requests), f"{sid}: .bin never fetched"
    assert not [u for u, s in responses if s >= 400], [(u, s) for u, s in responses if s >= 400]


def test_no_scenario_list_shows_an_error(static_site: tuple[str, Path], browser, tmp_path) -> None:
    """Without manifest and API the page must report the problem instead of hanging."""
    _, site = static_site
    bare = tmp_path / "bare"
    bare.mkdir()
    for name in ("index.html", "app.js", "style.css"):
        shutil.copy(site / name, bare / name)
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "--bind", "127.0.0.1", "-d", str(bare), str(port)],
        cwd=bare,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    context = browser.new_context(viewport=VIEWPORT)
    page = context.new_page()
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/index.html", proc, timeout=30.0)
        log = ConsoleLog(page)
        page.goto(f"http://127.0.0.1:{port}/", wait_until="load")
        page.wait_for_function(
            "() => window.__gravsim && window.__gravsim.stats.lastError !== null", timeout=30_000
        )
        assert page.evaluate(READY_JS) is False
        assert "scenario" in page.evaluate(LAST_ERROR_JS).lower()
        assert log.page_errors == [], log.page_errors
    finally:
        context.close()
        _stop(proc)
