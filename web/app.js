/* gravsim web front end — Three.js viewer for the scenarios served by `gravsim.web`.
 *
 * Physics stays in Python.  This module only (1) loads the scenario list
 * (`data/manifest.json`, falling back to `/api/scenarios`), (2) decodes the binary payload
 * written by `gravsim.web.payload.encode` -- a 4-byte little-endian uint32 header length,
 * a UTF-8 JSON header `{meta, index: [{name, dtype, shape, offset, nbytes}]}` and the raw
 * little-endian arrays -- into typed-array views, and (3) renders.
 *
 * Conventions mirrored from the package: positions in AU (or GM/c^2 for the strong-field
 * scenarios), time in days since J2000.0 (2000-01-01T12:00Z), barycentric J2000-ecliptic
 * frame with +z = ecliptic north.  Displayed N-body positions are
 *     p = newton + k * delta,   delta = gr - newton
 * with the amplification k (delta from the payload's float64-accurate `delta_gr` when it is
 * present, else the difference of the float32 positions), then shifted into the chosen
 * frame (barycentric, heliocentric = minus the Sun, or Earth-centred = minus the Earth).
 *
 * The perihelion trail marks the perihelia of the *displayed* path, not the samples listed
 * in `peri_idx_gr`: the GR run's passages lag the Newtonian ones, so at a GR perihelion
 * sample delta is mostly along-track and k * delta would drag a marker behind Newton's
 * perihelion (a retrograde arc once k != 1).  Each marker is therefore the local minimum of
 * the displayed Sun-Mercury distance nearest to the run's perihelion sample, with the
 * passage refined between samples by the parabola through the three distances around the
 * minimum (analysis.precession.perihelion_passages) and the position interpolated at that
 * time.  Body colours come from gravsim/viz/plots.py (BODY_COLORS) so the viewer matches
 * the lesson figures.
 *
 * `window.__gravsim` is the hook used by tests/test_web.py.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';

// ----------------------------------------------------------------------------- constants
const AU_KM = 149597870.7;
const JULIAN_YEAR_DAYS = 365.25;
const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);
const THREE_VERSION = '0.160.0';

/** gravsim.viz.plots.PALETTE: colour-blind-safe categorical palette, Mercury ... Neptune. */
const PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
const BODY_COLORS = {
  Mercury: PALETTE[0],
  Venus: PALETTE[1],
  EMB: PALETTE[2],
  Earth: PALETTE[2],
  Mars: PALETTE[3],
  Jupiter: PALETTE[4],
  Saturn: PALETTE[5],
  Uranus: PALETTE[6],
  Neptune: PALETTE[7],
  Sun: '#ffd27a',
  Moon: '#c9c6bd',
};
/** Non-physical but ordered marker sizes (relative). */
const BODY_SIZE = {
  Sun: 2.2, Mercury: 0.55, Venus: 0.8, EMB: 0.85, Earth: 0.85, Moon: 0.35, Mars: 0.6,
  Jupiter: 1.4, Saturn: 1.25, Uranus: 1.0, Neptune: 0.95,
};
const INNER_BODIES = new Set(['Sun', 'Mercury', 'Venus', 'EMB', 'Earth', 'Moon', 'Mars']);
const GREY = '#898781';
const GRID_COLOR = 0x2a2e38;
const PERI_NEWTON_COLOR = '#b9b7af';
const RING_COLORS = { horizon: '#e34948', photon: '#eb6834', isco: '#1baf7a' };
const RAY_RAMP = ['#2a78d6', '#1baf7a', '#eda100', '#eb6834', '#e34948'];

const EXPLANATIONS = {
  solar_system:
    'The Sun and eight planets are integrated for a century from the same J2000 state, once with ' +
    'Newtonian gravity and once with the Einstein–Infeld–Hoffmann 1PN equations used by the JPL ' +
    'ephemerides. At true scale the two runs are indistinguishable (Mercury drifts by a few thousand ' +
    'kilometres in a century), so the amplification slider scales the GR − Newton displacement until ' +
    'the extra 43″ per century of perihelion advance shows up as an arc of perihelion markers. The ' +
    'trail marks the perihelia of the displayed orbit, which run ahead of the Newtonian ones (grey ' +
    'markers in "both") by k × 43″ per century.',
  rosette:
    'Sun + Mercury with the speed of light reduced to c/100, which multiplies the 1PN precession by ' +
    '10⁴ (about 0.3° per orbit) while keeping v/c small. Over a few hundred orbits the perihelion ' +
    'sweeps around the Sun and the orbit traces a rosette — the same physics as the real 43″ per ' +
    'century, made visible.',
  black_hole:
    'An exact timelike geodesic of the Schwarzschild metric in units of GM/c² (horizon at r = 2, photon ' +
    'sphere at r = 3, innermost stable circular orbit at r = 6). Close to the photon sphere the orbit ' +
    'whirls around the hole several times near periapsis before zooming back out — the strong-field ' +
    'limit of Mercury\'s small precession.',
  light_rays:
    'Null geodesics (light rays) passing a compact mass at different impact parameters b, integrated ' +
    'exactly in the Schwarzschild metric. In the weak field the bending angle is 4GM/(c² b), twice the ' +
    'Newtonian corpuscle value; rays inside the critical impact parameter 3√3 GM/c² spiral into the ' +
    'photon sphere and are captured.',
  earth_moon:
    'The Sun, Earth and Moon (with the other planets) integrated for two years with Earth and Moon as ' +
    'separate bodies. Pick the "earth" camera to follow the Earth and watch the Moon\'s orbit around ' +
    'it; the lunar perigee advances about 40° per year, driven by the Sun\'s tidal pull rather than by ' +
    'general relativity.',
};
const GENERIC_EXPLANATION =
  'Trajectories precomputed in Python by gravsim and rendered here. ' +
  'The numbers below are taken from the scenario metadata.';

const META_RESERVED = new Set([
  'id', 'title', 'blurb', 'names', 'units', 'frame', 't0', 'n_samples', 'arrays', 'params', 'url',
]);

// ----------------------------------------------------------------------------- small helpers
const $ = (id) => document.getElementById(id);

// Info panel: these top-level meta keys lead, in this order; every other key folds into
// "Run details" (solver settings, GM values, grids, raw lists that the legend already shows).
const HEADLINE_KEYS = [
  'mercury', 'perihelion_rates', 'advance', 'moon', 'sun_wobble', 'orbit', 'c_scale',
  'advance_per_orbit_deg', 'advance_per_orbit_rad', 'turns_per_orbit', 'advance_1pn_rad',
  'E', 'L', 'r_peri', 'r_apo', 'r_horizon', 'r_photon_sphere', 'r_isco',
  'r_circular_stable', 'r_circular_unstable', 'b_crit', 'limb_deflection_arcsec',
];
const KEY_UNITS = [
  ['_arcsec_cy', '″/cy'], ['_rad_per_orbit', 'rad/orbit'], ['_deg_per_orbit', '°/orbit'],
  ['_deg_yr', '°/yr'], ['_au_day', 'AU/d'], ['_arcsec', '″'], ['_km', 'km'], ['_days', 'd'],
  ['_r_sun', 'R☉'], ['_au', 'AU'], ['_rad', 'rad'], ['_deg', '°'],
];
const KEY_WORDS = { gr: 'GR', '1pn': '1PN', pn1: '1PN', eih: 'EIH', err: '±', isco: 'ISCO', emb: 'EMB' };

/** 'gr_minus_newton_arcsec_cy' -> 'GR − Newton (″/cy)'. */
function humanKey(key) {
  let unit = '';
  let k = String(key);
  for (const [suffix, u] of KEY_UNITS) {
    if (k.endsWith(suffix) && k.length > suffix.length) {
      unit = u;
      k = k.slice(0, -suffix.length);
      break;
    }
  }
  const text = k.split('_').map((w) => KEY_WORDS[w] ?? w).join(' ')
    .replace(/\bminus newton\b/g, '− Newton').replace(/\bminus\b/g, '−')
    .replace(/\bv over c\b/g, 'v/c').replace(/\bover\b/g, '/');
  return unit ? `${text} (${unit})` : text;
}

/** Amplification factor for display: 2.50, 31.6, 1 000, 100 000. */
function fmtAmp(k) {
  if (k < 10) return k.toFixed(2);
  if (k < 100) return k.toFixed(1);
  return Math.round(k).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '\u2009');
}

function fmtNum(v) {
  if (!Number.isFinite(v)) return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(4);
  return String(Number(v.toPrecision(6)));
}

function fmtKm(km) {
  const a = Math.abs(km);
  if (a >= 1e6) return `${(km / 1e6).toFixed(3)} million km`;
  if (a >= 100) return `${km.toFixed(0)} km`;
  if (a >= 1) return `${km.toFixed(2)} km`;
  return `${(km * 1000).toFixed(1)} m`;
}

function fmtDays(d) {
  if (d >= 2 * JULIAN_YEAR_DAYS) return `${(d / JULIAN_YEAR_DAYS).toFixed(1)} yr`;
  if (d >= 30) return `${d.toFixed(0)} d`;
  if (d >= 1) return `${d.toFixed(1)} d`;
  return `${(d * 24).toFixed(1)} h`;
}

/** 'year +12.3 (2012-04-13)' for t in days since J2000.0. */
function fmtEpoch(days) {
  const yr = days / JULIAN_YEAR_DAYS;
  const date = new Date(J2000_MS + days * 86400e3);
  const iso = Number.isFinite(date.getTime()) ? date.toISOString().slice(0, 10) : '—';
  return `year ${yr >= 0 ? '+' : '−'}${Math.abs(yr).toFixed(2)} (${iso})`;
}

function isPlainObject(v) {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function isNumberList(v, max = 16) {
  return Array.isArray(v) && v.length > 0 && v.length <= max && v.every((x) => typeof x === 'number');
}

function unitsText(meta) {
  const u = meta.units;
  if (u == null) return '';
  if (typeof u === 'string') return u;
  if (isPlainObject(u)) return Object.entries(u).map(([k, v]) => `${k}: ${v}`).join(' · ');
  return JSON.stringify(u);
}

function lengthUnit(meta) {
  const u = meta.units;
  if (isPlainObject(u) && typeof u.length === 'string') return u.length;
  if (typeof u === 'string') return u;
  return /GM\/c/.test(unitsText(meta)) ? 'GM/c²' : 'AU';
}

function isGeometricUnits(meta) {
  return /GM\s*\/\s*c/i.test(unitsText(meta)) || /GM\s*\/\s*c/i.test(String(lengthUnit(meta)));
}

/** First key of `meta` whose name matches `re` and whose value passes `test`. */
function pickMeta(meta, re, test = () => true) {
  for (const key of Object.keys(meta)) {
    if (re.test(key) && test(meta[key])) return meta[key];
  }
  return undefined;
}

function rampColor(u, ramp = RAY_RAMP) {
  const n = ramp.length - 1;
  const x = Math.min(Math.max(u, 0), 1) * n;
  const i = Math.min(Math.floor(x), n - 1);
  const c = new THREE.Color(ramp[i]);
  c.lerp(new THREE.Color(ramp[i + 1]), x - i);
  return c;
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

// ----------------------------------------------------------------------------- payload decoder
const DTYPES = {
  float32: [Float32Array, 4],
  float64: [Float64Array, 8],
  int32: [Int32Array, 4],
  int16: [Int16Array, 2],
  int8: [Int8Array, 1],
  uint8: [Uint8Array, 1],
  uint16: [Uint16Array, 2],
  uint32: [Uint32Array, 4],
  int64: [BigInt64Array, 8],
};

/**
 * Decode a gravsim payload: uint32 LE header length, JSON header {meta, index}, raw arrays.
 * Returns {meta, arrays: {name: TypedArray}, shapes: {name: number[]}, index}.
 * Offsets are taken relative to the start of the array block (the contract); absolute file
 * offsets are accepted too.  Views are used when 4-byte aligned, copies otherwise.
 */
export function decodePayload(buffer) {
  if (!(buffer instanceof ArrayBuffer)) throw new TypeError('decodePayload expects an ArrayBuffer');
  if (buffer.byteLength < 4) throw new Error(`payload too short (${buffer.byteLength} bytes)`);
  const headerLen = new DataView(buffer).getUint32(0, true);
  const dataStart = 4 + headerLen;
  if (dataStart > buffer.byteLength) {
    throw new Error(`header length ${headerLen} exceeds the payload (${buffer.byteLength} bytes)`);
  }
  let header;
  try {
    header = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(buffer, 4, headerLen)));
  } catch (err) {
    throw new Error(`payload header is not valid JSON: ${err.message}`);
  }
  const meta = isPlainObject(header.meta) ? header.meta : {};
  const index = Array.isArray(header.index) ? header.index : [];
  const fits = (base) => index.every((e) => base + e.offset >= dataStart
    && base + e.offset + e.nbytes <= buffer.byteLength);
  let base;
  if (fits(dataStart)) base = dataStart;
  else if (fits(0)) base = 0;
  else throw new Error('payload index points outside the array block');

  const arrays = {};
  const shapes = {};
  for (const entry of index) {
    const spec = DTYPES[entry.dtype];
    if (!spec) throw new Error(`array '${entry.name}': unsupported dtype '${entry.dtype}'`);
    const [Ctor, itemsize] = spec;
    const shape = Array.isArray(entry.shape) ? entry.shape.map(Number) : [];
    const count = shape.reduce((a, b) => a * b, 1);
    if (count * itemsize !== entry.nbytes) {
      throw new Error(`array '${entry.name}': shape ${JSON.stringify(shape)} x ${itemsize} B != ${entry.nbytes} B`);
    }
    const start = base + entry.offset;
    let typed;
    if (start % itemsize === 0) typed = new Ctor(buffer, start, count);
    else typed = new Ctor(buffer.slice(start, start + entry.nbytes));
    if (Ctor === BigInt64Array) typed = Float64Array.from(typed, (x) => Number(x));
    arrays[entry.name] = typed;
    shapes[entry.name] = shape;
  }
  return { meta, arrays, shapes, index };
}

// ----------------------------------------------------------------------------- loading
async function fetchJSON(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error(`${url}: not JSON (${err.message})`);
  }
}

/**
 * Which scenario list to try first.  Both deployments are probed, but a 404 on the first
 * probe is logged by the browser as a console error, so when the page is served by
 * uvicorn (the gravsim FastAPI server) the API is asked first; a static server gets the
 * exported manifest first.  The probe re-reads the stylesheet, which the page has already
 * loaded (normally straight from the browser cache), so it never 404s.
 */
async function manifestOrder() {
  const order = ['data/manifest.json', '/api/scenarios'];
  try {
    const res = await fetch(new URL('style.css', document.baseURI), { cache: 'force-cache' });
    const server = (res.headers.get('server') || '').toLowerCase();
    if (/uvicorn|starlette|fastapi|hypercorn|gunicorn/.test(server)) order.reverse();
  } catch (_) {
    // keep the default order
  }
  return order;
}

/** Scenario list: data/manifest.json (static export) or /api/scenarios (FastAPI). */
async function loadManifest() {
  const errors = [];
  for (const url of await manifestOrder()) {
    try {
      const json = await fetchJSON(url);
      // a bare list (the API) or an object wrapping it under "scenarios" (the static export)
      const list = Array.isArray(json) ? json
        : isPlainObject(json) ? ['scenarios', 'entries', 'items'].map((k) => json[k]).find(Array.isArray) : undefined;
      if (!Array.isArray(list) || list.length === 0) throw new Error(`${url}: empty scenario list`);
      for (const e of list) {
        if (!e || typeof e.id !== 'string' || typeof e.url !== 'string') {
          throw new Error(`${url}: entries need 'id' and 'url'`);
        }
      }
      return { list, source: url };
    } catch (err) {
      errors.push(err.message);
    }
  }
  throw new Error(`no scenario list found.\n${errors.join('\n')}`);
}

async function fetchPayload(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return decodePayload(await res.arrayBuffer());
}

// ----------------------------------------------------------------------------- scene helpers
/**
 * Text label. CSS2DRenderer rewrites `style.transform` every frame, so offsets cannot live in
 * the stylesheet: `anchor` picks which corner of the label sits on the anchor point through
 * CSS2DObject.center ('ne' = label above-right of the point, 'se' below-right, 'nw', 'sw',
 * 'c' centred); the small pixel gap comes from the matching `.label.<anchor>` margin.
 */
const ANCHORS = { c: [0.5, 0.5], ne: [0, 1], se: [0, 0], nw: [1, 1], sw: [1, 0] };
function makeLabel(text, cls = '', anchor = 'ne') {
  const el = document.createElement('div');
  el.className = `label ${cls} ${anchor}`.replace(/\s+/g, ' ').trim();
  el.textContent = text;
  const obj = new CSS2DObject(el);
  const [cx, cy] = ANCHORS[anchor] ?? ANCHORS.c;
  obj.center.set(cx, cy);
  return obj;
}

/** Point on a circle of radius r at angle `deg` in the xy plane, for placing labels. */
function onCircle(r, deg) {
  const a = (deg * Math.PI) / 180;
  return [r * Math.cos(a), r * Math.sin(a), 0];
}

/**
 * CSS2DRenderer (three r160) decides a label's visibility from the label alone, ignoring its
 * ancestors, so hiding the grid group would leave the grid labels on screen. Each label keeps
 * the wanted state in `userData.show`; this pass combines it with the ancestors' visibility
 * right before the label render.
 */
function syncLabelVisibility(root) {
  root.traverse((o) => {
    if (!o.isCSS2DObject) return;
    if (o.userData.show === undefined) o.userData.show = true;
    let vis = o.userData.show;
    for (let p = o.parent; vis && p; p = p.parent) if (!p.visible) vis = false;
    o.visible = vis;
  });
}

function circleGeometry(radius, segments = 180) {
  const pts = new Float32Array((segments + 1) * 3);
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    pts[3 * i] = radius * Math.cos(a);
    pts[3 * i + 1] = radius * Math.sin(a);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  return g;
}

function ring(radius, color, opacity = 0.9) {
  const line = new THREE.LineLoop(circleGeometry(radius), new THREE.LineBasicMaterial({
    color, transparent: opacity < 1, opacity,
  }));
  line.frustumCulled = false;
  return line;
}

/** 1-2-5 series of radii up to `extent` (at least two rings). */
function niceRadii(extent) {
  const out = [];
  let e = Math.pow(10, Math.floor(Math.log10(Math.max(extent, 1e-9))) - 1);
  for (let guard = 0; guard < 60 && out.length < 40; guard++) {
    for (const m of [1, 2, 5]) {
      const r = m * e;
      if (r <= extent * 1.15) out.push(r);
    }
    e *= 10;
    if (e > extent * 2) break;
  }
  while (out.length > 7) out.splice(0, 1);
  return out;
}

/** Ecliptic (xy) grid: concentric rings at 1-2-5 radii, 12 spokes, ring labels. */
function makeEclipticGrid(extent, unit) {
  const group = new THREE.Group();
  group.name = 'grid';
  const radii = niceRadii(extent);
  const rmax = radii[radii.length - 1];
  const segs = [];
  for (let s = 0; s < 12; s++) {
    const a = (s / 12) * Math.PI * 2;
    segs.push(0, 0, 0, rmax * Math.cos(a), rmax * Math.sin(a), 0);
  }
  const spokes = new THREE.BufferGeometry();
  spokes.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs), 3));
  const mat = new THREE.LineBasicMaterial({ color: GRID_COLOR, transparent: true, opacity: 0.8 });
  const spokeLines = new THREE.LineSegments(spokes, mat);
  spokeLines.frustumCulled = false;
  group.add(spokeLines);
  for (const r of radii) {
    const ln = new THREE.LineLoop(circleGeometry(r, 128), mat);
    ln.frustumCulled = false;
    group.add(ln);
    const lab = makeLabel(`${fmtNum(r)} ${unit}`, 'grid', 'se');
    lab.position.set(...onCircle(r, -45));
    lab.userData.radius = r;
    group.add(lab);
  }
  return group;
}

function disposeObject(obj) {
  obj.traverse((o) => {
    if (o.isCSS2DObject && o.element && o.element.parentNode) o.element.parentNode.removeChild(o.element);
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) m.dispose();
    }
  });
  if (obj.parent) obj.parent.remove(obj);
}

/** Line over an interleaved (T, N, 3) buffer for body j. */
function bodyLine(ib, j, color, opacity) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, j * 3, false));
  const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity, depthWrite: false,
  }));
  line.frustumCulled = false;
  return line;
}

// ----------------------------------------------------------------------------- views
/** Shared state of the three scene kinds; a view owns one THREE.Group and knows its span. */
class BaseView {
  constructor(app, payload, entry) {
    this.app = app;
    this.meta = payload.meta;
    this.arrays = payload.arrays;
    this.shapes = payload.shapes;
    this.entry = entry;
    this.group = new THREE.Group();
    this.sphereGeo = new THREE.SphereGeometry(1, 28, 18);
    this.labels = [];
    this.grid = null;
    this.extent = 1;
    this.unit = lengthUnit(this.meta);
  }

  /** Playback span in "days" (or samples) used for the speed readout. */
  get span() { return 1; }

  setGridVisible(on) { if (this.grid) this.grid.visible = on; }

  setLabelsVisible(on) {
    this.showLabels = on;
    for (const l of this.labels) l.userData.show = on;
  }

  /**
   * Ring labels of the grid: drop those whose ring is smaller on screen than `minPx` (they
   * would pile up on the central body) or inside `gridLabelMinR` (named rings live there).
   */
  updateGridLabels(pxPerUnit) {
    if (!this.grid) return;
    const minPx = this.gridLabelMinPx(pxPerUnit);
    const minR = this.gridLabelMinR ?? 0;
    for (const o of this.grid.children) {
      if (!o.isCSS2DObject) continue;
      const r = o.userData.radius ?? 0;
      o.userData.show = r * pxPerUnit >= minPx && r > minR;
    }
  }

  gridLabelMinPx() { return 40; }

  /** Marker radius that keeps a roughly constant angular size with the camera distance. */
  markerScale(size) {
    const d = this.app.cameraDistance();
    return size * 0.011 * Math.pow(d, 0.9);
  }

  update() {}

  timeReadout() { return ''; }

  dispose() {
    disposeObject(this.group);
    this.sphereGeo.dispose();
  }
}

/** solar_system, rosette, earth_moon: pos_newton / pos_gr (T, N, 3) + peri_idx_*. */
class NBodyView extends BaseView {
  constructor(app, payload, entry) {
    super(app, payload, entry);
    const A = this.arrays;
    const S = this.shapes;
    if (!A.pos_newton || !S.pos_newton || S.pos_newton.length !== 3) {
      throw new Error("N-body scenario needs 'pos_newton' with shape (T, N, 3)");
    }
    [this.T, this.N] = S.pos_newton;
    this.newton = A.pos_newton;
    const sameShape = (shape) => Array.isArray(shape) && shape.length === 3
      && shape[0] === this.T && shape[1] === this.N;
    this.gr = A.pos_gr && sameShape(S.pos_gr) ? A.pos_gr : null;
    // gr - newton computed in float64 before the float32 cast (scenarios._nbody_arrays); the
    // float32 positions alone resolve only ~5 km at 0.4 AU
    this.delta = this.gr && A.delta_gr && sameShape(S.delta_gr) ? A.delta_gr : null;
    this.t = A.t && A.t.length === this.T ? A.t : Float32Array.from({ length: this.T }, (_, i) => i);
    const names = Array.isArray(this.meta.names) ? this.meta.names.map(String) : [];
    this.names = Array.from({ length: this.N }, (_, i) => names[i] ?? `body ${i}`);
    this.sunIdx = Math.max(this.names.indexOf('Sun'), 0);
    this.earthIdx = this.names.indexOf('Earth');
    if (this.earthIdx < 0) this.earthIdx = this.names.indexOf('EMB');
    this.mercIdx = this.names.indexOf('Mercury');
    if (this.mercIdx < 0) this.mercIdx = this.names.findIndex((_, i) => i !== this.sunIdx);
    this.periGR = A.peri_idx_gr || null;
    this.periN = A.peri_idx_newton || null;

    // epoch: days since J2000 of sample i = base + (t_i - t_0)
    const t0 = typeof this.meta.t0 === 'number' ? this.meta.t0 : this.t[0];
    this.epochBase = t0 > 1e6 ? t0 - 2451545.0 : t0;
    this.spanDays = Math.max(this.t[this.T - 1] - this.t[0], 1e-9);
    this.dtDays = this.spanDays / Math.max(this.T - 1, 1);

    // view state
    this.frame = 'bary';
    this.model = this.gr ? 'gr' : 'newton';
    this.k = 1;
    this.trailSamples = this.T;
    this.showPaths = true;
    this.showPeri = false;

    this.measureExtent();
    this.buildObjects();
    this.recompute();
  }

  get span() { return this.spanDays; }

  measureExtent() {
    const n = this.newton;
    let inner = 0;
    let outer = 0;
    const stride = Math.max(1, Math.floor(this.T / 400));
    for (let i = 0; i < this.T; i += stride) {
      for (let j = 0; j < this.N; j++) {
        const o = (i * this.N + j) * 3;
        const r = Math.hypot(n[o], n[o + 1], n[o + 2]);
        if (r > outer) outer = r;
        if (INNER_BODIES.has(this.names[j]) && r > inner) inner = r;
      }
    }
    this.outerR = outer > 0 ? outer : 1;
    this.innerR = inner > 0 ? inner : this.outerR;
    this.extent = this.outerR;
  }

  buildObjects() {
    const { T, N } = this;
    this.disp = new Float32Array(T * N * 3);
    this.dispN = new Float32Array(T * N * 3);
    this.ib = new THREE.InterleavedBuffer(this.disp, N * 3);
    this.ibN = new THREE.InterleavedBuffer(this.dispN, N * 3);
    this.ib.setUsage(THREE.DynamicDrawUsage);
    this.ibN.setUsage(THREE.DynamicDrawUsage);

    this.bodies = [];
    for (let j = 0; j < N; j++) {
      const name = this.names[j];
      const color = BODY_COLORS[name] ?? PALETTE[j % PALETTE.length];
      const size = BODY_SIZE[name] ?? 0.7;
      const isSun = name === 'Sun';
      const path = bodyLine(this.ib, j, color, 0.28);
      const trail = bodyLine(this.ib, j, color, 0.95);
      const ghost = bodyLine(this.ibN, j, color, 0.35);
      ghost.visible = false;
      const tint = new THREE.Color(color).lerp(new THREE.Color(0xffffff), 0.3);
      const material = isSun
        ? new THREE.MeshBasicMaterial({ color: BODY_COLORS.Sun })
        : new THREE.MeshLambertMaterial({ color: tint, emissive: tint.clone().multiplyScalar(0.35) });
      const mesh = new THREE.Mesh(this.sphereGeo, material);
      const ghostMesh = new THREE.Mesh(this.sphereGeo, new THREE.MeshBasicMaterial({
        color, wireframe: true, transparent: true, opacity: 0.45,
      }));
      ghostMesh.visible = false;
      // labels sit on the sphere's edge (mesh-local units scale with the drawn radius); the
      // Moon's goes below-right so it does not cover the Earth's above-right
      const below = name === 'Moon';
      const label = makeLabel(name === 'EMB' ? 'Earth (EMB)' : name, '', below ? 'se' : 'ne');
      label.position.set(...onCircle(1, below ? -45 : 45));
      mesh.add(label);
      this.labels.push(label);
      this.group.add(path, trail, ghost, mesh, ghostMesh);
      this.bodies.push({ name, j, color, size, isSun, path, trail, ghost, mesh, ghostMesh, label });
    }

    // the Sun lights the planets; no distance falloff so Neptune is lit too
    this.light = new THREE.PointLight(0xffffff, 2.2, 0, 0);
    this.group.add(this.light);
    this.ambient = new THREE.AmbientLight(0xffffff, 0.35);
    this.group.add(this.ambient);

    // perihelion markers (Mercury): the displayed (amplified) set in a light tint of Mercury's
    // colour so it stays readable over the band of amplified paths, the Newton set in grey
    const periMat = (color) => new THREE.PointsMaterial({
      color, size: 6, sizeAttenuation: false, depthWrite: false, transparent: true, opacity: 0.95,
    });
    const periColor = new THREE.Color(BODY_COLORS.Mercury).lerp(new THREE.Color(0xffffff), 0.55);
    this.periGRPts = this.makePeriPoints(this.periGR, periMat(periColor));
    this.periNPts = this.makePeriPoints(this.periN, periMat(PERI_NEWTON_COLOR));
    for (const p of [this.periGRPts, this.periNPts]) if (p) this.group.add(p);

    this.grid = makeEclipticGrid(this.outerR, this.unit);
    this.group.add(this.grid);
  }

  gridLabelMinPx(pxPerUnit) {
    const sun = this.markerScale(BODY_SIZE.Sun ?? 1) * pxPerUnit;
    return Math.max(40, 2.5 * sun);
  }

  /** Marker cloud for up to `seeds.length` perihelia; `fillPeri` places them. */
  makePeriPoints(seeds, material) {
    if (!seeds || seeds.length === 0 || this.mercIdx < 0 || this.mercIdx === this.sunIdx) return null;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seeds.length * 3), 3));
    geo.setDrawRange(0, 0);
    const pts = new THREE.Points(geo, material);
    pts.frustumCulled = false;
    pts.visible = false;
    pts.userData.seeds = seeds; // perihelion sample indices of the run (peri_idx_*)
    pts.userData.at = new Float64Array(seeds.length); // fractional sample index of each marker
    pts.userData.count = 0; // markers placed by the last fillPeri
    return pts;
  }

  /** Reference body index for the frame (-1 = barycentric). */
  refIndex() {
    if (this.frame === 'helio') return this.sunIdx;
    if (this.frame === 'earth' && this.earthIdx >= 0) return this.earthIdx;
    return -1;
  }

  /** Component `o` of newton + k * (gr - newton), from `delta_gr` when the payload has it. */
  amplified(o, k) {
    const n = this.newton[o];
    return n + k * (this.delta ? this.delta[o] : this.gr[o] - n);
  }

  /** Fill `out` with newton + k (gr - newton) in the current frame. */
  fillDisplay(out, k) {
    const { T, N, newton: n } = this;
    const ref = this.refIndex();
    const useGR = Boolean(this.gr) && k !== 0;
    const r = [0, 0, 0];
    for (let i = 0; i < T; i++) {
      const row = i * N * 3;
      if (ref >= 0) {
        const o = row + ref * 3;
        for (let c = 0; c < 3; c++) r[c] = useGR ? this.amplified(o + c, k) : n[o + c];
      }
      for (let j = 0; j < N; j++) {
        const o = row + j * 3;
        for (let c = 0; c < 3; c++) {
          const v = useGR ? this.amplified(o + c, k) : n[o + c];
          out[o + c] = ref >= 0 ? v - r[c] : v;
        }
      }
    }
  }

  /** Sun-Mercury distance at sample `i` of a display buffer (the frame shift cancels). */
  periDistance(src, i) {
    const a = (i * this.N + this.mercIdx) * 3;
    const b = (i * this.N + this.sunIdx) * 3;
    return Math.hypot(src[a] - src[b], src[a + 1] - src[b + 1], src[a + 2] - src[b + 2]);
  }

  /**
   * Place the markers of `points` at the perihelia of the path stored in `src`.
   *
   * Each seed (a perihelion sample of the Newton or GR run) walks downhill on the Sun-Mercury
   * distance to the nearest interior local minimum of the path actually drawn: on an amplified
   * path newton + k * delta the minimum sits a few samples away from the run's own perihelion,
   * and marking the seed sample itself puts the marker on the wrong side of it.  The passage is
   * then refined to the vertex of the parabola through the three distances around the minimum
   * (analysis.precession.perihelion_passages; the N-body sample grid is uniform, so the offset
   * `s` is in samples) and the position is the quadratic interpolant of the three samples at
   * that offset, so the markers do not scatter by the half-sample phase of the 2-day grid.
   * Markers are written in increasing time; seeds that converge on the same minimum or run off
   * the ends of the series are dropped.  Returns the number of markers placed.
   */
  fillPeri(points, src) {
    if (!points) return 0;
    const { seeds, at } = points.userData;
    const pos = points.geometry.attributes.position;
    const arr = pos.array;
    const { N, T, mercIdx } = this;
    const stride = N * 3;
    let count = 0;
    let last = -1;
    for (let q = 0; q < seeds.length; q++) {
      let i = Math.min(Math.max(seeds[q], 0), T - 1);
      let r = this.periDistance(src, i);
      for (let guard = 0; guard < T; guard++) {
        const rl = i > 0 ? this.periDistance(src, i - 1) : Infinity;
        const rr = i < T - 1 ? this.periDistance(src, i + 1) : Infinity;
        if (rl < r && rl <= rr) { i -= 1; r = rl; }
        else if (rr < r) { i += 1; r = rr; }
        else break;
      }
      if (i <= last || i === 0 || i === T - 1) continue;
      last = i;
      // r(s) = A s^2 + B s + r through (-1, r0), (0, r), (1, r2); vertex at s = -B / (2A)
      const r0 = this.periDistance(src, i - 1);
      const r2 = this.periDistance(src, i + 1);
      const A = 0.5 * (r0 + r2) - r;
      let s = A > 0 ? 0.25 * (r0 - r2) / A : 0;
      if (!(s >= -1 && s <= 1)) s = 0;
      const w0 = 0.5 * s * (s - 1);
      const w1 = 1 - s * s;
      const w2 = 0.5 * s * (s + 1);
      const o = (i * N + mercIdx) * 3;
      for (let c = 0; c < 3; c++) {
        arr[3 * count + c] = w0 * src[o - stride + c] + w1 * src[o + c] + w2 * src[o + stride + c];
      }
      at[count] = i + s;
      count += 1;
    }
    points.userData.count = count;
    pos.needsUpdate = true;
    return count;
  }

  /** Recompute the displayed arrays after a frame / model / amplification change. */
  recompute() {
    const kGR = this.model === 'newton' ? 0 : this.k;
    this.fillDisplay(this.disp, kGR);
    this.ib.needsUpdate = true;
    const both = this.model === 'both' && this.gr;
    if (both) {
      this.fillDisplay(this.dispN, 0);
      this.ibN.needsUpdate = true;
    }
    for (const b of this.bodies) {
      b.ghost.visible = Boolean(both) && this.showPaths;
      b.ghostMesh.visible = Boolean(both);
      b.path.visible = this.showPaths;
    }
    // perihelion markers: the GR set at the perihelia of the displayed (amplified) path, the
    // Newton set at those of the Newton path (dispN in "both", disp itself in "newton")
    if (this.periGRPts) {
      if (this.model !== 'newton') this.fillPeri(this.periGRPts, this.disp);
      this.periGRPts.visible = this.showPeri && this.model !== 'newton';
    }
    if (this.periNPts) {
      if (this.model !== 'gr') this.fillPeri(this.periNPts, this.model === 'both' ? this.dispN : this.disp);
      this.periNPts.visible = this.showPeri && this.model !== 'gr';
    }
  }

  setFrame(frame) {
    if (frame === 'earth' && this.earthIdx < 0) frame = 'helio';
    if (frame === this.frame) return;
    this.frame = frame;
    this.recompute();
  }

  setModel(model) {
    if (!this.gr) model = 'newton';
    if (model === this.model) return;
    this.model = model;
    this.recompute();
  }

  setAmplification(k) {
    k = Number(k);
    if (!Number.isFinite(k) || k <= 0) k = 1;
    if (k === this.k) return;
    this.k = k;
    this.recompute();
  }

  setTrailDays(days) {
    this.trailSamples = Math.max(2, Math.round(days / this.dtDays) + 1);
  }

  setPathOpacity(opacity) {
    for (const b of this.bodies) b.path.material.opacity = opacity;
  }

  setPathsVisible(on) {
    this.showPaths = on;
    for (const b of this.bodies) {
      b.path.visible = on;
      b.ghost.visible = on && this.model === 'both' && Boolean(this.gr);
    }
  }

  setPeriVisible(on) {
    this.showPeri = on;
    if (this.periGRPts) this.periGRPts.visible = on && this.model !== 'newton';
    if (this.periNPts) this.periNPts.visible = on && this.model !== 'gr';
  }

  /** |k (gr - newton)| and |gr - newton| of Mercury at sample index i (AU). */
  mercuryDisplacement(i) {
    if (!this.gr || this.mercIdx < 0) return null;
    const o = (i * this.N + this.mercIdx) * 3;
    const dc = (c) => (this.delta ? this.delta[o + c] : this.gr[o + c] - this.newton[o + c]);
    const d = Math.hypot(dc(0), dc(1), dc(2));
    return { trueAU: d, shownAU: (this.model === 'newton' ? 0 : this.k) * d };
  }

  sampleIndex(frac) {
    const x = Math.min(Math.max(frac, 0), 1) * (this.T - 1);
    const i0 = Math.min(Math.floor(x), this.T - 1);
    const i1 = Math.min(i0 + 1, this.T - 1);
    return { x, i0, i1, u: x - i0 };
  }

  update(frac) {
    const { x, i0, i1, u } = this.sampleIndex(frac);
    const { N } = this;
    const lerp = (src, j, c) => {
      const a = src[(i0 * N + j) * 3 + c];
      const b = src[(i1 * N + j) * 3 + c];
      return a + (b - a) * u;
    };
    const start = Math.max(0, i0 - this.trailSamples + 1);
    const count = Math.min(this.T, i1 + 1) - start;
    for (const b of this.bodies) {
      b.mesh.position.set(lerp(this.disp, b.j, 0), lerp(this.disp, b.j, 1), lerp(this.disp, b.j, 2));
      const s = this.markerScale(b.size);
      b.mesh.scale.setScalar(s);
      b.trail.geometry.setDrawRange(start, count);
      if (b.ghostMesh.visible) {
        b.ghostMesh.position.set(lerp(this.dispN, b.j, 0), lerp(this.dispN, b.j, 1), lerp(this.dispN, b.j, 2));
        b.ghostMesh.scale.setScalar(s * 1.05);
      }
      if (b.isSun) this.light.position.copy(b.mesh.position);
    }
    // markers up to the current (fractional) sample
    for (const pts of [this.periGRPts, this.periNPts]) {
      if (!pts || !pts.visible) continue;
      const { at, count } = pts.userData;
      let n = 0;
      while (n < count && at[n] <= x) n++;
      pts.geometry.setDrawRange(0, n);
    }
    this.currentIndex = i0;
  }

  timeReadout(frac) {
    const { x } = this.sampleIndex(frac);
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, this.T - 1);
    const tDays = this.t[i0] + (this.t[i1] - this.t[i0]) * (x - i0);
    return fmtEpoch(this.epochBase + (tDays - this.t[0]));
  }

  cameraPreset(name) {
    const app = this.app;
    const tilt = new THREE.Vector3(0, -0.74, 0.67).normalize();
    switch (name) {
      case 'inner': app.setCamera(tilt, 2.6 * this.innerR); break;
      case 'outer': app.setCamera(tilt, 2.4 * this.outerR); break;
      case 'top': app.setCamera(new THREE.Vector3(0, -1e-3, 1).normalize()); break;
      case 'edge': app.setCamera(new THREE.Vector3(0, -1, 0.07).normalize()); break;
      case 'earth':
        if (this.earthIdx >= 0) {
          app.ui.setFrame('earth');
          app.setCamera(tilt, 0.012);
        }
        break;
      default: break;
    }
  }
}

/** black_hole: geodesic (T, 3) in GM/c^2, rings at r = 2, 3, 6. */
class GeodesicView extends BaseView {
  constructor(app, payload, entry) {
    super(app, payload, entry);
    const A = this.arrays;
    const S = this.shapes;
    const name = A.geodesic ? 'geodesic' : Object.keys(A).find((k) => S[k].length === 2 && S[k][1] >= 2);
    if (!name) throw new Error("black-hole scenario needs a 'geodesic' (T, 3) array");
    const src = A[name];
    const [T, D] = S[name];
    this.T = T;
    this.unit = isGeometricUnits(this.meta) || !this.meta.units ? 'GM/c²' : this.unit;
    this.pts = new Float32Array(T * 3);
    for (let i = 0; i < T; i++) {
      this.pts[3 * i] = src[i * D];
      this.pts[3 * i + 1] = src[i * D + 1];
      this.pts[3 * i + 2] = D > 2 ? src[i * D + 2] : 0;
    }
    this.timeKey = ['tau', 'proper_time', 't', 'time'].find((k) => A[k] && A[k].length === T) || null;
    this.time = this.timeKey ? A[this.timeKey] : null;
    this.timeLabel = this.timeKey && /tau|proper/.test(this.timeKey) ? 'τ' : 't';
    this.periIdx = A.peri_idx && A.peri_idx.length ? A.peri_idx : null;
    this.showPeri = false;
    let rmax = 0;
    for (let i = 0; i < T; i++) {
      const r = Math.hypot(this.pts[3 * i], this.pts[3 * i + 1], this.pts[3 * i + 2]);
      if (r > rmax) rmax = r;
    }
    this.extent = Math.max(rmax, 8);
    this.trailSamples = T;
    this.showPaths = true;
    this.build();
  }

  get span() { return this.time ? this.time[this.T - 1] - this.time[0] : this.T; }

  build() {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pts, 3));
    this.path = new THREE.Line(geo, new THREE.LineBasicMaterial({
      color: PALETTE[0], transparent: true, opacity: 0.35, depthWrite: false,
    }));
    this.path.frustumCulled = false;
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', geo.attributes.position);
    this.trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: PALETTE[0] }));
    this.trail.frustumCulled = false;
    this.group.add(this.path, this.trail);

    const hole = new THREE.Mesh(new THREE.SphereGeometry(2, 48, 32), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    this.group.add(hole);
    // label angles chosen so the three ring labels and the grid's -45 deg labels never meet
    const rings = [
      [2, RING_COLORS.horizon, 'horizon r = 2', -135, 'sw'],
      [3, RING_COLORS.photon, 'photon sphere r = 3', 45, 'ne'],
      [6, RING_COLORS.isco, 'ISCO r = 6', 135, 'nw'],
    ];
    this.gridLabelMinR = 6 * 1.5;
    for (const [r, color, text, deg, anchor] of rings) {
      const ln = ring(r, color, 0.9);
      const lab = makeLabel(text, 'ring', anchor);
      lab.position.set(...onCircle(r, deg));
      ln.add(lab);
      this.labels.push(lab);
      this.group.add(ln);
    }
    this.marker = new THREE.Mesh(this.sphereGeo, new THREE.MeshBasicMaterial({
      color: new THREE.Color(PALETTE[0]).lerp(new THREE.Color(0xffffff), 0.45),
    }));
    const lab = makeLabel('test body', '', 'ne');
    lab.position.set(...onCircle(1, 45));
    this.marker.add(lab);
    this.labels.push(lab);
    this.group.add(this.marker);
    this.periPts = null;
    if (this.periIdx) {
      const pg = new THREE.BufferGeometry();
      const arr = new Float32Array(this.periIdx.length * 3);
      for (let q = 0; q < this.periIdx.length; q++) {
        const i = Math.min(Math.max(this.periIdx[q], 0), this.T - 1);
        arr[3 * q] = this.pts[3 * i];
        arr[3 * q + 1] = this.pts[3 * i + 1];
        arr[3 * q + 2] = this.pts[3 * i + 2];
      }
      pg.setAttribute('position', new THREE.BufferAttribute(arr, 3));
      this.periPts = new THREE.Points(pg, new THREE.PointsMaterial({
        color: BODY_COLORS.Sun, size: 7, sizeAttenuation: false, depthWrite: false, transparent: true, opacity: 0.95,
      }));
      this.periPts.frustumCulled = false;
      this.periPts.visible = false;
      this.group.add(this.periPts);
    }
    this.grid = makeEclipticGrid(this.extent, this.unit);
    this.group.add(this.grid);
  }

  setPeriVisible(on) {
    this.showPeri = on;
    if (this.periPts) this.periPts.visible = on;
  }

  setTrailDays(days) {
    const per = this.span / Math.max(this.T - 1, 1);
    this.trailSamples = Math.max(2, Math.round(days / per) + 1);
  }

  setPathsVisible(on) { this.path.visible = on; }

  update(frac) {
    const x = Math.min(Math.max(frac, 0), 1) * (this.T - 1);
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, this.T - 1);
    const u = x - i0;
    const p = this.pts;
    this.marker.position.set(
      p[3 * i0] + (p[3 * i1] - p[3 * i0]) * u,
      p[3 * i0 + 1] + (p[3 * i1 + 1] - p[3 * i0 + 1]) * u,
      p[3 * i0 + 2] + (p[3 * i1 + 2] - p[3 * i0 + 2]) * u,
    );
    this.marker.scale.setScalar(this.markerScale(1.0));
    const start = Math.max(0, i0 - this.trailSamples + 1);
    this.trail.geometry.setDrawRange(start, Math.min(this.T, i1 + 1) - start);
    if (this.periPts && this.periPts.visible) {
      let n = 0;
      while (n < this.periIdx.length && this.periIdx[n] <= i0) n++;
      this.periPts.geometry.setDrawRange(0, n);
    }
    this.currentIndex = i0;
  }

  timeReadout(frac) {
    const x = Math.min(Math.max(frac, 0), 1) * (this.T - 1);
    const i0 = Math.floor(x);
    const r = Math.hypot(this.pts[3 * i0], this.pts[3 * i0 + 1], this.pts[3 * i0 + 2]);
    const time = this.time
      ? `${this.timeLabel} = ${fmtNum(Number(this.time[i0].toPrecision(5)))} GM/c³ · `
      : `sample ${i0} / ${this.T - 1} · `;
    return `${time}r = ${r.toFixed(2)} GM/c²`;
  }

  cameraPreset(name) {
    const app = this.app;
    const tilt = new THREE.Vector3(0, -0.6, 0.8).normalize();
    switch (name) {
      case 'inner': app.setCamera(tilt, 2.4 * this.extent); break;
      case 'outer': app.setCamera(tilt, 5 * this.extent); break;
      case 'top': app.setCamera(new THREE.Vector3(0, -1e-3, 1).normalize()); break;
      case 'edge': app.setCamera(new THREE.Vector3(0, -1, 0.07).normalize()); break;
      default: break;
    }
  }
}

/** light_rays: ray_<k> polylines (P, 2|3), mass at the origin, photon-sphere ring. */
class RaysView extends BaseView {
  constructor(app, payload, entry) {
    super(app, payload, entry);
    const A = this.arrays;
    const S = this.shapes;
    const names = Object.keys(A).filter((k) => /^ray_/.test(k) && S[k].length === 2 && S[k][1] >= 2);
    names.sort((a, b) => Number(a.slice(4)) - Number(b.slice(4)) || a.localeCompare(b));
    if (names.length === 0) throw new Error("light-rays scenario needs 'ray_<k>' arrays");
    this.rayNames = names;
    const geometric = isGeometricUnits(this.meta);
    this.unit = geometric || !this.meta.units ? 'GM/c²' : this.unit;

    // impact parameters and deflections from meta (several spellings accepted)
    const n = names.length;
    const numList = (v) => isNumberList(v, 1e9) && v.length === n;
    let bList = pickMeta(this.meta, /^(b|b_values|bs|impact_parameters?|impact)$/i, numList)
      ?? pickMeta(this.meta, /impact/i, numList);
    // deflection: radians when the bending is strong (compact mass), arcseconds otherwise
    const deflKeys = Object.keys(this.meta).filter((k) => /deflect/i.test(k) && !/weak|newton|corpusc/i.test(k) && numList(this.meta[k]));
    const radKey = deflKeys.find((k) => /rad/i.test(k));
    const strong = radKey && Math.max(...this.meta[radKey].map(Math.abs)) >= 0.01;
    let deflKey = (strong && radKey)
      || deflKeys.find((k) => /arcsec/i.test(k)) || deflKeys.find((k) => /deg/i.test(k)) || radKey || deflKeys[0];
    let defl = deflKey ? this.meta[deflKey] : undefined;
    const rays = pickMeta(this.meta, /^rays?$/i, (v) => Array.isArray(v) && v.length === n && v.every(isPlainObject));
    if (rays) {
      const bk = Object.keys(rays[0]).find((k) => /^(b|impact)/i.test(k));
      const dk = Object.keys(rays[0]).find((k) => /deflect/i.test(k));
      if (bk && !bList) bList = rays.map((r) => Number(r[bk]));
      if (dk && !defl) { defl = rays.map((r) => Number(r[dk])); deflKey = dk; }
    }
    this.bList = bList || null;
    this.deflections = defl || null;
    this.usedMetaKeys = new Set(Object.keys(this.meta).filter((k) => this.meta[k] === bList || this.meta[k] === defl || this.meta[k] === rays));
    this.deflUnit = deflKey ? (/arcsec/i.test(deflKey) ? '″' : /deg/i.test(deflKey) ? '°' : /rad/i.test(deflKey) ? ' rad' : '') : '';
    if (deflKey && !this.deflUnit) {
      const du = isPlainObject(this.meta.units) ? this.meta.units.angle || this.meta.units.deflection : null;
      if (du) this.deflUnit = ` ${du}`;
    }

    // radii: photon sphere (3 GM/c^2) and the mass marker
    const gm = typeof this.meta.gm === 'number' ? this.meta.gm : null;
    const c = typeof this.meta.c === 'number' ? this.meta.c : null;
    const num = (v) => typeof v === 'number' && Number.isFinite(v) && v > 0;
    this.rPhoton = pickMeta(this.meta, /^(r_photon|r_photon_sphere|photon_sphere|r_ph|photon_sphere_radius)$/i, num)
      ?? (geometric ? 3 : gm && c ? 3 * gm / (c * c) : null);
    this.rMass = pickMeta(this.meta, /^(body_radius|r_mass|r_sun|radius|mass_radius|r_eq|r_horizon)$/i, num)
      ?? (geometric ? 2 : gm && c ? 2 * gm / (c * c) : null);
    this.compact = this.meta.compact === true
      || (this.meta.compact === undefined && (geometric || /hole|compact/i.test(String(this.meta.body || ''))));
    this.build();
  }

  get span() { return 1; }

  build() {
    const n = this.rayNames.length;
    const order = this.bList
      ? [...this.bList.keys()].sort((a, b) => Math.abs(this.bList[a]) - Math.abs(this.bList[b]))
      : [...Array(n).keys()];
    const rank = new Array(n);
    order.forEach((idx, r) => { rank[idx] = r; });
    this.rays = [];
    let bmax = 0;
    for (let q = 0; q < n; q++) {
      const name = this.rayNames[q];
      const src = this.arrays[name];
      const [P, D] = this.shapes[name];
      const pts = new Float32Array(P * 3);
      let minR = Infinity;
      for (let i = 0; i < P; i++) {
        pts[3 * i] = src[i * D];
        pts[3 * i + 1] = src[i * D + 1];
        pts[3 * i + 2] = D > 2 ? src[i * D + 2] : 0;
        const r = Math.hypot(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
        if (r < minR) minR = r;
      }
      const b = this.bList ? Math.abs(this.bList[q]) : minR;
      if (b > bmax) bmax = b;
      const color = rampColor(n > 1 ? rank[q] / (n - 1) : 0.5);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color }));
      line.frustumCulled = false;
      const marker = new THREE.Mesh(this.sphereGeo, new THREE.MeshBasicMaterial({ color }));
      this.group.add(line, marker);
      this.rays.push({ name, pts, P, color, line, marker, b, minR });
    }
    this.bmax = bmax;
    this.extent = Math.max(4 * bmax, (this.rPhoton || 0) * 6, 1e-9);

    if (this.rMass) {
      const color = this.compact ? 0x000000 : BODY_COLORS.Sun;
      const mass = new THREE.Mesh(new THREE.SphereGeometry(this.rMass, 40, 28), new THREE.MeshBasicMaterial({ color }));
      this.group.add(mass);
      if (this.compact) this.group.add(ring(this.rMass, RING_COLORS.horizon, 0.9));
    } else {
      this.massMarker = new THREE.Mesh(this.sphereGeo, new THREE.MeshBasicMaterial({ color: BODY_COLORS.Sun }));
      this.group.add(this.massMarker);
    }
    const bodyR = typeof this.meta.body_radius === 'number' ? this.meta.body_radius : 0;
    const massLabel = makeLabel(String(this.meta.body || 'mass'), 'ring', 'sw');
    massLabel.position.set(...onCircle(bodyR, -135));
    this.group.add(massLabel);
    this.labels.push(massLabel);
    if (this.rPhoton && this.rPhoton > 1e-3 * this.extent) {
      const ln = ring(this.rPhoton, RING_COLORS.photon, 0.9);
      const lab = makeLabel(`photon sphere r = ${fmtNum(this.rPhoton)}`, 'ring', 'ne');
      lab.position.set(...onCircle(this.rPhoton, 45));
      ln.add(lab);
      this.labels.push(lab);
      this.group.add(ln);
    }
    this.gridLabelMinR = 1.5 * Math.max(this.rPhoton || 0, bodyR);
    this.grid = makeEclipticGrid(this.extent, this.unit);
    this.group.add(this.grid);
  }

  /** Rows for the legend: [{color, b, deflection}] in the array order. */
  legend() {
    return this.rays.map((r, q) => ({
      color: `#${r.color.getHexString()}`,
      name: r.name,
      b: this.bList ? this.bList[q] : null,
      deflection: this.deflections ? this.deflections[q] : null,
      unit: this.deflUnit,
    }));
  }

  setTrailDays() {}

  setPathsVisible(on) { for (const r of this.rays) r.line.visible = on; }

  update(frac) {
    const s = this.markerScale(0.6);
    for (const r of this.rays) {
      const x = Math.min(Math.max(frac, 0), 1) * (r.P - 1);
      const i0 = Math.floor(x);
      const i1 = Math.min(i0 + 1, r.P - 1);
      const u = x - i0;
      const p = r.pts;
      r.marker.position.set(
        p[3 * i0] + (p[3 * i1] - p[3 * i0]) * u,
        p[3 * i0 + 1] + (p[3 * i1 + 1] - p[3 * i0 + 1]) * u,
        p[3 * i0 + 2] + (p[3 * i1 + 2] - p[3 * i0 + 2]) * u,
      );
      r.marker.scale.setScalar(s);
    }
    if (this.massMarker) this.massMarker.scale.setScalar(this.markerScale(1.2));
  }

  timeReadout(frac) {
    return `photons at ${(frac * 100).toFixed(1)} % of their paths`;
  }

  cameraPreset(name) {
    const app = this.app;
    const top = new THREE.Vector3(0, -1e-3, 1).normalize();
    switch (name) {
      case 'inner': app.setCamera(top, 2.2 * this.extent); break;
      case 'outer': app.setCamera(top, 12 * this.extent); break;
      case 'top': app.setCamera(top); break;
      case 'edge': app.setCamera(new THREE.Vector3(0, -1, 0.12).normalize()); break;
      default: break;
    }
  }
}

function viewKind(id, payload) {
  const A = payload.arrays;
  if (id === 'black_hole' || (A.geodesic && !A.pos_newton)) return 'geodesic';
  if (id === 'light_rays' || (!A.pos_newton && Object.keys(A).some((k) => /^ray_/.test(k)))) return 'rays';
  return 'nbody';
}

// ----------------------------------------------------------------------------- the app
class App {
  constructor() {
    this.stats = { frames: 0, lastError: null };
    this.view = null;
    this.entries = [];
    this.entryById = new Map();
    this.frac = 0;
    this.playing = false;
    this.speed = 1; // multiplier on "one full run per minute"
    this.lastTick = null;
    this.loadToken = 0;
    this.ui = null;
  }

  // -- status overlay -------------------------------------------------------------------
  showStatus(text, detail = null, error = false) {
    const box = $('status');
    box.hidden = false;
    box.classList.toggle('error', error);
    $('status-text').textContent = text;
    const d = $('status-detail');
    d.hidden = !detail;
    d.textContent = detail || '';
  }

  hideStatus() { $('status').hidden = true; }

  fail(context, err) {
    const msg = err && err.message ? err.message : String(err);
    this.stats.lastError = `${context}: ${msg}`;
    this.showStatus(context, msg, true);
  }

  // -- renderer ---------------------------------------------------------------------------
  initRenderer() {
    const host = $('canvas-host');
    const canvas = document.createElement('canvas');
    const hasWebGL = (() => {
      try {
        const probe = document.createElement('canvas');
        return Boolean(window.WebGLRenderingContext
          && (probe.getContext('webgl2') || probe.getContext('webgl')));
      } catch (_) {
        return false;
      }
    })();
    if (!hasWebGL) {
      // the usual cause is the browser, not the machine: Chrome turns hardware graphics off for
      // the rest of a session after its GPU process crashes a few times (and no longer falls
      // back to software WebGL), and both browsers can have acceleration switched off
      const isFirefox = /firefox/i.test(navigator.userAgent);
      const steps = isFirefox
        ? [
          '1. Open about:support and look at "WebGL 1 / 2 Driver Renderer" under Graphics.',
          '2. In about:config, webgl.disabled must be false.',
          '3. In Settings > General > Performance, allow hardware acceleration, then restart Firefox.',
        ]
        : [
          '1. Open chrome://gpu. If WebGL reads "Unavailable" or "Disabled", Chrome has turned hardware graphics off for this session, usually after its GPU process crashed.',
          '2. Restart Chrome with chrome://restart. Tabs are restored, and a fresh session normally gets WebGL back.',
          '3. In Settings > System, "Use graphics acceleration when available" must be on.',
        ];
      throw new Error(['WebGL is not available in this browser, so the 3-D viewer cannot run. '
        + 'The page itself is fine: it works in any browser with WebGL enabled.', '',
        ...steps].join('\n'));
    }
    // preserveDrawingBuffer keeps the back buffer readable (canvas.toDataURL for tests) and
    // avoids the MSAA-resolve ReadPixels stall that software GL reports as a console warning.
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: 'high-performance',
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--bg-canvas').trim() || '#0b0d12'), 1);
    host.appendChild(canvas);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.domElement.className = 'labels';
    host.appendChild(this.labelRenderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 1e-4, 1e6);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0, -3, 2.5);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 1e-5;
    this.controls.maxDistance = 1e5;
    this.controls.target.set(0, 0, 0);

    this.resize();
    if (typeof ResizeObserver === 'function') {
      new ResizeObserver(() => this.resize()).observe(host);
    }
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const host = $('canvas-host');
    const w = Math.max(host.clientWidth, 1);
    const h = Math.max(host.clientHeight, 1);
    this.renderer.setSize(w, h, false);
    this.labelRenderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  cameraDistance() {
    return Math.max(this.camera.position.distanceTo(this.controls.target), 1e-9);
  }

  /** Screen pixels per world unit at the orbit target. */
  pixelsPerUnit() {
    const halfH = Math.max($('canvas-host').clientHeight, 1) / 2;
    return halfH / (this.cameraDistance() * Math.tan((this.camera.fov * Math.PI) / 360));
  }

  /** Place the camera along `dir` from the origin at `dist` (default: keep the distance). */
  setCamera(dir, dist = null) {
    const d = dist ?? this.cameraDistance();
    this.controls.target.set(0, 0, 0);
    this.camera.position.copy(dir).multiplyScalar(d);
    this.camera.lookAt(0, 0, 0);
    this.controls.update();
  }

  updateClipPlanes() {
    const d = this.cameraDistance();
    const extent = this.view ? this.view.extent : 1;
    const near = Math.max(d * 2e-3, 1e-7);
    const far = Math.max(d * 50, extent * 8);
    if (Math.abs(this.camera.near - near) / near > 0.2 || Math.abs(this.camera.far - far) / far > 0.2) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  // -- playback ---------------------------------------------------------------------------
  setTime(frac) {
    const f = Number(frac);
    this.frac = Number.isFinite(f) ? Math.min(Math.max(f, 0), 1) : 0;
    if (this.ui) this.ui.syncTime();
  }

  play() {
    if (this.frac >= 1) this.frac = 0;
    this.playing = true;
    this.lastTick = null;
    if (this.ui) this.ui.syncPlay();
  }

  pause() {
    this.playing = false;
    if (this.ui) this.ui.syncPlay();
  }

  tick(now) {
    if (!this.playing || !this.view) return;
    if (this.lastTick === null) { this.lastTick = now; return; }
    const dt = Math.min((now - this.lastTick) / 1000, 0.25);
    this.lastTick = now;
    this.frac += dt * this.speed / 60; // one full run per minute at speed 1
    if (this.frac >= 1) {
      this.frac = 1;
      this.playing = false;
      if (this.ui) this.ui.syncPlay();
    }
    if (this.ui) this.ui.syncTime();
  }

  // -- scenarios --------------------------------------------------------------------------
  async loadScenario(id) {
    const entry = this.entryById.get(id);
    if (!entry) throw new Error(`unknown scenario '${id}'`);
    const token = ++this.loadToken;
    this.pause();
    this.showStatus(`Loading ${entry.title || id}…`);
    let payload;
    try {
      const url = new URL(entry.url, document.baseURI).href;
      payload = await fetchPayload(url);
    } catch (err) {
      if (token === this.loadToken) this.fail(`Could not load scenario '${id}'`, err);
      throw err;
    }
    if (token !== this.loadToken) return; // superseded by a newer request
    try {
      this.installView(id, entry, payload);
    } catch (err) {
      this.fail(`Could not render scenario '${id}'`, err);
      throw err;
    }
    this.hideStatus();
    await nextFrame();
    await nextFrame();
  }

  installView(id, entry, payload) {
    if (this.view) {
      this.view.dispose();
      this.view = null;
    }
    const kind = viewKind(id, payload);
    const View = kind === 'geodesic' ? GeodesicView : kind === 'rays' ? RaysView : NBodyView;
    const view = new View(this, payload, entry);
    this.scene.add(view.group);
    this.view = view;
    this.kind = kind;
    this.scenario = id;
    this.frac = 0;
    this.ui.bindView(view, kind, entry, payload);
    view.cameraPreset('inner');
    view.update(0);
  }

  // -- main loop --------------------------------------------------------------------------
  animate(now) {
    requestAnimationFrame((t) => this.animate(t));
    try {
      this.tick(now);
      this.controls.update();
      this.updateClipPlanes();
      if (this.view) {
        this.view.update(this.frac);
        this.view.updateGridLabels(this.pixelsPerUnit());
        this.ui.syncReadouts();
      }
      syncLabelVisibility(this.scene);
      this.renderer.render(this.scene, this.camera);
      this.labelRenderer.render(this.scene, this.camera);
      this.stats.frames += 1;
      if (this.view && !this.ready) {
        this.ready = true;
        if (window.__gravsim) window.__gravsim.ready = true;
      }
    } catch (err) {
      if (this.stats.lastError !== String(err)) {
        this.fail('Render error', err);
      }
    }
  }

  async start() {
    this.ui = new UI(this);
    try {
      this.initRenderer();
    } catch (err) {
      this.fail('3-D viewer unavailable', err);
      return;
    }
    requestAnimationFrame((t) => this.animate(t));
    let manifest;
    try {
      manifest = await loadManifest();
    } catch (err) {
      this.fail('Could not find the scenario list', err);
      return;
    }
    this.entries = manifest.list;
    this.entryById = new Map(this.entries.map((e) => [e.id, e]));
    this.ui.setScenarios(this.entries, manifest.source);
    const first = this.entries.find((e) => e.id === 'solar_system') || this.entries[0];
    try {
      await this.loadScenario(first.id);
    } catch (_) {
      // already reported in the status overlay
    }
  }
}

// ----------------------------------------------------------------------------- UI wiring
class UI {
  constructor(app) {
    this.app = app;
    this.view = null;
    this.kind = null;
    this.trailDays = 365;
    this.el = {
      select: $('scenario-select'), source: $('source'), play: $('play'), back: $('step-back'),
      fwd: $('step-fwd'), rewind: $('rewind'), speed: $('speed'), speedOut: $('speed-out'),
      time: $('time'), timeOut: $('time-out'), amp: $('amp'), ampOut: $('amp-out'),
      ampReadout: $('amp-readout'), ampField: $('amp-field'), peri: $('peri'), periWrap: $('peri-wrap'),
      periLabel: $('peri-label'), trail: $('trail'), trailOut: $('trail-out'),
      paths: $('paths'), labels: $('labels'), grid: $('grid'), presets: $('camera-presets'),
      presetEarth: $('preset-earth'), frameEarth: $('frame-earth-label'),
      groupModel: $('group-model'), groupFrame: $('group-frame'), groupDisplay: $('group-display'),
      topScenario: $('topbar-scenario'),
      infoTitle: $('info-title'), infoBlurb: $('info-blurb'), infoExplain: $('info-explain'),
      legendSec: $('info-legend'), legendBody: $('legend-table').querySelector('tbody'),
      numbers: $('numbers'), numbersMore: $('numbers-more'), numbersDetails: $('numbers-details'),
      notesSec: $('info-notes'), notes: $('notes'),
      paramsSec: $('info-params'), params: $('params'), arraysSec: $('info-arrays'), arrays: $('arrays'),
      controls: $('controls'), info: $('info'), toggleControls: $('toggle-controls'), toggleInfo: $('toggle-info'),
    };
    this.wire();
  }

  wire() {
    const e = this.el;
    const app = this.app;
    e.select.addEventListener('change', () => {
      const id = e.select.value;
      if (id) app.loadScenario(id).catch(() => {});
    });
    e.play.addEventListener('click', () => (app.playing ? app.pause() : app.play()));
    e.rewind.addEventListener('click', () => { app.pause(); app.setTime(0); });
    const step = (dir) => {
      if (!this.view || !this.view.T) return;
      app.pause();
      const i = Math.round(app.frac * (this.view.T - 1)) + dir;
      app.setTime(Math.min(Math.max(i, 0), this.view.T - 1) / Math.max(this.view.T - 1, 1));
    };
    e.back.addEventListener('click', () => step(-1));
    e.fwd.addEventListener('click', () => step(1));
    e.speed.addEventListener('input', () => {
      app.speed = Math.pow(10, Number(e.speed.value));
      this.syncSpeed();
    });
    e.time.addEventListener('input', () => {
      app.frac = Number(e.time.value) / Number(e.time.max);
    });
    for (const r of document.querySelectorAll('input[name="model"]')) {
      r.addEventListener('change', () => { if (r.checked && this.view && this.view.setModel) this.view.setModel(r.value); });
    }
    for (const r of document.querySelectorAll('input[name="frame"]')) {
      r.addEventListener('change', () => { if (r.checked && this.view && this.view.setFrame) this.view.setFrame(r.value); });
    }
    e.amp.addEventListener('input', () => {
      const k = Math.pow(10, Number(e.amp.value));
      if (this.view && this.view.setAmplification) this.view.setAmplification(k);
      this.syncAmp();
    });
    e.peri.addEventListener('change', () => { if (this.view && this.view.setPeriVisible) this.view.setPeriVisible(e.peri.checked); });
    e.trail.addEventListener('input', () => this.applyTrail());
    e.paths.addEventListener('change', () => { if (this.view) this.view.setPathsVisible(e.paths.checked); });
    e.labels.addEventListener('change', () => { if (this.view) this.view.setLabelsVisible(e.labels.checked); });
    e.grid.addEventListener('change', () => { if (this.view) this.view.setGridVisible(e.grid.checked); });
    e.presets.addEventListener('click', (ev) => {
      const btn = ev.target.closest('button[data-preset]');
      if (btn && this.view) this.view.cameraPreset(btn.dataset.preset);
    });
    const toggle = (panel, btn) => {
      const open = panel.classList.toggle('open');
      btn.setAttribute('aria-expanded', String(open));
      const other = panel === e.controls ? e.info : e.controls;
      const otherBtn = panel === e.controls ? e.toggleInfo : e.toggleControls;
      if (open && other.classList.contains('open')) {
        other.classList.remove('open');
        otherBtn.setAttribute('aria-expanded', 'false');
      }
    };
    e.toggleControls.addEventListener('click', () => toggle(e.controls, e.toggleControls));
    e.toggleInfo.addEventListener('click', () => toggle(e.info, e.toggleInfo));
    window.addEventListener('keydown', (ev) => {
      if (ev.code === 'Space' && !['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(document.activeElement?.tagName)) {
        ev.preventDefault();
        if (app.view) (app.playing ? app.pause() : app.play());
      }
    });
    this.syncSpeed();
  }

  setScenarios(entries, source) {
    const e = this.el;
    e.select.innerHTML = '';
    for (const entry of entries) {
      const opt = document.createElement('option');
      opt.value = entry.id;
      opt.textContent = entry.title || entry.id;
      e.select.appendChild(opt);
    }
    e.select.disabled = false;
    e.source.textContent = `scenarios from ${source}`;
  }

  /** Reconfigure the panels for a freshly built view. */
  bindView(view, kind, entry, payload) {
    this.view = view;
    this.kind = kind;
    const e = this.el;
    const meta = payload.meta;
    const id = entry.id;
    e.select.value = id;
    e.topScenario.textContent = entry.title || meta.title || id;
    for (const b of [e.play, e.rewind]) b.disabled = false;
    // the rays view animates a continuous fraction, it has no samples to step through
    for (const b of [e.back, e.fwd]) b.disabled = kind === 'rays';

    const nbody = kind === 'nbody';
    e.groupModel.hidden = !nbody || !view.gr;
    // amplifying GR - Newton only helps when the difference is invisible at true scale; in the
    // rosette (c/100) it is already millions of km and k = 1000 throws Mercury off-screen
    e.ampField.hidden = !nbody || !view.gr || id === 'rosette';
    e.groupFrame.hidden = !nbody;
    e.frameEarth.hidden = !(nbody && view.earthIdx >= 0);
    e.presetEarth.hidden = !(nbody && view.earthIdx >= 0);
    e.trail.disabled = kind === 'rays';
    const hasPeri = (nbody && Boolean(view.periGR || view.periN)) || (kind === 'geodesic' && Boolean(view.periIdx));
    e.periWrap.hidden = !hasPeri;
    e.peri.disabled = !hasPeri;
    // the rosette's point is the perihelion sweep, so its markers start on
    e.peri.checked = hasPeri && id === 'rosette';
    e.periLabel.textContent = kind === 'geodesic' ? ' Periapsis markers' : ' Perihelion trail (Mercury)';
    if (view.setPeriVisible) view.setPeriVisible(e.peri.checked);

    // defaults per scenario
    const frameRadio = document.querySelector('input[name="frame"][value="bary"]');
    if (frameRadio) frameRadio.checked = true;
    const modelRadio = document.querySelector(`input[name="model"][value="${view.gr ? 'gr' : 'newton'}"]`);
    if (modelRadio) modelRadio.checked = true;
    e.amp.value = '0';
    if (view.setAmplification) view.setAmplification(1);
    this.syncAmp();

    if (nbody) {
      const span = view.spanDays;
      // rosette: the full path (faint) shows the rosette, a two-orbit bright trail the current
      // ellipse; a full-span bright trail would paint one thick band
      const period = meta.orbit && typeof meta.orbit.period_days === 'number' ? meta.orbit.period_days : 88;
      this.trailDays = id === 'rosette' ? Math.min(span, 2 * period) : Math.min(span, id === 'earth_moon' ? 60 : 365);
      if (view.setPathOpacity) view.setPathOpacity(id === 'rosette' ? 0.35 : 0.28);
      e.trail.value = String(this.trailFraction(this.trailDays, span));
    } else if (kind === 'geodesic') {
      e.trail.value = '0.87';
    }
    this.applyTrail();
    if (kind === 'rays') e.trailOut.textContent = '—';
    e.paths.checked = true;
    e.labels.checked = true;
    e.grid.checked = true;
    view.setPathsVisible(true);
    view.setLabelsVisible(true);
    view.setGridVisible(true);

    this.app.speed = 1;
    e.speed.value = '0';
    this.syncSpeed();
    this.syncPlay();
    this.syncTime();
    this.renderInfo(view, kind, entry, payload);
  }

  // trail slider maps [0, 1] -> [min, span] days on a log scale
  trailFraction(days, span) {
    const lo = Math.log(Math.max(span / 1e4, 1e-6));
    const hi = Math.log(span);
    return Math.min(Math.max((Math.log(days) - lo) / (hi - lo), 0), 1);
  }

  applyTrail() {
    if (!this.view || this.kind === 'rays') return;
    const span = this.view.span;
    const lo = Math.log(Math.max(span / 1e4, 1e-6));
    const hi = Math.log(span);
    const days = Math.exp(lo + (hi - lo) * Number(this.el.trail.value));
    this.trailDays = days;
    this.view.setTrailDays(days);
    if (this.kind === 'nbody') this.el.trailOut.textContent = fmtDays(days);
    else if (this.view.time) this.el.trailOut.textContent = `${fmtNum(Number(days.toPrecision(3)))} GM/c³`;
    else this.el.trailOut.textContent = `${Math.round(days)} samples`;
  }

  setFrame(frame) {
    const r = document.querySelector(`input[name="frame"][value="${frame}"]`);
    if (r) r.checked = true;
    if (this.view && this.view.setFrame) this.view.setFrame(frame);
  }

  syncPlay() {
    this.el.play.textContent = this.app.playing ? 'Pause' : 'Play';
  }

  syncSpeed() {
    const s = this.app.speed;
    let rate = '';
    if (this.view && this.kind === 'nbody') rate = ` · ${fmtDays(this.view.span * s / 60)}/s`;
    this.el.speedOut.textContent = `×${s >= 10 ? s.toFixed(0) : s.toFixed(2)}${rate}`;
  }

  syncTime() {
    const e = this.el;
    e.time.value = String(Math.round(this.app.frac * Number(e.time.max)));
  }

  syncAmp() {
    this.el.ampOut.textContent = fmtAmp(Math.pow(10, Number(this.el.amp.value)));
  }

  /** Per-frame readouts (time, Mercury displacement); the slider follows while playing. */
  syncReadouts() {
    const v = this.view;
    if (!v) return;
    this.el.timeOut.textContent = v.timeReadout(this.app.frac);
    if (this.kind === 'nbody' && v.gr && !this.el.groupModel.hidden) {
      const d = v.mercuryDisplacement(v.currentIndex ?? 0);
      if (d) {
        const name = v.names[v.mercIdx];
        this.el.ampReadout.textContent = this.el.ampField.hidden
          ? `${name}: GR − Newton = ${fmtKm(d.trueAU * AU_KM)}`
          : `${name}: GR − Newton = ${fmtKm(d.trueAU * AU_KM)}, shown × k = ${fmtKm(d.shownAU * AU_KM)}`;
      }
    }
    if (this.app.playing) this.syncTime();
  }

  renderInfo(view, kind, entry, payload) {
    const e = this.el;
    const meta = payload.meta;
    e.infoTitle.textContent = entry.title || meta.title || entry.id;
    e.infoBlurb.textContent = entry.blurb || meta.blurb || '';
    e.infoExplain.textContent = EXPLANATIONS[entry.id] || GENERIC_EXPLANATION;

    // legend (rays)
    e.legendBody.innerHTML = '';
    if (kind === 'rays') {
      for (const row of view.legend()) {
        const tr = document.createElement('tr');
        const sw = document.createElement('td');
        sw.className = 'sw';
        const s = document.createElement('span');
        s.className = 'swatch';
        s.style.background = row.color;
        sw.appendChild(s);
        const tb = document.createElement('td');
        tb.textContent = row.b === null ? row.name : `b = ${fmtNum(row.b)} ${view.unit}`;
        const td = document.createElement('td');
        td.textContent = row.deflection === null ? '' : `δ = ${fmtNum(row.deflection)}${row.unit}`;
        tr.append(sw, tb, td);
        e.legendBody.appendChild(tr);
      }
    }
    e.legendSec.hidden = kind !== 'rays';

    // numbers and notes from meta: headline keys first, the rest under "Run details"
    e.numbers.innerHTML = '';
    e.numbersMore.innerHTML = '';
    e.notes.innerHTML = '';
    const notes = [];
    const addRow = (dl, key, value, sub = false) => {
      const dt = document.createElement('dt');
      dt.textContent = humanKey(key);
      dt.title = key;
      if (sub) dt.className = 'sub';
      const dd = document.createElement('dd');
      dd.textContent = value;
      dl.append(dt, dd);
    };
    const addGroup = (dl, path) => {
      const dt = document.createElement('dt');
      dt.className = 'group-row';
      dt.textContent = path.map(humanKey).join(' · ');
      dl.append(dt);
    };
    // nested dicts become a group header (full path) followed by their leaves (short keys)
    const walk = (dl, obj, path) => {
      const nested = [];
      for (const key of Object.keys(obj)) {
        const v = obj[key];
        if (v === null || v === undefined) continue;
        const sub = path.length > 0;
        const label = [...path, key].join('.');
        if (typeof v === 'number') addRow(dl, key, fmtNum(v), sub);
        else if (typeof v === 'boolean') addRow(dl, key, String(v), sub);
        else if (typeof v === 'string') notes.push([label, v]);
        else if (isNumberList(v)) addRow(dl, key, v.map(fmtNum).join(', '), sub);
        else if (Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === 'string')) {
          v.forEach((x, i) => notes.push([v.length > 1 ? `${label} ${i + 1}` : label, x]));
        } else if (isPlainObject(v) && path.length < 3) nested.push([key, v]);
      }
      for (const [key, v] of nested) {
        const hasLeaf = Object.values(v).some((x) => typeof x === 'number' || typeof x === 'boolean' || isNumberList(x));
        if (hasLeaf) addGroup(dl, [...path, key]);
        walk(dl, v, [...path, key]);
      }
    };
    const used = view.usedMetaKeys || new Set();
    const shown = (key) => !META_RESERVED.has(key) && !used.has(key) && meta[key] !== null && meta[key] !== undefined;
    const head = {};
    const rest = {};
    for (const key of HEADLINE_KEYS) if (key in meta && shown(key)) head[key] = meta[key];
    for (const key of Object.keys(meta)) if (!(key in head) && shown(key)) rest[key] = meta[key];
    walk(e.numbers, head, []);
    walk(e.numbersMore, rest, []);
    const nSamples = view.T ?? meta.n_samples;
    if (typeof nSamples === 'number') addRow(e.numbersMore, 'samples', String(nSamples));
    e.numbersDetails.hidden = e.numbersMore.childElementCount === 0;
    e.numbersDetails.open = e.numbers.childElementCount === 0;
    if (typeof meta.frame === 'string') notes.unshift(['frame', meta.frame]);
    if (meta.units) notes.unshift(['units', unitsText(meta)]);
    for (const [key, text] of notes) {
      const p = document.createElement('p');
      const k = document.createElement('span');
      k.className = 'key';
      k.textContent = `${key} · `;
      p.appendChild(k);
      p.appendChild(document.createTextNode(text));
      e.notes.appendChild(p);
    }
    e.notesSec.hidden = notes.length === 0;

    // parameters (from the manifest entry)
    e.params.innerHTML = '';
    const params = entry.params;
    if (isPlainObject(params) && Object.keys(params).length > 0) {
      for (const [key, spec] of Object.entries(params)) {
        const dt = document.createElement('dt');
        dt.textContent = key;
        if (isPlainObject(spec) && typeof spec.description === 'string') dt.title = spec.description;
        const dd = document.createElement('dd');
        if (isPlainObject(spec)) {
          const def = spec.default ?? spec.value;
          const range = spec.min !== undefined && spec.max !== undefined ? ` [${fmtNum(spec.min)}, ${fmtNum(spec.max)}]` : '';
          dd.textContent = `${def === undefined ? '' : typeof def === 'number' ? fmtNum(def) : String(def)}${range}`;
        } else {
          dd.textContent = typeof spec === 'number' ? fmtNum(spec) : String(spec);
        }
        e.params.append(dt, dd);
      }
      e.paramsSec.hidden = false;
    } else if (Array.isArray(params) && params.length > 0) {
      for (const spec of params) {
        if (!isPlainObject(spec)) continue;
        const dt = document.createElement('dt');
        dt.textContent = String(spec.name ?? spec.id ?? '?');
        const dd = document.createElement('dd');
        const def = spec.default ?? spec.value;
        dd.textContent = def === undefined ? '' : typeof def === 'number' ? fmtNum(def) : String(def);
        e.params.append(dt, dd);
      }
      e.paramsSec.hidden = false;
    } else {
      e.paramsSec.hidden = true;
    }

    // array index
    e.arrays.innerHTML = '';
    const docs = isPlainObject(meta.arrays) ? meta.arrays : {};
    for (const entryIdx of payload.index) {
      const dt = document.createElement('dt');
      dt.textContent = entryIdx.name;
      const raw = docs[entryIdx.name];
      const doc = typeof raw === 'string' ? raw : isPlainObject(raw) && typeof raw.description === 'string' ? raw.description : null;
      if (doc) dt.title = doc;
      const dd = document.createElement('dd');
      dd.textContent = `${entryIdx.dtype} (${entryIdx.shape.join(', ')})`;
      e.arrays.append(dt, dd);
    }
    e.arraysSec.hidden = payload.index.length === 0;
  }
}

// ----------------------------------------------------------------------------- boot
const app = new App();
window.__gravsim = {
  ready: false,
  get scenario() { return app.scenario ?? null; },
  version: THREE_VERSION,
  loadScenario: (id) => app.loadScenario(id),
  setTime: (frac) => app.setTime(frac),
  setAmplification: (k) => {
    // a no-op where the slider is hidden (rosette, black hole, light rays), as in the UI
    if (!app.view || !app.view.setAmplification || app.ui.el.ampField.hidden) return;
    app.view.setAmplification(k);
    app.ui.el.amp.value = String(Math.log10(Math.max(Number(k) || 1, 1)));
    app.ui.syncAmp();
  },
  play: () => app.play(),
  pause: () => app.pause(),
  stats: app.stats,
  app,
};

app.start().catch((err) => app.fail('Start-up failed', err));
