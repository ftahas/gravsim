"""Binary payload of a web scenario: one JSON header followed by raw little-endian arrays.

Layout (``encode``), every multi-byte integer little-endian::

    bytes [0, 4)            uint32  H        length in bytes of the JSON header
    bytes [4, 4 + H)        UTF-8 JSON       {"version": 1, "meta": {...}, "index": [...]}
    bytes [4 + H, ...)      array block      the arrays of ``index``, concatenated in order

``index`` is a list of ``{"name", "dtype", "shape", "offset", "nbytes"}`` with ``dtype`` either
``"float32"`` or ``"int32"``, ``shape`` a list of ints (C order) and ``offset`` the byte position
of the array *relative to the start of the array block*, i.e. its absolute position in the
payload is ``4 + H + offset``.  The JSON text is padded with trailing spaces so that ``4 + H`` is
a multiple of 4; since every element is 4 bytes wide, every array therefore starts at an absolute
offset that is a multiple of 4 and a ``Float32Array``/``Int32Array`` view can be created directly
on the ``ArrayBuffer`` without copying.  The JS loader mirrors ``decode``::

    const H = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, H)));
    for (const a of header.index) {
      const n = a.nbytes / 4;
      const ctor = a.dtype === "int32" ? Int32Array : Float32Array;
      arrays[a.name] = new ctor(buf, 4 + H + a.offset, n);   // reshape with a.shape
    }

``meta`` is any JSON-serialisable dict (numpy scalars and arrays are converted); the scenario
builders put the display numbers and the array documentation there (``meta["arrays"]``).
Float arrays are stored as float32 and integer arrays as int32, which is what the browser
consumes; other dtypes are rejected rather than silently converted.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

__all__ = ["VERSION", "encode", "decode"]

VERSION = 1
_DTYPES = {"float32": np.dtype("<f4"), "int32": np.dtype("<i4")}
_ALIGN = 4


def _jsonable(obj: Any) -> Any:
    """Fallback for ``json.dumps``: numpy scalars/arrays and other array-likes become plain JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"meta value of type {type(obj).__name__} is not JSON-serialisable")


def _storage_dtype(arr: np.ndarray, name: str) -> str:
    kind = arr.dtype.kind
    if kind == "f":
        return "float32"
    if kind in "iu":
        return "int32"
    raise TypeError(
        f"array {name!r} has dtype {arr.dtype}; only float (-> float32) and integer "
        "(-> int32) arrays can be encoded"
    )


def encode(payload: dict) -> bytes:
    """Serialise ``{"meta": dict, "arrays": {name: ndarray}}`` to the binary format above.

    Parameters
    ----------
    payload : dict
        ``"meta"`` (JSON-serialisable; missing means ``{}``) and ``"arrays"`` (mapping of
        name to array-like; missing means no arrays).  Float arrays are cast to float32 and
        integer arrays to int32; other dtypes raise ``TypeError``.  Insertion order is kept.

    Returns
    -------
    bytes
        The payload; ``decode`` inverts it exactly (up to the float32/int32 casts).
    """
    meta = payload.get("meta", {}) if payload else {}
    arrays = payload.get("arrays", {}) if payload else {}
    if not isinstance(meta, dict):
        raise TypeError("payload['meta'] must be a dict")
    index: list[dict] = []
    blocks: list[bytes] = []
    offset = 0
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("array names must be non-empty strings")
        arr = np.asarray(value)
        arr = np.ascontiguousarray(arr.astype(_DTYPES[_storage_dtype(arr, name)], copy=False))
        raw = arr.tobytes(order="C")
        index.append(
            {
                "name": name,
                "dtype": _storage_dtype(arr, name),
                "shape": [int(s) for s in arr.shape],
                "offset": offset,
                "nbytes": len(raw),
            }
        )
        blocks.append(raw)
        offset += len(raw)
    header = json.dumps(
        {"version": VERSION, "meta": meta, "index": index},
        ensure_ascii=False,
        separators=(",", ":"),
        default=_jsonable,
    ).encode("utf-8")
    pad = (-(4 + len(header))) % _ALIGN
    header += b" " * pad
    return struct.pack("<I", len(header)) + header + b"".join(blocks)


def decode(data: bytes | bytearray | memoryview) -> dict:
    """Parse a payload produced by ``encode``.

    Parameters
    ----------
    data : bytes-like
        The whole payload.

    Returns
    -------
    dict
        ``{"meta": dict, "arrays": {name: ndarray}}``; arrays are fresh, C-contiguous,
        native-endian float32 or int32 arrays with the recorded shapes.

    Raises
    ------
    ValueError
        If the bytes are truncated, the header is not valid JSON of the expected form, or an
        index entry is inconsistent (unknown dtype, size mismatch, out of range).
    """
    buf = memoryview(data).cast("B")
    if len(buf) < 4:
        raise ValueError("payload shorter than the 4-byte header length")
    (header_len,) = struct.unpack("<I", buf[:4])
    if 4 + header_len > len(buf):
        raise ValueError(f"header length {header_len} exceeds the payload size {len(buf)}")
    try:
        header = json.loads(bytes(buf[4 : 4 + header_len]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"payload header is not UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict) or "index" not in header:
        raise ValueError("payload header must be a JSON object with an 'index'")
    meta = header.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("payload header 'meta' must be a JSON object")
    block = buf[4 + header_len :]
    arrays: dict[str, np.ndarray] = {}
    for entry in header["index"]:
        try:
            name = entry["name"]
            dtype = _DTYPES[entry["dtype"]]
            shape = tuple(int(s) for s in entry["shape"])
            offset = int(entry["offset"])
            nbytes = int(entry["nbytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed index entry {entry!r}: {exc}") from exc
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if nbytes != count * dtype.itemsize:
            raise ValueError(
                f"array {name!r}: nbytes {nbytes} does not match shape {shape} of {entry['dtype']}"
            )
        if offset < 0 or offset + nbytes > len(block):
            raise ValueError(
                f"array {name!r}: byte range [{offset}, {offset + nbytes}) out of bounds"
            )
        arr = np.frombuffer(block[offset : offset + nbytes], dtype=dtype, count=count)
        arrays[name] = np.ascontiguousarray(arr.reshape(shape).astype(dtype.newbyteorder("=")))
    return {"meta": meta, "arrays": arrays}
