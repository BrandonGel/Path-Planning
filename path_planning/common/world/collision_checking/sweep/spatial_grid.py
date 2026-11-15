"""Uniform grid over axis-aligned boxes (2D or 3D), built and queried in numba.

``BoxGrid(lo, hi)`` bins each item's box (a point when ``lo == hi``) into the cells it covers.
``query(qlo, qhi)`` returns every ``(query, item)`` pair whose boxes overlap, for many query boxes in
parallel. Each pair is reported once, from the cell holding the lower corner of the two boxes'
intersection. It is a candidate filter for the exact tests in ``disc_sweep`` / ``polytope_sweep``.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _cell(x, origin, inv_h, n):
    c = int(math.floor((x - origin) * inv_h))
    return 0 if c < 0 else (n - 1 if c >= n else c)


@njit(cache=True)
def _build(lo, hi, origin, inv_h, dims):
    """CSR of items per cell: ``(cell_start, entries)``."""
    m = lo.shape[0]
    nx, ny, nz = dims[0], dims[1], dims[2]
    counts = np.zeros(nx * ny * nz + 1, dtype=np.int64)
    for j in range(m):
        x0, x1 = _cell(lo[j, 0], origin[0], inv_h, nx), _cell(hi[j, 0], origin[0], inv_h, nx)
        y0, y1 = _cell(lo[j, 1], origin[1], inv_h, ny), _cell(hi[j, 1], origin[1], inv_h, ny)
        z0, z1 = _cell(lo[j, 2], origin[2], inv_h, nz), _cell(hi[j, 2], origin[2], inv_h, nz)
        for z in range(z0, z1 + 1):
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    counts[x + nx * (y + ny * z) + 1] += 1
    cell_start = np.cumsum(counts)
    fill = cell_start[:-1].copy()
    entries = np.empty(cell_start[-1], dtype=np.int64)
    for j in range(m):
        x0, x1 = _cell(lo[j, 0], origin[0], inv_h, nx), _cell(hi[j, 0], origin[0], inv_h, nx)
        y0, y1 = _cell(lo[j, 1], origin[1], inv_h, ny), _cell(hi[j, 1], origin[1], inv_h, ny)
        z0, z1 = _cell(lo[j, 2], origin[2], inv_h, nz), _cell(hi[j, 2], origin[2], inv_h, nz)
        for z in range(z0, z1 + 1):
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    c = x + nx * (y + ny * z)
                    entries[fill[c]] = j
                    fill[c] += 1
    return cell_start, entries


@njit(cache=True, inline="always")
def _visit(q, qlo, qhi, lo, hi, origin, inv_h, dims, cell_start, entries, out, pos):
    """Items overlapping query box ``q``, each once: written to ``out[pos:]`` when ``out`` is non-empty
    (fill pass), only counted otherwise. Returns the count."""
    nx, ny, nz = dims[0], dims[1], dims[2]
    x0, x1 = _cell(qlo[q, 0], origin[0], inv_h, nx), _cell(qhi[q, 0], origin[0], inv_h, nx)
    y0, y1 = _cell(qlo[q, 1], origin[1], inv_h, ny), _cell(qhi[q, 1], origin[1], inv_h, ny)
    z0, z1 = _cell(qlo[q, 2], origin[2], inv_h, nz), _cell(qhi[q, 2], origin[2], inv_h, nz)
    k = 0
    write = out.shape[0] > 0
    for z in range(z0, z1 + 1):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                c = x + nx * (y + ny * z)
                for e in range(cell_start[c], cell_start[c + 1]):
                    j = entries[e]
                    if (lo[j, 0] > qhi[q, 0] or hi[j, 0] < qlo[q, 0] or lo[j, 1] > qhi[q, 1] or hi[j, 1] < qlo[q, 1]
                            or lo[j, 2] > qhi[q, 2] or hi[j, 2] < qlo[q, 2]):
                        continue
                    # Report from the cell holding the intersection's lower corner only.
                    if (_cell(max(lo[j, 0], qlo[q, 0]), origin[0], inv_h, nx) != x
                            or _cell(max(lo[j, 1], qlo[q, 1]), origin[1], inv_h, ny) != y
                            or _cell(max(lo[j, 2], qlo[q, 2]), origin[2], inv_h, nz) != z):
                        continue
                    if write:
                        out[pos + k] = j
                    k += 1
    return k


@njit(cache=True, parallel=True)
def _query(qlo, qhi, lo, hi, origin, inv_h, dims, cell_start, entries):
    nq = qlo.shape[0]
    cnt = np.empty(nq, dtype=np.int64)
    none = np.empty(0, dtype=np.int64)
    for q in prange(nq):
        cnt[q] = _visit(q, qlo, qhi, lo, hi, origin, inv_h, dims, cell_start, entries, none, 0)
    offs = np.empty(nq + 1, dtype=np.int64)
    offs[0] = 0
    for q in range(nq):
        offs[q + 1] = offs[q] + cnt[q]
    rows = np.empty(offs[nq], dtype=np.int64)
    ids = np.empty(offs[nq], dtype=np.int64)
    for q in prange(nq):
        _visit(q, qlo, qhi, lo, hi, origin, inv_h, dims, cell_start, entries, ids, offs[q])
        for k in range(offs[q], offs[q + 1]):
            rows[k] = q
    return rows, ids


def _pad3(x):
    x = np.asarray(x, dtype=float)
    if x.shape[1] == 3:
        return np.ascontiguousarray(x)
    return np.ascontiguousarray(np.column_stack([x, np.zeros((len(x), 3 - x.shape[1]))]))


class BoxGrid:
    """Uniform grid over the boxes ``[lo[j], hi[j]]`` (``(m, D)``, D = 2 or 3). ``cell_size`` defaults to
    the median of the boxes' largest side (points: about two per cell), and is enlarged so there are at most
    ``max_cells_per_item`` cells per item."""

    def __init__(self, lo, hi=None, cell_size=None, max_cells_per_item: float = 4.0):
        lo = np.asarray(lo, dtype=float)
        hi = lo if hi is None else np.asarray(hi, dtype=float)
        if lo.ndim != 2 or lo.shape[1] not in (2, 3) or lo.shape != hi.shape:
            raise ValueError(f"BoxGrid: lo / hi must be matching (m, 2) or (m, 3) arrays, got {lo.shape}, {hi.shape}")
        m, D = lo.shape
        self.dim = D
        self.lo, self.hi = _pad3(lo), _pad3(hi)
        gmin = self.lo.min(axis=0) if m else np.zeros(3)
        gmax = self.hi.max(axis=0) if m else np.zeros(3)
        ext = np.maximum(gmax - gmin, 0.0)[:D]
        if cell_size is None:
            side = (hi - lo).max(axis=1) if m else np.zeros(1)
            cell_size = float(np.median(side)) if m and np.median(side) > 0 else 0.0
            if cell_size <= 0.0:                                 # points: about two per cell
                vol = float(np.prod(np.maximum(ext, 1e-12)))
                cell_size = (2.0 * vol / max(m, 1)) ** (1.0 / D)
        h = max(float(cell_size), 1e-12)
        cap = max_cells_per_item * max(m, 1) + 64
        while np.prod(np.floor(ext / h) + 1) > cap:
            h *= 1.25
        dims = np.ones(3, dtype=np.int64)
        dims[:D] = (np.floor(ext / h) + 1).astype(np.int64)
        self.cell_size, self.dims, self.origin = h, dims, gmin
        self.cell_start, self.entries = _build(self.lo, self.hi, self.origin, 1.0 / h, self.dims)

    @property
    def nbytes(self) -> int:
        """Memory held by the grid's arrays (boxes, cell offsets, entries)."""
        return int(self.lo.nbytes + self.hi.nbytes + self.cell_start.nbytes + self.entries.nbytes)

    def query(self, qlo, qhi):
        """``(rows, ids)``: every query box ``rows[k]`` of ``[qlo, qhi]`` (``(n, D)``) overlapping item
        ``ids[k]``, grouped by query, each pair once."""
        if len(qlo) == 0:
            z = np.empty(0, dtype=np.int64)
            return z, z
        return _query(_pad3(qlo), _pad3(qhi), self.lo, self.hi, self.origin, 1.0 / self.cell_size, self.dims,
                      self.cell_start, self.entries)
