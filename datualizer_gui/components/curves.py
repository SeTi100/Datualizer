"""Fast line curves for large series: peak downsampling, clip to view, gap breaks.

Drawing every sample on every repaint makes panning and the crosshair lag with large data.
pyqtgraph can draw only the visible part (clip to view) and reduce it to a min/max pair per
screen column (peak downsampling), which keeps spikes and dropouts visible. Plotting only;
the dataset is never changed.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QGraphicsItem

MOUSE_RATE_LIMIT_HZ = 60


def _version(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


# Before 0.14, pyqtgraph does not clip or downsample a connect array together with the data,
# so line breaks would land in the wrong place; such an install draws every point (slow but right).
FAST_CURVES = _version(pg.__version__) >= (0, 14)


def connect_mask(y: np.ndarray, gap_rows: np.ndarray) -> np.ndarray:
    """Per row: join sample i to i+1 only if both are finite and no time gap starts at i+1."""
    ok = np.isfinite(np.asarray(y, dtype=np.float64))
    n = len(ok)
    connect = np.zeros(n, dtype=bool)
    if n > 1:
        connect[:-1] = ok[:-1] & ok[1:]
    gaps = np.asarray(gap_rows, dtype=np.int64)
    gaps = gaps[(gaps > 0) & (gaps < n)]
    connect[gaps - 1] = False
    return connect


def plot_arrays(x: np.ndarray, y: np.ndarray, gap_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Arrays ready for a fast curve: (x, y, connect, x_sorted), finite points only.

    Rows without a finite time or value are left out and the line is broken there (as at a
    time gap). Peak downsampling takes max/min per block, and a single NaN would blank a
    whole block, so scattered dropouts could hide a channel in the overview.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    keep = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    connect = np.zeros(len(keep), dtype=bool)
    if len(keep) > 1:
        nxt = keep[1:]
        gap_start = np.zeros(n, dtype=bool)
        gaps = np.asarray(gap_rows, dtype=np.int64)
        gap_start[gaps[(gaps >= 0) & (gaps < n)]] = True
        connect[:-1] = (nxt == keep[:-1] + 1) & ~gap_start[nxt]
    xs, ys = x[keep], y[keep]
    x_sorted = bool(len(xs) < 2 or np.all(xs[1:] >= xs[:-1]))
    return xs, ys, connect, x_sorted


def add_curve(plot: pg.PlotItem, x: np.ndarray, y: np.ndarray, gap_rows: np.ndarray, pen) -> pg.PlotDataItem:
    """Add a line curve that stays fast for millions of points."""
    x, y, connect, x_sorted = plot_arrays(x, y, gap_rows)
    curve = plot.plot(x, y, pen=pen, connect=connect)
    cache_static_parts(plot, curve)
    if not FAST_CURVES:
        return curve
    # One min/max pair per screen column is all a line can show (pyqtgraph default: 5 per pixel)
    curve.opts["autoDownsampleFactor"] = 1.0
    # Set after adding: pyqtgraph fails if clipToView is given while the item joins the scene.
    curve.setDownsampling(auto=True, method="peak")
    # Clip to view needs increasing x (binary search); otherwise everything is drawn.
    curve.setClipToView(x_sorted)
    return curve


def cache_static_parts(plot: pg.PlotItem, curve: pg.PlotDataItem) -> None:
    """Keep curve and axes (with grid) as pixmaps, so moving the crosshair only blits them.

    The cache is redrawn whenever the item changes (new data, pan, zoom, resize).
    """
    cache = QGraphicsItem.CacheMode.DeviceCoordinateCache
    curve.curve.setCacheMode(cache)
    for name in ("left", "bottom"):
        plot.getAxis(name).setCacheMode(cache)


def rate_limited_mouse(scene, slot) -> pg.SignalProxy:
    """Deliver scene mouse moves at most MOUSE_RATE_LIMIT_HZ times per second (latest wins).

    Keep the returned proxy referenced, otherwise the connection is dropped.
    """
    return pg.SignalProxy(scene.sigMouseMoved, rateLimit=MOUSE_RATE_LIMIT_HZ, slot=lambda args: slot(args[0]))
