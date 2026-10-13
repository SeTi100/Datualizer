"""Fast line curves for large series: peak downsampling, clip to view, gap breaks.

Drawing every sample on every repaint makes panning and the crosshair lag with large data.
`FastCurve` draws only the visible part and reduces it to a min/max pair per screen column
(peak downsampling), which keeps spikes visible. It never merges samples across a line break
(missing value, time gap): pyqtgraph's own peak downsampling forms blocks across a break and
then draws a line over a logger dropout. Plotting only; the dataset is never changed.

Measured costs per frame with 8 subplots (offscreen, software raster): lines wider than 1 px
cost about 3x, a grid over minor ticks adds about 50 %, and a pixmap cache that is too small
for all plots makes Qt redraw everything on every crosshair move.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor, QGuiApplication, QPen, QPixmapCache
from PySide6.QtWidgets import QGraphicsItem

MOUSE_RATE_LIMIT_HZ = 60
CACHED_LAYERS = 4  # pixmaps per screen area: curves, axes with grid, spare
MIN_PIXMAP_CACHE_KB = 64 * 1024
DEFAULT_COLUMNS = 2000  # before the view has a size


def curve_pen(color: str | QColor) -> QPen:
    """1 px cosmetic pen: Qt strokes it on a fast path, wider lines cost about three times as much."""
    pen = pg.mkPen(color, width=1)
    pen.setCosmetic(True)
    return pen


def light_grid(plot: pg.PlotItem, alpha: float) -> None:
    """Grid lines at the labelled ticks only; lines at every minor tick cost much more to draw."""
    plot.showGrid(x=True, y=True, alpha=alpha)
    for name in ("left", "bottom"):
        plot.getAxis(name).setStyle(maxTickLevel=0)


def ensure_pixmap_cache() -> int:
    """Raise Qt's pixmap cache so cached curves and axes of all subplots fit (limit in KB).

    Qt's default of 10 MB holds the pixmaps of only a few subplots; beyond that the cache
    evicts on every frame and the whole view is redrawn on each crosshair move.
    """
    app = QGuiApplication.instance()
    screens = app.screens() if app is not None else []
    pixels = sum(
        sc.geometry().width() * sc.geometry().height() * sc.devicePixelRatio() ** 2 for sc in screens
    )
    needed_kb = int(pixels * 4 * CACHED_LAYERS / 1024)
    limit = max(QPixmapCache.cacheLimit(), MIN_PIXMAP_CACHE_KB, needed_kb)
    QPixmapCache.setCacheLimit(limit)
    return limit


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


def peak_lod(
    x: np.ndarray,
    y: np.ndarray,
    connect: np.ndarray,
    x_range: tuple[float, float] | None,
    columns: int,
    x_sorted: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What a line can show of (x, y, connect) in `x_range` on `columns` screen columns.

    Clips to the range (plus one sample each side, so the line reaches the edges) and, if more
    than a few samples fall on one column, keeps the max and min per column (drawn as a vertical
    stroke). Two columns are joined only if the last sample of one is joined to the first sample
    of the next, so a dropout or time gap wider than a column is never bridged by a line.
    Unsorted x is returned unchanged (neither clipping nor columns are defined then).
    """
    n = len(x)
    if n == 0 or not x_sorted:
        return x, y, connect
    i0, i1 = 0, n
    if x_range is not None:
        i0 = max(int(np.searchsorted(x, x_range[0], side="left")) - 1, 0)
        i1 = min(int(np.searchsorted(x, x_range[1], side="right")) + 1, n)
    xs, ys, cs = x[i0:i1], y[i0:i1], connect[i0:i1].copy()
    if len(cs):
        cs[-1] = False
    m = len(xs)
    columns = max(int(columns), 1)
    if m <= 4 * columns:
        return xs, ys, cs
    span = float(xs[-1] - xs[0]) or 1.0
    col = ((xs - xs[0]) * (columns / span)).astype(np.int64)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(col) != 0) + 1))
    y_max = np.maximum.reduceat(ys, starts)
    y_min = np.minimum.reduceat(ys, starts)
    out_x = np.repeat(xs[starts], 2)
    out_y = np.column_stack((y_max, y_min)).ravel()
    out_c = np.ones(len(out_x), dtype=bool)
    # a column joins the next only if its last sample joins the next column's first sample
    out_c[1::2] = np.append(cs[starts[1:] - 1], False)
    return out_x, out_y, out_c


class FastCurve(pg.PlotCurveItem):
    """Line curve that recomputes what it draws when the view range or size changes."""

    def __init__(self, x: np.ndarray, y: np.ndarray, gap_rows: np.ndarray, pen) -> None:
        super().__init__(pen=pen, antialias=True)
        self._full = plot_arrays(x, y, gap_rows)
        self._lod_key: tuple | None = None
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self._update_lod(force=True)

    def getOriginalDataset(self) -> tuple[np.ndarray, np.ndarray]:
        """All finite samples (the drawn data is reduced to the screen)."""
        return self._full[0], self._full[1]

    def viewRangeChanged(self) -> None:
        super().viewRangeChanged()
        self._update_lod()

    def viewTransformChanged(self) -> None:
        super().viewTransformChanged()
        self._update_lod()  # also called on resize, when the number of columns changes

    def _update_lod(self, force: bool = False) -> None:
        x, y, connect, x_sorted = self._full
        vb = self.getViewBox()
        x_range, columns = None, DEFAULT_COLUMNS
        if vb is not None and hasattr(vb, "viewRange") and hasattr(vb, "autoRangeEnabled"):
            if not vb.autoRangeEnabled()[0]:
                x_range = tuple(vb.viewRange()[0])
            if vb.width() > 1:
                columns = int(vb.width())
        key = (x_range, columns)
        if not force and key == self._lod_key:
            return
        self._lod_key = key
        xs, ys, cs = peak_lod(x, y, connect, x_range, columns, x_sorted)
        self.setData(xs, ys, connect=cs)


def add_curve(plot: pg.PlotItem, x: np.ndarray, y: np.ndarray, gap_rows: np.ndarray, pen) -> FastCurve:
    """Add a line curve that stays fast for millions of points."""
    curve = FastCurve(x, y, gap_rows, pen)
    plot.addItem(curve)
    cache_static_parts(plot, curve)
    return curve


def cache_static_parts(plot: pg.PlotItem, curve: pg.GraphicsObject) -> None:
    """Keep curve and axes (with grid) as pixmaps, so moving the crosshair only blits them.

    The cache is redrawn whenever the item changes (new data, pan, zoom, resize).
    """
    cache = QGraphicsItem.CacheMode.DeviceCoordinateCache
    curve.setCacheMode(cache)
    for name in ("left", "bottom"):
        plot.getAxis(name).setCacheMode(cache)


def rate_limited_mouse(scene, slot) -> pg.SignalProxy:
    """Deliver scene mouse moves at most MOUSE_RATE_LIMIT_HZ times per second (latest wins).

    Keep the returned proxy referenced, otherwise the connection is dropped.
    """
    return pg.SignalProxy(scene.sigMouseMoved, rateLimit=MOUSE_RATE_LIMIT_HZ, slot=lambda args: slot(args[0]))
