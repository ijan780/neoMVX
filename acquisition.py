#!/usr/bin/env python3
"""
acquisition.py
==============
AcquisitionEngine: one unified run() method that combines multi-channel,
Z-stack, timelapse, and tiling based on which options are enabled.

Result dict structure
---------------------
  Single / Z-stack:
    { "mode":     "single",
      "czyx":     np.ndarray (C, Z, H, W),
      "channels": List[str] }

  Tiling:
    { "mode":       "tiled",
      "tile_data":  Dict[str, List[np.ndarray]],   # ch → [ZYX, …]
      "tile_pos":   List[Tuple[float, float]],
      "grid_shape": Tuple[int, int],
      "channels":   List[str],
      "overlap_x":  float,
      "overlap_y":  float }

  Timelapse finished sentinel:
    { "mode": "timelapse", "n_timepoints": int }

Timelapse memory strategy
--------------------------
  Each time-point: acquired → timepoint_ready emitted → widget saves &
  releases viewer layers → result dict goes out of scope → GC frees arrays.
  RAM stays bounded regardless of total time-points.

Interval units
--------------
  interval_min is in MINUTES (float). Sub-minute values are supported
  (e.g. 0.5 = 30 s).

Z-position direction
--------------------
  _z_positions() handles both ascending (start < end) and descending
  (start > end) ranges by signing the step automatically.  Passing
  start=-1273.70, end=-1400.60, step=1.0 correctly yields a descending
  list from -1273.70 to -1400.60.

Countdown signal
----------------
  countdown(str) is emitted once per second during the inter-timepoint
  wait.  Format: "Next acquisition in  MM:SS  (N of M complete)"
  An empty string "" is emitted immediately before each acquisition
  starts so the widget can clear the label.

Inter-timepoint BF rest
-----------------------
  After all channels are acquired for a time-point, the system switches
  to the brightfield channel during the countdown wait.
  The LED is always OFF during the wait.
  The LED is only turned on at the start of the next time-point if the
  BF channel is included in the selected acquisition channels.

Overlap auto-computation
------------------------
  TilingEngine.compute_grid() automatically determines the optimal
  overlap percentage for both X and Y axes independently based on the
  recorded top-left and bottom-right corner positions and the FOV size.

  Algorithm (per axis):
    span = center-to-center distance between corner tiles (µm)
    If span == 0: one tile, overlap = MIN_OVERLAP (10%)
    Otherwise find minimum n ≥ 2 such that overlap ≥ MIN_OVERLAP:
      n  = ceil(span / (fov * (1 - MIN_OVERLAP))) + 1   (minimum n)
      ov = 1 - span / ((n-1) * fov)

Tile ordering and stage coordinate convention
---------------------------------------------
  Stage coordinate system:
    +X  →  right of imaged area
    -X  →  left  of imaged area
    -Y  →  top   of imaged area
    +Y  →  bottom of imaged area

  Two corners are recorded:
    top_left     = (x_left,  y_top)    ← most -X, most -Y
    bottom_right = (x_right, y_bottom) ← most +X, most +Y

  Tile grid:
    Tile (0,        0       ) → (x_left,  y_top)    [top-left]
    Tile (0,        n_cols-1) → (x_right, y_top)    [top-right]
    Tile (n_rows-1, 0       ) → (x_left,  y_bottom) [bottom-left]
    Tile (n_rows-1, n_cols-1) → (x_right, y_bottom) [bottom-right]

  Col centers: evenly spaced from x_left  → x_right  (X increasing)
  Row centers: evenly spaced from y_top   → y_bottom (Y increasing)

Stitching canvas convention
---------------------------
  Stage X maps directly to canvas/pixel X:
    stage -X → canvas left  (smaller canvas X)
    stage +X → canvas right (larger  canvas X)

  Stage Y maps directly to canvas Y:
    stage -Y → canvas top    (smaller canvas Y)
    stage +Y → canvas bottom (larger  canvas Y)

  This guarantees:
    tile (row=0,        col=0       ) → canvas (0,     0    )  [top-left,     smallest]
    tile (row=0,        col=n_cols-1) → canvas (max_x, 0    )  [top-right]
    tile (row=n_rows-1, col=0       ) → canvas (0,     max_y)  [bottom-left]
    tile (row=n_rows-1, col=n_cols-1) → canvas (max_x, max_y)  [bottom-right, largest]

tile_ready signal
-----------------
  tile_ready(dict) is emitted immediately after each tile position is
  fully acquired (all channels + all Z-slices).  The widget exports the
  tile to disk immediately so RAM is not held for the full tiling run.

  Payload keys:
    "czyx"        : np.ndarray (C, Z, H, W)
    "row"         : int
    "col"         : int
    "tile_index"  : int  (row-major index)
    "channels"    : List[str]
    "x_um"        : float  (stage X, µm)
    "y_um"        : float  (stage Y, µm)
    "t_index"     : int or None

LED management during acquisition
----------------------------------
  BF channel   → LED on at led_current_ma (only if BF is selected)
  Fluo channel → LED always disabled before snap
  Between timepoints → LED always off regardless of channel

"""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from qtpy.QtCore import QObject, Signal

from config import HardwareConfig, ChannelConfig
from hardware import MMController, DCS100E

log = logging.getLogger(__name__)

MIN_OVERLAP = 0.10   # 10 % minimum overlap


# ══════════════════════════════════════════════════════════════════════════════
#  TILING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class TilingEngine:
    """
    Records two stage corners (top-left, bottom-right) and computes the
    full XY tile grid with automatic overlap ≥ MIN_OVERLAP per axis.

    Stage coordinate convention
    ---------------------------
      +X  →  right of imaged area  (x_right is most positive X)
      -X  →  left  of imaged area  (x_left  is most negative X)
      -Y  →  top   of imaged area  (y_top   is most negative Y)
      +Y  →  bottom of imaged area (y_bottom is most positive Y)

    Corner tile positions (exact, not approximated)
    -----------------------------------------------
      Tile (0,        0       ) acquired at (x_left,  y_top)
      Tile (0,        n_cols-1) acquired at (x_right, y_top)
      Tile (n_rows-1, 0       ) acquired at (x_left,  y_bottom)
      Tile (n_rows-1, n_cols-1) acquired at (x_right, y_bottom)

    Col centers: x_left → x_right  (n_cols evenly spaced, X increases)
    Row centers: y_top  → y_bottom (n_rows evenly spaced, Y increases)

    Overlap is computed independently per axis from the center-to-center
    span and FOV size.  Minimum overlap is MIN_OVERLAP (10%) on each axis.
    """

    def __init__(self, cfg: HardwareConfig):
        self.cfg          = cfg
        self.top_left:     Optional[Tuple[float, float]] = None
        self.bottom_right: Optional[Tuple[float, float]] = None
        self._n_rows      = 0
        self._n_cols      = 0
        self._overlap_x:  float = MIN_OVERLAP
        self._overlap_y:  float = MIN_OVERLAP

    # ── corner management ────────────────────────────────────────────────

    def set_top_left(self, x: float, y: float):
        """Record the top-left corner — stage position of tile (0, 0)."""
        self.top_left = (x, y)
        log.info("Top-left corner set: (%.2f, %.2f)", x, y)

    def set_bottom_right(self, x: float, y: float):
        """Record the bottom-right corner — stage position of tile (n-1, n-1)."""
        self.bottom_right = (x, y)
        log.info("Bottom-right corner set: (%.2f, %.2f)", x, y)

    def clear_corners(self):
        self.top_left     = None
        self.bottom_right = None
        self._n_rows      = 0
        self._n_cols      = 0
        self._overlap_x   = MIN_OVERLAP
        self._overlap_y   = MIN_OVERLAP

    @property
    def ready(self) -> bool:
        return self.top_left is not None and self.bottom_right is not None

    # ── overlap ───────────────────────────────────────────────────────────

    @property
    def overlap_x(self) -> float:
        return self._overlap_x

    @property
    def overlap_y(self) -> float:
        return self._overlap_y

    @property
    def computed_overlap(self) -> float:
        """Larger of the two per-axis overlaps, for display."""
        return max(self._overlap_x, self._overlap_y)

    @staticmethod
    def _auto_n(span: float, fov: float) -> Tuple[int, float]:
        """
        Return (n_tiles, overlap) for one axis.

        Parameters
        ----------
        span : center-to-center distance between the two corner tile
               positions on this axis (µm).  0 → single tile.
        fov  : field-of-view size on this axis (µm).

        Returns
        -------
        n_tiles : number of tiles needed (1 if span == 0, else ≥ 2)
        overlap : fractional overlap on this axis (≥ MIN_OVERLAP)
        """
        if span <= 0:
            return 1, MIN_OVERLAP

        if span <= fov * (1 - MIN_OVERLAP):
            # span fits in two tiles with ≥ MIN_OVERLAP
            n = 2
        else:
            n = math.ceil(span / (fov * (1 - MIN_OVERLAP))) + 1

        overlap = 1.0 - span / ((n - 1) * fov)
        overlap = max(overlap, MIN_OVERLAP)
        return n, overlap

    @staticmethod
    def _axis_centers(boundary_a: float,
                      boundary_b: float,
                      n: int,
                      fov: float) -> List[float]:
        """
        Return n evenly-spaced tile centers along one axis such that:
          - centers[0]   is exactly boundary_a  (the user-set corner tile)
          - centers[n-1] is exactly boundary_b  (the user-set corner tile)

        boundary_a / boundary_b are the recorded stage positions of the
        corner TILES themselves (centers), not the outer edge of the
        mosaic — the user physically moved the stage there and expects
        the tile to be acquired at exactly that position.

        Works for both ascending and descending boundary pairs.

        For n == 1: single center at the midpoint of the span.
        For n > 1 : evenly spaced from boundary_a to boundary_b inclusive.
        """
        if n == 1:
            return [(boundary_a + boundary_b) / 2.0]

        step = (boundary_b - boundary_a) / (n - 1)
        centers = [boundary_a + i * step for i in range(n)]
        return centers

    # ── grid computation ─────────────────────────────────────────────────

    def compute_grid(self) -> Tuple[List[Tuple[float, float]], Tuple[int, int]]:
        """
        Compute the full XY tile grid from the two recorded corners.

        Returns
        -------
        positions  : List of (x, y) stage positions in row-major order
                     positions[0] = tile (0,0) = top-left corner
        grid_shape : (n_rows, n_cols)
        """
        if not self.ready:
            raise RuntimeError("Both corners must be set before computing grid.")

        x_left,  y_top    = self.top_left
        x_right, y_bottom = self.bottom_right

        fov_x = self.cfg.image_width_px  * self.cfg.pixel_size_um
        fov_y = self.cfg.image_height_px * self.cfg.pixel_size_um

        # Span = center-to-center distance between corner tiles
        span_x = abs(x_right - x_left)    # X increases left→right
        span_y = abs(y_bottom - y_top)     # Y increases top→bottom

        n_cols, self._overlap_x = self._auto_n(span_x, fov_x)
        n_rows, self._overlap_y = self._auto_n(span_y, fov_y)

        self._n_rows = n_rows
        self._n_cols = n_cols

        # Col centers: x_left → x_right  (X increases col-by-col)
        col_centers = self._axis_centers(x_left,  x_right,  n_cols, fov_x)
        # Row centers: y_top  → y_bottom (Y increases row-by-row)
        row_centers = self._axis_centers(y_top,   y_bottom, n_rows, fov_y)

        # Build positions in row-major order
        positions: List[Tuple[float, float]] = []
        for row, y in enumerate(row_centers):
            for col, x in enumerate(col_centers):
                positions.append((x, y))

        return positions, (n_rows, n_cols)


# ══════════════════════════════════════════════════════════════════════════════
#  ACQUISITION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AcquisitionEngine(QObject):
    """
    Unified acquisition engine for all imaging modes.

    Signals
    -------
    progress(int)              : 0–100 percent complete
    result_ready(dict)         : emitted when a non-timelapse acquisition finishes
    timepoint_ready(dict, int) : emitted after each timelapse timepoint
    tile_ready(dict)           : emitted after each tile is acquired (see payload below)
    countdown(str)             : countdown string during timelapse wait (or "" to clear)
    finished()                 : always emitted when the run completes
    error(str)                 : emitted on exception

    tile_ready payload keys
    -----------------------
      "czyx"        : np.ndarray (C, Z, H, W)
      "row"         : int
      "col"         : int
      "tile_index"  : int   row-major index (row * n_cols + col)
      "channels"    : List[str]
      "x_um"        : float  stage X position (µm)
      "y_um"        : float  stage Y position (µm)
      "t_index"     : int or None
    """

    progress         = Signal(int)
    result_ready     = Signal(dict)
    timepoint_ready  = Signal(dict, int)
    tile_ready       = Signal(dict)
    countdown        = Signal(str)
    finished         = Signal()
    error            = Signal(str)

    def __init__(self,
                 cfg:    HardwareConfig,
                 mmc:    MMController,
                 led:    DCS100E,
                 tiling: TilingEngine):
        super().__init__()
        self.cfg    = cfg
        self.mmc    = mmc
        self.led    = led
        self.tiling = tiling

        self._stop_flag = threading.Event()
        self._thread:   Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────

    def run(self,
            channels:         List[ChannelConfig],
            enable_z:         bool  = False,
            z_start_um:       float = 0.0,
            z_end_um:         float = 0.0,
            z_step_um:        float = 1.0,
            enable_tiling:    bool  = False,
            enable_timelapse: bool  = False,
            n_timepoints:     int   = 1,
            interval_min:     float = 1.0):
        """Start acquisition in a background thread."""
        if self._thread and self._thread.is_alive():
            log.warning("Acquisition already running.")
            return

        self._stop_flag.clear()
        kwargs = dict(
            channels          = channels,
            enable_z          = enable_z,
            z_start_um        = z_start_um,
            z_end_um          = z_end_um,
            z_step_um         = z_step_um,
            enable_tiling     = enable_tiling,
            enable_timelapse  = enable_timelapse,
            n_timepoints      = n_timepoints,
            interval_min      = interval_min,
        )
        self._thread = threading.Thread(
            target=self._safe_run, kwargs=kwargs, daemon=True)
        self._thread.start()

    def stop(self):
        """Request a graceful stop."""
        self._stop_flag.set()

    # ── internal ─────────────────────────────────────────────────────────

    def _safe_run(self, **kwargs):
        try:
            self._run_combined(**kwargs)
        except Exception as e:
            log.error("Acquisition error", exc_info=True)
            self.error.emit(str(e))
        finally:
            self.mmc.close_shutter()
            self.led.disable(self.cfg.led_channel)
            self.finished.emit()

    def _run_combined(self,
                      channels:         List[ChannelConfig],
                      enable_z:         bool,
                      z_start_um:       float,
                      z_end_um:         float,
                      z_step_um:        float,
                      enable_tiling:    bool,
                      enable_timelapse: bool,
                      n_timepoints:     int,
                      interval_min:     float):

        z_positions = (self._z_positions(z_start_um, z_end_um, z_step_um)
                       if enable_z else [None])

        if enable_tiling:
            tile_positions, grid_shape = self.tiling.compute_grid()
        else:
            tile_positions = [None]
            grid_shape     = (1, 1)

        n_tiles   = len(tile_positions)
        n_z       = len(z_positions)
        n_ch      = len(channels)
        n_tp      = n_timepoints if enable_timelapse else 1

        total_steps = n_tp * n_tiles * n_z * n_ch
        steps_done  = [0]

        bf_in_channels = any(ch.is_bf for ch in channels)

        def step_cb(n: int = 1):
            steps_done[0] += n
            pct = int(steps_done[0] / total_steps * 100)
            self.progress.emit(min(pct, 100))

        for t_idx in range(n_tp):
            if self._stop_flag.is_set():
                break

            self.countdown.emit("")   # clear countdown label

            if enable_timelapse:
                log.info("Time-point %d / %d", t_idx + 1, n_tp)

            if enable_tiling:
                result = self._acquire_tiled(
                    channels, z_positions, tile_positions,
                    grid_shape, bf_in_channels, step_cb, t_idx)
            else:
                result = self._acquire_czyx(
                    channels, z_positions, bf_in_channels, step_cb)

            if enable_timelapse:
                self.timepoint_ready.emit(result, t_idx)
                if t_idx < n_tp - 1:
                    self._timelapse_rest(channels)
                    self._wait_with_countdown(interval_min, t_idx + 1, n_tp)
            else:
                self.result_ready.emit(result)

        if enable_timelapse:
            self.result_ready.emit({
                "mode":         "timelapse",
                "n_timepoints": n_tp,
            })

    # ── snap helper ───────────────────────────────────────────────────────

    def _snap(self) -> np.ndarray:
        """Snap one frame."""
        return self.mmc.snap()

    # ── Z positions ───────────────────────────────────────────────────────

    @staticmethod
    def _z_positions(z_start: float,
                     z_end:   float,
                     z_step:  float) -> List[float]:
        step = abs(z_step)
        if step == 0:
            raise ValueError("Z step must be non-zero.")
        if z_start <= z_end:
            positions = list(np.arange(z_start, z_end + step * 0.5,  step))
        else:
            positions = list(np.arange(z_start, z_end - step * 0.5, -step))
        if not positions:
            raise ValueError(
                f"Z range produced no positions: start={z_start}, "
                f"end={z_end}, step={z_step}")
        return positions

    # ── single-position CZYX acquisition ─────────────────────────────────

    def _acquire_czyx(self,
                      channels:       List[ChannelConfig],
                      z_positions:    List[Optional[float]],
                      bf_in_channels: bool,
                      step_cb) -> dict:
        """Acquire a (C, Z, H, W) array at the current XY position."""
        c_arrays = []
        for ch in channels:
            if self._stop_flag.is_set():
                break
            z_arrays = []
            for z in z_positions:
                if self._stop_flag.is_set():
                    break
                if z is not None:
                    self.mmc.move_z(z)
                self._prepare_channel(ch, bf_in_channels)
                img = self._snap()
                z_arrays.append(img)
                step_cb()
            c_arrays.append(np.stack(z_arrays, axis=0))   # (Z, H, W)

        czyx = np.stack(c_arrays, axis=0)   # (C, Z, H, W)
        return {
            "mode":     "single",
            "czyx":     czyx,
            "channels": [ch.name for ch in channels],
        }

    # ── tiled acquisition ─────────────────────────────────────────────────

    def _acquire_tiled(self,
                       channels:       List[ChannelConfig],
                       z_positions:    List[Optional[float]],
                       tile_positions: List[Tuple[float, float]],
                       grid_shape:     Tuple[int, int],
                       bf_in_channels: bool,
                       step_cb,
                       t_idx:          Optional[int] = None) -> dict:
        """
        Acquire all tiles in row-major order; emit tile_ready after each tile.
        """
        n_rows, n_cols = grid_shape
        tile_data: Dict[str, List[np.ndarray]] = {
            ch.name: [] for ch in channels}
        tile_pos_list: List[Tuple[float, float]] = []

        for tile_idx, (x, y) in enumerate(tile_positions):
            if self._stop_flag.is_set():
                break

            row = tile_idx // n_cols
            col = tile_idx  % n_cols

            self.mmc.move_xy(x, y)

            c_arrays = []
            for ch in channels:
                if self._stop_flag.is_set():
                    break
                z_arrays = []
                for z in z_positions:
                    if self._stop_flag.is_set():
                        break
                    if z is not None:
                        self.mmc.move_z(z)
                    self._prepare_channel(ch, bf_in_channels)
                    img = self._snap()
                    z_arrays.append(img)
                    step_cb()
                c_arrays.append(np.stack(z_arrays, axis=0))   # (Z, H, W)

            czyx_tile = np.stack(c_arrays, axis=0)   # (C, Z, H, W)

            for i, ch in enumerate(channels):
                tile_data[ch.name].append(czyx_tile[i])

            tile_pos_list.append((x, y))

            log.debug(
                "tile_ready row=%d col=%d  stage=(%.2f, %.2f)",
                row, col, x, y)

            self.tile_ready.emit({
                "czyx":        czyx_tile,
                "row":         row,
                "col":         col,
                "tile_index":  tile_idx,
                "channels":    [ch.name for ch in channels],
                "x_um":        x,
                "y_um":        y,
                "t_index":     t_idx,
            })

        return {
            "mode":       "tiled",
            "tile_data":  tile_data,
            "tile_pos":   tile_pos_list,
            "grid_shape": grid_shape,
            "channels":   [ch.name for ch in channels],
            "overlap_x":  self.tiling.overlap_x,
            "overlap_y":  self.tiling.overlap_y,
        }

    # ── channel / LED preparation ─────────────────────────────────────────

    def _prepare_channel(self, ch: ChannelConfig, bf_in_channels: bool):
        """Set filter wheels, shutter, and LED for one channel snap."""
        self.mmc.set_channel(ch)
        self.mmc.set_exposure(ch.exposure_ms)
        if ch.is_bf:
            if bf_in_channels:
                ma = min(ch.led_current_ma, self.cfg.led_max_current_ma)
                self.led.set_current_ma(self.cfg.led_channel, ma)
            else:
                self.led.disable(self.cfg.led_channel)
        else:
            self.led.disable(self.cfg.led_channel)

    # ── inter-timepoint helpers ───────────────────────────────────────────

    def _timelapse_rest(self,
                        channels: List[ChannelConfig]):
        """
        Switch to filter set of first selected channel with
        shutter closed and LED off during the wait period.
        """
        first_ch = channels[0]

        self.mmc.set_channel(first_ch, shut_closed=True)
        self.led.disable(self.cfg.led_channel)

    def _wait_with_countdown(self,
                              interval_min: float,
                              completed:    int,
                              total:        int):
        """
        Sleep for interval_min minutes, emitting a countdown string every second.
        """
        total_secs = int(interval_min * 60)
        for remaining in range(total_secs, 0, -1):
            if self._stop_flag.is_set():
                self.countdown.emit("")
                return
            mins, secs = divmod(remaining, 60)
            self.countdown.emit(
                f"Next acquisition in  {mins:02d}:{secs:02d}"
                f"  ({completed} of {total} complete)")
            time.sleep(1.0)
        self.countdown.emit("")
