#!/usr/bin/env python3
"""
export.py
=========
ExportEngine: saves acquisition results to disk.

Export rules (mirrors widget logic) — all 8 flag combinations
-------------------------------------------------------------
  No flags               → <base_name>.ome.tif                    (CZYX)
  Z only                 → <base_name>.ome.tif                    (CZYX, Z>1)
  Tiling only            → <base_name>_tile<R>x<C>.ome.tif per tile (CZYX)
                           <base_name>_stitched.ome.tif composite CZYX
  Timelapse only         → <base_name>_timepoint<NNNN>.ome.tif    (CZYX)
  Z + Tiling             → per-tile CZYX (ZYX preserved) + stitched
  Z + Timelapse          → <base_name>_timepoint<NNNN>.ome.tif    (CZYX, Z>1)
  Tiling + Timelapse     → per-tile and stitched each carry _timepoint<NNNN> suffix
  Z + Tiling + Timelapse → per-tile CZYX (ZYX preserved) and stitched each carry
                           _timepoint<NNNN> suffix

Tile naming
-----------
  Each tile is saved as a single CZYX OME-TIFF containing all channels:
    <base_name>[_timepoint<NNNN>]_tile<RRRR>x<CCCC>.ome.tif
  e.g.  acquisition_tile0000x0001.ome.tif
        acquisition_timepoint0003_tile0000x0001.ome.tif

Timepoint numbering
-------------------
  Time-points are numbered from 0 (timepoint0000, timepoint0001, …).

Stitching output
----------------
  A single composite CZYX (or CZYX with Z>1) stitched file is written per
  acquisition / time-point.  Per-channel stitched files are NOT written —
  all channels are available in the composite file.

OME-TIFF axes metadata
-----------------------
  Every file is written with explicit axes='CZYX' metadata and channel
  names so ImageJ / FIJI correctly identifies C as channels, not Z slices.
  Data is always reshaped to a true 4-D (C, Z, Y, X) array before writing
  — even when Z=1 — to guarantee consistent dimension labelling.

Acquisition metadata
--------------------
  save_acquisition_metadata() writes a companion JSON file alongside the
  OME-TIFFs containing all acquisition settings:
    <base_name>_metadata.json
  The same settings are embedded in the OME-TIFF Description field so
  they travel with the image file.
  Returns the JSON string so widgets.py can cache it as _metadata_str.

Stitching algorithm
-------------------
  Tile placement uses actual stage positions + pixel size (µm/px) to
  compute pixel offsets — NOT overlap percentages.  This eliminates
  cumulative integer-rounding errors that would cut off mosaic edges.

  Stage coordinate convention:
    -X  →  left  of imaged area   (col 0 is at x_min)
    +X  →  right of imaged area   (col increases as X increases)
    -Y  →  top   of imaged area   (row 0 is at y_min)
    +Y  →  bottom of imaged area  (row increases as Y increases)

  Pixel offset from reference tile (row 0, col 0):
    col_offset_px = round((x - ref_x) / pixel_size_um)   # +x is right
    row_offset_px = round((y - ref_y) / pixel_size_um)   # +y is down

  Reference tile is always the tile with minimum X (leftmost) and
  minimum Y (topmost) — i.e. tile (0, 0) — selected explicitly rather
  than assuming positions[0] is the reference.

  Feathered blending: a 2-D linear ramp weight (distance from tile edge)
  is applied to each tile before accumulation. The weighted average in
  overlap regions produces smooth, seamless transitions.

Helper
------
  build_path(out_dir, base_name, suffix) builds the full Path with suffix
  inserted before the .ome.tif extension.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

from config import HardwareConfig, ChannelConfig

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_path(out_dir: Path, base_name: str, suffix: str = "") -> Path:
    """
    Assemble an output path.

    Examples
    --------
    build_path("/data", "exp1", "")
        → /data/exp1.ome.tif

    build_path("/data", "exp1", "_timepoint0000")
        → /data/exp1_timepoint0000.ome.tif

    build_path("/data", "exp1", "_timepoint0003_tile0002x0001")
        → /data/exp1_timepoint0003_tile0002x0001.ome.tif
    """
    return out_dir / f"{base_name}{suffix}.ome.tif"


def _to_czyx(data: np.ndarray) -> np.ndarray:
    """
    Ensure data is always a 4-D (C, Z, Y, X) array.

    Handles:
      (C, Y, X)    → (C, 1, Y, X)
      (C, Z, Y, X) → unchanged
    """
    if data.ndim == 3:
        return data[:, np.newaxis, :, :]   # (C, 1, Y, X)
    if data.ndim == 4:
        return data
    raise ValueError(f"Expected 3-D or 4-D array, got shape {data.shape}")


def _write_ome_tiff(path: Path,
                    data: np.ndarray,
                    channel_names: List[str],
                    description: str = ""):
    """
    Write a CZYX OME-TIFF with explicit axes and channel name metadata
    so ImageJ / FIJI correctly identifies C as channels (not Z).

    Parameters
    ----------
    path          : output file path (must end in .ome.tif)
    data          : array of shape (C, Z, Y, X) or (C, Y, X)
    channel_names : list of strings, one per channel
    description   : optional JSON string embedded in OME Description field
    """
    czyx = _to_czyx(data)   # always 4-D

    metadata = {
        'axes': 'CZYX',
        'Channel': {'Name': channel_names},
    }
    if description:
        metadata['Description'] = description

    tifffile.imwrite(
        str(path),
        czyx,
        photometric='minisblack',
        metadata=metadata,
    )
    log.info("Saved %s  shape=%s  channels=%s",
             path.name, czyx.shape, channel_names)


# ══════════════════════════════════════════════════════════════════════════════
#  STITCHING
# ══════════════════════════════════════════════════════════════════════════════

class StitchingEngine:
    """
    Feathered (weighted-average) stitching using actual stage positions.

    Tile placement is computed from stage coordinates + pixel size (µm/px)
    rather than from overlap percentages.  This eliminates cumulative
    integer-rounding errors that grow with the number of tiles.

    Stage coordinate convention
    ---------------------------
      -X  →  left  of imaged area   col_offset increases as X increases
      +X  →  right of imaged area
      -Y  →  top   of imaged area   row_offset increases as Y increases
      +Y  →  bottom of imaged area

    Reference tile
    --------------
      The reference tile is always tile (0, 0) — the tile with the
      minimum X value (leftmost) and minimum Y value (topmost).
      It is selected explicitly by finding min(x) and min(y) across all
      positions rather than assuming positions[0] is the reference.

    Feathered blending
    ------------------
      Each tile is multiplied by a 2-D linear ramp weight before being
      accumulated into the canvas.  The weight is proportional to the
      distance from the nearest tile edge — pixels near the center of a
      tile have the highest weight.  Dividing the accumulated canvas by
      the accumulated weights produces a smooth weighted average in all
      overlap regions with no sharp seams.
    """

    def __init__(self, cfg: HardwareConfig):
        self.cfg = cfg

    # ── public API ────────────────────────────────────────────────────────

    def stitch(self,
               tiles:      List[np.ndarray],
               positions:  List[Tuple[float, float]],
               grid_shape: Tuple[int, int],
               overlap_x:  float,
               overlap_y:  float) -> np.ndarray:
        """
        Stitch a list of 2-D (HW) tiles into a mosaic.

        If tiles are 3-D (ZYX), the middle Z-slice is used for stitching;
        call stitch_zyx() to get a full ZYX stitched volume.
        """
        if not tiles:
            raise ValueError("No tiles to stitch.")

        # Collapse Z to middle slice for 2-D stitching
        flat = []
        for t in tiles:
            if t is None:
                flat.append(None)
            elif t.ndim == 3:
                flat.append(t[t.shape[0] // 2])
            else:
                flat.append(t)

        return self._stitch_2d(flat, positions)

    def stitch_zyx(self,
                   tiles:      List[np.ndarray],
                   positions:  List[Tuple[float, float]],
                   grid_shape: Tuple[int, int],
                   overlap_x:  float,
                   overlap_y:  float) -> np.ndarray:
        """
        Stitch a list of 3-D (ZYX) tiles into a ZYX volume.
        Each Z-plane is stitched independently then stacked.
        """
        if not tiles:
            raise ValueError("No tiles to stitch.")

        # Find a reference tile to get Z depth
        ref_tile = next((t for t in tiles if t is not None), None)
        if ref_tile is None:
            raise ValueError("All tiles are None.")

        n_z = ref_tile.shape[0] if ref_tile.ndim == 3 else 1

        planes = []
        for z in range(n_z):
            slice_tiles = []
            for t in tiles:
                if t is None:
                    slice_tiles.append(None)
                elif t.ndim == 3:
                    slice_tiles.append(t[z])
                else:
                    slice_tiles.append(t)
            planes.append(self._stitch_2d(slice_tiles, positions))

        return np.stack(planes, axis=0)   # (Z, H_out, W_out)

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _feather_weight(h: int, w: int) -> np.ndarray:
        """
        2-D linear ramp weight — proportional to distance from nearest edge.
        Shape: (H, W), dtype float32, values in (0, 1].
        """
        wy = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1)).astype(np.float32)
        wx = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(np.float32)
        weight = np.outer(wy, wx)
        weight /= weight.max()
        return weight

    def _stitch_2d(self,
                   tiles:     List[Optional[np.ndarray]],
                   positions: List[Tuple[float, float]]) -> np.ndarray:
        """
        Feathered stitch of 2-D (HW) tiles using stage positions.

        Reference tile is the one with minimum X (leftmost in image space)
        and minimum Y (topmost in image space) — tile (0, 0).
        """
        pixel_size = self.cfg.pixel_size_um
        if pixel_size <= 0:
            pixel_size = 1.0

        # Find reference tile: min X (left edge), min Y (top edge)
        ref_x = min(x for x, y in positions)
        ref_y = min(y for x, y in positions)

        # Find a valid tile for shape reference
        ref_tile = next((t for t in tiles if t is not None), None)
        if ref_tile is None:
            raise ValueError("All tiles are None — nothing to stitch.")
        h_tile, w_tile = ref_tile.shape[:2]

        # Compute pixel offsets for each tile
        offsets: List[Tuple[int, int]] = []
        for (x, y) in positions:
            col_px = int(round((x - ref_x) / pixel_size))   # +x is right → col increases right
            row_px = int(round((y - ref_y) / pixel_size))   # +y is down → row increases down
            offsets.append((row_px, col_px))

        # Canvas size — sized to contain all tiles exactly
        h_out = max(row_px + h_tile for row_px, col_px in offsets)
        w_out = max(col_px + w_tile for row_px, col_px in offsets)

        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"Invalid canvas size ({h_out}, {w_out}). "
                f"Check stage positions and pixel_size_um ({pixel_size} µm/px).")

        log.info("Stitching canvas: (%d, %d)  tiles: %d  pixel_size: %.4f µm/px",
                 h_out, w_out, len(tiles), pixel_size)

        canvas  = np.zeros((h_out, w_out), dtype=np.float64)
        weights = np.zeros((h_out, w_out), dtype=np.float64)
        weight_tile = self._feather_weight(h_tile, w_tile)

        for tile, (row_px, col_px) in zip(tiles, offsets):
            if tile is None:
                continue

            # Clip tile to canvas bounds (safety — should not be needed)
            r0, c0 = row_px, col_px
            r1 = min(r0 + h_tile, h_out)
            c1 = min(c0 + w_tile, w_out)
            tr = r1 - r0
            tc = c1 - c0
            if tr <= 0 or tc <= 0:
                continue

            tile_f = tile[:tr, :tc].astype(np.float64)
            wt     = weight_tile[:tr, :tc]

            canvas [r0:r1, c0:c1] += tile_f * wt
            weights[r0:r1, c0:c1] += wt

        # Weighted average — avoid division by zero
        mask = weights > 0
        canvas[mask] /= weights[mask]

        return canvas.astype(np.uint16)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ExportEngine:
    """Saves acquisition results to disk as OME-TIFF files."""

    def __init__(self, cfg: HardwareConfig):
        self.cfg      = cfg
        self.stitcher = StitchingEngine(cfg)

    # ── public API ────────────────────────────────────────────────────────

    def export_single(self,
                      result:    dict,
                      out_dir:   Path,
                      base_name: str,
                      metadata:  str  = "",
                      t_index:   Optional[int] = None) -> Path:
        """
        Export a non-tiled acquisition (single, Z-stack, or one timelapse
        timepoint) as a CZYX OME-TIFF.
        """
        czyx      = result["czyx"]
        ch_names  = result["channels"]   # List[str]

        tp_suffix = f"_timepoint{t_index:04d}" if t_index is not None else ""
        path      = build_path(out_dir, base_name, tp_suffix)

        _write_ome_tiff(path, czyx, ch_names, description=metadata)
        return path

    def export_tile(self,
                    payload:   dict,
                    out_dir:   Path,
                    base_name: str,
                    metadata:  str = "") -> Path:
        """
        Export a single tile immediately after it is acquired.

        payload keys (from tile_ready signal):
            "czyx"       : np.ndarray (C, Z, H, W)
            "row"        : int
            "col"        : int
            "tile_index" : int
            "channels"   : List[str]
            "x_um"       : float
            "y_um"       : float
            "t_index"    : int or None
        """
        czyx     = payload["czyx"]
        row      = payload["row"]
        col      = payload["col"]
        ch_names = payload["channels"]   # List[str]
        t_index  = payload.get("t_index", None)

        tp_suffix   = f"_timepoint{t_index:04d}" if t_index is not None else ""
        tile_suffix = f"{tp_suffix}_tile{row:04d}x{col:04d}"
        path        = build_path(out_dir, base_name, tile_suffix)

        _write_ome_tiff(path, czyx, ch_names, description=metadata)
        return path

    def export_stitched(self,
                        result:    dict,
                        out_dir:   Path,
                        base_name: str,
                        metadata:  str  = "",
                        t_index:   Optional[int] = None) -> Path:
        """
        Stitch all tiles from a tiling acquisition and save as CZYX OME-TIFF.
        Tiles must have already been exported individually via export_tile().

        result keys:
            "tile_data"  : Dict[str, List[np.ndarray]]  ch_name → [ZYX or YX, …]
            "tile_pos"   : List[Tuple[float, float]]
            "grid_shape" : Tuple[int, int]
            "channels"   : List[str]
            "overlap_x"  : float
            "overlap_y"  : float
        """
        tile_data  = result["tile_data"]
        tile_pos   = result["tile_pos"]
        grid_shape = result["grid_shape"]
        ch_names   = result["channels"]   # List[str]
        overlap_x  = result.get("overlap_x", 0.10)
        overlap_y  = result.get("overlap_y", 0.10)

        tp_suffix = f"_timepoint{t_index:04d}" if t_index is not None else ""

        # Detect whether tiles have a Z dimension
        ref_tile = next(
            (t for tiles in tile_data.values() for t in tiles if t is not None),
            None)
        if ref_tile is None:
            log.error("No valid tiles found for stitching.")
            return None

        has_z = (ref_tile.ndim == 3 and ref_tile.shape[0] > 1)

        stitched_channels = []
        for ch_name in ch_names:
            tiles = tile_data.get(ch_name, [])
            try:
                if has_z:
                    stitched = self.stitcher.stitch_zyx(
                        tiles, tile_pos, grid_shape, overlap_x, overlap_y)
                else:
                    # Flatten (1, H, W) → (H, W) for 2-D stitching
                    flat = [t[0] if (t is not None and t.ndim == 3) else t
                            for t in tiles]
                    stitched = self.stitcher.stitch(
                        flat, tile_pos, grid_shape, overlap_x, overlap_y)
            except Exception as e:
                log.error("Stitching failed for channel %s: %s", ch_name, e)
                continue
            stitched_channels.append(stitched)

        if not stitched_channels:
            log.error("No channels stitched — stitched file not written.")
            return None

        composite = np.stack(stitched_channels, axis=0)   # (C, [Z,] H, W)

        path = build_path(out_dir, base_name, f"{tp_suffix}_stitched")
        _write_ome_tiff(path, composite, ch_names, description=metadata)
        return path

    def save_acquisition_metadata(self,
                                  out_dir:   Path,
                                  base_name: str,
                                  settings:  dict) -> str:
        """
        Write acquisition settings to a JSON sidecar file and return
        the JSON string (for embedding in OME-TIFF Description fields).

        Parameters
        ----------
        out_dir   : directory where the OME-TIFFs are being saved
        base_name : acquisition base name
        settings  : dict built by widgets._build_metadata_settings()
        """
        settings["acquisition_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        settings["magnification"]    = self.cfg.magnification
        settings["pixel_size_um"]    = self.cfg.pixel_size_um

        json_str  = json.dumps(settings, indent=2)
        json_path = out_dir / f"{base_name}_metadata.json"

        try:
            with open(json_path, "w") as f:
                f.write(json_str)
            log.info("Metadata saved → %s", json_path.name)
        except Exception as e:
            log.error("Failed to write metadata: %s", e)

        return json_str
