#!/usr/bin/env python3
"""
widgets.py
==========
MicroscopeMainWindow: standalone PyQt5 QMainWindow.

Layout
------
  Left panel  — pyqtgraph live view canvas (fills available space)
  Right panel — QTabWidget with two tabs:

  Tab 1 – Hardware
      Magnification drop-down → auto-fills pixel size (read-only)
      Channel selector, exposure (ms), LED current (mA)
      Live view controls (Start / Stop)
      Contrast sliders (min / max) with auto-scale button
      Histogram (toggleable, pyqtgraph PlotWidget, 5 fps)
      Home Stage button

  Tab 2 – Acquire
      Output directory + base filename
      Channel table  – checkbox + manual exposure + ↑/↓ order buttons
      Z-stack        – enable toggle, start / end / step (µm)
      Tiling         – enable toggle, top-left + bottom-right corner buttons,
                       auto overlap display
      Time-lapse     – enable toggle, # time-points, interval (min)
      Run / Stop, progress bar, countdown label

Colormap
--------
  Live view is always displayed in grayscale — no per-channel colormap.

Contrast
--------
  Two QSlider widgets (Min / Max) map linearly over [0, 65535].
  "Auto" reads the current frame's 0.1–99.9th percentile and snaps
  the sliders to those values.  Sliders update the pyqtgraph ImageItem
  setLevels() call directly — zero overhead, sub-millisecond response.

Histogram
---------
  pyqtgraph PlotWidget replaces matplotlib — rendered in the same OpenGL
  context as the image canvas so there is no cross-context overhead.
  Toggleable via a checkbox; when off the widget is hidden and the update
  timer is stopped.  Bin count = HISTOGRAM_BINS (default 256).

Home Stage
----------
  Runs mmc.home_stage() in a QThread so the UI stays responsive.

Channel ordering
----------------
  ↑ / ↓ buttons in the channel table reorder acquisition channels.

Tiling corners
--------------
  Two buttons: "Set Top-Left" and "Set Bottom-Right".
  Top-left  = stage position of tile (row=0, col=0)  — most +X, most -Y
  Bottom-right = stage position of tile (n-1, n-1)   — most -X, most +Y
  Overlap is computed automatically by TilingEngine.compute_grid() once
  both corners are recorded.  The result is displayed as a read-only
  label showing both per-axis values:
    "Auto: X=12.3%  Y=15.7%"
  Minimum overlap is always 10 % on each axis.

Export / metadata
-----------------
  Acquisition settings are written to a JSON sidecar via
  ExportEngine.save_acquisition_metadata().  The settings dict includes:
    - Per-channel name, exposure, LED current, colormap, shutter state
    - Z-stack: start, end, step, and full list of Z positions (µm)
    - Tiling: grid shape, overlap per axis, XY stage position of every tile,
              corner coordinates
    - Time-lapse: number of timepoints, interval in minutes and seconds

_on_finished() routing
-----------------------
  Timelapse:  each timepoint is exported in _on_timepoint_ready();
              _on_finished() only cleans up UI — NO additional export.
  Non-timelapse: export happens in _on_finished() via result_ready signal.
  This prevents double-export and the 12% GUI freeze.

Tile export
-----------
  tile_ready signal is connected to _on_tile_ready() which exports each
  tile immediately as it is acquired.  After all tiles are done,
  _on_finished() (non-timelapse) or _on_timepoint_ready() (timelapse)
  exports the stitched composite.

Canvas background
-----------------
  The pyqtgraph ViewBox background is set to dark gray (80, 80, 80) so
  non-image areas outside the camera FOV are visually distinct from the
  black pixels at the image edges.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, QThread, Signal, QObject
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QTabWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QCheckBox, QFileDialog, QProgressBar,
    QGridLayout, QFormLayout, QSizePolicy, QMessageBox,
    QScrollArea, QLineEdit, QSlider, QFrame,
)

from config import HardwareConfig, ChannelConfig, MAGNIFICATION_PIXEL_SIZES
from hardware import MMController, DCS100E
from acquisition import AcquisitionEngine, TilingEngine
from live_view import LiveViewManager
from export import ExportEngine

log = logging.getLogger(__name__)

# pyqtgraph global settings
pg.setConfigOptions(antialias=False, useOpenGL=True)

HISTOGRAM_BINS = 128

# Gray background color for the canvas (R, G, B) — distinguishes FOV from image edges
_CANVAS_BG = (80, 80, 80)


# ══════════════════════════════════════════════════════════════════════════════
#  HOME STAGE WORKER
# ══════════════════════════════════════════════════════════════════════════════

class HomeStageWorker(QObject):
    """Runs home_stage() in a background QThread."""
    finished = Signal()

    def __init__(self, mmc: MMController):
        super().__init__()
        self._mmc = mmc

    def run(self):
        self._mmc.home_stage()
        self.finished.emit()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MicroscopeMainWindow(QMainWindow):
    """Standalone PyQt5 window — no napari dependency."""

    def __init__(self, cfg, mmc, led, acq, tiling, live, exporter):
        super().__init__()
        self.cfg      = cfg
        self.mmc      = mmc
        self.led      = led
        self.acq      = acq
        self.tiling   = tiling
        self.live     = live
        self.exporter = exporter

        self._current_frame: Optional[np.ndarray] = None
        self._acq_layer_names: List[str]          = []
        self._t_index: int                        = 0

        # cached flags set at run-start — used by signal handlers
        self._enable_tiling_cache:    bool         = False
        self._enable_timelapse_cache: bool         = False
        self._enable_z_cache:         bool         = False
        self._out_dir_cache:          Optional[Path] = None
        self._base_name_cache:        str          = "acquisition"
        self._metadata_str:           str          = ""

        # cached result for non-timelapse export in _on_finished()
        self._last_result:            Optional[dict] = None

        self.setWindowTitle("Microscope Control")
        self._build_ui()
        self._connect_signals()

    # ══════════════════════════════════════════════════════════════════════
    #  UI BUILD
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ── LEFT: live view canvas ────────────────────────────────────────
        splitter.addWidget(self._build_canvas())
        splitter.setStretchFactor(0, 3)

        # ── RIGHT: control panel ──────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(380)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        tabs.addTab(self._tab_hardware(), "Hardware")
        tabs.addTab(self._tab_acquire(),  "Acquire")
        right_layout.addWidget(tabs)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

    def _build_canvas(self) -> QWidget:
        """pyqtgraph live view canvas with gray background."""
        self._img_view = pg.GraphicsLayoutWidget()
        self._img_view.setBackground(_CANVAS_BG)

        vp = self._img_view.addViewBox(row=0, col=0)
        vp.setAspectLocked(True)
        vp.setBackgroundColor(_CANVAS_BG)

        self._img_item = pg.ImageItem()
        vp.addItem(self._img_item)

        return self._img_view

    # ── Hardware tab ──────────────────────────────────────────────────────

    def _tab_hardware(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Objective
        obj_grp = QGroupBox("Objective")
        obj_lay = QFormLayout(obj_grp)
        self.mag_combo = QComboBox()
        for mag in sorted(MAGNIFICATION_PIXEL_SIZES.keys()):
            self.mag_combo.addItem(f"×{mag}", mag)
        default_idx = self.mag_combo.findData(self.cfg.magnification)
        if default_idx >= 0:
            self.mag_combo.setCurrentIndex(default_idx)
        self.px_label = QLabel(f"{self.cfg.pixel_size_um:.4f} µm/px")
        self.px_label.setStyleSheet("color: gray;")
        obj_lay.addRow("Magnification:", self.mag_combo)
        obj_lay.addRow("Pixel size:",    self.px_label)
        lay.addWidget(obj_grp)

        # Channel
        ch_grp = QGroupBox("Channel")
        ch_lay = QFormLayout(ch_grp)
        self.ch_combo    = QComboBox()
        self.exp_spin    = QDoubleSpinBox()
        self.led_spin    = QDoubleSpinBox()
        for ch in self.cfg.channels:
            self.ch_combo.addItem(ch.name)
        self.exp_spin.setRange(0.1, 60000.0)
        self.exp_spin.setSuffix(" ms")
        self.exp_spin.setValue(self.cfg.channels[0].exposure_ms)
        self.led_spin.setRange(0.0, self.cfg.led_max_current_ma)
        self.led_spin.setSuffix(" mA")
        # Initialise from the BF channel; 0 mA default if no BF channel exists
        _bf_init = next((ch for ch in self.cfg.channels if ch.is_bf), None)
        self.led_spin.setValue(_bf_init.led_current_ma if _bf_init else 0.0)
        # Enable only when the BF channel is currently selected
        self.led_spin.setEnabled(
            bool(self.cfg.channels) and self.cfg.channels[0].is_bf)
        ch_lay.addRow("Channel:",        self.ch_combo)
        ch_lay.addRow("Exposure:",       self.exp_spin)
        ch_lay.addRow("BF LED Current:", self.led_spin)
        lay.addWidget(ch_grp)

        # Live view
        live_grp = QGroupBox("Live View")
        live_lay = QHBoxLayout(live_grp)
        self.btn_live_start = QPushButton("▶  Start")
        self.btn_live_stop  = QPushButton("■  Stop")
        self.btn_live_stop.setEnabled(False)
        live_lay.addWidget(self.btn_live_start)
        live_lay.addWidget(self.btn_live_stop)
        lay.addWidget(live_grp)

        # Contrast
        contrast_grp = QGroupBox("Contrast")
        contrast_lay = QGridLayout(contrast_grp)
        self.contrast_min = QSlider(Qt.Horizontal)
        self.contrast_max = QSlider(Qt.Horizontal)
        self.contrast_min.setRange(0, 65535)
        self.contrast_max.setRange(0, 65535)
        self.contrast_min.setValue(0)
        self.contrast_max.setValue(65535)
        self.btn_auto_contrast = QPushButton("Auto")
        self.lbl_min = QLabel("0")
        self.lbl_max = QLabel("65535")
        contrast_lay.addWidget(QLabel("Min:"),         0, 0)
        contrast_lay.addWidget(self.contrast_min,      0, 1)
        contrast_lay.addWidget(self.lbl_min,           0, 2)
        contrast_lay.addWidget(QLabel("Max:"),         1, 0)
        contrast_lay.addWidget(self.contrast_max,      1, 1)
        contrast_lay.addWidget(self.lbl_max,           1, 2)
        contrast_lay.addWidget(self.btn_auto_contrast, 2, 0, 1, 3)
        lay.addWidget(contrast_grp)

        # Histogram
        hist_grp = QGroupBox("Histogram")
        hist_lay = QVBoxLayout(hist_grp)
        self.btn_hist_toggle = QPushButton("■  Histogram Off")
        self.btn_hist_toggle.setCheckable(True)
        self.btn_hist_toggle.setChecked(False)
        self._hist_widget = pg.PlotWidget(background=(30, 30, 30))
        self._hist_widget.setMaximumHeight(120)
        self._hist_widget.hide()
        hist_lay.addWidget(self.btn_hist_toggle)
        hist_lay.addWidget(self._hist_widget)
        lay.addWidget(hist_grp)

        # Histogram timer — starts only when histogram is shown
        self._hist_timer = QTimer(self)
        self._hist_timer.setInterval(200)
        self._hist_timer.timeout.connect(self._update_histogram)

        # Home Stage
        stage_grp = QGroupBox("Home Stage")
        stage_lay = QFormLayout(stage_grp)
        self.btn_home = QPushButton("⌂  Home Stage")
        stage_lay.addRow(self.btn_home)
        lay.addWidget(stage_grp)

        lay.addStretch()
        return w

    # ── Acquire tab ───────────────────────────────────────────────────────

    def _tab_acquire(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        lay   = QVBoxLayout(inner)
        lay.setSpacing(6)

        # Output directory + base name
        out_grp = QGroupBox("Output")
        out_lay = QFormLayout(out_grp)
        self.dir_edit  = QLineEdit()
        self.dir_edit.setPlaceholderText("Select output directory…")
        self.base_edit = QLineEdit("acquisition")
        btn_dir = QPushButton("Browse…")
        btn_dir.clicked.connect(self._browse_dir)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(btn_dir)
        out_lay.addRow("Directory:", dir_row)
        out_lay.addRow("Base name:", self.base_edit)
        lay.addWidget(out_grp)

        # Channel table
        ch_grp = QGroupBox("Channels  (check to include · ↑/↓ to reorder)")
        self._ch_table_layout = QVBoxLayout(ch_grp)
        self._ch_rows: List[dict] = []
        for ch in self.cfg.channels:
            self._add_channel_row(ch)
        lay.addWidget(ch_grp)

        # Z-Stack
        z_grp = QGroupBox("Z-Stack")
        z_grp.setCheckable(True)
        z_grp.setChecked(False)
        self.z_grp = z_grp
        z_lay = QFormLayout(z_grp)
        self.z_start_spin = QDoubleSpinBox()
        self.z_end_spin   = QDoubleSpinBox()
        self.z_step_spin  = QDoubleSpinBox()
        for sp in (self.z_start_spin, self.z_end_spin):
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(3)
            sp.setSuffix(" µm")
        self.z_step_spin.setRange(0.001, 10000.0)
        self.z_step_spin.setDecimals(3)
        self.z_step_spin.setSuffix(" µm")
        self.z_step_spin.setValue(1.0)
        btn_z_start = QPushButton("Set from stage")
        btn_z_end   = QPushButton("Set from stage")
        btn_z_start.clicked.connect(
            lambda: self.z_start_spin.setValue(self.mmc.get_z()))
        btn_z_end.clicked.connect(
            lambda: self.z_end_spin.setValue(self.mmc.get_z()))
        z_start_row = QHBoxLayout()
        z_start_row.addWidget(self.z_start_spin)
        z_start_row.addWidget(btn_z_start)
        z_end_row = QHBoxLayout()
        z_end_row.addWidget(self.z_end_spin)
        z_end_row.addWidget(btn_z_end)
        z_lay.addRow("Start:", z_start_row)
        z_lay.addRow("End:",   z_end_row)
        z_lay.addRow("Step:",  self.z_step_spin)
        lay.addWidget(z_grp)

        # Tiling
        tile_grp = QGroupBox("Tiling")
        tile_grp.setCheckable(True)
        tile_grp.setChecked(False)
        self.tile_grp = tile_grp
        tile_lay = QVBoxLayout(tile_grp)

        btn_tl = QPushButton("Set Top-Left from Stage")
        btn_br = QPushButton("Set Bottom-Right from Stage")
        self._lbl_tl = QLabel("Top-Left:     not set")
        self._lbl_br = QLabel("Bottom-Right: not set")
        self._lbl_derived_tr = QLabel("Top-Right (0×n):  not set")
        self._lbl_derived_bl = QLabel("Bottom-Left (n×0): not set")
        self._lbl_overlap = QLabel("Overlap (auto): —")
        self._lbl_overlap.setStyleSheet("color: gray;")

        btn_tl.clicked.connect(self._set_top_left)
        btn_br.clicked.connect(self._set_bottom_right)

        tile_lay.addWidget(btn_tl)
        tile_lay.addWidget(self._lbl_tl)
        tile_lay.addWidget(btn_br)
        tile_lay.addWidget(self._lbl_br)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        tile_lay.addWidget(sep)
        tile_lay.addWidget(QLabel("Derived corners:"))
        tile_lay.addWidget(self._lbl_derived_tr)
        tile_lay.addWidget(self._lbl_derived_bl)
        tile_lay.addWidget(self._lbl_overlap)
        lay.addWidget(tile_grp)

        # Time-lapse
        tl_grp = QGroupBox("Time-Lapse")
        tl_grp.setCheckable(True)
        tl_grp.setChecked(False)
        self.tl_grp = tl_grp
        tl_lay = QFormLayout(tl_grp)
        self.n_timepoints_spin = QSpinBox()
        self.n_timepoints_spin.setRange(1, 100000)
        self.n_timepoints_spin.setValue(10)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.0, 1440.0)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setSuffix(" min")
        self.interval_spin.setValue(5.0)
        tl_lay.addRow("Time-points:", self.n_timepoints_spin)
        tl_lay.addRow("Interval:",    self.interval_spin)
        lay.addWidget(tl_grp)

        # Run / Stop
        run_grp = QGroupBox()
        run_lay = QVBoxLayout(run_grp)
        btn_row = QHBoxLayout()
        self.btn_run  = QPushButton("▶  Run")
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        self.progress_bar    = QProgressBar()
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "color: #f0a500; font-weight: bold;")
        run_lay.addLayout(btn_row)
        run_lay.addWidget(self.progress_bar)
        run_lay.addWidget(self.countdown_label)
        lay.addWidget(run_grp)

        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    # ── channel table helpers ─────────────────────────────────────────────

    def _add_channel_row(self, ch: ChannelConfig):
        row_widget = QWidget()
        row_lay    = QHBoxLayout(row_widget)
        row_lay.setContentsMargins(0, 0, 0, 0)

        chk = QCheckBox(ch.name)
        chk.setChecked(True)
        chk.setFixedWidth(90)

        exp = QDoubleSpinBox()
        exp.setRange(0.1, 60000.0)
        exp.setSuffix(" ms")
        exp.setValue(ch.exposure_ms)
        exp.setFixedWidth(100)

        btn_up   = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_up.setFixedWidth(28)
        btn_down.setFixedWidth(28)

        row_lay.addWidget(chk)
        row_lay.addWidget(exp)
        row_lay.addWidget(btn_up)
        row_lay.addWidget(btn_down)

        entry = {"widget": row_widget, "chk": chk, "exp": exp, "ch": ch}
        self._ch_rows.append(entry)
        self._ch_table_layout.addWidget(row_widget)

        btn_up.clicked.connect(lambda _, e=entry: self._move_channel_row(e, -1))
        btn_down.clicked.connect(lambda _, e=entry: self._move_channel_row(e, +1))

    def _move_channel_row(self, entry: dict, direction: int):
        idx = self._ch_rows.index(entry)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self._ch_rows):
            return
        self._ch_rows[idx], self._ch_rows[new_idx] = \
            self._ch_rows[new_idx], self._ch_rows[idx]
        self._refresh_channel_layout()

    def _refresh_channel_layout(self):
        # Remove all widgets from layout
        while self._ch_table_layout.count():
            item = self._ch_table_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        # Re-add in new order
        for entry in self._ch_rows:
            self._ch_table_layout.addWidget(entry["widget"])
            entry["widget"].show()

    def _selected_channels(self) -> List[ChannelConfig]:
        selected = []
        for entry in self._ch_rows:
            if entry["chk"].isChecked():
                ch = entry["ch"]
                ch.exposure_ms = entry["exp"].value()
                selected.append(ch)
        return selected

    # ══════════════════════════════════════════════════════════════════════
    #  SIGNAL CONNECTIONS
    # ══════════════════════════════════════════════════════════════════════

    def _connect_signals(self):
        # Hardware tab
        self.mag_combo.currentIndexChanged.connect(self._on_mag_changed)
        self.ch_combo.currentIndexChanged.connect(self._on_channel_changed)
        self.exp_spin.valueChanged.connect(self._on_exposure_changed)
        self.led_spin.valueChanged.connect(self._on_led_changed)
        self.btn_live_start.clicked.connect(self._on_live_start)
        self.btn_live_stop.clicked.connect(self._on_live_stop)
        self.contrast_min.valueChanged.connect(self._on_contrast_changed)
        self.contrast_max.valueChanged.connect(self._on_contrast_changed)
        self.btn_auto_contrast.clicked.connect(self._on_auto_contrast)
        self.btn_hist_toggle.toggled.connect(self._on_hist_toggled)
        self.btn_home.clicked.connect(self._on_home_stage)

        # Acquire tab
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop.clicked.connect(self._on_stop)

        # Acquisition signals
        self.live.frame_ready.connect(self._on_live_frame)
        self.acq.result_ready.connect(self._on_result_ready)
        self.acq.tile_ready.connect(self._on_tile_ready)
        self.acq.timepoint_ready.connect(self._on_timepoint_ready)
        self.acq.progress.connect(self.progress_bar.setValue)
        self.acq.finished.connect(self._on_finished)
        self.acq.error.connect(self._on_error)
        self.acq.countdown.connect(self._on_countdown)

    # ══════════════════════════════════════════════════════════════════════
    #  HARDWARE HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_mag_changed(self, idx: int):
        mag = self.mag_combo.itemData(idx)
        from config import MAGNIFICATION_PIXEL_SIZES
        px  = MAGNIFICATION_PIXEL_SIZES.get(mag, self.cfg.pixel_size_um)
        self.cfg.magnification = mag
        self.cfg.pixel_size_um = px
        self.px_label.setText(f"{px:.4f} µm/px")

    # ── BF channel helper ─────────────────────────────────────────────────

    def _bf_channel(self) -> Optional[ChannelConfig]:
        """Return the brightfield ChannelConfig, or None if none is defined."""
        return next((ch for ch in self.cfg.channels if ch.is_bf), None)

    def _on_channel_changed(self, idx: int):
        if idx < 0 or idx >= len(self.cfg.channels):
            return
        ch = self.cfg.channels[idx]
        self.exp_spin.blockSignals(True)
        self.exp_spin.setValue(ch.exposure_ms)
        self.exp_spin.blockSignals(False)
        # BF LED spin always reflects the BF channel's current, regardless
        # of which channel is selected.  Disable it for non-BF channels so
        # only brightfield LED current is ever editable here.
        bf_ch = self._bf_channel()
        self.led_spin.blockSignals(True)
        self.led_spin.setValue(bf_ch.led_current_ma if bf_ch else 0.0)
        self.led_spin.blockSignals(False)
        self.led_spin.setEnabled(ch.is_bf)
        if self.live.running:
            self.live.set_channel(ch)

    def _on_exposure_changed(self, val: float):
        idx = self.ch_combo.currentIndex()
        if 0 <= idx < len(self.cfg.channels):
            self.cfg.channels[idx].exposure_ms = val
        if self.live.running:
            self.live.set_exposure(val)

    def _on_led_changed(self, val: float):
        # Always update the BF channel's LED current only.
        # Other channels keep their default 0 mA and are never touched here.
        bf_ch = self._bf_channel()
        if bf_ch is not None:
            bf_ch.led_current_ma = val
        # Apply to live view only when the BF channel is currently active
        if self.live.running:
            idx = self.ch_combo.currentIndex()
            if 0 <= idx < len(self.cfg.channels) and self.cfg.channels[idx].is_bf:
                self.live.set_led_current(val)

    def _on_live_start(self):
        idx = self.ch_combo.currentIndex()
        if idx < 0:
            return
        ch = self.cfg.channels[idx]
        self.live.start(ch)
        self.btn_live_start.setEnabled(False)
        self.btn_live_stop.setEnabled(True)

    def _on_live_stop(self):
        self.live.stop()
        self.btn_live_start.setEnabled(True)
        self.btn_live_stop.setEnabled(False)

    def _on_home_stage(self):
        self.btn_home.setEnabled(False)
        self.btn_home.setText("⌂  Homing…")
        self._home_worker  = HomeStageWorker(self.mmc)
        self._home_thread  = QThread()
        self._home_worker.moveToThread(self._home_thread)
        self._home_thread.started.connect(self._home_worker.run)
        self._home_worker.finished.connect(self._home_thread.quit)
        self._home_worker.finished.connect(self._on_home_done)
        self._home_thread.start()

    def _on_home_done(self):
        self.btn_home.setEnabled(True)
        self.btn_home.setText("⌂  Home Stage")

    # ══════════════════════════════════════════════════════════════════════
    #  LIVE VIEW HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _on_live_frame(self, img: np.ndarray):
        self._current_frame = img
        lo = self.contrast_min.value()
        hi = self.contrast_max.value()
        display = (img)
        self._img_item.setImage(display, autoLevels=False, levels=(lo, hi))

    def _on_contrast_changed(self):
        lo = self.contrast_min.value()
        hi = self.contrast_max.value()
        self.lbl_min.setText(str(lo))
        self.lbl_max.setText(str(hi))
        if self._current_frame is not None:
            self._img_item.setLevels((lo, hi))

    def _on_auto_contrast(self):
        if self._current_frame is None:
            return
        lo = int(np.percentile(self._current_frame, 0.1))
        hi = int(np.percentile(self._current_frame, 99.9))
        self.contrast_min.setValue(lo)
        self.contrast_max.setValue(hi)

    def _on_hist_toggled(self, checked: bool):
        if checked:
            self._hist_widget.show()
            self._hist_timer.start()
            self.btn_hist_toggle.setText("▶  Histogram On")
        else:
            self._hist_timer.stop()
            self._hist_widget.hide()
            self.btn_hist_toggle.setText("■  Histogram Off")

    def _update_histogram(self):
        if self._current_frame is None:
            return
        counts, edges = np.histogram(self._current_frame, bins=HISTOGRAM_BINS)
        self._hist_widget.clear()
        self._hist_widget.plot(
            edges[:-1], counts,
            stepMode=False, fillLevel=0,
            brush=(100, 200, 100, 120))
        self._hist_widget.setXRange(0, 65535, padding=0)
        self._hist_widget.getViewBox().setLimits(xMin=0, xMax=65535)
        self._hist_widget.enableAutoRange(axis='x', enable=False)

    # ══════════════════════════════════════════════════════════════════════
    #  TILING CORNER BUTTONS
    # ══════════════════════════════════════════════════════════════════════

    def _set_top_left(self):
        try:
            x, y = self.mmc.get_xy()
            self.tiling.set_top_left(x, y)
            self._lbl_tl.setText(f"Top-Left:     X={x:.2f}  Y={y:.2f} µm")
            self._update_derived_corners()
        except Exception as e:
            log.error("Could not set top-left corner: %s", e)

    def _set_bottom_right(self):
        try:
            x, y = self.mmc.get_xy()
            self.tiling.set_bottom_right(x, y)
            self._lbl_br.setText(f"Bottom-Right: X={x:.2f}  Y={y:.2f} µm")
            self._update_derived_corners()
        except Exception as e:
            log.error("Could not set bottom-right corner: %s", e)

    def _update_derived_corners(self):
        if not self.tiling.ready:
            return
        tl = self.tiling.top_left
        br = self.tiling.bottom_right
        x_left,  y_top    = tl
        x_right, y_bottom = br

        # Derived corners
        self._lbl_derived_tr.setText(
            f"Top-Right  (0×n): X={x_right:.2f}  Y={y_top:.2f} µm")
        self._lbl_derived_bl.setText(
            f"Bottom-Left (n×0): X={x_left:.2f}  Y={y_bottom:.2f} µm")

        # Compute grid to get overlap
        try:
            self.tiling.compute_grid()
            ox = self.tiling.overlap_x * 100
            oy = self.tiling.overlap_y * 100
            nr, nc = self.tiling._n_rows, self.tiling._n_cols
            self._lbl_overlap.setText(
                f"Grid: {nr} rows × {nc} cols  |  "
                f"Overlap: X={ox:.1f}%  Y={oy:.1f}%")
            self._lbl_overlap.setStyleSheet("color: #00aa00; font-weight: bold;")
        except Exception as e:
            self._lbl_overlap.setText(f"Overlap: error ({e})")
            self._lbl_overlap.setStyleSheet("color: red;")

    # ══════════════════════════════════════════════════════════════════════
    #  ACQUISITION HANDLERS
    # ══════════════════════════════════════════════════════════════════════

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self.dir_edit.setText(d)

    def _on_run(self):
        channels = self._selected_channels()
        if not channels:
            QMessageBox.warning(self, "No channels", "Select at least one channel.")
            return

        out_dir_str = self.dir_edit.text().strip()
        if not out_dir_str:
            QMessageBox.warning(self, "No directory", "Select an output directory.")
            return
        out_dir = Path(out_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_name       = self.base_edit.text().strip() or "acquisition"
        enable_z        = self.z_grp.isChecked()
        enable_tiling   = self.tile_grp.isChecked()
        enable_timelapse = self.tl_grp.isChecked()

        # Validate tiling
        if enable_tiling and not self.tiling.ready:
            QMessageBox.warning(self, "Tiling",
                "Set both top-left and bottom-right corners first.")
            return

        # Z positions
        z_start_um = self.z_start_spin.value() if enable_z else None
        z_end_um   = self.z_end_spin.value()   if enable_z else None
        z_step_um  = self.z_step_spin.value()  if enable_z else None

        # Timelapse
        n_timepoints = self.n_timepoints_spin.value() if enable_timelapse else 1
        interval_min = self.interval_spin.value()     if enable_timelapse else 0.0

        # Cache run parameters for signal handlers
        self._enable_tiling_cache    = enable_tiling
        self._enable_timelapse_cache = enable_timelapse
        self._enable_z_cache         = enable_z
        self._out_dir_cache          = out_dir
        self._base_name_cache        = base_name
        self._last_result            = None
        self._t_index                = 0

        # Build and save metadata
        settings = self._build_metadata_settings(
            channels, enable_z, z_start_um, z_end_um, z_step_um,
            enable_tiling, enable_timelapse, n_timepoints, interval_min)
        self._metadata_str = self.exporter.save_acquisition_metadata(
            out_dir, base_name, settings)

        # UI state
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.countdown_label.setText("")

        # Stop live view if running
        if self.live.running:
            self.live.stop()
            self.btn_live_start.setEnabled(True)
            self.btn_live_stop.setEnabled(False)

        # Launch acquisition
        self.acq.run(
            channels         = channels,
            enable_z         = enable_z,
            z_start_um       = z_start_um,
            z_end_um         = z_end_um,
            z_step_um        = z_step_um,
            enable_tiling    = enable_tiling,
            enable_timelapse = enable_timelapse,
            n_timepoints     = n_timepoints,
            interval_min     = interval_min,
        )

    def _on_stop(self):
        self.acq.stop()

    def _on_result_ready(self, result: dict):
        """Cache result for non-timelapse export in _on_finished()."""
        if not self._enable_timelapse_cache:
            self._last_result = result

    def _on_tile_ready(self, payload: dict):
        """Export each tile immediately as it is acquired."""
        if self._out_dir_cache is None:
            return
        try:
            self.exporter.export_tile(
                payload   = payload,
                out_dir   = self._out_dir_cache,
                base_name = self._base_name_cache,
                metadata  = self._metadata_str,
            )
        except Exception as e:
            log.error("Tile export error: %s", e)

    def _on_timepoint_ready(self, result: dict, t_index: int):
        """Export each time-point immediately, then release memory."""
        if self._out_dir_cache is None:
            return
        try:
            if self._enable_tiling_cache:
                self.exporter.export_stitched(
                    result    = result,
                    out_dir   = self._out_dir_cache,
                    base_name = self._base_name_cache,
                    metadata  = self._metadata_str,
                    t_index   = t_index,
                )
            else:
                self.exporter.export_single(
                    result    = result,
                    out_dir   = self._out_dir_cache,
                    base_name = self._base_name_cache,
                    metadata  = self._metadata_str,
                    t_index   = t_index,
                )
        except Exception as e:
            log.error("Timepoint export error: %s", e)

    def _on_finished(self):
        """
        Export for non-timelapse runs only.
        Timelapse exports are handled per-timepoint in _on_timepoint_ready().
        """
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.countdown_label.setText("")

        if self._enable_timelapse_cache:
            # Timelapse already exported per timepoint — nothing to do
            return

        if self._out_dir_cache is None or self._last_result is None:
            return

        try:
            if self._enable_tiling_cache:
                self.exporter.export_stitched(
                    result    = self._last_result,
                    out_dir   = self._out_dir_cache,
                    base_name = self._base_name_cache,
                    metadata  = self._metadata_str,
                    t_index   = None,
                )
            else:
                self.exporter.export_single(
                    result    = self._last_result,
                    out_dir   = self._out_dir_cache,
                    base_name = self._base_name_cache,
                    metadata  = self._metadata_str,
                    t_index   = None,
                )
        except Exception as e:
            log.error("Export error: %s", e)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.countdown_label.setText("")
        QMessageBox.critical(self, "Acquisition Error", msg)

    def _on_countdown(self, msg: str):
        self.countdown_label.setText(msg)

    # ══════════════════════════════════════════════════════════════════════
    #  METADATA BUILDER
    # ══════════════════════════════════════════════════════════════════════

    def _build_metadata_settings(
            self,
            channels:         List[ChannelConfig],
            enable_z:         bool,
            z_start_um:       Optional[float],
            z_end_um:         Optional[float],
            z_step_um:        Optional[float],
            enable_tiling:    bool,
            enable_timelapse: bool,
            n_timepoints:     int,
            interval_min:     float) -> dict:

        settings: dict = {}

        # Channels
        settings["channels"] = [
            {
                "name":           ch.name,
                "exposure_ms":    ch.exposure_ms,
                "led_current_ma": ch.led_current_ma if ch.is_bf else None,
                "shutter_state":  ch.shutter_state,
                "is_bf":          ch.is_bf,
            }
            for ch in channels
        ]

        # Z-stack
        if enable_z and z_start_um is not None:
            if z_start_um <= z_end_um:
                z_positions = list(np.arange(
                    z_start_um, z_end_um + z_step_um * 0.5, z_step_um))
            else:
                z_positions = list(np.arange(
                    z_start_um, z_end_um - z_step_um * 0.5, -z_step_um))
            settings["z_stack"] = {
                "start_um":     z_start_um,
                "end_um":       z_end_um,
                "step_um":      z_step_um,
                "n_slices":     len(z_positions),
                "positions_um": z_positions,
            }

        # Tiling
        if enable_tiling and self.tiling.ready:
            try:
                tile_positions, grid_shape = self.tiling.compute_grid()
                tl = self.tiling.top_left
                br = self.tiling.bottom_right
                settings["tiling"] = {
                    "grid_shape":     list(grid_shape),
                    "n_tiles":        len(tile_positions),
                    "overlap_x_pct":  round(self.tiling.overlap_x * 100, 2),
                    "overlap_y_pct":  round(self.tiling.overlap_y * 100, 2),
                    "top_left_um":    list(tl),
                    "bottom_right_um": list(br),
                    "tile_positions": [
                        {
                            "tile_index": i,
                            "row":        i // grid_shape[1],
                            "col":        i %  grid_shape[1],
                            "x_um":       x,
                            "y_um":       y,
                        }
                        for i, (x, y) in enumerate(tile_positions)
                    ],
                }
            except Exception as e:
                log.error("Metadata tiling error: %s", e)

        # Time-lapse
        if enable_timelapse:
            settings["timelapse"] = {
                "n_timepoints": n_timepoints,
                "interval_min": interval_min,
                "interval_sec": round(interval_min * 60, 1),
            }

        return settings
