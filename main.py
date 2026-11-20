#!/usr/bin/env python3
"""
main.py
=======
Entry point — wires all modules together and launches a standalone PyQt5 window.
pyqtgraph handles all image display.

Usage
-----
    python main.py                          # default config
    python main.py --config my_scope.json   # load JSON config
    python main.py --save-config scope.json # write default config and exit

Module layout
-------------
    main.py          ← you are here
    config.py        ← HardwareConfig, ChannelConfig
    hardware.py      ← MMController (pycromanager), DCS100E (TCP LED)
    acquisition.py   ← AcquisitionEngine, TilingEngine
    live_view.py     ← LiveViewManager (background thread → Qt signal)
    export.py        ← ExportEngine, StitchingEngine
    widgets.py       ← MicroscopeMainWindow (QMainWindow + pyqtgraph)
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings

# Suppress noisy third-party warnings before any imports trigger them
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Tight layout not applied")
warnings.filterwarnings("ignore", message="constrained_layout not applied")

import os

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import Qt

from config      import HardwareConfig
from hardware    import MMController, DCS100E
from acquisition import AcquisitionEngine, TilingEngine
from live_view   import LiveViewManager
from export      import ExportEngine
from widgets     import MicroscopeMainWindow

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream  = sys.stdout,
)
log = logging.getLogger("main")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Microscope control — PyQt5 + pyqtgraph + Micro-Manager + DCS-100E")
    p.add_argument("--config",      metavar="PATH",
                   help="Path to JSON hardware config file.")
    p.add_argument("--save-config", metavar="PATH",
                   help="Write default config to PATH and exit.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── write default config and exit ─────────────────────────────────────
    if args.save_config:
        HardwareConfig().to_json(args.save_config)
        print(f"Default config written to: {args.save_config}")
        sys.exit(0)

    # ── load config ───────────────────────────────────────────────────────
    if args.config:
        log.info("Loading config: %s", args.config)
        cfg = HardwareConfig.from_json(args.config)
    else:
        log.info("Using default (built-in) config.")
        cfg = HardwareConfig()

    # ── hardware init ─────────────────────────────────────────────────────
    mmc  = MMController(cfg)
    led = DCS100E(cfg.led_ip, cfg.led_port)

    mmc_ok  = mmc.connect()
    led_ok = led.connect()

    if not mmc_ok:
        log.warning("Micro-Manager unavailable — running in DEMO mode.")
    if not led_ok:
        log.warning("DCS-100E unavailable  — LED in DEMO mode.")

    # ── Qt application ────────────────────────────────────────────────────
    os.environ["QT_SCALE_FACTOR"] = "1.25"
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ── application objects ───────────────────────────────────────────────
    tiling   = TilingEngine(cfg)
    acq      = AcquisitionEngine(cfg, mmc, led, tiling)
    live     = LiveViewManager(cfg, mmc, led)
    exporter = ExportEngine(cfg)

    # ── main window ───────────────────────────────────────────────────────
    window = MicroscopeMainWindow(
        cfg      = cfg,
        mmc      = mmc,
        led      = led,
        acq      = acq,
        tiling   = tiling,
        live     = live,
        exporter = exporter,
    )
    window.showMaximized()

    # ── clean shutdown ────────────────────────────────────────────────────
    def _on_quit():
        log.info("Shutting down…")
        live.stop()
        acq.stop()
        bf_ch = cfg.channels[0]
        mmc.set_channel(bf_ch)
        mmc.close_shutter()
        mmc.disconnect()
        led.disable(cfg.led_channel)
        led.disconnect()
        log.info("Hardware released.")

    app.aboutToQuit.connect(_on_quit)

    log.info("Application ready.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
