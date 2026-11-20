#!/usr/bin/env python3
"""
config.py
=========
Hardware and channel configuration dataclasses.
Serialisable to / from JSON via HardwareConfig.to_json() / from_json().

Shutter convention
------------------
  shutter_state is stored explicitly per ChannelConfig:
    0  →  brightfield path   (BF channel)
    1  →  fluorescence path  (all other channels)
  MMController.set_channel() writes this value directly to the shutter
  device — no derivation from is_bf at runtime.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List

log = logging.getLogger(__name__)


# ── Magnification → pixel size lookup (µm/px) ─────────────────────────────
MAGNIFICATION_PIXEL_SIZES: dict[float, float] = {
     1.26: 5.1587,
     1.6:  4.0625,
     2.0:  3.2244,
     2.5:  2.6558,
     3.2:  2.0680,
     4.0:  1.6535,
     5.0:  1.3163,
     6.4:  1.0257,
     8.0:  0.8203,
    10.0:  0.6522,
    12.6:  0.5247,
}

DEFAULT_MAGNIFICATION: float = 1.26


@dataclass
class ChannelConfig:
    """Per-channel hardware settings."""
    name:           str
    excitation_pos: int            # excitation filter wheel position (0-based)
    emission_pos:   int            # emission  filter wheel position  (0-based)
    shutter_state:  int   = 1      # 0 = closed, 1 = open
    exposure_ms:    float = 100.0
    is_bf:          bool  = False  # True → BF LED via DCS-100E
    led_current_ma: float = 0      # milliamps

    def __post_init__(self):
        self.shutter_state  = int(self.shutter_state)
        self.exposure_ms    = float(self.exposure_ms)
        self.led_current_ma = float(self.led_current_ma)


@dataclass
class HardwareConfig:
    """Top-level hardware configuration."""

    # ── Micro-Manager ──────────────────────────────────────────────────────
    mm_config_file:          str   = r"C:\Users\Allbritton_Lab\OneDrive - UW\Allbritton Lab\MVX\NeoMVX System Code\Micro-Manager Config\InvertedMVX_NewPC_8900.cfg"
    camera_device:           str   = "Camera"
    xy_stage_device:         str   = "XYStage"
    z_stage_device:          str   = "ZStage"

    # ── Filter wheels ──────────────────────────────────────────────────────
    excitation_wheel_device: str   = "Wheel_EX"
    emission_wheel_device:   str   = "Wheel_EM"

    # ── Shutter ────────────────────────────────────────────────────────────
    shutter_device:          str   = "Shutter"

    # ── DCS-100E LED controller ────────────────────────────────────────────
    led_ip:                  str   = "192.168.0.1"
    led_port:                int   = 777
    led_channel:             int   = 1
    led_max_current_ma:      float = 800.0

    # ── Magnification & pixel size ─────────────────────────────────────────
    magnification:           float = DEFAULT_MAGNIFICATION
    pixel_size_um:           float = MAGNIFICATION_PIXEL_SIZES[DEFAULT_MAGNIFICATION]

    # ── Image / stage geometry ────────────────────────────────────────────
    image_width_px:          int   = 2048
    image_height_px:         int   = 2048

    # ── Stage timeouts ────────────────────────────────────────────────────
    # xy_stage_timeout_ms: how long to wait for XY stage to complete a move.
    #   Increase this if tiling raises "Wait for device timed out".
    #   Rule of thumb: (max_tile_distance_um / stage_speed_um_per_s) * 1000 + 2000
    xy_stage_timeout_ms:     int   = 30_000   # 30 s default — covers slow stages
    z_stage_timeout_ms:      int   = 10_000   # 10 s default

    # ── Channels ──────────────────────────────────────────────────────────
    channels: List[ChannelConfig] = field(default_factory=lambda: [
        # name, ex, em, shutter, exp_ms, bf?, led_mA
        ChannelConfig("BF", 0, 0, 0, 10.0, True, 100.0),
        ChannelConfig("Blue(UV)", 0, 1, 1, 100.0, False, 0.0),
        ChannelConfig("Blue(Violet)", 1, 1, 1, 100.0, False, 0.0),
        ChannelConfig("Green", 2, 2, 1, 100.0, False, 0.0),
        ChannelConfig("Red", 3, 3, 1, 100.0, False, 0.0),
        ChannelConfig("FarRed", 4, 4, 1, 100.0, False, 0.0),
    ])

    # ── JSON I/O ──────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str) -> "HardwareConfig":
        with open(path) as f:
            d = json.load(f)
        raw_ch = d.pop("channels", [])
        valid  = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg    = cls(**valid)
        if raw_ch:
            cfg.channels = [ChannelConfig(**c) for c in raw_ch]
        return cfg

    def to_json(self, path: str):
        d = {k: v for k, v in self.__dict__.items() if k != "channels"}
        d["channels"] = [c.__dict__ for c in self.channels]
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        log.info("Config saved → %s", path)
