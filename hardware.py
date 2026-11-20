#!/usr/bin/env python3
"""
hardware.py
===========
Low-level hardware drivers:

  MMController  – wraps pymmcore_plus.CMMCorePlus; falls back to demo mode.
  DCS100E       – DCS-100E LED controller over TCP/IP.

DCS-100E command protocol
-------------------------
  SET:LEVEL:CHANNELc,nnnn;
      c    = channel number (1-based integer)
      nnnn = current in milliamps (0000 – max rated current)

  SET:MODE:CHANNELc,n;
      n = 0  off
      n = 1  continuous
      n = 2  strobe
      n = 3  gated continuous

  *IDN?;                query device identity / firmware
  *CHANNEL:CONFIGS?;    query all channel settings
  DISCONNECT;           disconnect communication

  The DCS-100E closes the TCP connection after every command (one-shot
  server).  A fresh socket is opened for each command and closed after.
  WinError 10054 (ConnectionResetError) on recv() is normal and ignored.
  A 0.5 s pause is inserted after every send to allow the device to
  process the command before the next one is issued.

Shutter convention (ALWAYS manual):
  shutter_state is read directly from ChannelConfig:
    0  →  brightfield path
    1  →  fluorescence path
  MMController.set_channel() writes core.setShutterOpen(state)
  and waits for the device — no auto-shutter at any point.

wait_for_system()
-----------------
  core.waitForSystem() is called after each logical group of hardware
  commands (channel switch, stage move, snap) to ensure all devices have
  settled before the next command is issued.

Stage timeouts
--------------
  cfg.xy_stage_timeout_ms and cfg.z_stage_timeout_ms control how long
  waitForDevice() blocks before raising a timeout exception.
  The default (30 s XY, 10 s Z) covers most motorised stages.
  Increase these values in config.py if tiling raises:
    "Wait for device XYStage timed out after 5000ms"
"""
from __future__ import annotations

import logging
import socket
import time
from pathlib import Path

import numpy as np

from config import HardwareConfig, ChannelConfig

log = logging.getLogger(__name__)

try:
    from pymmcore_plus import CMMCorePlus
    MM_AVAILABLE = True
except ImportError:
    MM_AVAILABLE = False
    log.warning("pymmcore-plus not found – running in DEMO mode")

# Pause after every DCS-100E command (seconds)
_DCS_POST_SEND_PAUSE = 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  MICRO-MANAGER CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class MMController:
    """
    Thin wrapper around pymmcore_plus.CMMCorePlus.

    Uses CMMCorePlus() directly rather than the shared CMMCorePlus.instance()
    singleton so that each MMController owns its own core.  If you want to
    share a single core across the application, replace CMMCorePlus() with
    CMMCorePlus.instance() and remove the explicit connect/disconnect.
    """

    SHUTTER_BF   = 0
    SHUTTER_FLUO = 1

    def __init__(self, cfg: HardwareConfig):
        self.cfg:  HardwareConfig      = cfg
        self.core: CMMCorePlus | None  = None
        self.demo: bool                = True

    # ── connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        
        if not MM_AVAILABLE:
            log.info("DEMO mode: pymmcore_plus not installed.")
            self.demo = True
            return False
        
        try:
            self.core = CMMCorePlus()

            cfg_path = self.cfg.mm_config_file.strip()
            if cfg_path:
                p = Path(cfg_path)
                if not p.exists():
                    raise FileNotFoundError(
                        f"MM config not found: {p.resolve()}")
                log.info("Loading MM config: %s", p.resolve())
                self.core.loadSystemConfiguration(str(p))
                log.info("MM config loaded.")
            else:
                log.info("No mm_config_file set – using MM's current state.")

            # Disable auto-shutter.
            self.core.setAutoShutter(False)
            self._apply_stage_timeouts()

            # Settle all devices after the full connect sequence.
            self.core.waitForSystem()

            log.info("MM connected.")
            self.demo = False
            return True

        except FileNotFoundError as e:
            log.error(str(e))
            self.demo = True
            return False
        except Exception as e:
            log.warning("MM connection failed (%s) – DEMO mode", e)
            self.demo = True
            return False

    def _apply_stage_timeouts(self):
        """Set per-device timeout on XY and Z stage devices."""
        if self.demo or self.core is None:
            return
        try:
            self.core.setProperty(
                self.cfg.xy_stage_device, "TimeoutMs",
                str(self.cfg.xy_stage_timeout_ms))
            log.info("XY stage timeout → %d ms", self.cfg.xy_stage_timeout_ms)
        except Exception as e:
            log.debug("Could not set XY stage timeout (%s) – using MM default", e)
        try:
            self.core.setProperty(
                self.cfg.z_stage_device, "TimeoutMs",
                str(self.cfg.z_stage_timeout_ms))
            log.info("Z stage timeout → %d ms", self.cfg.z_stage_timeout_ms)
        except Exception as e:
            log.debug("Could not set Z stage timeout (%s) – using MM default", e)

    def disconnect(self):
        if self.core:
            try:
                self._set_shutter_raw(self.SHUTTER_BF)
                self.core.waitForSystem()
            except Exception:
                pass
        self.core = None

    # ── shutter ───────────────────────────────────────────────────────────

    def _set_shutter_raw(self, state: int):
        """Write shutter state (0 or 1) directly to MM."""
        if self.demo or self.core is None:
            log.debug("DEMO shutter → %d", state)
            return
        self.core.setShutterOpen(self.cfg.shutter_device, bool(state))
        self.core.waitForDevice(self.cfg.shutter_device)

    def set_channel(self,
                    ch: ChannelConfig,
                    shut_closed = False):
        """
        Move both filter wheels then set the shutter state stored in ch.
        Calls waitForSystem() after all channel hardware has settled.
        """
        if self.demo or self.core is None:
            log.debug(f"DEMO set_channel → {ch.name}  ex={ch.excitation_pos} em={ch.emission_pos} shutter={ch.shutter_state}")
            return

        # 1. Shutter
        if shut_closed:
            self._set_shutter_raw(0)
        else:
            self._set_shutter_raw(ch.shutter_state)
        
        # 2. Excitation wheel
        self.core.setState(self.cfg.excitation_wheel_device, ch.excitation_pos)

        # 3. Emission wheel
        self.core.setState(self.cfg.emission_wheel_device, ch.emission_pos)
        
        # 4. Settle all devices.
        self.core.waitForSystem()

        log.debug(f"Channel → {ch.name}")

    def close_shutter(self):
        """Closed between acquisitions."""
        self._set_shutter_raw(0)
        if self.core is not None and not self.demo:
            self.core.waitForSystem()
        log.debug("Shutter closed")

    # ── camera ────────────────────────────────────────────────────────────

    def set_exposure(self, exposure_ms: float):
        if self.demo or self.core is None:
            return
        self.core.setExposure(exposure_ms)

    def snap(self) -> np.ndarray:
        """
        Snap a single image and return it as a 2-D numpy array.

        pymmcore-plus provides CMMCorePlus.snap() which snaps the image
        and returns the numpy array in one call — no manual reshaping needed.
        """
        if self.demo or self.core is None:
            return self._demo_image()
        # snap() returns np.ndarray (H × W) directly.
        img: np.ndarray = self.core.snap()
        # Settle all devices after the snap sequence.
        self.core.waitForSystem()
        return np.fliplr(img) # flip frame to match user looking down at sample.

    # ── stage ─────────────────────────────────────────────────────────────

    def get_xy(self) -> tuple[float, float]:
        if self.demo or self.core is None:
            return 0.0, 0.0
        x = self.core.getXPosition(self.cfg.xy_stage_device)
        y = self.core.getYPosition(self.cfg.xy_stage_device)
        return float(x), float(y)

    def move_xy(self, x: float, y: float):
        """Move XY stage, wait with extended timeout, then waitForSystem()."""
        if self.demo or self.core is None:
            log.debug("DEMO move_xy → (%.2f, %.2f)", x, y)
            return
        self.core.setXYPosition(self.cfg.xy_stage_device, x, y)
        self._wait_xy()
        # Settle all devices after the full XY move sequence.
        self.core.waitForSystem()

    def _wait_xy(self):
        """Poll waitForDevice() with a Python-side timeout budget."""
        deadline = time.monotonic() + self.cfg.xy_stage_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                self.core.waitForDevice(self.cfg.xy_stage_device)
                return
            except Exception as e:
                if "timed out" in str(e).lower():
                    time.sleep(0.05)
                    continue
                raise
        raise TimeoutError(
            f"XY stage did not settle within {self.cfg.xy_stage_timeout_ms} ms")

    def get_z(self) -> float:
        if self.demo or self.core is None:
            return 0.0
        return float(self.core.getPosition(self.cfg.z_stage_device))

    def move_z(self, z: float):
        """Move Z stage, wait with extended timeout, then waitForSystem()."""
        if self.demo or self.core is None:
            log.debug("DEMO move_z → %.2f", z)
            return
        self.core.setPosition(self.cfg.z_stage_device, z)
        self._wait_z()
        # Settle all devices after the full Z move sequence.
        self.core.waitForSystem()

    def _wait_z(self):
        """Poll waitForDevice() with a Python-side timeout budget."""
        deadline = time.monotonic() + self.cfg.z_stage_timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                self.core.waitForDevice(self.cfg.z_stage_device)
                return
            except Exception as e:
                if "timed out" in str(e).lower():
                    time.sleep(0.05)
                    continue
                raise
        raise TimeoutError(
            f"Z stage did not settle within {self.cfg.z_stage_timeout_ms} ms")

    def home_stage(self):
        """
        Home both axes — Z first (safety), then XY.
        Each move already calls waitForSystem() internally.
        """
        log.info("Homing stage — Z first, then XY")
        self.move_z(-1000.0) # user-set home position (reduces travel path each time)
        self.move_xy(0.0, 0.0)
        log.info("Stage homed to (0, 0, 0)")

    # ── demo ──────────────────────────────────────────────────────────────

    @staticmethod
    def _demo_image() -> np.ndarray:
        """Synthetic 512×512 noisy image for demo mode."""
        base = np.random.randint(100, 300, (512, 512), dtype=np.uint16)
        cx, cy = np.random.randint(50, 462, 2)
        y, x   = np.ogrid[:512, :512]
        r      = np.sqrt((x - cx)**2 + (y - cy)**2)
        base  += (3000 * np.exp(-r**2 / (2 * 40**2))).astype(np.uint16)
        return base.clip(0, 65535).astype(np.uint16)


# ══════════════════════════════════════════════════════════════════════════════
#  DCS-100E LED CONTROLLER  (unchanged — pure TCP/IP, no MM dependency)
# ══════════════════════════════════════════════════════════════════════════════

class DCS100E:
    """
    DCS-100E LED controller over TCP/IP.

    One-shot server — the device closes the TCP connection after every
    command, so a fresh socket is opened for each send.  WinError 10054
    (ConnectionResetError) on recv() is normal and silently ignored.

    A _DCS_POST_SEND_PAUSE second pause is inserted after every send so
    the device has time to process the command before the socket closes.
    """

    def __init__(self, ip: str, port: int, timeout: float = 2.0):
        self.ip      = ip
        self.port    = port
        self.timeout = timeout
        self.demo    = False

    # ── connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Probe reachability with a bare TCP connect (no data sent)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.ip, self.port))
            s.close()
            log.info("DCS-100E connected  %s:%d", self.ip, self.port)
            self.demo = False
            return True
        except OSError as e:
            log.warning("DCS-100E unreachable (%s) – LED in DEMO mode", e)
            self.demo = True
            return False

    def disconnect(self):
        try:
            self._send("DISCONNECT;")
        except Exception:
            pass
        log.info("DCS-100E disconnected")

    # ── internal send ─────────────────────────────────────────────────────

    def _send(self, cmd: str):
        """
        Open a fresh socket, send cmd, pause, drain response, close.
        ConnectionResetError at any stage is silently ignored — the
        DCS-100E closes the connection from its side after processing.
        """
        if self.demo:
            log.debug("DEMO DCS-100E → %r", cmd)
            return

        log.info("DCS-100E → %r", cmd)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.timeout)
        try:
            s.connect((self.ip, self.port))
            try:
                s.sendall(cmd.encode())
            except ConnectionResetError:
                # RST during send — device already processed the command.
                log.debug("DCS-100E: RST during sendall (cmd delivered)")
                return
            # Drain any response.
            try:
                data = s.recv(256)
                if data:
                    log.debug("DCS-100E ← %r", data.decode(errors="replace").strip())
            except (socket.timeout, ConnectionResetError, OSError):
                pass  # normal for one-shot server
        except ConnectionResetError:
            log.debug("DCS-100E: RST on connect (cmd likely delivered)")
        except OSError as e:
            log.error("DCS-100E send error: %s", e)
        finally:
            try:
                s.close()
            except OSError:
                pass

    # ── public API ────────────────────────────────────────────────────────

    def set_current_ma(self, channel: int, current_ma: float):
        """
        Enable continuous mode then set current in mA.
        """
        ma = max(0, int(round(current_ma)))
        # self._send(f"SET:MODE:CHANNEL{channel},1")
        self._send(f"SET:LEVEL:CHANNEL{channel},{ma:04d};")
        time.sleep(_DCS_POST_SEND_PAUSE)
        log.info(f"DCS-100E ch={channel} → {ma} mA")

    def disable(self, channel: int):
        ma = 0
        self._send(f"SET:LEVEL:CHANNEL{channel},{ma:04d};")
        time.sleep(_DCS_POST_SEND_PAUSE)
        log.info(f"DCS-100E ch={channel} → {ma} mA")
