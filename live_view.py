#!/usr/bin/env python3
"""
live_view.py
============
LiveViewManager: continuous acquisition via pymmcore-plus
startContinuousSequenceAcquisition.

Architecture
------------
  Acquisition thread  →  frame_ready signal  →  MicroscopeMainWindow
        ↑                                                ↓
   CMMCorePlus                                pyqtgraph ImageItem update
   continuous seq

Key design points
-----------------
  1. Uses mmCore.startContinuousSequenceAcquisition(50) — the camera feeds
     its circular buffer continuously; the loop polls getLastImage() to
     always display the most recent frame, skipping stale ones.
  2. Channel / exposure changes set a _reconfigure flag; the acquisition
     thread handles the sequence stop → hardware reconfigure → restart,
     keeping all MM calls on a single thread and avoiding concurrent
     core access from the UI thread.
  3. Demo mode (no hardware): falls back to a synthetic frame loop at the
     same cadence as _SEQ_INTERVAL_MS.

Shutter behaviour
-----------------
  • On start: opens the correct shutter path for the selected channel via
    mmCore.set_channel() (which respects ch.shutter_state).
  • On stop : always closes the shutter (state 0 = closed.

DCS-100E LED
------------
  BF channel  → on
  Fluo channel→ off

Channel / exposure switching during live view
---------------------------------------------
  set_channel() and set_exposure() only update state and set the
  _reconfigure event.  The acquisition thread detects the flag at the top
  of its poll loop, performs stopSequenceAcquisition, applies the new
  hardware state, and calls startContinuousSequenceAcquisition again.
  This guarantees no two threads ever call CMMCorePlus concurrently.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np
from qtpy.QtCore import QObject, Signal

from config import HardwareConfig, ChannelConfig
from hardware import MMController, DCS100E

log = logging.getLogger(__name__)

# Interval (ms) passed to startContinuousSequenceAcquisition.
# The camera will insert a new frame into the circular buffer no faster than
# this; the display loop picks up the latest one via getLastImage().
_SEQ_INTERVAL_MS: int = 50


class LiveViewManager(QObject):
    """
    Low-latency live acquisition using pymmcore-plus continuous sequence
    acquisition (startContinuousSequenceAcquisition).

    Signals
    -------
    frame_ready(np.ndarray)
        Emitted from the acquisition thread whenever a fresh frame is
        available in the CMMCorePlus circular buffer.
    """

    frame_ready = Signal(object)   # np.ndarray — no colormap

    def __init__(self,
                 cfg: HardwareConfig,
                 mmc:  MMController,
                 led: DCS100E) -> None:
        super().__init__()
        self.cfg = cfg
        self.mmc  = mmc
        self.led = led

        self._stop_event  = threading.Event()
        # Set by set_channel() / set_exposure() to request a sequence restart.
        self._reconfigure = threading.Event()

        self._thread:  Optional[threading.Thread] = None
        self._channel: Optional[ChannelConfig]    = None
        self._running: bool = False

    # ── public API ────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    def start(self, channel: ChannelConfig) -> None:
        """Begin live view for *channel*."""
        if self._running:
            self.stop()

        self._channel = channel

        self._stop_event.clear()
        self._reconfigure.clear()

        self._thread = threading.Thread(
            target=self._acquire_loop, daemon=True, name="LiveAcqThread")
        self._thread.start()

        self._running = True

    def stop(self) -> None:
        """Stop live view, close shutter, disable LED."""
        if not self._running:
            return

        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        self._running = False
        self.mmc.close_shutter()
        self.led.disable(self.cfg.led_channel)
        log.info("Live view stopped  (shutter closed, LED off)")

    def set_channel(self, channel: ChannelConfig) -> None:
        """
        Switch channel while live view is running (zero thread teardown).

        LED state is updated immediately from the calling thread (pure
        TCP/IP, independent of MM).  Filter wheels + shutter + sequence
        restart are handled by the acquisition thread via _reconfigure.
        """
        if not self._running:
            return
        log.info(f"Live view channel switch → channel={channel.name}  shutter_state={channel.shutter_state}")
        self._channel = channel
        self._apply_led(channel)
        self._reconfigure.set()

    def set_exposure(self, exposure_ms: float) -> None:
        """
        Update exposure time live.

        Persists into the current ChannelConfig and triggers a sequence
        restart (acquisition thread applies mmCore.setExposure before
        restarting startContinuousSequenceAcquisition).
        """
        if self._channel is not None:
            self._channel.exposure_ms = exposure_ms
        self._reconfigure.set()

    def set_led_current(self, current_ma: float) -> None:
        """Update LED current live and persist into ChannelConfig."""
        if self._channel is not None and self._channel.is_bf:
            self._channel.led_current_ma = current_ma
            ma = min(current_ma, self.cfg.led_max_current_ma)
            self.led.set_current_ma(self.cfg.led_channel, ma)

    # ── internal helpers ──────────────────────────────────────────────────

    def _apply_led(self, channel: ChannelConfig) -> None:
        """Apply correct LED state for *channel* (called from any thread)."""
        if channel.is_bf:
            ma = min(channel.led_current_ma, self.cfg.led_max_current_ma)
            self.led.set_current_ma(self.cfg.led_channel, ma)
        else:
            self.led.disable(self.cfg.led_channel)

    def _seq_start(self, mmCore, channel: ChannelConfig) -> None:
        """
        Set exposure and start continuous sequence acquisition.
        Must be called from the acquisition thread only.
        """
        mmCore.setExposure(channel.exposure_ms)
        mmCore.startContinuousSequenceAcquisition(_SEQ_INTERVAL_MS)
        log.debug(f"Sequence started  channel={channel.name}  exposure={channel.exposure_ms} ms")

    def _seq_stop(self, mmCore) -> None:
        """
        Stop continuous sequence acquisition if it is currently running.
        Must be called from the acquisition thread only.
        """
        try:
            if mmCore.isSequenceRunning():
                mmCore.stopSequenceAcquisition()
                log.debug("Sequence stopped")
        except Exception as exc:
            log.warning(f"stopSequenceAcquisition error: {exc}")

    # ── acquisition loop ──────────────────────────────────────────────────

    def _acquire_loop(self) -> None:
        """
        Continuous acquisition loop running in a daemon thread.

        Live hardware path
        ~~~~~~~~~~~~~~~~~~
        1. Calls mmCore.setExposure / mmCore.startContinuousSequenceAcquisition(_SEQ_INTERVAL_MS)
           so the camera feeds the CMMCorePlus circular buffer automatically.
        2. Polls mmCore.getRemainingImageCount(); when at least one frame is
           present, calls mmCore.getLastImage() (non-destructive — always the
           most recent frame) and emits frame_ready.
        3. On _reconfigure: stops the sequence, lets mmCore.set_channel() apply
           new filter / shutter state, then restarts the sequence with the
           updated exposure.  All MM calls remain on this single thread.
        4. On _stop_event: exits the loop; _seq_stop() is called in finally.

        Demo mode fallback
        ~~~~~~~~~~~~~~~~~~
        When no hardware is connected (mmc is None or demo=True), emits a
        synthetic noisy image at the same _SEQ_INTERVAL_MS cadence.
        """
        mmCore = self.mmc.core

        # ── demo mode ──────────────────────────────────────────────────
        if mmCore is None or self.mmc.demo:
            log.info("Live view running in DEMO mode")
            while not self._stop_event.is_set():
                img = MMController._demo_image()
                self.frame_ready.emit(img)
                self._stop_event.wait(_SEQ_INTERVAL_MS / 1000.0)
            return

        # ── live hardware path ─────────────────────────────────────────
        channel = self._channel
        log.info(f"Live view started  channel={channel.name}  shutter_state={channel.shutter_state}") 
        self.mmc.set_channel(channel)
        self._apply_led(channel)
        self._seq_start(mmCore, channel)

        try:
            while not self._stop_event.is_set():

                # ── handle channel / exposure change ───────────────────
                if self._reconfigure.is_set():
                    self._reconfigure.clear()
                    self._seq_stop(mmCore)
                    channel = self._channel          # pick up updated reference
                    self.mmc.set_channel(channel)    # filter wheels + shutter
                    self._seq_start(mmCore, channel) # new exposure + restart

                # ── deliver latest frame ───────────────────────────────
                try:
                    if mmCore.getRemainingImageCount() > 0:
                        img: np.ndarray = mmCore.getLastImage()
                        img = np.fliplr(np.rot90(img, 1)) # flip frame to match user looking down at sample
                        self.frame_ready.emit(img)
                        self._stop_event.wait(_SEQ_INTERVAL_MS / 1000.0) # threading.Event.wait() accepts seconds
                    else:
                        # Buffer empty — yield briefly to avoid busy-spin.
                        time.sleep(0.005)
                except Exception as exc:
                    log.error("Live view frame error: %s", exc)
                    time.sleep(0.05)   # back-off on repeated errors

        finally:
            # Always stop the sequence, even if the loop exits via exception.
            self._seq_stop(mmCore)
