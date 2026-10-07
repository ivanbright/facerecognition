"""
Part 3: horizontal tracking -> calibration -> MQTT feed.

Thin layer on top of the existing Part 2 pipeline (`face_tracking.py`). It does
NOT re-detect or re-recognize faces. It consumes the existing
`LockedFaceTracker.update()` result and adds the three things Part 3 needs:

  1. HorizontalTracker   - smoothed face-center X, LEFT/CENTER/RIGHT state and
                           recognition confidence from the EXISTING tracker.
  2. PositionMapper      - calibrated camera_x -> motor-position mapping,
                           piecewise-linear interpolation, clamped to the
                           calibrated range (no naive x/W*180 assumption).
  3. MQTTPublisher       - compact JSON on topic `face_track` via Paho
                           (non-blocking loop so the camera loop is never
                           blocked by the network).

Config lives in face_needle_tracker/mqtt_config.json, same convention as
camera_config.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parent


def load_config(path: Optional[str] = None) -> dict:
    cfg_path = Path(path) if path else ROOT / "mqtt_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"MQTT config not found: {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


# -------------------------
# 1. Horizontal tracking state (reuses tracker output)
# -------------------------
class HorizontalTracker:
    """Wraps the existing LockedFaceTracker result into the horizontal signal.

    Only called when the locked face is the trained target (the tracker already
    filters unknown faces), so 'recognized' here means exactly that.
    """

    def __init__(self, frame_width: int, dead_zone_ratio: float = 0.07):
        self.frame_width = int(frame_width)
        self.dead_zone_ratio = float(dead_zone_ratio)
        self.raw_x = 0.0
        self.smoothed_x = 0.0
        self.state = "CENTER"
        self.confidence = 0.0

    def update(self, locked_face, tracker) -> bool:
        """Update from one tracker result. Returns True if target is locked."""
        if locked_face is None:
            return False
        box = tracker.box(locked_face)
        self.raw_x = float(box[0] + box[2]) / 2.0
        # Reuse the tracker's existing EMA smoothing - not a second pipeline.
        self.smoothed_x = float(tracker.smooth_center[0])

        center = self.frame_width / 2.0
        dz = self.dead_zone_ratio * center  # dead zone in pixels
        if self.smoothed_x < center - dz:
            self.state = "LEFT"
        elif self.smoothed_x > center + dz:
            self.state = "RIGHT"
        else:
            self.state = "CENTER"

        match = getattr(tracker, "last_match", None)
        self.confidence = float(match.similarity) if match is not None else 0.0
        return True


# -------------------------
# 2. Calibrated camera_x -> motor position
# -------------------------
class PositionMapper:
    """Piecewise-linear camera_x -> motor_position map, clamped at the ends.

    Calibration points are (camera_x, motor_position). Points outside the
    calibrated camera range clamp to the first/last motor position, so the
    needle is never commanded beyond its calibrated physical range.
    """

    def __init__(self, points: Optional[List[Tuple[float, float]]] = None):
        self.points: List[Tuple[float, float]] = sorted(
            [(float(a), float(b)) for a, b in (points or [])]
        )

    @property
    def min_motor(self) -> float:
        return self.points[0][1] if self.points else 0.0

    @property
    def max_motor(self) -> float:
        return self.points[-1][1] if self.points else 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "PositionMapper":
        points = cfg.get("calibration", {}).get("points", [])
        return cls([(p["camera_x"], p["motor_position"]) for p in points])

    def load(self, path) -> None:
        p = Path(path)
        if not p.exists():
            # No saved calibration yet -> keep the current points (config
            # defaults from mqtt_config.json). Wiping them here is what showed
            # "0..0 steps / uncalibrated" even though defaults existed.
            return
        try:
            data = json.loads(p.read_text())
            self.points = sorted(
                (float(d["camera_x"]), float(d["motor_position"]))
                for d in data.get("points", [])
            )
        except Exception:
            pass

    def map(self, camera_x: float) -> float:
        """Interpolate motor position for a camera x (clamped). Returns raw float steps."""
        if not self.points:
            return 0.0
        xs = [a for a, _ in self.points]
        ms = [b for _, b in self.points]
        return float(np.interp(camera_x, np.array(xs), np.array(ms)))

    def map_int(self, camera_x: float) -> int:
        return int(round(self.map(camera_x)))

    def is_calibrated(self) -> bool:
        return len(self.points) >= 2


# -------------------------
# 3. MQTT publisher (Paho, non-blocking)
# -------------------------
class MQTTPublisher:
    """Publishes compact tracking JSON to topic `topic` on a Mosquitto broker.

    connect_async + loop_start keep a background network thread, so the camera
    loop never blocks on the broker. publish() throttles to publish_interval_s.
    """

    def __init__(
        self,
        broker: str,
        port: int = 1883,
        topic: str = "face_track",
        interval_s: float = 0.10,
        client_id: Optional[str] = None,
    ):
        self.topic = topic
        self.interval_s = float(interval_s)
        self._last_sent = 0.0
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            # A fixed client_id makes a second process (e.g. needle_test.py)
            # kick this client off the broker (duplicate-id policy). Scope it
            # per process so trackers and test tools can coexist.
            client_id=client_id or f"facelock_pub_{os.getpid()}",
        )
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        self._client.connect_async(broker, int(port))
        self._client.loop_start()

    def publish(
        self,
        recognized: bool,
        x: Optional[float] = None,
        position: Optional[str] = None,
        confidence: Optional[float] = None,
        frame_width: Optional[int] = None,
    ) -> None:
        now = time.time()
        if now - self._last_sent < self.interval_s:
            return
        self._last_sent = now
        msg: Dict[str, object] = {"recognized": bool(recognized), "timestamp": int(now)}
        if recognized:
            msg["x"] = int(round(x)) if x is not None else 0
            msg["position"] = position or "CENTER"
            if confidence is not None:
                msg["confidence"] = round(float(confidence), 4)
        if frame_width is not None:
            msg["frame_width"] = int(frame_width)
        self._client.publish(self.topic, json.dumps(msg, sort_keys=True))

    def wait_connected(self, timeout: float = 10.0) -> bool:
        """Block up to `timeout` s until the background connect_async finishes.

        Returns True if connected. Publishers built on connect_async buffer (and
        eventually drop) messages sent before the socket is up, so callers should
        warn when this returns False to avoid silently losing messages."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._client.is_connected():
                return True
            time.sleep(0.15)
        return self._client.is_connected()

    def close(self) -> None:
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass