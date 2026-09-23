"""
Face signals for Part 2: EAR (blink / eyes-closed) and smile score.

Faithful port of the reference GA2 `src/face_signals.py`, with one internal
change: the landmarking runs through MediaPipe's Tasks FaceLandmarker (the
same backend the rest of this repo already uses, see src/haar_5pt.py /
src/recognize.py) instead of the legacy `mp.solutions.face_mesh` API. The
468-point topology, the indices, the blink state machine and the smile
ratio are identical, so the signals mean the same thing.

Signals produced per locked face:
    ear          mean Eye Aspect Ratio of the two eyes (Soukupova & Cech 2016)
    blink        True when a short low-EAR dip (>= blink_min_frames,
                 <= blink_max_frames) closes and re-opens
    eyes_closed  True when the low-EAR dip lasts >= closed_frames, i.e. a
                 sustained close, not a blink
    smile_score  mouth_width / face_width (raw measure, shown on screen)
    smiling      adaptive: True when the current smile_score rises at least
                 smile_delta_on above the person's own slowly-tracked neutral
                 ratio and STAYS there for smile_on_frames consecutive frames
                 (talking/mouth-jitter spikes do not fire it), and stays True
                 until it sits below smile_delta_off for smile_off_frames.
                 A fixed threshold is unreliable because neutral mouth width
                 differs per person and per camera.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as _e:  # pragma: no cover
    mp = None
    _MP_IMPORT_ERROR = _e

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
FACE_LEFT, FACE_RIGHT = 234, 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, idx: Tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)
    width = max(distance(p1, p4), 1e-6)
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)


@dataclass
class FaceSignals:
    ear: float
    blink: bool
    eyes_closed: bool
    smile_score: float
    smiling: bool
    smile_neutral: float = 0.0
    smile_delta: float = 0.0


class FaceSignalExtractor:
    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        smile_delta_on: float = 0.05,
        smile_delta_off: float = 0.035,
        neutral_rate: float = 0.05,
        smile_on_frames: int = 4,
        smile_off_frames: int = 6,
        landmarker_path: Optional[os.PathLike] = None,
    ):
        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames
        self.smile_delta_on = smile_delta_on
        self.smile_delta_off = smile_delta_off
        self.neutral_rate = neutral_rate
        self.smile_on_frames = max(1, int(smile_on_frames))
        self.smile_off_frames = max(1, int(smile_off_frames))
        self.low_ear_frames = 0
        self.smiling = False
        self._neutral = None
        self._above_frames = 0
        self._below_frames = 0

        if mp is None:
            raise RuntimeError(f"mediapipe import failed: {_MP_IMPORT_ERROR}")

        if landmarker_path is None:
            landmarker_path = ROOT / "models" / "face_landmarker.task"

        if not os.path.exists(landmarker_path):
            print("[SIG] Downloading face_landmarker.task ...")
            url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                   "face_landmarker/float16/latest/face_landmarker.task")
            import urllib.request
            os.makedirs(os.path.dirname(landmarker_path), exist_ok=True)
            urllib.request.urlretrieve(url, str(landmarker_path))
            print(f"[SIG] Model at {landmarker_path}")

        base_options = python.BaseOptions(model_asset_path=str(landmarker_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def reset(self) -> None:
        self.low_ear_frames = 0
        self.smiling = False
        self._neutral = None
        self._above_frames = 0
        self._below_frames = 0

    def close(self) -> None:
        try:
            self.landmarker.close()
        except Exception:
            pass

    def analyze(self, frame: np.ndarray, bbox) -> Optional[FaceSignals]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.12 * bw), int(0.18 * bh)
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(roi, cv2.COLOR_BGR2RGB),
        )
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        rh, rw = roi.shape[:2]
        lm = result.face_landmarks[0]
        points = np.array(
            [[p.x * rw + rx1, p.y * rh + ry1] for p in lm],
            dtype=np.float32,
        )

        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)
        ear = 0.5 * (left_ear + right_ear)

        blink = False
        if ear < self.ear_threshold:
            self.low_ear_frames += 1
        else:
            if self.blink_min_frames <= self.low_ear_frames <= self.blink_max_frames:
                blink = True
            self.low_ear_frames = 0
        eyes_closed = self.low_ear_frames >= self.closed_frames

        face_width = max(distance(points[FACE_LEFT], points[FACE_RIGHT]), 1e-6)
        mouth_width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
        smile_score = mouth_width / face_width

        self._update_smile(smile_score)

        delta = 0.0 if self._neutral is None else smile_score - self._neutral
        return FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
            smile_neutral=0.0 if self._neutral is None else self._neutral,
            smile_delta=delta,
        )

    def _update_smile(self, score: float) -> None:
        """
        Adaptive smile: learn the person's neutral mouth-width ratio and treat
        a rise above it (by smile_delta_on) as a smile. The rise must hold for
        smile_on_frames consecutive frames so talking / mouth-jitter spikes
        (single-frame width blips) do not fire it. Once smiling it persists
        until the score stays below smile_delta_off for smile_off_frames,
        so it neither chatters nor latches. Neutral only creeps on calm frames
        (score near neutral), so it anchors to the resting mouth instead of
        chasing a grin or a big mouth-open.
        """
        if self._neutral is None:
            self._neutral = score
            return
        delta = score - self._neutral
        if not self.smiling:
            if abs(delta) <= self.smile_delta_off:
                self._neutral += self.neutral_rate * delta
            if delta > self.smile_delta_on:
                self._above_frames += 1
            else:
                self._above_frames = 0
            if self._above_frames >= self.smile_on_frames:
                self.smiling = True
                self._below_frames = 0
        else:
            if delta < self.smile_delta_off:
                self._below_frames += 1
            else:
                self._below_frames = 0
            if self._below_frames >= self.smile_off_frames:
                self.smiling = False
                self._above_frames = 0