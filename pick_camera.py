#!/usr/bin/env python3
"""
Pick which camera index is the physical webcam you want to use.
Shows a live labeled preview of each detected camera:
  SPACE = use this camera
  Q     = skip to next camera
Saves the choice to camera_config.json (read by working_face_recognition.py).
"""
import cv2
import json
import numpy as np


CONFIG_FILE = "camera_config.json"


def scan_working_indices():
    working = []
    for i in range(6):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        except Exception:
            continue
        if cap.isOpened():
            try:
                ok, frame = cap.read()
                cv2.waitKey(1)
                if ok and frame is not None:
                    working.append(i)
            except Exception:
                pass
        cap.release()
    return working


def main():
    indices = scan_working_indices()
    print(f"[PICK] Detected working camera indices: {indices}")
    if not indices:
        print("[ERR] No working cameras found")
        return

    chosen = None
    for idx in indices:
        print(f"[PICK] Showing camera {idx} (SPACE=use this, Q=skip)")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        win = f"Camera index {idx} (SPACE=use, Q=skip)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        while True:
            ok, frame = cap.read()
            if ok and frame is not None:
                display = frame.copy()
                label = f"INDEX {idx} - SPACE=use this, Q=skip"
                color = (0, 255, 255)
            else:
                display = np.zeros((480, 640, 3), dtype=np.uint8)
                label = f"INDEX {idx} - NO FRAME"
                color = (0, 0, 255)
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord(" "), 13):
                chosen = idx
                break
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyWindow(win)
        if chosen is not None:
            break

    if chosen is None:
        print("[PICK] No camera selected (nothing saved)")
        return

    with open(CONFIG_FILE, "w") as f:
        json.dump({"camera_index": chosen}, f)
    print(f"[PICK] Saved camera index {chosen} -> {CONFIG_FILE}")


if __name__ == "__main__":
    main()