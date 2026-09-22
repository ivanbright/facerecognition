#!/usr/bin/env python3
"""
Complete working face recognition system that actually uses the ArcFace ONNX model.
This is the real face recognition + servo tracking system you want.
"""
import cv2
import json
import numpy as np
import onnxruntime as ort
import pickle
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.haar_5pt import Haar5ptDetector, align_face_5pt
from src.embed import ArcFaceEmbedderONNX

# For servo tracking
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

CONFIG_FILE = "camera_config.json"


def load_camera_index():
    """Read the camera index chosen by pick_camera.py (default 2)."""
    try:
        with open(CONFIG_FILE) as f:
            return int(json.load(f)["camera_index"])
    except Exception:
        return 2


def open_camera(index):
    """Open a camera, keeping its native format/field of view."""
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        try:
            cap = cv2.VideoCapture(index, api)
        except Exception:
            continue
        if not cap.isOpened():
            cap.release()
            continue
        try:
            ok, frame = cap.read()
            good = ok and frame is not None and float(frame.mean()) > 10
        except Exception:
            good = False
        if good:
            print(f"[CAM] Camera index {index} via {cap.getBackendName()}")
            return cap
        cap.release()

    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"[CAM] Camera index {index} (default backend)")
        return cap
    return None

class ArcFaceRecognizer:
    """ArcFace recognition with five-point alignment and a shared database."""
    
    def __init__(self, model_path="models/embedder_arcface.onnx"):
        print("[AI] Initializing ArcFace Recognition System...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")
        
        self.detector = Haar5ptDetector(
            min_size=(70, 70),
            smooth_alpha=0.80,
            debug=False,
        )
        self.embedder = ArcFaceEmbedderONNX(
            model_path=model_path,
            input_size=(112, 112),
            debug=False,
        )
        self.face_db = {}
        self.load_database()
        self.threshold = 0.6
        
        print(f"[STATS] Loaded {len(self.face_db)} enrolled faces")
    
    def get_aligned_face(self, frame):
        """Return the largest detected face aligned to 112x112."""
        faces = self.detector.detect(frame, max_faces=1)
        if not faces:
            return None, None
        
        face = faces[0]
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return aligned, face

    def detect_faces(self, frame, max_faces=5):
        """Detect and return all confirmed five-point faces."""
        return self.detector.detect(frame, max_faces=max_faces)

    def recognize_frame(self, frame):
        """Recognize each detected face and return aligned, name, similarity."""
        results = []
        for face in self.detect_faces(frame, max_faces=5):
            aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
            name, similarity = self.recognize_face(aligned)
            results.append((face, name, similarity))
        return results
    
    def get_embedding(self, face_img):
        """Get a normalized 512D embedding from an aligned face."""
        try:
            if face_img.shape[:2] != (112, 112):
                face_img = cv2.resize(face_img, (112, 112))
            result = self.embedder.embed(face_img)
            return result.embedding
        except Exception as e:
            print(f"[ERR] Error getting embedding: {e}")
            return None
    
    def cosine_similarity(self, emb1, emb2):
        return float(np.dot(emb1, emb2))
    
    def recognize_face(self, face_img):
        embedding = self.get_embedding(face_img)
        if embedding is None:
            return None, 0.0
        
        best_match = None
        best_similarity = -1.0
        for name, enrolled_embedding in self.face_db.items():
            similarity = self.cosine_similarity(embedding, enrolled_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name
        
        if best_similarity > self.threshold:
            return best_match, best_similarity
        return None, best_similarity
    
    def enroll_person(self, name, auto_save_every=5, save_on_quit=True):
        """Enroll a person using five-point aligned faces and save the shared DB."""
        print(f"\n[CAPTURE] Enrolling: {name}")
        print("[CAPTURE] SPACE=capture | s=save | q=quit")

        cap = open_camera(load_camera_index())
        if cap is None:
            print("[ERR] USB camera not available")
            return False
        
        captured_embeddings = []
        last_auto_save = 0
        cv2.destroyAllWindows()
        cv2.namedWindow("ArcFace Enrollment", cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                aligned, face = self.get_aligned_face(frame)
                display = frame.copy()
                
                if face is not None:
                    cv2.rectangle(display, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                    for (px, py) in face.kps.astype(int):
                        cv2.circle(display, (int(px), int(py)), 4, (0, 255, 0), -1)
                    cv2.putText(display, "5PT LOCKED", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "NO FACE - POSITION FACE", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(display, f"Captured: {len(captured_embeddings)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display, "SPACE=capture | s=save | q=quit", (10, frame.shape[0] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow("ArcFace Enrollment", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" ") and aligned is not None:
                    embedding = self.get_embedding(aligned)
                    if embedding is not None:
                        captured_embeddings.append(embedding)
                        print(f"[CAPTURE] Face {len(captured_embeddings)} embedded")
                        
                        if len(captured_embeddings) % auto_save_every == 0:
                            if self.save_template(name, captured_embeddings):
                                last_auto_save = len(captured_embeddings)
                                print(f"[SAVE] Progress saved after {len(captured_embeddings)} samples")
                    
                    continue
                
                if key == ord("s") and captured_embeddings:
                    return self.finalize_enrollment(name, captured_embeddings)
                
                if key == ord("q"):
                    if captured_embeddings and save_on_quit:
                        return self.finalize_enrollment(name, captured_embeddings)
                    print("[WARN] Enrollment cancelled")
                    return False
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def save_template(self, name, embeddings):
        if not embeddings:
            return False
        template = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(template)
        if norm == 0:
            return False
        self.face_db[name] = template / norm
        self.save_database()
        return True
    
    def finalize_enrollment(self, name, embeddings):
        if not embeddings:
            print("[ERR] No embeddings to save")
            return False
        if self.save_template(name, embeddings):
            print(f"[OK] {name} enrolled with {len(embeddings)} aligned samples")
            print("[TARGET] Run mode 2 to recognize and follow this trained face")
            return True
        print("[ERR] Failed to create enrollment template")
        return False
    
    def save_database(self, db_path="data/face_database.pkl"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, "wb") as file:
            pickle.dump(self.face_db, file)
        print(f"[SAVE] Saved {len(self.face_db)} faces to {db_path}")
    
    def load_database(self, db_path="data/face_database.pkl"):
        if not os.path.exists(db_path):
            print("[DB] No existing database found")
            self.face_db = {}
            return
        
        try:
            with open(db_path, "rb") as file:
                loaded = pickle.load(file)
            normalized = {}
            for name, embedding in loaded.items():
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    normalized[name] = vector / norm
            self.face_db = normalized
            print(f"[DB] Loaded {len(self.face_db)} enrolled faces")
        except Exception as e:
            print(f"[WARN] Could not load database: {e}")
            self.face_db = {}

class ServoController:
    """ESP8266 servo controller for face tracking."""
    
    def __init__(self, port="COM4", baud=115200):
        self.port = port
        self.baud = baud
        self.serial_conn = None
        self.servo_angle = 90.0
        self.smoothed_angle = 90.0
        self.last_command_time = 0.0
        self.min_command_interval = 0.05
        self.angle_threshold = 0.5
        self.face_lost_timeout = 1.0
        self.smoothing_factor = 0.30
        self.max_step_per_command = 4.0
        self.last_face_time = 0.0
        
        if SERIAL_AVAILABLE:
            try:
                print(f"[SERIAL] Connecting to servo on {port}...")
                import serial.tools.list_ports
                ports = [p.device for p in serial.tools.list_ports.comports()]
                if port not in ports:
                    print(f"[WARN]  Port {port} not found")
                    self.serial_conn = None
                else:
                    self.serial_conn = serial.Serial(port, baud, timeout=3, write_timeout=2)
                    time.sleep(3)
                    self.serial_conn.reset_input_buffer()
                    time.sleep(0.5)
                    
                    test_success = self._test_communication()
                    if test_success:
                        print(f"[SIGNAL] Servo controller connected: {port}")
                    else:
                        print(f"[WARN]  Servo connected but not responding properly")
                        
            except Exception as e:
                print(f"[WARN]  Could not connect to servo: {e}")
                print(f"[WARN]  Try power cycling the ESP8266")
                self.serial_conn = None
        else:
            print("[WARN]  PySerial not available - servo tracking disabled")
    
    def _test_communication(self):
        """Test if ESP8266 is responding properly (retries while it boots)."""
        for attempt in range(5):
            try:
                self.serial_conn.reset_input_buffer()
                self.serial_conn.write(b"90\r\n")
                self.serial_conn.flush()
                time.sleep(0.8)
                if self.serial_conn.in_waiting > 0:
                    response = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if "tracker" in response or "angle" in response:
                        return True
                print(f"[WARN]  Servo handshake attempt {attempt + 1}/5 no valid response")
            except Exception as e:
                print(f"[WARN]  Handshake attempt {attempt + 1}/5 failed: {e}")
                time.sleep(0.5)
                try:
                    self.serial_conn.close()
                    self.serial_conn = serial.Serial(self.port, self.baud, timeout=3, write_timeout=2)
                except Exception:
                    pass
            time.sleep(0.5)
        return False
    
    def move_to_angle(self, angle):
        """Move the servo smoothly toward a target angle."""
        if self.serial_conn is None:
            return

        angle = max(0.0, min(180.0, float(angle)))
        current_time = time.time()
        if current_time - self.last_command_time < self.min_command_interval:
            return

        self.smoothed_angle += (angle - self.smoothed_angle) * self.smoothing_factor
        step = self.smoothed_angle - self.servo_angle
        step = max(-self.max_step_per_command, min(self.max_step_per_command, step))
        self.smoothed_angle = self.servo_angle + step
        if abs(self.smoothed_angle - self.servo_angle) < self.angle_threshold:
            return

        try:
            command = f"{int(round(self.smoothed_angle))}\r\n"
            self.serial_conn.write(command.encode())
            self.serial_conn.flush()

            self.servo_angle = self.smoothed_angle
            self.last_command_time = current_time
        except Exception:
            if not hasattr(self, 'error_count'):
                self.error_count = 0
            self.error_count += 1
            if self.error_count > 20:
                try:
                    self.serial_conn.close()
                except:
                    pass
                self.serial_conn = None

    def track_face(self, face_center_x, frame_width):
        """Track a face smoothly or return the servo to center when it is lost."""
        if face_center_x is None:
            if time.time() - self.last_face_time >= self.face_lost_timeout:
                self.last_face_time = time.time()
                self.center()
            return

        self.last_face_time = time.time()
        offset = (face_center_x - frame_width / 2) / (frame_width / 2)
        target_angle = 90 - offset * 60
        self.move_to_angle(target_angle)

    def center(self):
        """Return the servo smoothly to the center position."""
        self.move_to_angle(90.0)
    
    def close(self):
        """Close serial connection."""
        if self.serial_conn is not None and self.serial_conn.is_open:
            try:
                self.center()  # Return to center
                time.sleep(0.2)
                self.serial_conn.close()
                print("[SIGNAL] Servo connection closed")
            except Exception:
                pass

def main_enrollment():
    """Enrollment mode - train faces for recognition."""
    print("[CAPTURE] ARCFACE ENROLLMENT MODE")
    
    recognizer = ArcFaceRecognizer()
    name = input("Enter person name to enroll: ").strip()
    if not name:
        return
    
    recognizer.enroll_person(name)

def main_recognition():
    """Recognition mode - detect, recognize, and track enrolled faces."""
    print("[TARGET] ARCFACE RECOGNITION + SERVO TRACKING")
    
    recognizer = ArcFaceRecognizer()
    servo = ServoController()
    
    if len(recognizer.face_db) == 0:
        print("[ERR] No enrolled faces found! Run enrollment first.")
        return
    
    cap = open_camera(load_camera_index())
    if cap is None:
        print("[ERR] Camera not available")
        return
    
    print(f"[TARGET] Tracking enrolled faces: {list(recognizer.face_db.keys())}")
    print("Press 'q' to quit")
    
    cv2.destroyAllWindows()
    cv2.namedWindow("ArcFace Recognition + Tracking", cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                retries = getattr(main_recognition, '_retries', 0) + 1
                main_recognition._retries = retries
                if retries > 30:
                    print("[ERR] Camera lost")
                    break
                continue
            main_recognition._retries = 0
            
            h, w = frame.shape[:2]
            results = recognizer.recognize_frame(frame)
            display_frame = frame.copy()
            recognized_faces = []
            
            for face, name, similarity in results:
                if name is not None:
                    recognized_faces.append(face)
                    color = (0, 255, 0)
                    label = f"{name} ({similarity:.2f})"
                else:
                    color = (0, 0, 255)
                    label = f"Unknown ({similarity:.2f})"
                
                cv2.rectangle(display_frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
                for (px, py) in face.kps.astype(int):
                    cv2.circle(display_frame, (int(px), int(py)), 3, color, -1)
                cv2.putText(
                    display_frame,
                    label,
                    (face.x1, max(0, face.y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            
            if recognized_faces:
                tracked_face = recognized_faces[0]
                face_center_x = (tracked_face.x1 + tracked_face.x2) // 2
                servo.track_face(face_center_x, w)
                cv2.circle(display_frame, (face_center_x, tracked_face.y1), 10, (0, 255, 255), 3)
                cv2.putText(display_frame, "TRACKING", (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                servo.track_face(None, w)
                cv2.putText(display_frame, "NO TRAINED FACE - CENTERING SERVO", (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.putText(display_frame, f"ArcFace Recognition | Enrolled: {len(recognizer.face_db)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("ArcFace Recognition + Tracking", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo.close()

if __name__ == "__main__":
    print("[AI] ArcFace Face Recognition System")
    print("=" * 50)
    
    mode = input("Choose mode:\n1. Enroll faces (train)\n2. Recognition + tracking\nEnter (1 or 2): ").strip()
    
    if mode == "1":
        main_enrollment()
    elif mode == "2":
        main_recognition()
    else:
        print("Invalid choice")