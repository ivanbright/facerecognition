#!/usr/bin/env python3
"""
Face recognition only - no servo control.
Use this to test face recognition first, then add servo later.
"""
import cv2
import numpy as np
import onnxruntime as ort
import pickle
import os

class ArcFaceRecognizer:
    """ArcFace ONNX face recognition system - recognition only."""
    
    def __init__(self, model_path="models/embedder_arcface.onnx"):
        print("🤖 Initializing ArcFace Recognition System...")
        
        # Load ONNX model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")
        
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        print(f"✅ ArcFace ONNX loaded: {input_shape} → 512D embeddings")
        
        # Face detector (Haar - reliable)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)
        
        # Face database
        self.face_db = {}
        self.load_database()
        
        # Recognition threshold
        self.threshold = 0.6
        
        print(f"📊 Loaded {len(self.face_db)} enrolled faces")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for ArcFace ONNX model."""
        face_resized = cv2.resize(face_img, (112, 112))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_batch = np.expand_dims(face_norm, axis=0)
        return face_batch
    
    def get_embedding(self, face_img):
        """Get 512D embedding from face image using ArcFace ONNX."""
        try:
            input_data = self.preprocess_face(face_img)
            embedding = self.session.run([self.output_name], {self.input_name: input_data})[0]
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return None
    
    def cosine_similarity(self, emb1, emb2):
        """Calculate cosine similarity between embeddings."""
        return np.dot(emb1, emb2)
    
    def recognize_face(self, face_img):
        """Recognize face using ArcFace embeddings."""
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
        else:
            return None, best_similarity
    
    def load_database(self, db_path="data/face_database.pkl"):
        """Load face database from disk."""
        if os.path.exists(db_path):
            try:
                with open(db_path, 'rb') as f:
                    self.face_db = pickle.load(f)
                print(f"📂 Loaded database: {len(self.face_db)} faces")
            except Exception as e:
                print(f"⚠️  Could not load database: {e}")
                self.face_db = {}
        else:
            print("📂 No existing database found")
            self.face_db = {}

def main():
    """Recognition mode - detect and recognize faces (no servo)."""
    print("🎯 ARCFACE FACE RECOGNITION (NO SERVO)")
    
    recognizer = ArcFaceRecognizer()
    
    if len(recognizer.face_db) == 0:
        print("❌ No enrolled faces found! Run enrollment first.")
        return
    
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Camera not available")
        return
    
    print(f"🎯 Recognizing enrolled faces: {list(recognizer.face_db.keys())}")
    print("Press 'q' to quit")
    
    target_face_position = None  # Track where target face is
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = recognizer.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(60, 60),
            maxSize=(400, 400),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        display_frame = frame.copy()
        target_face_position = None
        
        for (x, y, face_w, face_h) in faces:
            # Extract face for recognition
            face_crop = frame[y:y+face_h, x:x+face_w]
            
            # Recognize face
            name, similarity = recognizer.recognize_face(face_crop)
            
            if name:  # Recognized enrolled person
                color = (0, 255, 0)  # Green for known person
                label = f"{name} ({similarity:.2f})"
                target_face_position = (x + face_w//2, y + face_h//2)  # Face center
                
                # Calculate servo angle for display
                face_center_x = x + face_w//2
                offset = (face_center_x - w/2) / (w/2)  # -1 to +1
                servo_angle = 90 + offset * 60  # ±60° range
                servo_angle = max(0, min(180, int(servo_angle)))
                
                # Show servo angle that would be sent
                cv2.putText(display_frame, f"Servo: {servo_angle}°", (x, y+face_h+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
            else:
                color = (0, 0, 255)  # Red for unknown person
                label = f"Unknown ({similarity:.2f})"
            
            # Draw face rectangle and label
            cv2.rectangle(display_frame, (x, y), (x+face_w, y+face_h), color, 2)
            cv2.putText(display_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show tracking status
        if target_face_position:
            face_center_x, face_center_y = target_face_position
            cv2.circle(display_frame, (face_center_x, face_center_y), 10, (0, 255, 255), 3)
            cv2.putText(display_frame, "TARGET ACQUIRED", (10, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "NO TARGET", (10, h-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Status overlay
        cv2.putText(display_frame, f"ArcFace Recognition | Enrolled: {len(recognizer.face_db)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition Only", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()