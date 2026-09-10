#!/usr/bin/env python3
"""
Test basic Haar cascade face detection to see if the issue is with MediaPipe or Haar.
"""
import cv2
import numpy as np

def test_basic_face_detection():
    print("🔍 Testing basic Haar cascade face detection...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
        
    # Load Haar cascade
    try:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(haar_path)
        if face_cascade.empty():
            print(f"❌ Failed to load Haar cascade from {haar_path}")
            return
        print(f"✅ Loaded Haar cascade: {haar_path}")
    except Exception as e:
        print(f"❌ Error loading Haar cascade: {e}")
        return
    
    print("\n📹 Starting basic face detection...")
    print("- Position your face in front of camera")
    print("- Press 'q' to quit, 's' to test servo angle calculation")
    
    frame_count = 0
    faces_detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Convert to grayscale for Haar detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(70, 70)
        )
        
        faces_detected_count += len(faces)
        
        # Draw results
        vis = frame.copy()
        
        for i, (x, y, w_face, h_face) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(vis, (x, y), (x + w_face, y + h_face), (255, 0, 0), 2)
            
            # Calculate face center and servo angle
            face_center_x = x + w_face / 2.0
            offset = (face_center_x - (w / 2.0)) / (w / 2.0)  # -1 to +1
            servo_angle = 90 + 60 * offset  # Center + gain * offset
            servo_angle = max(0, min(180, servo_angle))
            
            # Display info
            cv2.putText(vis, f"Face {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(vis, f"Center: {face_center_x:.0f}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(vis, f"Offset: {offset:.2f}", (x, y + h_face + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(vis, f"Servo: {servo_angle:.0f}°", (x, y + h_face + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw center line and face center
            cv2.line(vis, (w//2, 0), (w//2, h), (0, 255, 255), 1)  # Screen center
            cv2.circle(vis, (int(face_center_x), int(y + h_face/2)), 5, (0, 0, 255), -1)  # Face center
        
        # Status info
        status_color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        cv2.putText(vis, f"Haar Faces: {len(faces)} | Frame: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis, f"Total detected: {faces_detected_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        if len(faces) == 0:
            cv2.putText(vis, "NO FACE - Try facing camera directly", 
                       (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(vis, f"✓ FACE DETECTED - Servo would track", 
                       (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Basic Face Detection", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Test servo command
            x, y, w_face, h_face = faces[0]  # Use first face
            face_center_x = x + w_face / 2.0
            offset = (face_center_x - (w / 2.0)) / (w / 2.0)
            servo_angle = 90 + 60 * offset
            servo_angle = max(0, min(180, servo_angle))
            print(f"📡 Would send to ESP8266: {int(servo_angle)}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Basic Detection Results:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Faces detected: {faces_detected_count}")
    print(f"   Detection rate: {faces_detected_count/max(1,frame_count)*100:.1f}%")
    
    if faces_detected_count > 0:
        print(f"\n✅ Basic Haar detection WORKS!")
        print(f"   The issue might be with MediaPipe integration")
        print(f"   Let's try fixing the MediaPipe pipeline...")
    else:
        print(f"\n❌ Even basic Haar detection failed")
        print(f"   Check: lighting, camera angle, face visibility")

if __name__ == "__main__":
    test_basic_face_detection()