#!/usr/bin/env python3
"""
Test face detection to debug why tracking isn't working.
"""
import cv2
import numpy as np
import os
import time

# Import the face detection class
try:
    from src.recognize import HaarFaceMesh5pt
    print("✅ Successfully imported face detection")
except ImportError as e:
    print(f"❌ Failed to import face detection: {e}")
    exit(1)

def test_face_detection():
    print("🔍 Testing face detection for servo tracking...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
        
    # Fix camera settings for better detection
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
    
    # Initialize face detector
    try:
        print("🤖 Initializing face detector...")
        detector = HaarFaceMesh5pt(min_size=(70, 70), debug=True)
        print("✅ Face detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize face detector: {e}")
        return
    
    print("\n📹 Starting face detection test...")
    print("Instructions:")
    print("- Position your face in front of camera")
    print("- Move left and right to test tracking")
    print("- Press 'q' to quit")
    
    frame_count = 0
    faces_detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
            
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Check frame brightness
        brightness = np.mean(frame)
        
        # Detect faces
        try:
            faces = detector.detect(frame, max_faces=5)
            faces_detected_count += len(faces)
            
            # Draw face detection results
            vis = frame.copy()
            
            for i, face in enumerate(faces):
                # Draw bounding box
                cv2.rectangle(vis, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
                
                # Draw facial landmarks
                for j, (x, y) in enumerate(face.kps.astype(int)):
                    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][j]
                    cv2.circle(vis, (x, y), 3, color, -1)
                
                # Calculate face center and servo angle
                face_center_x = (face.x1 + face.x2) / 2.0
                offset = (face_center_x - (w / 2.0)) / (w / 2.0)  # -1 to +1
                servo_angle = 90 + 60 * offset  # Center (90) + gain (60) * offset
                servo_angle = max(0, min(180, servo_angle))
                
                # Display tracking info
                cv2.putText(vis, f"Face {i+1}: Center={face_center_x:.0f}", 
                           (face.x1, face.y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f"Offset: {offset:.2f}", 
                           (face.x1, face.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(vis, f"Servo: {servo_angle:.0f}°", 
                           (face.x1, face.y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Display status
            status_color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
            cv2.putText(vis, f"Faces: {len(faces)} | Frame: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(vis, f"Brightness: {brightness:.1f} | Total detected: {faces_detected_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if len(faces) == 0:
                cv2.putText(vis, "NO FACE DETECTED", 
                           (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(vis, "Try: Better lighting, face camera directly", 
                           (10, h//2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(vis, f"✓ TRACKING ACTIVE", 
                           (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Face Detection Test", vis)
            
        except Exception as e:
            print(f"❌ Error in face detection: {e}")
            cv2.putText(frame, f"ERROR: {str(e)[:50]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Test Results:")
    print(f"   Total frames: {frame_count}")
    print(f"   Total faces detected: {faces_detected_count}")
    print(f"   Detection rate: {faces_detected_count/frame_count*100:.1f}% of frames")
    
    if faces_detected_count == 0:
        print(f"\n❌ No faces detected! Possible issues:")
        print(f"   - Camera too dark (brightness was ~{brightness:.1f})")
        print(f"   - Face not visible to camera")
        print(f"   - MediaPipe model issues")
        print(f"   - Lighting conditions")
    else:
        print(f"\n✅ Face detection working! Should enable servo tracking.")

if __name__ == "__main__":
    test_face_detection()