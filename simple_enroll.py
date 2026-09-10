#!/usr/bin/env python3
"""
Simplified enrollment that works better with face detection issues.
"""
import cv2
import numpy as np
import os
from pathlib import Path

def simple_enrollment():
    name = input("Enter person name to enroll: ").strip()
    if not name:
        print("No name provided. Exiting.")
        return
    
    # Create directories
    data_dir = Path("data")
    crops_dir = data_dir / "crops" / name
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎯 Simple enrollment for: {name}")
    print("This will capture face images for training")
    
    # Initialize camera
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.7)
    
    # Load basic face detector for enrollment
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_path)
    
    if face_cascade.empty():
        print("❌ Could not load face detector")
        return
    
    print(f"\n📸 SIMPLE ENROLLMENT MODE")
    print(f"Controls:")
    print(f"  SPACE = Capture face (when detected)")
    print(f"  'a' = Auto-capture mode (captures automatically)")
    print(f"  's' = Save all captured images") 
    print(f"  'q' = Quit")
    print(f"\nTarget: Capture 5-10 good face images")
    
    captured_images = []
    capture_count = 0
    auto_mode = False
    auto_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with Haar cascade (more reliable)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        
        # Create display frame
        display_frame = frame.copy()
        
        face_detected = len(faces) > 0
        
        if face_detected:
            # Draw face rectangles
            for (x, y, w_face, h_face) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
                cv2.putText(display_frame, "FACE DETECTED", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Status overlay
        cv2.putText(display_frame, f"ENROLL: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Captured: {capture_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if auto_mode:
            cv2.putText(display_frame, "AUTO: ON", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "AUTO: OFF", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if face_detected:
            cv2.putText(display_frame, "Press SPACE to capture", (10, h-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Position face in camera", (10, h-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(display_frame, "SPACE=capture | a=auto | s=save | q=quit", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Simple Enrollment", display_frame)
        
        # Auto capture logic
        if auto_mode and face_detected:
            auto_counter += 1
            if auto_counter >= 30:  # Capture every 30 frames (1 second at 30fps)
                # Capture largest face
                if len(faces) > 0:
                    areas = [w_face * h_face for (x, y, w_face, h_face) in faces]
                    best_idx = np.argmax(areas)
                    x, y, w_face, h_face = faces[best_idx]
                    
                    # Crop face with some padding
                    padding = 0.3
                    px, py = int(w_face * padding), int(h_face * padding)
                    crop_x1 = max(0, x - px)
                    crop_y1 = max(0, y - py)
                    crop_x2 = min(w, x + w_face + px)
                    crop_y2 = min(h, y + h_face + py)
                    
                    cropped_face = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if cropped_face.size > 0:
                        captured_images.append(cropped_face.copy())
                        capture_count += 1
                        print(f"📸 Auto-captured image {capture_count}")
                        auto_counter = 0
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and face_detected:  # Space to capture
            # Capture largest face
            if len(faces) > 0:
                areas = [w_face * h_face for (x, y, w_face, h_face) in faces]
                best_idx = np.argmax(areas)
                x, y, w_face, h_face = faces[best_idx]
                
                # Crop face with padding
                padding = 0.3
                px, py = int(w_face * padding), int(h_face * padding)
                crop_x1 = max(0, x - px)
                crop_y1 = max(0, y - py)
                crop_x2 = min(w, x + w_face + px)
                crop_y2 = min(h, y + h_face + py)
                
                cropped_face = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                if cropped_face.size > 0:
                    captured_images.append(cropped_face.copy())
                    capture_count += 1
                    print(f"📸 Captured image {capture_count}")
        
        elif key == ord('a'):  # Toggle auto mode
            auto_mode = not auto_mode
            auto_counter = 0
            print(f"🤖 Auto mode: {'ON' if auto_mode else 'OFF'}")
        
        elif key == ord('s'):  # Save captured images
            if len(captured_images) > 0:
                print(f"\n💾 Saving {len(captured_images)} captured images...")
                for i, img in enumerate(captured_images):
                    filename = crops_dir / f"{name}_{i:03d}.jpg"
                    cv2.imwrite(str(filename), img)
                    print(f"   Saved: {filename}")
                
                print(f"✅ Successfully saved {len(captured_images)} face images for '{name}'")
                print(f"📁 Location: {crops_dir}")
                print(f"\n🎯 Now you can run face recognition:")
                print(f"   python -m src.recognize")
                break
            else:
                print("❌ No images captured yet!")
        
        elif key == ord('q'):  # Quit
            print("❌ Enrollment cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    simple_enrollment()