#!/usr/bin/env python3
"""
Manual enrollment - captures images without requiring face detection.
Just position yourself and press SPACE to capture.
"""
import cv2
import numpy as np
import os
from pathlib import Path

def manual_enrollment():
    name = input("Enter person name to enroll: ").strip()
    if not name:
        print("No name provided. Exiting.")
        return
    
    # Create directories
    data_dir = Path("data")
    crops_dir = data_dir / "crops" / name
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📸 MANUAL ENROLLMENT for: {name}")
    print("This captures images directly without face detection")
    print("Just position yourself and press SPACE to capture!")
    
    # Initialize camera
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"\nControls:")
    print(f"  SPACE = Capture current frame")
    print(f"  's' = Save all captured images")
    print(f"  'q' = Quit")
    print(f"\nInstructions:")
    print(f"  1. Position your face in center of camera")
    print(f"  2. Press SPACE to capture (works anytime!)")
    print(f"  3. Move slightly and capture from different angles")
    print(f"  4. Capture 5-10 images total")
    print(f"  5. Press 's' to save for face recognition")
    
    captured_images = []
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        # Create display frame with overlay
        display_frame = frame.copy()
        
        # Add crosshair to help positioning
        center_x, center_y = w // 2, h // 2
        cv2.line(display_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(display_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        
        # Add face guide rectangle
        guide_size = 200
        guide_x1 = center_x - guide_size // 2
        guide_y1 = center_y - guide_size // 2
        guide_x2 = center_x + guide_size // 2
        guide_y2 = center_y + guide_size // 2
        cv2.rectangle(display_frame, (guide_x1, guide_y1), (guide_x2, guide_y2), (255, 255, 0), 2)
        
        # Status overlay
        cv2.putText(display_frame, f"MANUAL ENROLL: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Captured: {capture_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Position face in yellow box", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, "Press SPACE to capture anytime!", (10, h-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE=capture | s=save | q=quit", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Manual Enrollment", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space - always captures!
            # Crop a face-sized region from center
            crop_size = 300  # Larger crop area
            crop_x1 = max(0, center_x - crop_size // 2)
            crop_y1 = max(0, center_y - crop_size // 2)
            crop_x2 = min(w, center_x + crop_size // 2)
            crop_y2 = min(h, center_y + crop_size // 2)
            
            cropped_face = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped_face.size > 0:
                captured_images.append(cropped_face.copy())
                capture_count += 1
                print(f"📸 Captured image {capture_count} - crop size: {cropped_face.shape}")
                
                # Show brief flash to confirm capture
                flash_frame = display_frame.copy()
                cv2.rectangle(flash_frame, (0, 0), (w, h), (255, 255, 255), 20)
                cv2.imshow("Manual Enrollment", flash_frame)
                cv2.waitKey(100)
        
        elif key == ord('s'):  # Save all captured images
            if len(captured_images) > 0:
                print(f"\n💾 Saving {len(captured_images)} captured images...")
                for i, img in enumerate(captured_images):
                    # Resize to standard size for face recognition
                    resized = cv2.resize(img, (224, 224))
                    filename = crops_dir / f"{name}_{i:03d}.jpg"
                    cv2.imwrite(str(filename), resized)
                    print(f"   Saved: {filename} ({resized.shape})")
                
                print(f"\n✅ SUCCESS! Saved {len(captured_images)} face images for '{name}'")
                print(f"📁 Location: {crops_dir}")
                
                # Now create a simple face database entry
                print(f"\n🤖 Creating face database entry...")
                try:
                    # This is a simplified approach - just save the images
                    # The recognition system will need to be updated to use these images
                    db_file = Path("data/db") / f"{name}_manual.txt"
                    db_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(db_file, 'w') as f:
                        f.write(f"Manual enrollment for {name}\n")
                        f.write(f"Images: {len(captured_images)}\n")
                        f.write(f"Location: {crops_dir}\n")
                    
                    print(f"📝 Database entry created: {db_file}")
                except Exception as e:
                    print(f"⚠️  Could not create database entry: {e}")
                
                print(f"\n🎯 NEXT STEPS:")
                print(f"1. Your face images are now saved")
                print(f"2. The system needs these images to create face embeddings")
                print(f"3. Try running: python -m src.recognize")
                print(f"   (May need to update recognition to use manual enrollment)")
                break
            else:
                print("❌ No images captured yet! Press SPACE to capture.")
        
        elif key == ord('q'):  # Quit
            print("❌ Enrollment cancelled")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_enrollment()