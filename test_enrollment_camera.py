#!/usr/bin/env python3
"""
Test the camera feed in enrollment to debug the black screen issue.
"""
import cv2
import numpy as np

def test_enrollment_camera():
    print("🎥 Testing enrollment camera feed...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("📹 Camera opened successfully")
    print("Press 'q' to quit, 'r' to show raw frame, 'g' to show grayscale")
    
    frame_count = 0
    show_mode = "normal"  # normal, raw, gray
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Debug info
        h, w = frame.shape[:2]
        
        if show_mode == "raw":
            # Show completely unprocessed frame
            cv2.putText(display_frame, f"RAW FRAME {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif show_mode == "gray":
            # Show grayscale version
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            cv2.putText(display_frame, f"GRAYSCALE {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Normal mode with debug info
            cv2.putText(display_frame, f"Frame {frame_count} - {w}x{h}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Mode: {show_mode} (r=raw, g=gray, n=normal)", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Check if frame is all black
        if np.all(frame == 0):
            cv2.putText(display_frame, "WARNING: ALL BLACK FRAME!", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Check frame statistics
        mean_val = np.mean(frame)
        cv2.putText(display_frame, f"Mean brightness: {mean_val:.1f}", 
                   (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Enrollment Camera Test", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            show_mode = "raw"
            print("📸 Switched to RAW mode")
        elif key == ord('g'):
            show_mode = "gray"
            print("📸 Switched to GRAYSCALE mode")
        elif key == ord('n'):
            show_mode = "normal"
            print("📸 Switched to NORMAL mode")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed")

if __name__ == "__main__":
    test_enrollment_camera()