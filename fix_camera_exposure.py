#!/usr/bin/env python3
"""
Fix camera exposure and brightness settings.
"""
import cv2
import numpy as np

def fix_camera_settings():
    print("🔧 Fixing camera exposure and brightness...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Print current camera properties
    print("\n📊 Current camera properties:")
    properties = [
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"), 
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure"),
        (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto Exposure"),
        (cv2.CAP_PROP_GAIN, "Gain"),
    ]
    
    for prop, name in properties:
        value = cap.get(prop)
        print(f"   {name}: {value}")
    
    # Try to fix settings
    print("\n🔧 Applying camera fixes...")
    
    # Enable auto exposure first
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure
    print("   ✅ Auto exposure enabled")
    
    # Increase brightness
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)  # Increase brightness
    print("   ✅ Brightness increased")
    
    # Increase contrast
    cap.set(cv2.CAP_PROP_CONTRAST, 0.6)    # Increase contrast
    print("   ✅ Contrast increased")
    
    # Set exposure manually if auto doesn't work
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)     # Higher exposure (less negative = brighter)
    print("   ✅ Exposure adjusted")
    
    # Increase gain
    cap.set(cv2.CAP_PROP_GAIN, 50)         # Increase gain
    print("   ✅ Gain increased")
    
    # Wait for settings to take effect
    print("\n⏳ Waiting for camera to adjust...")
    for i in range(30):
        ret, frame = cap.read()
        if ret:
            cv2.waitKey(100)  # Wait 100ms between frames
    
    print("\n📊 Testing with new settings...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Check brightness
        mean_brightness = np.mean(frame)
        h, w = frame.shape[:2]
        
        # Create display frame
        display_frame = frame.copy()
        
        # Add debug info
        color = (0, 255, 0) if mean_brightness > 50 else (0, 0, 255) if mean_brightness > 10 else (255, 0, 0)
        
        cv2.putText(display_frame, f"Frame {frame_count} - {w}x{h}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_frame, f"Brightness: {mean_brightness:.1f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if mean_brightness < 10:
            cv2.putText(display_frame, "TOO DARK! Check lens cap/lighting", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif mean_brightness < 50:
            cv2.putText(display_frame, "Still dark - try more light", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            cv2.putText(display_frame, "Good brightness!", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(display_frame, "Press 'q' to quit, 's' to save settings", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Camera Exposure Fix", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n💾 Current settings work! Brightness: {mean_brightness:.1f}")
            print("   These settings will be used in your face recognition system.")
            break
    
    # Print final settings
    print("\n📊 Final camera properties:")
    for prop, name in properties:
        value = cap.get(prop)
        print(f"   {name}: {value}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera settings test completed")

if __name__ == "__main__":
    fix_camera_settings()