#!/usr/bin/env python3
"""
Fix camera focus and quality settings for better face enrollment and recognition.
"""
import cv2
import numpy as np

def fix_camera_focus():
    print("🎯 Fixing camera focus and quality for face recognition...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
        
    print("📊 Original camera settings:")
    settings = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_FOCUS, "Focus"),
        (cv2.CAP_PROP_AUTOFOCUS, "Auto Focus"),
        (cv2.CAP_PROP_SHARPNESS, "Sharpness"),
        (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto Exposure"),
        (cv2.CAP_PROP_EXPOSURE, "Exposure"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
    ]
    
    for prop, name in settings:
        value = cap.get(prop)
        print(f"   {name}: {value}")
    
    print("\n🔧 Applying optimal settings for face recognition...")
    
    # Set resolution for good quality but manageable processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("   ✅ Resolution: 640x480")
    
    # Enable autofocus for sharp images
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    print("   ✅ Autofocus enabled")
    
    # Set manual focus to infinity (good for face distances)
    cap.set(cv2.CAP_PROP_FOCUS, 0)  # Focus to infinity
    print("   ✅ Focus set to infinity")
    
    # Increase sharpness for better edge detection
    cap.set(cv2.CAP_PROP_SHARPNESS, 100)  # Max sharpness
    print("   ✅ Sharpness maximized")
    
    # Optimize exposure for faces
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Slightly underexpose to avoid blur
    print("   ✅ Exposure optimized")
    
    # Good contrast and brightness for face features
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Moderate brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 0.7)    # High contrast for better features
    print("   ✅ Brightness and contrast optimized")
    
    # Set FPS for stable processing
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("   ✅ FPS set to 30")
    
    print("\n⏳ Letting camera adjust to new settings...")
    # Let camera settle with new settings
    for i in range(60):  # 2 seconds at 30fps
        ret, frame = cap.read()
        cv2.waitKey(33)
    
    print("\n📹 Testing new camera quality...")
    print("Instructions:")
    print("- Position face 1-3 feet from camera")
    print("- Ensure good lighting on your face")
    print("- Press 's' to save good settings")
    print("- Press 'f' to try different focus")
    print("- Press 'q' to quit")
    
    focus_values = [0, 25, 50, 75, 100, 150, 200, 255]  # Different focus distances
    current_focus_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate image sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Add quality metrics overlay
        h, w = frame.shape[:2]
        brightness = np.mean(frame)
        
        # Quality indicators
        if sharpness > 100:
            sharpness_color = (0, 255, 0)  # Green = good
            sharpness_text = "SHARP"
        elif sharpness > 50:
            sharpness_color = (0, 165, 255)  # Orange = ok
            sharpness_text = "OK"
        else:
            sharpness_color = (0, 0, 255)  # Red = blurry
            sharpness_text = "BLURRY"
        
        cv2.putText(frame, f"Sharpness: {sharpness:.1f} ({sharpness_text})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sharpness_color, 2)
        cv2.putText(frame, f"Brightness: {brightness:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Focus setting: {focus_values[current_focus_idx]}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Good for faces: Sharpness > 100, Brightness 80-150", 
                   (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press: 'f'=focus, 's'=save settings, 'q'=quit", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Camera Quality Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            # Try next focus setting
            current_focus_idx = (current_focus_idx + 1) % len(focus_values)
            new_focus = focus_values[current_focus_idx]
            cap.set(cv2.CAP_PROP_FOCUS, new_focus)
            print(f"🎯 Trying focus: {new_focus}")
            # Wait for focus to settle
            for i in range(10):
                ret, frame = cap.read()
                cv2.waitKey(50)
        elif key == ord('s'):
            print(f"\n💾 Saving current settings:")
            print(f"   Sharpness: {sharpness:.1f}")
            print(f"   Brightness: {brightness:.1f}")
            print(f"   Focus: {focus_values[current_focus_idx]}")
            if sharpness > 100:
                print("✅ Good quality for face recognition!")
            else:
                print("⚠️  May need better focus/lighting for optimal results")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n🎯 Camera optimization complete!")
    print("Now try running enrollment again with better image quality.")

if __name__ == "__main__":
    fix_camera_focus()