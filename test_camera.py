#!/usr/bin/env python3
"""
Simple camera test script to debug camera issues.
"""
import cv2
import numpy as np

def test_camera():
    print("🎥 Testing camera access...")
    
    # Try different camera indices
    camera_indices = [0, 1, 2, 3]
    
    for idx in camera_indices:
        print(f"\n📹 Trying camera index {idx}...")
        
        try:
            cap = cv2.VideoCapture(idx)
            
            # Set some properties to help with initialization
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                print(f"❌ Camera {idx} could not be opened")
                continue
                
            # Try to read a frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                print(f"❌ Camera {idx} opened but couldn't read frame")
                cap.release()
                continue
                
            # Check frame properties
            h, w = frame.shape[:2]
            print(f"✅ Camera {idx} working!")
            print(f"   Frame size: {w}x{h}")
            print(f"   Frame shape: {frame.shape}")
            print(f"   Frame dtype: {frame.dtype}")
            
            # Show a test frame
            print(f"\n🖼️  Displaying test window for camera {idx}")
            print("   Press 'q' to quit, 's' to save test image, or any other key to try next camera")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                    
                frame_count += 1
                
                # Add some text overlay to confirm the frame is updating
                cv2.putText(frame, f"Camera {idx} - Frame {frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(f"Camera Test - Index {idx}", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("✅ Quitting camera test")
                    cap.release()
                    cv2.destroyAllWindows()
                    return idx  # Return working camera index
                elif key == ord('s'):
                    # Save test image
                    cv2.imwrite(f"test_camera_{idx}.jpg", frame)
                    print(f"💾 Saved test image: test_camera_{idx}.jpg")
                elif key != 255:  # Any other key pressed
                    print(f"🔄 Trying next camera...")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"❌ Error with camera {idx}: {e}")
            
    print(f"\n❌ No working cameras found")
    return None

def check_system_info():
    """Check system information that might affect camera access."""
    print("\n🔍 System Information:")
    
    try:
        import platform
        print(f"   OS: {platform.system()} {platform.release()}")
        print(f"   Python: {platform.python_version()}")
    except:
        pass
        
    try:
        print(f"   OpenCV: {cv2.__version__}")
    except:
        print("   OpenCV: Version unknown")
        
    # Check OpenCV backend info
    try:
        backends = []
        if hasattr(cv2, 'CAP_DSHOW'):
            backends.append("DirectShow")
        if hasattr(cv2, 'CAP_MSMF'):
            backends.append("Media Foundation")
        if hasattr(cv2, 'CAP_V4L2'):
            backends.append("V4L2")
        print(f"   OpenCV backends: {', '.join(backends) if backends else 'Unknown'}")
    except:
        pass

if __name__ == "__main__":
    print("📹 Camera Diagnostic Tool")
    print("=" * 50)
    
    check_system_info()
    
    working_camera = test_camera()
    
    if working_camera is not None:
        print(f"\n🎉 SUCCESS! Camera {working_camera} is working")
        print(f"💡 Use camera index {working_camera} in your face recognition scripts")
    else:
        print(f"\n💡 Troubleshooting tips:")
        print(f"1. Check if another application is using the camera")
        print(f"2. Make sure camera drivers are installed")
        print(f"3. Try running as administrator")
        print(f"4. Check Windows camera privacy settings")
        print(f"5. Try unplugging and reconnecting USB cameras")
        print(f"6. Restart your computer")