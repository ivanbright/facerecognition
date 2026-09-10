#!/usr/bin/env python3
"""
Test servo after firmware flash - handles initialization properly.
"""
import serial
import time

def test_servo_after_flash():
    print("🎯 Testing Servo After Firmware Flash")
    
    try:
        # Connect with longer timeout for initialization
        print("🔌 Connecting to ESP8266...")
        ser = serial.Serial("COM3", 115200, timeout=3)
        time.sleep(3)  # Wait for ESP8266 boot and servo initialization
        
        print("✅ Serial connection established")
        
        # Clear any startup garbage
        ser.reset_input_buffer()
        time.sleep(1)
        
        # Look for the ready message
        print("\n📖 Waiting for servo ready message...")
        ready_found = False
        
        for attempt in range(10):  # Try for 10 seconds
            try:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"   ESP8266: {line}")
                        if "Ready" in line or "tracker" in line:
                            ready_found = True
                            break
            except:
                pass  # Ignore decode errors during startup
            time.sleep(1)
        
        if not ready_found:
            print("⚠️  No ready message found, but continuing with servo test...")
        
        # Test servo movement
        print(f"\n🎯 Testing servo movement:")
        test_angles = [90, 0, 180, 45, 135, 90]  # Center, extremes, then center
        
        for i, angle in enumerate(test_angles):
            print(f"\n   Test {i+1}: Moving to {angle}°")
            
            # Clear input buffer
            ser.reset_input_buffer()
            
            # Send command
            command = f"{angle}\n"
            ser.write(command.encode())
            ser.flush()
            
            # Wait for response
            response_received = False
            for wait in range(20):  # Wait up to 2 seconds
                try:
                    if ser.in_waiting > 0:
                        response = ser.readline().decode('utf-8', errors='ignore').strip()
                        if response:
                            print(f"   ESP8266 response: {response}")
                            if f"angle={angle}" in response:
                                print(f"   ✅ Command confirmed!")
                                response_received = True
                                break
                except:
                    pass
                time.sleep(0.1)
            
            if not response_received:
                print(f"   ⚠️  No response for angle {angle}")
            
            print(f"   ⏳ Waiting 3 seconds for servo movement...")
            time.sleep(3)  # Give time to physically observe movement
        
        print(f"\n🎯 Final test - quick sweep:")
        for angle in [0, 30, 60, 90, 120, 150, 180]:
            ser.write(f"{angle}\n".encode())
            time.sleep(0.5)
        
        ser.close()
        print(f"\n✅ Servo test completed!")
        print(f"\n💡 What to check:")
        print(f"   - Did you see the servo physically move?")
        print(f"   - Check servo wiring if no movement")
        print(f"   - Servo power supply (5V, adequate current)")
        
    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_servo_after_flash()