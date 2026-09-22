#!/usr/bin/env python3
"""
Simple servo test - just check if servo moves physically
"""
import serial
import time

def simple_servo_test():
    print("🎯 Simple Servo Movement Test")
    print("First test: Did the servo move to 90° from the first command?")
    
    try:
        # Connect
        print("🔌 Connecting to ESP8266...")
        ser = serial.Serial("COM4", 115200, timeout=2)
        time.sleep(2)
        
        # Clear buffer
        ser.reset_input_buffer()
        
        print("\n✋ MANUAL TEST:")
        print("I'll send one command at a time.")
        print("Tell me if you see the servo move physically!")
        
        # Test each angle individually
        test_angles = [90, 45, 135, 0, 180]
        
        for angle in test_angles:
            input(f"\nPress ENTER to move servo to {angle}°...")
            
            try:
                # Send command
                command = f"{angle}\n"
                ser.write(command.encode())
                ser.flush()
                
                print(f"📤 Sent: {angle}°")
                
                # Wait a bit and check for response
                time.sleep(0.5)
                if ser.in_waiting > 0:
                    try:
                        response = ser.readline().decode('utf-8', errors='ignore').strip()
                        print(f"📥 ESP8266: {response}")
                    except:
                        print("📥 ESP8266: (garbled response)")
                
                print("👁️ Did you see the servo move? (physically)")
                response = input("   Type 'y' for yes, 'n' for no: ").lower()
                
                if response == 'y':
                    print("✅ Servo is working!")
                else:
                    print("❌ No movement detected")
                
            except Exception as e:
                print(f"❌ Error sending command: {e}")
                break
        
        ser.close()
        print("\n🎯 Test complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    simple_servo_test()