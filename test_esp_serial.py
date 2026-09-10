#!/usr/bin/env python3
"""
Test ESP8266 serial communication and servo response.
"""
import serial
import time

def test_esp8266_communication():
    print("🔌 Testing ESP8266 Serial Communication")
    
    try:
        # Connect to ESP8266
        ser = serial.Serial("COM3", 115200, timeout=2)
        time.sleep(3)  # Wait for ESP8266 to boot
        
        print("✅ Serial connection established")
        
        # Read any startup messages
        print("\n📖 Reading startup messages:")
        for i in range(10):  # Try to read startup messages
            if ser.in_waiting > 0:
                msg = ser.readline().decode().strip()
                print(f"   ESP8266: {msg}")
            time.sleep(0.1)
        
        # Send test servo commands
        test_angles = [90, 45, 135, 0, 180, 90]
        
        print(f"\n🎯 Sending test servo commands:")
        for angle in test_angles:
            print(f"   Sending: {angle}°")
            
            # Send command
            ser.write(f"{angle}\n".encode())
            ser.flush()
            
            # Wait and read response
            time.sleep(0.5)
            if ser.in_waiting > 0:
                response = ser.readline().decode().strip()
                print(f"   ESP8266 response: {response}")
            else:
                print("   ❌ No response from ESP8266")
            
            time.sleep(2)  # Wait between moves
        
        ser.close()
        print("\n✅ Test completed")
        
    except serial.SerialException as e:
        print(f"❌ Serial error: {e}")
        print("💡 Check:")
        print("   - ESP8266 connected to COM3")
        print("   - Correct baud rate (115200)")
        print("   - No other programs using COM3")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_esp8266_communication()