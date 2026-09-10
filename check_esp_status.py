#!/usr/bin/env python3
"""
Check ESP8266 status and connection
"""
import serial
import time

def check_esp_status():
    print("🔍 Checking ESP8266 Status")
    
    try:
        # Try to connect
        print("🔌 Attempting connection...")
        ser = serial.Serial("COM3", 115200, timeout=1)
        time.sleep(2)
        
        print("✅ Serial port opened successfully")
        
        # Check if ESP is responding
        print("\n📡 Checking for ESP8266 messages...")
        
        # Read any startup messages
        for i in range(10):
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"   ESP: {line}")
                except Exception as e:
                    print(f"   Decode error: {e}")
            time.sleep(0.5)
        
        print("\n🧪 Testing basic communication...")
        
        # Try to send a simple command
        try:
            ser.write(b"90\n")
            ser.flush()
            print("📤 Sent test command: 90")
            
            time.sleep(1)
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"📥 Response: {response}")
            else:
                print("📥 No response received")
                
        except Exception as e:
            print(f"❌ Communication error: {e}")
        
        ser.close()
        
    except serial.SerialException as e:
        print(f"❌ Serial connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if Arduino IDE Serial Monitor is closed")
        print("   2. Try unplugging/replugging the ESP8266")
        print("   3. Check if COM3 is the correct port")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    check_esp_status()