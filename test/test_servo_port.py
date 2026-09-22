import serial
import serial.tools.list_ports
import time
import sys

print('Checking COM4...')
ports = serial.tools.list_ports.comports()
for p in ports:
    if 'COM4' in p.device:
        print(f'Found: {p.device} - {p.description}')

print()
print('Attempting connection...')
try:
    ser = serial.Serial('COM4', 115200, timeout=2, write_timeout=2)
    time.sleep(1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print('Connected, sending test...')
    
    for i in range(3):
        ser.write(b'90\n')
        print(f'Sent command {i+1}')
        time.sleep(0.3)
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            print('Response:', data.decode(errors='ignore'))
    
    ser.close()
    print('Done')
except Exception as e:
    print('Error:', e)
