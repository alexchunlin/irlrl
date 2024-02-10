import socket
import time
import math
import numpy as np

server_ip = "192.168.0.104"  # Replace with your ESP32's IP address
server_port = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((server_ip, server_port))

def send_motor_command(power_level):
    message = f"{power_level}\n"
    sock.sendall(message.encode())

def receive_reply():
    reply = sock.recv(1024)  # Buffer size is 1024 bytes
    data = reply.decode().strip()  # Decode and strip whitespace
    position, velocity = map(float, data.split(','))  # Split by comma and convert to float
    return np.array([position, velocity])

try:
    for t in range(100):  # Example duration
        # Generate a sine wave power level
        power_level = 50 + 50 * math.sin(2 * math.pi * 0.1 * t)
        send_motor_command(power_level)
        pos_vel_array = receive_reply()
        print(f"Received: {pos_vel_array}")

        # If you want to use position and velocity individually
        position, velocity = pos_vel_array
        print(f"Position: {position}, Velocity: {velocity}")

        time.sleep(0.01)  # Adjust as necessary
finally:
    sock.close()
