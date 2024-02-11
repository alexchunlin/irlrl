import socket
import time
import math
import numpy as np
import ipdb

server_ip = "192.168.0.104"  # Replace with your ESP32's IP address
server_port = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((server_ip, server_port))

def reset():
    send_motor_command(0); ##centered on 0-100 so 50 is no power
    ## TODO start training on data
    vel_is_zero = False
    while not vel_is_zero:
        send_motor_command(0)
        position, velocity = receive_reply()
        vel_is_zero = (velocity == 0)

def send_motor_command(power_level):
    message = str(power_level)+"\n"
    sock.sendall(message.encode())

def send_motor_position_reset():
    message = "zero_pos"+"\n"
    sock.sendall(message.encode())


def receive_reply():
    reply = sock.recv(1024)  # Buffer size is 1024 bytes
    data = reply.decode().strip()  # Decode and strip whitespace
    try:
        position, velocity = map(float, data.split(','))  # Split by comma and convert to float
    except:
        ipdb.set_trace()
    return np.array([position, velocity])


if __name__ == '__main__':
    
    for t in range(100):  # Example duration
        # Generate a sine wave power level
        power_level = 50 + 50 * math.sin(2 * math.pi * 0.1 * t)
        send_motor_command(power_level)
        pos_vel_array = receive_reply()
        # print(f"Received: {pos_vel_array}")
        print("Received: ", pos_vel_array)
        # If you want to use position and velocity individually
        position, velocity = pos_vel_array
        print("Position: ",position, "Velocity: ", velocity)

        time.sleep(0.01)  # Adjust as necessary



    send_motor_command(0)
    pos_vel_array = receive_reply()
    time.sleep(5.)
    send_motor_position_reset()

    for t in range(100):  # Example duration
        # Generate a sine wave power level
        power_level = 50 + 50 * math.sin(2 * math.pi * 0.1 * t)
        send_motor_command(power_level)
        pos_vel_array = receive_reply()
        # print(f"Received: {pos_vel_array}")
        print("Received: ", pos_vel_array)
        # If you want to use position and velocity individually
        position, velocity = pos_vel_array
        print("Position: ",position, "Velocity: ", velocity)

        time.sleep(0.01)  # Adjust as necessary
    
    
    sock.close()
