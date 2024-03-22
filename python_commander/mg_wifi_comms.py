import socket

# TCP client setup
TCP_IP = '192.168.1.100'  # IP address of ESP1
TCP_PORT = 8080

# Create a TCP socket
tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcp_socket.connect((TCP_IP, TCP_PORT))

while True:
    # Send data to ESP1
    data_to_send = input("Enter data to send: ")
    tcp_socket.sendall(data_to_send.encode())

    # Receive data from ESP1
    received_data = tcp_socket.recv(1024)
    print(f"Received data: {received_data.decode()}")

# Clean up the socket when done
tcp_socket.close()
