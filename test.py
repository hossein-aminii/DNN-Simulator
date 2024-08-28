import serial
import threading
import time


def send_data_through_uart(serial_port):
    """
    Send data through UART in a loop.

    Args:
    - serial_port (serial.Serial): The serial port object.
    """
    try:
        while True:
            data = input("")
            serial_port.write(data.encode())
            # print(f"Data sent: {data}")
            time.sleep(1)  # Wait for 1 second before sending the next data
    except serial.SerialException as e:
        print(f"Error sending data: {e}")


def receive_data_from_uart(serial_port):
    """
    Receive data from UART in a loop.

    Args:
    - serial_port (serial.Serial): The serial port object.
    """
    try:
        while True:
            if serial_port.in_waiting > 0:
                data = serial_port.read(serial_port.in_waiting).decode()
                print(f"Data received from Zynq: {data}")
            time.sleep(0.1)  # Check for received data every 100 ms
    except serial.SerialException as e:
        print(f"Error receiving data: {e}")


def main():
    # Replace 'COM3' with your UART port (e.g., '/dev/ttyUSB0' for Linux)
    uart_port = "COM7"
    baud_rate = 115200  # Set your baud rate here

    try:
        # Open the serial port
        with serial.Serial(uart_port, baud_rate, timeout=1) as serial_port:
            print(f"Connected to {uart_port} at {baud_rate} baud.")

            # Create threads for sending and receiving data
            send_thread = threading.Thread(target=send_data_through_uart, args=(serial_port,))
            receive_thread = threading.Thread(target=receive_data_from_uart, args=(serial_port,))

            # Start the threads
            send_thread.start()
            receive_thread.start()

            # Join the threads to the main thread
            send_thread.join()
            receive_thread.join()

    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")


if __name__ == "__main__":
    main()
