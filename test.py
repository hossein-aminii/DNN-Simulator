# import serial
# import threading
# import time


# def send_data_through_uart(serial_port):
#     """
#     Send data through UART in a loop.

#     Args:
#     - serial_port (serial.Serial): The serial port object.
#     """
#     try:
#         while True:
#             data = input("")
#             serial_port.write(data.encode())
#             # print(f"Data sent: {data}")
#             time.sleep(1)  # Wait for 1 second before sending the next data
#     except serial.SerialException as e:
#         print(f"Error sending data: {e}")


# def receive_data_from_uart(serial_port):
#     """
#     Receive data from UART in a loop.

#     Args:
#     - serial_port (serial.Serial): The serial port object.
#     """
#     try:
#         while True:
#             if serial_port.in_waiting > 0:
#                 data = serial_port.read(serial_port.in_waiting).decode()
#                 print(f"Data received from Zynq: {data}")
#             time.sleep(0.1)  # Check for received data every 100 ms
#     except serial.SerialException as e:
#         print(f"Error receiving data: {e}")


# def main():
#     # Replace 'COM3' with your UART port (e.g., '/dev/ttyUSB0' for Linux)
#     uart_port = "COM7"
#     baud_rate = 115200  # Set your baud rate here

#     try:
#         # Open the serial port
#         with serial.Serial(uart_port, baud_rate, timeout=1) as serial_port:
#             print(f"Connected to {uart_port} at {baud_rate} baud.")

#             # Create threads for sending and receiving data
#             send_thread = threading.Thread(target=send_data_through_uart, args=(serial_port,))
#             receive_thread = threading.Thread(target=receive_data_from_uart, args=(serial_port,))

#             # Start the threads
#             send_thread.start()
#             receive_thread.start()

#             # Join the threads to the main thread
#             send_thread.join()
#             receive_thread.join()

#     except serial.SerialException as e:
#         print(f"Error opening serial port: {e}")


# if __name__ == "__main__":
#     main()

from quantization.utils.fix_point_converter import FixedPointConverter

fx_converter = FixedPointConverter(int_bits=2, frac_bits=6)
value = "10000000"
print(fx_converter.sign_magnitude_to_float(binary_str=value))


import os

INQ_1_7 = [
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 1,
            "fraction_bits": 7,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-1,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-1-7_fault_injection_ratio{1e-1}-int{1}-fraction{7}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{1}_frac{7}_ER{1e-1}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 1,
            "fraction_bits": 7,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-2,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-1-7_fault_injection_ratio{1e-2}-int{1}-fraction{7}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{1}_frac{7}_ER{1e-2}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 1,
            "fraction_bits": 7,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-3,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-1-7_fault_injection_ratio{1e-3}-int{1}-fraction{7}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{1}_frac{7}_ER{1e-3}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 1,
            "fraction_bits": 7,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-4,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-1-7_fault_injection_ratio{1e-4}-int{1}-fraction{7}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{1}_frac{7}_ER{1e-4}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 1,
            "fraction_bits": 7,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-5,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-1-7_fault_injection_ratio{1e-5}-int{1}-fraction{7}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{1}_frac{7}_ER{1e-5}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 1,
            "fraction_bits": 7,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-6,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-1-7_fault_injection_ratio{1e-6}-int{1}-fraction{7}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{1}_frac{7}_ER{1e-6}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{2}_fraction{6}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 2,
            "fraction_bits": 6,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-1,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-2-6_fault_injection_ratio{1e-1}-int{2}-fraction{6}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{2}_frac{6}_ER{1e-1}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{2}_fraction{6}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 2,
            "fraction_bits": 6,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-5,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-2-6_fault_injection_ratio{1e-5}-int{2}-fraction{6}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{2}_frac{6}_ER{1e-5}.json",
        },
    },
    {
        "model_filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{2}_fraction{6}.h5"),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 2,
            "fraction_bits": 6,
            "total_bits": 8,
            "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-6,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"INQ-2-6_fault_injection_ratio{1e-6}-int{2}-fraction{6}-format-INQ"
            ),
            "accuracy_results_filename": f"accuracy_results_int{2}_frac{6}_ER{1e-6}.json",
        },
    },
]


fixed_4_12 = [
    {
        "model_filepath": os.path.join(
            "results", "models", "PTQ", "16-bit", f"fixed-point-model-int{4}-frac{12}-format-twos_complement.h5"
        ),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 4,
            "fraction_bits": 12,
            "total_bits": 16,
            "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-1,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"Fixed-4-12_fault_injection_ratio{1e-1}-int{4}-fraction{12}-format-twos_complement"
            ),
            "accuracy_results_filename": f"accuracy_results_int{4}_frac{12}_ER{1e-1}.json",
        },
    },
    {
        "model_filepath": os.path.join(
            "results", "models", "PTQ", "16-bit", f"fixed-point-model-int{4}-frac{12}-format-twos_complement.h5"
        ),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 4,
            "fraction_bits": 12,
            "total_bits": 16,
            "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-2,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"Fixed-4-12_fault_injection_ratio{1e-2}-int{4}-fraction{12}-format-twos_complement"
            ),
            "accuracy_results_filename": f"accuracy_results_int{4}_frac{12}_ER{1e-2}.json",
        },
    },
    {
        "model_filepath": os.path.join(
            "results", "models", "PTQ", "16-bit", f"fixed-point-model-int{4}-frac{12}-format-twos_complement.h5"
        ),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 4,
            "fraction_bits": 12,
            "total_bits": 16,
            "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-3,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"Fixed-4-12_fault_injection_ratio{1e-3}-int{4}-fraction{12}-format-twos_complement"
            ),
            "accuracy_results_filename": f"accuracy_results_int{4}_frac{12}_ER{1e-3}.json",
        },
    },
    {
        "model_filepath": os.path.join(
            "results", "models", "PTQ", "16-bit", f"fixed-point-model-int{4}-frac{12}-format-twos_complement.h5"
        ),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 4,
            "fraction_bits": 12,
            "total_bits": 16,
            "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-4,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"Fixed-4-12_fault_injection_ratio{1e-4}-int{4}-fraction{12}-format-twos_complement"
            ),
            "accuracy_results_filename": f"accuracy_results_int{4}_frac{12}_ER{1e-4}.json",
        },
    },
    {
        "model_filepath": os.path.join(
            "results", "models", "PTQ", "16-bit", f"fixed-point-model-int{4}-frac{12}-format-twos_complement.h5"
        ),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 4,
            "fraction_bits": 12,
            "total_bits": 16,
            "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-5,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"Fixed-4-12_fault_injection_ratio{1e-5}-int{4}-fraction{12}-format-twos_complement"
            ),
            "accuracy_results_filename": f"accuracy_results_int{4}_frac{12}_ER{1e-5}.json",
        },
    },
    {
        "model_filepath": os.path.join(
            "results", "models", "PTQ", "16-bit", f"fixed-point-model-int{4}-frac{12}-format-twos_complement.h5"
        ),
        "action": "fault-injection",
        "fault_injector": "basic_fault_injector",
        "fault_injector_config": {
            "int_bits": 4,
            "fraction_bits": 12,
            "total_bits": 16,
            "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
            "fault_injection_ratio": 1e-6,
            "mode": "single_bit_flip",
            "model_results_directory": os.path.join(
                "results", f"Fixed-4-12_fault_injection_ratio{1e-6}-int{4}-fraction{12}-format-twos_complement"
            ),
            "accuracy_results_filename": f"accuracy_results_int{4}_frac{12}_ER{1e-6}.json",
        },
    },
]
