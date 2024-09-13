import numpy as np
from quantization.fix_point_pt_quantizer import FixedPointConverter


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def hard_sigmoid(x):
    return np.clip(0.25 * x + 0.5, 0, 1)


def hard_tanh(x):
    return np.clip(x, -1, 1)


def run_tanh():
    results = []
    start = -5
    end = 5
    num_expriment = 200
    step = (end - start) / num_expriment
    inputs = []
    while True:
        inputs.append(start)
        results.append(tanh(start))
        start += step
        if start > end + step:
            break

    for inp in inputs:
        print("{:.2f}".format(inp))

    print("\n")
    print("-" * 80)
    print("\n")
    for value in results:
        print("{:.4f}".format(value))


def run_hard_tanh():
    results = []
    start = -5
    end = 5
    num_expriment = 200
    step = (end - start) / num_expriment
    inputs = []
    while True:
        inputs.append(start)
        results.append(hard_tanh(start))
        start += step
        if start > end + step:
            break

    for inp in inputs:
        print("{:.2f}".format(inp))

    print("\n")
    print("-" * 80)
    print("\n")
    for value in results:
        print("{:.4f}".format(value))


def run_sigmoid():
    results = []
    start = -5
    end = 5
    num_expriment = 200
    step = (end - start) / num_expriment
    inputs = []
    while True:
        inputs.append(start)
        results.append(sigmoid(start))
        start += step
        if start > end + step:
            break

    for inp in inputs:
        print("{:.2f}".format(inp))

    print("\n")
    print("-" * 80)
    print("\n")
    for value in results:
        print("{:.4f}".format(value))


def run_hard_sigmoid():
    results = []
    start = -5
    end = 5
    num_expriment = 200
    step = (end - start) / num_expriment
    inputs = []
    while True:
        inputs.append(start)
        results.append(hard_sigmoid(start))
        start += step
        if start > end + step:
            break

    for inp in inputs:
        print("{:.2f}".format(inp))

    print("\n")
    print("-" * 80)
    print("\n")
    for value in results:
        print("{:.4f}".format(value))


def save_hex(flat_array, filename: str, int_bits=2, frac_bits=6, type="sign_magnitude"):
    converter = FixedPointConverter(int_bits=int_bits, frac_bits=frac_bits)
    hex_length = (int_bits + frac_bits) // 4
    results = ""
    for value in flat_array:
        if type == "sign_magnitude":
            binary_str = converter.float_to_sign_magnitude(value=value)
        else:
            binary_str = converter.float_to_twos_complement(value=value)
        hex_value = str(hex(int(binary_str, 2))[2:])
        while len(hex_value) < hex_length:
            hex_value = "0" + hex_value
        hex_value = "x <= 16'h" + hex_value + ";\n" + '$display("%h", y);\n' "#50;"
        results += hex_value
        results += "\n"
    with open(filename, "w") as mf:
        mf.write(results)


def save_inputs():
    start = -5
    end = 5
    num_expriment = 200
    step = (end - start) / num_expriment
    inputs = []
    while True:
        inputs.append(start)
        start += step
        if start > end + step:
            break
    save_hex(flat_array=inputs, filename="sigmoid_hex_inputs.mem", int_bits=4, frac_bits=12, type="twos_complement")


def process_h_sigmoud_out():

    filename = "sigmoid_h_output.mem"

    with open(filename, "r") as f:
        data = f.readlines()
    preprocessed_data = []
    for d in data:
        preprocessed_data.append(d[1 : len(d) - 1])
    # print(preprocessed_data)
    converter = FixedPointConverter(int_bits=4, frac_bits=12)
    hw_values = []
    for hv in preprocessed_data:
        binary_str = bin(int(hv, 16))[2:].zfill(16)
        value = converter.twos_complement_to_float(binary_str=binary_str)
        hw_values.append(value)
    results = ""
    for value in hw_values:
        results += str(value) + "\n"
    with open("HW_hard_sigmoid_output.mem", "w") as mf:
        mf.write(results)


def process_h_tanh_out():

    filename = "tanh_output.mem"

    with open(filename, "r") as f:
        data = f.readlines()
    preprocessed_data = []
    for d in data:
        preprocessed_data.append(d[1 : len(d) - 1])
    # print(preprocessed_data)
    converter = FixedPointConverter(int_bits=4, frac_bits=12)
    hw_values = []
    for hv in preprocessed_data:
        binary_str = bin(int(hv, 16))[2:].zfill(16)
        value = converter.twos_complement_to_float(binary_str=binary_str)
        hw_values.append(value)
    results = ""
    for value in hw_values:
        results += str(value) + "\n"
    with open("HW_hard_tanh_output.mem", "w") as mf:
        mf.write(results)
