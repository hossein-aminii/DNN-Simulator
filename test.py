# # # from quantization.utils import FixedPointConverter

# # # converter = FixedPointConverter(int_bits=6, frac_bits=10)

# # # binary_str = "1111111000000000"

# # # print(converter.twos_complement_to_float(binary_str=binary_str))

from quantization.fix_point_pt_quantizer import FixedPointConverter

# # weights = "88 04 84 04 00 08 04 84 88 01 82 08 04 10 08 90 82 88 08 a0 08 10 04 88 08 01 08 04 04 88 08 10"
# # inputs = "0008 0005 00fb 00f9 0005 00fe 00ff 00fa 00fd 0002 0003 00fc 0004 00fc 00f9 00ff 00fe 00fb 00fe 0013 00f8 00fd 00fd 00fc 00fa 00fd 00f8 0000 00f7 0004 0005 0009"

# # weights = weights.split(" ")

# # binary_weights = []
# # for w in weights:
# #     binary_str = bin(int(w, 16))[2:].zfill(8)
# #     binary_weights.append(binary_str)

# # float_weights = []
# # converter = FixedPointConverter(int_bits=2, frac_bits=6)
# # for w in binary_weights:
# #     float_weights.append(converter.sign_magnitude_to_float(binary_str=w))
# # print(float_weights)

# # inputs = inputs.split(" ")
# # binary_inputs = []
# # for i in inputs:
# #     binary_str = bin(int(i, 16))[2:].zfill(16)
# #     binary_inputs.append(binary_str)

# # float_inputs = []
# # converter = FixedPointConverter(int_bits=4, frac_bits=12)
# # for i in binary_inputs:
# #     float_inputs.append(converter.twos_complement_to_float(binary_str=i))
# # print(float_inputs)

# import numpy as np

# a = np.random.random(32)
# print(a)
converter = FixedPointConverter(int_bits=4, frac_bits=12)
# print(converter.float_to_twos_complement(value=1))

filename = "i_gate_results.mem"
# filename = "i_gate_mvm.mem"
# filename = "i_gate_recurrent_mvm.mem"
filename = "i_gate_hard_sigmoid.mem"
filename = "i_gate_sigmoid.mem"
filename = "CTBuffer.mem"
filename = "HBuffer.mem"
with open(filename, "r") as f:
    data = f.readlines()

# print(data)
values = []
for d in data:
    d = d[: len(d) - 1]
    binary_str = bin(int(d, 16))[2:].zfill(16)
    value = converter.twos_complement_to_float(binary_str=binary_str)
    values.append(value)

print(values)

hw_hex_values = [
    "13D",
    "3D4",
    "AD",
    "F93D",
    "FFA7",
    "FA94",
    "FF09",
    "34",
    "17",
    "FFAC",
    "42B",
    "283",
    "14E",
    "63D",
    "FE91",
    "22",
    "FB7D",
    "279",
    "FE06",
    "FF37",
    "FE44",
    "FE0A",
    "F201",
    "FD06",
    "278",
    "4C9",
    "276",
    "30B",
    "FBDF",
    "56C",
    "126",
    "FE24",
]

hw_values = []
for hv in hw_hex_values:
    binary_str = bin(int(hv, 16))[2:].zfill(16)
    value = converter.twos_complement_to_float(binary_str=binary_str)
    hw_values.append(value)

print(hw_values)
mse = 0
for i in range(len(hw_values)):
    print(f"SW/HW: {values[i]} --> {hw_values[i]} ===== {abs(values[i] - hw_values[i])}   idx: {i+1}")
    mse += (values[i] - hw_values[i]) ** 2

print(mse / len(hw_values))


# """
# [-0.416717529296875, 0.4263916015625, 0.4527015686035156, -0.049365997314453125, -0.2520866394042969, 0.3854827880859375, -0.21562957763671875, -0.4262580871582031, -0.19380950927734375, 0.41866302490234375, 0.21907806396484375, 0.3275642395019531, -0.6983184814453125, 0.3172416687011719, 0.3131256103515625, -0.3149070739746094, -0.2686958312988281, 0.4146537780761719, 0.31107330322265625, 0.2588462829589844, 0.24974822998046875, 0.12712478637695312, 0.1005859375, 0.5382232666015625, 0.08317184448242188, -0.173126220703125, -0.47139739990234375, 0.2896728515625, -0.22777175903320312, 0.21377182006835938, 0.38887786865234375, -0.8401069641113281]
# """

# from utils.activation_function import *

# run_sigmoid()

# run_hard_sigmoid()

# save_inputs()

# process_h_sigmoud_out()
# run_tanh()

# run_hard_tanh()

# process_h_tanh_out()
