IGateMVMOut[0] <= IGateMVMOut[0] + IGateMultResult[0]
IGateRMVMOut[0] <= IGateRMVMOut[0] + IGateRMultResult[0]

results = ""

for i in range(32):
    results += (
        f"IGateMVMOut[{i}] <= IGateMVMOut[{i}] + IGateMultResult[{i}];\n"
        + f"OGateRWeight[{i}] <= OGateRWBuffer[w_ptr + {i*32}];\n"
    )

with open("circuit.mem", "w") as f:
    f.write(results)
