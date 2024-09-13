import tensorflow as tf
from tensorflow import keras
from model.utils import ModelLoader
from config import Config
import numpy as np
import time
import json

config = Config()

base_model = ModelLoader(config=config).load()

kernel = None
recurrent_kernel = None
biases = None

for layer in base_model.layers:
    if layer.name == "lstm":
        for idx, weight in enumerate(layer.get_weights()):
            if idx == 0:
                kernel = weight
            elif idx == 1:
                recurrent_kernel = weight
            elif idx == 2:
                biases = weight

single_input = [
    0.45929628,
    0.79989361,
    0.8695671,
    0.41531773,
    0.26924149,
    0.12415365,
    0.82781691,
    0.33033358,
    0.01066194,
    0.35340493,
    0.64590161,
    0.65769779,
    0.22291437,
    0.83282802,
    0.96214574,
    0.45689524,
    0.9945435,
    0.56380107,
    0.80128656,
    0.47115956,
    0.27279179,
    0.29572027,
    0.80724564,
    0.37955554,
    0.85338941,
    0.41140614,
    0.53338419,
    0.36498607,
    0.66152791,
    0.71917363,
    0.22816605,
    0.22987329,
]
single_input = np.array(single_input)
single_input = single_input.reshape((1, 1, 32))

model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=32))
model.build((None, 128, 32))
print(model.summary())
for layer in model.layers:
    print(layer.name)
    if layer.name == "lstm":
        weights = [kernel, recurrent_kernel, biases]
        layer.set_weights(weights)


@tf.function
def inference_step(input_tensor):
    return model(input_tensor)


def tf_lstm(single_input):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=32))
    model.build((None, 128, 32))
    print(model.summary())
    for layer in model.layers:
        print(layer.name)
        if layer.name == "lstm":
            weights = [kernel, recurrent_kernel, biases]
            layer.set_weights(weights)
    # print(model.get_weights())
    single_input = np.random.rand(1, 1, 32).astype(np.float32)
    input_tensor = tf.convert_to_tensor(single_input)
    # warp up!
    for _ in range(50):
        _ = inference_step(input_tensor)
    # Measure the inference time for one timestep
    times = []
    for _ in range(100):
        start_time = time.time()
        # Perform inference for just one timestep
        _ = inference_step(input_tensor)
        end_time = time.time()
        elapsed_time = end_time - start_time
        times.append(elapsed_time)
    with open("TF_GPU_T4_times.json", "w") as jf:
        json.dump(times, jf)
    # print(f"output is: {output[0]}")


tf_lstm(single_input=single_input)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def manual_lstm(single_input):
    single_input = single_input.reshape((1, 32))
    h = np.zeros((1, 32))
    c_prev = np.zeros((1, 32))
    f_kernel = kernel[:, 32:64]
    f_recurrent_kernel = recurrent_kernel[:, 32:64]
    f_biases = biases[32:64]
    # --------------------------------------------------------
    i_kernel = kernel[:, :32]
    i_recurrent_kernel = recurrent_kernel[:, :32]
    i_biases = biases[:32]
    # --------------------------------------------------------
    c_kernel = kernel[:, 64:96]
    c_recurrent_kernel = recurrent_kernel[:, 64:96]
    c_biases = biases[64:96]
    # --------------------------------------------------------
    o_kernel = kernel[:, 96:128]
    o_recurrent_kernel = recurrent_kernel[:, 96:128]
    o_biases = biases[96:128]
    # ----------------------------------------------------------
    forget_gate = sigmoid(np.matmul(single_input, f_kernel) + np.matmul(h, f_recurrent_kernel) + f_biases)
    input_gate = sigmoid(np.matmul(single_input, i_kernel) + np.matmul(h, i_recurrent_kernel) + i_biases)
    cell_gate = tanh(np.matmul(single_input, c_kernel) + np.matmul(h, c_recurrent_kernel) + c_biases)
    output_gate = sigmoid(np.matmul(single_input, o_kernel) + np.matmul(h, o_recurrent_kernel) + o_biases)
    # -----------------------------------------------------------
    cell = cell_gate * input_gate + forget_gate * c_prev
    h = output_gate * tanh(cell)

    print(f"cell: {cell}")
    print(f"h: {h}")


# manual_lstm(single_input=single_input)
