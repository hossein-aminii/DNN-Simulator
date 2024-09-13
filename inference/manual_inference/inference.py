from config import Config
from model.utils import ModelLoader
from datasets import DataLoaderDispatcher
import pickle
import random
import numpy as np
import tensorflow as tf
from quantization.fix_point_pt_quantizer import FixedPointConverter


class ManualInference:

    def __init__(self, config: Config, name="A module to do inference of model manually!"):
        self.name = name

        self.config = config
        model_loader = ModelLoader(config=self.config)
        self.model = model_loader.load()

        # self.data_loader = self.get_data_loader(info=self.config.dataset_info)
        # self.train_dataset = self.data_loader.train_dataset
        # self.validation_dataset = self.data_loader.validation_dataset

        self.manual_inference_config = self.config.manual_inference_config
        self.target_layer = self.manual_inference_config["target_layer"]
        self.prev_target_layer = self.manual_inference_config["prev_target_layer"]
        self.weights_info = self.manual_inference_config["weights_info"]
        self.bias_info = self.manual_inference_config["bias_info"]

        self.target_layer_parameters = {}

    def load_tokenizer(self, path: str):
        try:
            with open(path, "rb") as handle:
                tokenizer = pickle.load(handle)
            print("INFO: Loading tokenizer successfully done!")
        except:
            tokenizer = None
            print("INFO: Can not load tokenizer. tokenizer set to None!")
        return tokenizer

    def get_data_loader(self, info: dict):
        dataset = info["name"]
        file_path = info["filepath"]
        tokenizer = self.load_tokenizer(path=info["tokenizer_filepath"])
        dispatcher = DataLoaderDispatcher(dataset=dataset)
        return dispatcher.dispatch(filepath=file_path, params={"tokenizer": tokenizer})

    # def get_random_input(self, _from="train"):
    #     if _from == "train":
    #         dataset = self.train_dataset
    #     else:
    #         dataset = self.validation_dataset
    #     dataset_list = list(dataset.unbatch().as_numpy_iterator())
    #     input_idx = random.randint(0, len(dataset_list) - 1)
    #     return dataset_list[input_idx][0]

    def get_random_input(self):
        shape = (128,)
        return np.random.random(shape)

    def extract_target_layer_parameters(self, target_layer):
        for idx, weight in enumerate(target_layer.get_weights()):
            if idx in list(self.bias_info.keys()):
                key = self.bias_info[idx]
                self.target_layer_parameters[key] = weight
            elif idx in list(self.weights_info.keys()):
                key = self.weights_info[idx]
                self.target_layer_parameters[key] = weight

    def get_tf_results(self, model_input: np.ndarray):
        model_input = model_input.reshape((1, model_input.shape[0]))  # for spatial dropout!
        input_layer = self.model.input
        target_layer_input = self.model.get_layer(name=self.prev_target_layer).output
        target_layer = self.model.get_layer(name=self.target_layer)
        self.extract_target_layer_parameters(target_layer=target_layer)
        temp_model = tf.keras.Model(inputs=input_layer, outputs=[target_layer_input, target_layer.output])
        output = temp_model.predict(model_input)
        print(output)
        target_layer_input, target_layer_output = temp_model.predict(model_input)

        return self.adjust_shape(target_layer_input), self.adjust_shape(target_layer_output)

    def adjust_shape(self, array: np.ndarray):
        target_shape = []
        for dim in array.shape:
            if dim > 1:
                target_shape.append(dim)
        return array.reshape(target_shape)

    def save_hex(self, flat_array, filename: str, int_bits=2, frac_bits=6, type="sign_magnitude"):
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
            results += hex_value
            results += "\n"
        with open(filename, "w") as mf:
            mf.write(results)

    def mac(self, inputs, weights):
        binary_weights = []
        converter = FixedPointConverter(int_bits=2, frac_bits=6)
        for w in weights:
            binary_weights.append(converter.float_to_sign_magnitude(value=w))
        # --------------------------------------------------------------------------
        binary_inputs = []
        converter = FixedPointConverter(int_bits=4, frac_bits=12)
        for i in inputs:
            binary_inputs.append(converter.float_to_twos_complement(value=i))
        # ---------------------------------------------------------------------------
        for i in range(len(binary_inputs)):
            weight = binary_weights[i]
            weight_value: str = weight[2:]
            # inp = int("0b" + binary_inputs[i], 2)
            num_shift = weight_value.find("1")
            print(weight, num_shift)

    def _hex(self, value):
        converter = FixedPointConverter(int_bits=4, frac_bits=12)
        binary_str = converter.float_to_twos_complement(value=value)
        hex_value = str(hex(int(binary_str, 2))[2:])
        return hex_value

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def hard_sigmoid(self, x):
        return np.clip(0.25 * x + 0.5, 0, 1)

    def hard_tanh(self, x):
        return np.clip(x, -1, 1)

    def get_manual_results(self, layer_input: np.ndarray):
        biases: np.ndarray = self.target_layer_parameters["biases"]
        kernel: np.ndarray = self.target_layer_parameters["kernel"]
        recurrent_kernel: np.ndarray = self.target_layer_parameters["recurrent_kernel"]
        print(
            f"INFO: input shape: {layer_input.shape}, bias shape: {biases.shape}, kernel shape: {kernel.shape}, recurrent kernel shape: {recurrent_kernel.shape}"
        )
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
        single_input = single_input.reshape((1, 32))
        single_input_flat = single_input.flatten()
        self.save_hex(
            flat_array=single_input_flat, filename="lstm_inputs.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        # ----------------------------------------------------------------------
        i_gate_kernel = kernel[:, :32]
        i_gate_recurrent_kernel = recurrent_kernel[:, :32]
        i_gate_bias = biases[:32]
        # -----------------------------------------------------------------------
        f_gate_kernel = kernel[:, 32:64]
        f_gate_recurrent_kernel = recurrent_kernel[:, 32:64]
        f_gate_biases = biases[32:64]
        # ------------------------------------------------------------------------
        c_gate_kernel = kernel[:, 64:96]
        c_gate_recurrent_kernel = recurrent_kernel[:, 64:96]
        c_gate_biases = biases[64:96]
        # ------------------------------------------------------------------------
        o_gate_kernel = kernel[:, 96:128]
        o_gate_recurrent_kernel = recurrent_kernel[:, 96:128]
        o_gate_biases = biases[96:128]
        print(
            f"input shape: {single_input.shape}, kernel_shape: {i_gate_kernel.shape}, recurrent shape: {i_gate_recurrent_kernel.shape}, biase shape: {i_gate_bias.shape}"
        )
        # ----------------------------------------------------------------------------
        i_gate_kernel_trans = i_gate_kernel.T.flatten()
        i_gate_recurrent_kernel_trans = i_gate_recurrent_kernel.T.flatten()
        self.save_hex(flat_array=i_gate_kernel_trans, filename="input_gate_weights.mem")
        self.save_hex(flat_array=i_gate_recurrent_kernel_trans, filename="input_gate_recurrent_weights.mem")
        self.save_hex(
            flat_array=i_gate_bias, filename="input_gate_biases.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        # -------------------------------------------------------------------------------------
        f_gate_kernel_trans = f_gate_kernel.T.flatten()
        f_gate_recurrent_kernel_trans = f_gate_recurrent_kernel.T.flatten()
        self.save_hex(flat_array=f_gate_kernel_trans, filename="forget_gate_weights.mem")
        self.save_hex(flat_array=f_gate_recurrent_kernel_trans, filename="forget_gate_recurrent_weights.mem")
        self.save_hex(
            flat_array=f_gate_biases,
            filename="forget_gate_biases.mem",
            int_bits=4,
            frac_bits=12,
            type="twos_complement",
        )
        # --------------------------------------------------------------------------------------
        c_gate_kernel_trans = c_gate_kernel.T.flatten()
        c_gate_recurrent_kernel_trans = c_gate_recurrent_kernel.T.flatten()
        self.save_hex(flat_array=c_gate_kernel_trans, filename="cell_gate_weights.mem")
        self.save_hex(flat_array=c_gate_recurrent_kernel_trans, filename="cell_gate_recurrent_weights.mem")
        self.save_hex(
            flat_array=c_gate_biases, filename="cell_gate_biases.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        # --------------------------------------------------------------------------------------
        o_gate_kernel_trans = o_gate_kernel.T.flatten()
        o_gate_recurrent_kernel_trans = o_gate_recurrent_kernel.T.flatten()
        self.save_hex(flat_array=o_gate_kernel_trans, filename="output_gate_weights.mem")
        self.save_hex(flat_array=o_gate_recurrent_kernel_trans, filename="output_gate_recurrent_weights.mem")
        self.save_hex(
            flat_array=o_gate_biases,
            filename="output_gate_biases.mem",
            int_bits=4,
            frac_bits=12,
            type="twos_complement",
        )
        # ---------------------------------------------------------------------------------------
        h_prev = np.zeros((32,))
        c_prev = np.zeros((32,))
        self.save_hex(
            flat_array=h_prev, filename="h_iitial_values.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        self.save_hex(
            flat_array=c_prev, filename="c_iitial_values.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        # return
        # ---------------------------------------------------------------------------------------------------
        h_prev = h_prev.reshape((1, 32))
        i_gate_sigmoid = self.hard_sigmoid(
            np.matmul(single_input, i_gate_kernel) + np.matmul(h_prev, i_gate_recurrent_kernel) + i_gate_bias
        )
        f_gate_sigmoid = self.hard_sigmoid(
            np.matmul(single_input, f_gate_kernel) + np.matmul(h_prev, f_gate_recurrent_kernel) + f_gate_biases
        )
        o_gate_sigmoid = self.hard_sigmoid(
            np.matmul(single_input, o_gate_kernel) + np.matmul(h_prev, o_gate_recurrent_kernel) + o_gate_biases
        )
        c_gate_tanh = self.hard_tanh(
            np.matmul(single_input, c_gate_kernel) + np.matmul(h_prev, c_gate_recurrent_kernel) + c_gate_biases
        )
        ct = f_gate_sigmoid * c_prev + i_gate_sigmoid * c_gate_tanh
        flat_ct = ct.flatten()
        self.save_hex(flat_array=flat_ct, filename="CTBuffer.mem", int_bits=4, frac_bits=12, type="twos_complement")
        tanh_ct = self.hard_tanh(flat_ct)
        self.save_hex(
            flat_array=tanh_ct, filename="CT_TanhBuffer.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        ht = tanh_ct * o_gate_sigmoid
        flat_ht = ht.flatten()
        self.save_hex(flat_array=flat_ht, filename="HBuffer.mem", int_bits=4, frac_bits=12, type="twos_complement")
        return
        h = np.zeros((1, 32))
        recurrent_kernel_mvm = np.matmul(h, i_gate_recurrent_kernel)
        recurrent_kernel_mvm = recurrent_kernel_mvm.flatten()
        kernel_mvm = np.matmul(single_input, i_gate_kernel)
        kernel_mvm = kernel_mvm.flatten()
        # ----------------------------------------------------------------------------------
        # conv = h.flatten() * i_gate_recurrent_kernel_trans[:32]
        # # self.mac(inputs=single_input.flatten(), weights=i_gate_kernel_trans[:32])
        # results = sum(conv)
        # mac_result = 0
        # for i in range(32):
        #     inp = h.flatten()[i]
        #     w = i_gate_recurrent_kernel_trans[:32][i]
        #     print(f"w: {w}, inp: {inp}")
        #     mult = inp * w
        #     print(self._hex(mult))
        #     print("-" * 60)
        #     mac_result += mult
        # self.save_hex(
        #     flat_array=[mac_result],
        #     filename="i_gate_recurrent_mvm.mem",
        #     int_bits=4,
        #     frac_bits=12,
        #     type="twos_complement",
        # )
        # return
        # --------------------------------------------------------------------------------------
        self.save_hex(
            flat_array=kernel_mvm, filename="i_gate_mvm.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        self.save_hex(
            flat_array=recurrent_kernel_mvm,
            filename="i_gate_recurrent_mvm.mem",
            int_bits=4,
            frac_bits=12,
            type="twos_complement",
        )
        self.save_hex(
            flat_array=single_input.flatten(),
            filename="single_input.mem",
            int_bits=4,
            frac_bits=12,
            type="twos_complement",
        )
        self.save_hex(flat_array=i_gate_kernel_trans, filename="single_weights.mem")
        print(f"mvm shape: {kernel_mvm.shape}")
        print(f"recurrent mvm shape: {recurrent_kernel_mvm.shape}")
        results = kernel_mvm + recurrent_kernel_mvm + i_gate_bias
        self.save_hex(
            flat_array=results, filename="i_gate_results.mem", int_bits=4, frac_bits=12, type="twos_complement"
        )
        self.save_hex(
            flat_array=self.sigmoid(results),
            filename="i_gate_sigmoid",
            int_bits=4,
            frac_bits=12,
            type="twos_complement",
        )
        self.save_hex(
            flat_array=self.hard_sigmoid(results),
            filename="i_gate_hard_sigmoid",
            int_bits=4,
            frac_bits=12,
            type="twos_complement",
        )

    def run(self):
        print("INFO: manual inference called!")
        # model_input: np.ndarray = self.get_random_input(_from="validation")
        # print(model_input.shape)
        model_input: np.ndarray = self.get_random_input()
        # model_input = [
        #     0.45929628,
        #     0.79989361,
        #     0.8695671,
        #     0.41531773,
        #     0.26924149,
        #     0.12415365,
        #     0.82781691,
        #     0.33033358,
        #     0.01066194,
        #     0.35340493,
        #     0.64590161,
        #     0.65769779,
        #     0.22291437,
        #     0.83282802,
        #     0.96214574,
        #     0.45689524,
        #     0.9945435,
        #     0.56380107,
        #     0.80128656,
        #     0.47115956,
        #     0.27279179,
        #     0.29572027,
        #     0.80724564,
        #     0.37955554,
        #     0.85338941,
        #     0.41140614,
        #     0.53338419,
        #     0.36498607,
        #     0.66152791,
        #     0.71917363,
        #     0.22816605,
        #     0.22987329,
        # ]
        # model_input = np.array(model_input)
        tf_target_input, tf_target_output = self.get_tf_results(model_input=model_input)
        print(f"INFO: TF Target input shape: {tf_target_input.shape}, TF Target output shape: {tf_target_output.shape}")
        manual_target_output = self.get_manual_results(layer_input=tf_target_input)
