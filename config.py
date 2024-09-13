import os


class Config:
    """
    simulator parameter:
        - must be a string
        - supported values: ["LSTM"]
        - future values: ["CNN", "MLP"]
    """

    simulator = "LSTM"

    model_info = {
        "type": "LSTM",
        "initialize": False,
        "layers_info": [
            {"type": "Embedding", "input_dim": 10000, "output_dim": 32, "input_length": 128},
            {"type": "SpatialDropout1D", "rate": 0.2},
            {
                "type": "LSTM",
                "units": 32,
                "use_bias": True,
                "dropout": 0,
                "recurrent_dropout": 0,
                "input_shape": (128, 32),
            },
            {"type": "Dense", "units": 2, "use_bias": False, "activation": "softmax"},
        ],  # needed when initialize is True (crete new model)
        # "filepath": os.path.join(
        #     "results", "models", "base_model", "IMDB_LSTM_Base_epoch#1.h5"
        # ),  # nedded when initialize is False (load model from a file)
        # "filepath": os.path.join("results", "models", "base_model", "IMDB_LSTM_Base_epoch#1.h5"),
        # "filepath": "IMDB_LSTM_INQ_step#3_epoch#1.h5",
        "filepath": os.path.join("results", "models", "QAT", "INQ", f"IMDB_LSTM_INQ_int{2}_fraction{6}.h5"),
    }

    """
    action parameter:
        - must be a string
        - supported values: ["train", "inference", "quantization", "visualization", "test", "fault-injection", "manual-inference"]
    """
    action = "visualization"

    """
    1- train
    2- validation
    """
    inference_data = "validation"

    dataset_info = {
        "name": "IMDB",
        "filepath": os.path.join("datasets", "IMDB", "data", "IMDB-Dataset.csv"),
        "tokenizer_filepath": os.path.join("results", "models", "base_model", "tokenizer.pickle"),
    }

    train_params = {"epochs": 5}

    """
    options:
        1- weight_distribution
        2- BER
        3- bar_chart
    """
    visualizer = "BER"
    visualizer_config = {
        "important_layers": ["lstm", "dense"],
        "filepath": os.path.join("results", "BER", "BER.json"),
    }

    """
    options:
        1- fix-point-PT-quantizer
        2- QAT-INQ
    """
    quantizer = "QAT-INQ"
    quantizer_config = {
        "int_bits": 2,
        "fraction_bits": 2,
        "total_bits": 4,
        "important_tensors": {"lstm": [0, 1, 2], "dense": [0]},  # layer_name: indexes
        "format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
        "model_results_filepath": os.path.join(
            "results", f"fixed-point-model-int{2}-frac{2}-format-twos_complement.h5"
        ),
    }
    INQ_quantizer_config = {
        "accumulated_portion": [0.5, 0.75, 0.875, 1],
        "fraction_bits": 6,
        "important_tensors": {
            "lstm": [0, 1],
            "dense": [0],
        },
        "tensor_mapping": {"lstm-0": 1, "lstm-1": 2, "dense-0": 4},
    }

    """
    options:
        1- basic_fault_injector
        2- accuracy_checker
    """
    fault_injector = "basic_fault_injector"
    fault_injector_config = {
        "injection_mode": "normal",  # 1-normal  2-map_to_zero  3-map_to_less  4-specific_bit
        "specific_bit_index": 0,
        "log": False,
        "injection_type": "single_bit_flip",
        "int_bits": 2,
        "fraction_bits": 6,
        "total_bits": 8,
        "important_tensors": {"lstm": [0, 1, 2]},  # layer_name: indexes
        "fix_point_format": "sign_magnitude",  # 1-twos_complement  2-sign_magnitude
        "fault_injection_ratio": 1e-1,
        "model_results_directory": os.path.join(
            "results", f"INQ-{2}-{6}_fault_injection_ratio{1e-1}-int{2}-fraction{6}-format-sign_magnitude"
        ),
        "num_sample": 400,
        "accuracy_results_filename": f"accuracy_results_int{2}_frac{6}_ER{1e-1}.json",
        "remove_generated_files": True,
        "sample_index_range": (0, 10),
    }

    manual_inference_config = {
        "target_layer": "lstm",
        "prev_target_layer": "spatial_dropout1d",
        "weights_info": {0: "kernel", 1: "recurrent_kernel"},
        "bias_info": {2: "biases"},
    }
