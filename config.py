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
        "filepath": os.path.join(
            "results", "models", "base_model", "IMDB_LSTM_Base_epoch#1.h5"
        ),  # nedded when initialize is False (load model from a file)
        # "filepath": os.path.join("results", "models", "QAT", "INQ", "IMDB_LSTM_INQ_step#4_epoch#1.h5"),
        # "filepath": "IMDB_LSTM_INQ_step#3_epoch#1.h5",
        # "filepath": os.path.join("results", "models", "PTQ", "8-bit", f"fixed-point-model-int{12}-frac{4}-format-twos_complement.h")
    }

    """
    action parameter:
        - must be a string
        - supported values: ["train", "inference", "quantization", "visualization", "test", "fault-injection", "FI-accuracy"]
    """
    action = "quantization"

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
    """
    visualizer = "BER"
    visualizer_config = {"important_layers": ["lstm", "dense"], "filepath": os.path.join("results", "BER", "BER.json")}

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
    fault_injector = "accuracy_checker"
    fault_injector_config = {
        "int_bits": 12,
        "fraction_bits": 4,
        "total_bits": 16,
        "important_tensors": {"lstm": [0, 1, 2]},  # layer_name: indexes
        "fix_point_format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
        "fault_injection_ratio": 1e-1,
        "mode": "single_bit_flip",
        "model_results_directory": os.path.join(
            "results", f"fault_injection_ratio{1e-1}-int{12}-fraction{4}-format-twos_complement"
        ),
        "num_sample": 400,
        "accuracy_results_filename": "accuracy_results.json",
    }
