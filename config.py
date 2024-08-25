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
        # "filepath": "results\\models\\base_model\\IMDB_LSTM_Base_epoch#1.h5",  # nedded when initialize is False (load model from a file)
        "filepath": f"results\\fixed-point-model-int{12}-frac{4}-format-twos_complement.h5",
    }

    """
    action parameter:
        - must be a string
        - supported values: ["train", "inference", "quantization", "visualization", "test", "fault-injection", "FI-accuracy"]
    """
    action = "fault-injection"

    """
    1- train
    2- validation
    """
    inference_data = "validation"

    dataset_info = {
        "name": "IMDB",
        "filepath": "datasets\\IMDB\\data\\IMDB-Dataset.csv",
        "tokenizer_filepath": "results\\models\\base_model\\tokenizer.pickle",
        # "tokenizer_filepath": "results\\models\\4\\tokenizer.pic",
    }

    train_params = {"epochs": 5}

    visualizer = "weight_distribution"
    visualizer_config = {"important_layers": ["lstm", "dense"]}

    quantizer = "fix-point-PT-quantizer"
    quantizer_config = {
        "int_bits": 2,
        "fraction_bits": 2,
        "total_bits": 4,
        "important_tensors": {"lstm": [0, 1, 2], "dense": [0]},  # layer_name: indexes
        "format": "twos_complement",  # 1-twos_complement  2-sign_magnitude
        "model_results_filepath": f"results\\fixed-point-model-int{2}-frac{2}-format-twos_complement.h5",
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
        "model_results_directory": f"results\\fault_injection_ratio{1e-1}-int{12}-fraction{4}-format-twos_complement",
        "num_sample": 400,
        "accuracy_results_filename": "accuracy_results.json",
    }
