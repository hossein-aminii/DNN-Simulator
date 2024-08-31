from config import Config
import multiprocessing
import time
from simulator import SimulatorDispatcher
import os

# import tensorflow as tf


# gpu_index = 1
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_visible_devices(physical_devices[gpu_index], "GPU")


def run_simulator(config):
    # start chronometer
    start = time.time()
    dispatcher = SimulatorDispatcher(config=config)
    simulator = dispatcher.dispatch()
    simulator.run()
    # stop chronometer
    end = time.time()
    # ----------------------------------------------
    # calculate running time
    take_long = end - start
    hours = int(take_long // 3600)
    take_long = take_long % 3600
    minutes = int(take_long // 60)
    take_long = take_long % 60
    seconds = int(take_long)
    # ------------------------------------------------
    # show running time
    print("this simulation take {:02d}:{:02d}:{:02d} long!".format(hours, minutes, seconds))


def run_simulation_in_background(config: Config):

    run_simulator(config)
    # config.fault_injector = "accuracy_checker"
    # run_simulator(config)

    # int_bits = 2
    # fraction_bits = 6
    # injection_mode = "specific_bit"  # 1-normal  2-map_to_zero  3-map_to_less  4-specific_bit
    # specific_bit_index = 0
    # fix_point_format = "sign_magnitude"  # 1-twos_complement  2-sign_magnitude
    # ratios = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    # quantization_type = "QAT"  # 1-QAT  2-PTQ
    # quantization = "INQ"  # 1-INQ  2-16-bit  3-12-bit  4-8-bit
    # model_name = f"IMDB_LSTM_INQ_int{int_bits}_fraction{fraction_bits}.h5"


def fault_injection_runner():
    int_bits = 2
    fraction_bits = 6
    injection_mode = "normal"  # 1-normal  2-map_to_zero  3-map_to_less  4-specific_bit
    specific_bit_index = 0
    fix_point_format = "sign_magnitude"  # 1-twos_complement  2-sign_magnitude
    ratios = [1e-1]
    quantization_type = "QAT"  # 1-QAT  2-PTQ
    quantization = "INQ"  # 1-INQ  2-16-bit  3-12-bit  4-8-bit
    model_name = f"IMDB_LSTM_INQ_int{int_bits}_fraction{fraction_bits}.h5"
    fault_injector = "basic_fault_injector"  # 1-basic_fault_injector  2- accuracy_checker
    configs_data = []
    num_process = 40
    start = -1
    end = 10
    for idx in range(num_process):
        for ratio in ratios:
            configs_data.append(
                {
                    "model_filepath": os.path.join("results", "models", quantization_type, quantization, model_name),
                    "action": "fault-injection",
                    "fault_injector": fault_injector,
                    "fault_injector_config": {
                        "injection_mode": injection_mode,
                        "specific_bit_index": specific_bit_index,
                        "log": False,
                        "int_bits": int_bits,
                        "fraction_bits": fraction_bits,
                        "total_bits": int_bits + fraction_bits,
                        "fix_point_format": fix_point_format,
                        "fault_injection_ratio": ratio,
                        "mode": "single_bit_flip",
                        "model_results_directory": os.path.join(
                            "results",
                            f"{injection_mode}#{specific_bit_index}-{int_bits}-{fraction_bits}_fault_injection_ratio{ratio}-int{int_bits}-fraction{fraction_bits}-format-{fix_point_format}",
                        ),
                        "accuracy_results_filename": f"accuracy_results_int{int_bits}_frac{fraction_bits}_ER{ratio}.json",
                        "sample_index_range": (start + 1, end),
                    },
                },
            )
        start = end
        end += 10
    for config_info in configs_data:
        config = Config()
        config.model_info["filepath"] = config_info["model_filepath"]
        config.action = config_info["action"]
        config.fault_injector = config_info["fault_injector"]
        config.fault_injector_config.update(config_info["fault_injector_config"])

        p = multiprocessing.Process(target=run_simulation_in_background, args=(config,))
        p.start()
    # run_simulation_in_background(config=config)


# def fault_injection_runner():
#     int_bits = 1
#     fraction_bits = 7
#     injection_mode = "specific_bit"  # 1-normal  2-map_to_zero  3-map_to_less  4-specific_bit
#     specific_bit_index = 2
#     fix_point_format = "sign_magnitude"  # 1-twos_complement  2-sign_magnitude
#     ratios = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
#     quantization_type = "QAT"  # 1-QAT  2-PTQ
#     quantization = "INQ"  # 1-INQ  2-16-bit  3-12-bit  4-8-bit
#     model_name = f"IMDB_LSTM_INQ_int{int_bits}_fraction{fraction_bits}.h5"
#     fault_injector = "basic_fault_injector"  # 1-basic_fault_injector  2- accuracy_checker
#     configs_data = []
#     for ratio in ratios:
#         configs_data.append(
#             {
#                 "model_filepath": os.path.join("results", "models", quantization_type, quantization, model_name),
#                 "action": "fault-injection",
#                 "fault_injector": fault_injector,
#                 "fault_injector_config": {
#                     "injection_mode": injection_mode,
#                     "specific_bit_index": specific_bit_index,
#                     "log": False,
#                     "int_bits": int_bits,
#                     "fraction_bits": fraction_bits,
#                     "total_bits": int_bits + fraction_bits,
#                     "fix_point_format": fix_point_format,
#                     "fault_injection_ratio": ratio,
#                     "mode": "single_bit_flip",
#                     "model_results_directory": os.path.join(
#                         "results",
#                         f"SpecificBit#2-{int_bits}-{fraction_bits}_fault_injection_ratio{ratio}-int{int_bits}-fraction{fraction_bits}-format-INQ",
#                     ),
#                     "accuracy_results_filename": f"accuracy_results_int{int_bits}_frac{fraction_bits}_ER{ratio}.json",
#                 },
#             },
#         )
#     for config_info in configs_data:
#         config = Config()
#         config.model_info["filepath"] = config_info["model_filepath"]
#         config.action = config_info["action"]
#         config.fault_injector = config_info["fault_injector"]
#         config.fault_injector_config.update(config_info["fault_injector_config"])

#         p = multiprocessing.Process(target=run_simulation_in_background, args=(config,))
#         p.start()
#         # run_simulation_in_background(config=config)


if __name__ == "__main__":
    fault_injection_runner()


"""
accuracy checker

    int_bits = 1
    fraction_bits = 7
    injection_mode = "normal"  # 1-normal  2-map_to_zero  3-map_to_less  4-specific_bit
    specific_bit_index = 0
    fix_point_format = "sign_magnitude"  # 1-twos_complement  2-sign_magnitude
    ratios = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    quantization_type = "QAT"  # 1-QAT  2-PTQ
    quantization = "INQ"  # 1-INQ  2-16-bit  3-12-bit  4-8-bit
    model_name = f"IMDB_LSTM_INQ_int{1}_fraction{7}.h5"
    fault_injector = "accuracy_checker"  # 1-basic_fault_injector  2- accuracy_checker
    configs_data = []
    for ratio in ratios:
        configs_data.append(
            {
                "model_filepath": os.path.join("results", "models", quantization_type, quantization, model_name),
                "action": "fault-injection",
                "fault_injector": fault_injector,
                "fault_injector_config": {
                    "injection_mode": injection_mode,
                    "specific_bit_index": specific_bit_index,
                    "log": False,
                    "int_bits": int_bits,
                    "fraction_bits": fraction_bits,
                    "total_bits": int_bits + fraction_bits,
                    "fix_point_format": fix_point_format,
                    "fault_injection_ratio": ratio,
                    "mode": "single_bit_flip",
                    "model_results_directory": os.path.join(
                        "results",
                        f"INQ-{int_bits}-{fraction_bits}_fault_injection_ratio{ratio}-int{int_bits}-fraction{fraction_bits}-format-INQ",
                    ),
                    "accuracy_results_filename": f"accuracy_results_int{int_bits}_frac{fraction_bits}_ER{ratio}.json",
                },
            },

"""
