from config import Config
import multiprocessing
import time
from simulator import SimulatorDispatcher
import os


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
    config.fault_injector = "accuracy_checker"
    run_simulator(config)


def fault_injection_runner():
    configs_data = []
    for config_info in configs_data:
        config = Config()
        config.model_info["filepath"] = config_info["model_filepath"]
        config.action = config_info["action"]
        config.fault_injector = config_info["fault_injector"]
        config.fault_injector_config.update(config_info["fault_injector_config"])

        p = multiprocessing.Process(target=run_simulation_in_background, args=(config,))
        p.start()


if __name__ == "__main__":
    fault_injection_runner()
