from config import Config
from .fault_injection import FaultInjector
from .check_accuracy import AccuracyChecker


class FaultInjectionDispatcher:

    def __init__(self, config: Config, name="A module to select the right quantizer based on config") -> None:
        self.name = name
        self.config = config

    def dispatch(self):
        if self.config.fault_injector == "basic_fault_injector":
            print(f"INFO: Selected fault injector is {self.config.fault_injector}")
            return FaultInjector(config=self.config)
        elif self.config.fault_injector == "accuracy_checker":
            print(f"INFO: Selected fault injector is {self.config.fault_injector}")
            return AccuracyChecker(config=self.config)
        else:
            raise Exception(
                f"ERROR: Selected fault injector is invalid. There is no fault injector with this tag: {self.config.fault_injector}"
            )
