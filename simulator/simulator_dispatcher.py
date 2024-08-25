from config import Config
from .lstm_simulator import LSTMSimulator


class SimulatorDispatcher:

    def __init__(self, config: Config, name="A module to select the right simulator basedon config") -> None:
        self.name = name
        self.config = config

    def dispatch(self):
        if self.config.simulator == "LSTM":
            print(f"INFO: Selected simulator is {self.config.simulator}")
            return LSTMSimulator(config=self.config)
        else:
            raise Exception(
                f"ERROR: Selected simulator is invalid. There is no simulator with this tag: {self.config.simulator}"
            )
