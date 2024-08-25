from config import Config
from .weight_distribution import WeightDistributionVisualizer


class VisualizerDispatcher:

    def __init__(self, config: Config, name="A module to select the right visualizer based on config") -> None:
        self.name = name
        self.config = config

    def dispatch(self):
        if self.config.visualizer == "weight_distribution":
            print(f"INFO: Selected visualizer is {self.config.visualizer}")
            return WeightDistributionVisualizer(config=self.config)
        else:
            raise Exception(
                f"ERROR: Selected visualizer is invalid. There is no visualizer with this tag: {self.config.visualizer}"
            )
