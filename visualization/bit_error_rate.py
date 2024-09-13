import json
import pandas as pd
from matplotlib import pyplot as plt
from config import Config


class BERVisualizer:

    def __init__(self, config: Config, name="A module to plot accuracy or loss info according to BER") -> None:
        self.name = name
        self.config = config
        visualizer_config = self.config.visualizer_config
        self.filepath = visualizer_config["filepath"]
        self.data: dict = self.read_json(filepath=self.filepath)

    @staticmethod
    def read_json(filepath: str):
        with open(filepath, "r") as jf:
            return json.load(jf)

    def run(self):
        ber = self.data.pop("BER")
        x_title = self.data.pop("x-title")
        y_title = self.data.pop("y-title")
        df = pd.DataFrame(self.data, index=ber)
        ax = df.plot(kind="line", logx=True, linewidth=4)
        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)
        plt.show()
