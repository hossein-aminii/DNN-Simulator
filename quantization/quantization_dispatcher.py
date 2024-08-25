from config import Config
from .fix_point_pt_quantizer import FixPointPTQuantizer


class QuantizationDispatcher:

    def __init__(self, config: Config, name="A module to select the right quantizer based on config") -> None:
        self.name = name
        self.config = config

    def dispatch(self):
        if self.config.quantizer == "fix-point-PT-quantizer":
            print(f"INFO: Selected quantizer is {self.config.quantizer}")
            return FixPointPTQuantizer(config=self.config)
        else:
            raise Exception(
                f"ERROR: Selected quantizer is invalid. There is no quantizer with this tag: {self.config.quantizer}"
            )
