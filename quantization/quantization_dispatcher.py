from config import Config
from .fix_point_pt_quantizer import FixPointPTQuantizer
from .inq_quantizer import INQ


class QuantizationDispatcher:

    def __init__(self, config: Config, name="A module to select the right quantizer based on config") -> None:
        self.name = name
        self.config = config

    def dispatch(self):
        if self.config.quantizer == "fix-point-PT-quantizer":
            print(f"INFO: Selected quantizer is {self.config.quantizer}")
            return FixPointPTQuantizer(config=self.config)
        elif self.config.quantizer == "QAT-INQ":
            print(f"INFO: Selected quantizer is {self.config.quantizer}")
            return INQ(config=self.config)
        else:
            raise Exception(
                f"ERROR: Selected quantizer is invalid. There is no quantizer with this tag: {self.config.quantizer}"
            )
