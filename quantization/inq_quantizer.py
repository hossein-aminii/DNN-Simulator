from config import Config
from model.utils import ModelLoader
import numpy as np
from train import CustomTrainer


class INQ:

    def __init__(self, config: Config, name="A module to apply INQ quantization on model") -> None:
        self.name = name
        self.model_loader = ModelLoader(config=config)
        self.config = config
        self.inq_config: dict = self.config.INQ_quantizer_config
        self.accumulated_portion: list = self.inq_config["accumulated_portion"]
        self.fraction_bits: int = self.inq_config["fraction_bits"]
        self.tensors: dict = self.inq_config["important_tensors"]
        self.tensor_mapping: dict = self.inq_config["tensor_mapping"]
        self.masks = []

        self.available_weights: np.ndarray = self.get_available_weights(fraction_bits=self.fraction_bits)
        print(f"INFO: available weights in INQ: {self.available_weights}")

    def get_available_weights(self, fraction_bits: int):
        weights = [-1, 0, 1]
        for idx in range(fraction_bits):
            weights.append(2 ** -(idx + 1))
            weights.append(-(2 ** -(idx + 1)))
        weights = np.array(weights)
        weights.sort()
        return weights

    def apply_quantization_on_array(self, array: np.ndarray, portion: float, array_tag: str):
        mask_idx = self.tensor_mapping[array_tag]
        mask = self.masks[mask_idx]
        if mask is None:
            mask = np.ones(shape=array.shape)
        # ---------------------------------------------------------------------------
        flat_array = array.flatten()
        flat_mask = mask.flatten()
        # ----------------------------------------------------------------------------
        ones_indices = np.where(flat_mask == 1)[0]
        num_quantized_value = int(np.ceil(len(flat_mask) * portion))
        if num_quantized_value >= len(flat_mask):
            selected_indices = ones_indices
        else:
            np.random.shuffle(ones_indices)
            selected_indices = ones_indices[:num_quantized_value]
        # -----------------------------------------------------------------------------
        flat_mask[selected_indices] = 0
        selected_values = flat_array[selected_indices]
        for idx, value in enumerate(selected_values):
            if value < self.available_weights[0]:
                selected_values[idx] = self.available_weights[0]
            elif value > self.available_weights[-1]:
                selected_values[idx] = self.available_weights[-1]
            else:
                diffs = np.abs(self.available_weights - value)
                closest_value = self.available_weights[np.argmin(diffs)]
                selected_values[idx] = closest_value
        # --------------------------------------------------------------------------------
        flat_array[selected_indices] = selected_values
        mask = flat_mask.reshape(mask.shape)
        self.masks[mask_idx] = mask
        array = flat_array.reshape(array.shape)
        return array

    def apply_quantization(self, model, portion):
        for layer in model.layers:
            if layer.name not in list(self.tensors.keys()):
                continue
            indexes = self.tensors[layer.name]
            new_weights = []
            for idx, w in enumerate(layer.get_weights()):
                if idx in indexes:
                    new_weights.append(
                        self.apply_quantization_on_array(array=w, portion=portion, array_tag=f"{layer.name}-{idx}")
                    )
                else:
                    new_weights.append(w)
            layer.set_weights(new_weights)
        return model

    def initialize_masks(self, model):
        self.masks = [np.ones_like(var) for var in model.trainable_variables]

    def test_model(self, model):
        for layer in model.layers:
            if layer.name not in list(self.tensors.keys()):
                continue
            indexes = self.tensors[layer.name]
            for idx, w in enumerate(layer.get_weights()):
                if idx in indexes:
                    num_error = 0
                    no_error = 0
                    flat_w = w.flatten()
                    for value in flat_w:
                        if value not in self.available_weights:
                            print(f"ERROR: value {value} not in list!")
                            num_error += 1
                        else:
                            no_error += 1
                    print(f"num errors: {num_error}")
                    print(f"no error: {no_error}")

    def run(self):
        prev_portion = 0
        model = self.model_loader.load()
        self.initialize_masks(model=model)
        trainer = CustomTrainer(config=self.config)
        for idx, new_portion in enumerate(self.accumulated_portion):
            print(f"INFO: Step {idx + 1}/{len(self.accumulated_portion)} in INQ process...")
            this_step_portion = new_portion - prev_portion if new_portion != 1 else 1
            model = self.apply_quantization(model=model, portion=this_step_portion)
            prev_portion = new_portion
            trainer.epochs = 1
            model = trainer.retrain(model=model, masks=self.masks, inq_step=idx + 1)
