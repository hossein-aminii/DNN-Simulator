import numpy as np
from quantization.utils import FixedPointConverter
import random


class BaseFaultInjector:

    def __init__(
        self,
        fault_injection_ratio: float,
        int_bits: int,
        fraction_bits: int,
        fix_point_format: str = "twos_complement",
        mode: str = "single_bit_flip",
        name="A module to inject faults into weights array",
    ) -> None:
        self.name = name
        self.mode = mode
        self.fault_injection_ratio = fault_injection_ratio
        self.int_bits = int_bits
        self.fraction_bits = fraction_bits
        self.total_bits = int_bits + fraction_bits
        self.fix_point_format = fix_point_format
        self.converter = FixedPointConverter(int_bits=self.int_bits, frac_bits=self.fraction_bits)

    def get_num_bits_to_flip(self, array: np.ndarray):
        num_bits = np.prod(array.shape) * self.total_bits
        return int(num_bits * self.fault_injection_ratio) + 1

    def _inject_single_bit_faults(self, binary_array: np.ndarray, num_bits_to_flip: int):
        selected_weight_indexes = []
        unselected_weight_indexes = [i for i in range(len(binary_array))]
        for _ in range(num_bits_to_flip):
            if len(unselected_weight_indexes) == 0:
                return binary_array
            random.shuffle(unselected_weight_indexes)
            array_idx = unselected_weight_indexes[0]
            selected_weight_indexes.append(array_idx)
            unselected_weight_indexes.remove(array_idx)
            bit_idx = random.randint(0, self.converter.total_bits - 1)
            binary_str = binary_array[array_idx]
            # Flip the selected bit
            new_bit = "0" if binary_str[bit_idx] == "1" else "1"
            binary_array[array_idx] = binary_str[:bit_idx] + new_bit + binary_str[bit_idx + 1 :]
        return binary_array

    def _inject_multi_bit_faults(self, binary_array: np.ndarray, num_bits_to_flip: int):
        for _ in range(num_bits_to_flip):
            array_idx = random.randint(0, len(binary_array) - 1)
            bit_idx = random.randint(0, self.converter.total_bits - 1)
            binary_str = binary_array[array_idx]
            # Flip the selected bit
            new_bit = "0" if binary_str[bit_idx] == "1" else "1"
            binary_array[array_idx] = binary_str[:bit_idx] + new_bit + binary_str[bit_idx + 1 :]
        return binary_array

    def inject_faults(self, weights: np.ndarray):

        fixed_point_array = self.converter.array_to_fixed_point(array=weights, format_type=self.fix_point_format)
        faulty_array = np.copy(fixed_point_array)

        num_bits_to_flip = self.get_num_bits_to_flip(array=fixed_point_array)

        print(f"num bits to flip: {num_bits_to_flip}")

        flat_array = fixed_point_array.flatten()

        binary_array = np.array([b.decode("utf-8") for b in flat_array])

        if self.mode == "single_bit_flip":
            binary_array = self._inject_single_bit_faults(binary_array=binary_array, num_bits_to_flip=num_bits_to_flip)
        elif self.mode == "multi_bit_flip":
            binary_array = self._inject_multi_bit_faults(binary_array=binary_array, num_bits_to_flip=num_bits_to_flip)
        else:
            raise ValueError("Invalid mode. Choose 'single_bit_flip' or 'multi_bit_flip'.")

        faulty_array = np.array([b.encode("utf-8") for b in binary_array]).reshape(fixed_point_array.shape)
        float_faulty_array = self.converter.fixed_point_array_to_float(
            array=faulty_array, format_type=self.fix_point_format
        )
        return float_faulty_array
