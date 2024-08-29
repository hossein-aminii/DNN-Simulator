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
        injection_type: str = "single_bit_flip",
        injection_mode: str = "normal",
        specific_bit_index: int = 1000,
        log: bool = False,
        name="A module to inject faults into weights array",
    ) -> None:
        self.name = name
        self.injection_type = injection_type
        self.injection_mode = injection_mode
        self.specific_bit_index = specific_bit_index
        self.fault_injection_ratio = fault_injection_ratio
        self.int_bits = int_bits
        self.fraction_bits = fraction_bits
        self.total_bits = int_bits + fraction_bits
        self.fix_point_format = fix_point_format
        self.converter = FixedPointConverter(int_bits=self.int_bits, frac_bits=self.fraction_bits)
        self.log = log

    def get_num_bits_to_flip(self, array: np.ndarray):
        num_bits = np.prod(array.shape) * self.total_bits
        return int(num_bits * self.fault_injection_ratio) + 1

    def print_log_info(self, fault_free_binary: str, normal_faulty_binary: str, faulty_binary: str, bit_index: int):
        if self.log:
            print(
                f"fault free value: {fault_free_binary} -> {self.converter.sign_magnitude_to_float(fault_free_binary)}"
            )
            print(f"bit index to inject: {bit_index}")
            print(
                f"normal faulty value: {normal_faulty_binary} -> {self.converter.sign_magnitude_to_float(normal_faulty_binary)}"
            )
            print(f"faulty value: {faulty_binary} -> {self.converter.sign_magnitude_to_float(faulty_binary)}")
            print("-" * 80)

    def inject_fault_to_weight(self, fault_free_weight: str, bit_index: int):
        """
        fault_free_weight: is a string of bits (ex. 01000000)
        bit_index: position of bit to flip. must be less than "fault_free_weight" length!
        """
        # handle specific bit selection for fault injection
        if self.injection_mode == "specific_bit":
            bit_index = self.specific_bit_index
        # -----------------------------------------------------------------------------------------------
        # do bit flip!
        new_bit = "0" if fault_free_weight[bit_index] == "1" else "1"
        normal_faulty_weight = fault_free_weight[:bit_index] + new_bit + fault_free_weight[bit_index + 1 :]
        # ------------------------------------------------------------------------------------------------
        if self.injection_mode == "normal" or self.injection_mode == "specific_bit":  # normal bit flip
            self.print_log_info(
                fault_free_binary=fault_free_weight,
                normal_faulty_binary=normal_faulty_weight,
                faulty_binary=normal_faulty_weight,
                bit_index=bit_index,
            )
            return normal_faulty_weight
        # -------------------------------------------------------------------------------------------------
        # handle bit flip detection and fault tolerance improvement mechanism. remember, for now, this is only works with sign magnitude only
        sign_bit = normal_faulty_weight[0]
        value_bits = normal_faulty_weight[1:]
        count_of_ones = value_bits.count("1")
        if self.injection_mode == "map_to_zero":  # if the fault is detectable, return zero as faulty weight
            if count_of_ones == 1:  # fault is not detectable
                self.print_log_info(
                    fault_free_binary=fault_free_weight,
                    normal_faulty_binary=normal_faulty_weight,
                    faulty_binary=normal_faulty_weight,
                    bit_index=bit_index,
                )
                return normal_faulty_weight
            else:  # faulty value equal to zero, or fault is detectable
                faulty_weight = "0" * len(normal_faulty_weight)
                self.print_log_info(
                    fault_free_binary=fault_free_weight,
                    normal_faulty_binary=normal_faulty_weight,
                    faulty_binary=faulty_weight,
                    bit_index=bit_index,
                )
                return faulty_weight

        elif (
            self.injection_mode == "map_to_less"
        ):  # if the fault is detectable, return well format weight that has the least value
            if count_of_ones > 1:  # fault is detectable
                rightmost_one_index = value_bits.rfind("1")
                if rightmost_one_index != -1:
                    # Create a new string with all '0's except for the rightmost '1'
                    new_value_bits = "0" * rightmost_one_index + "1" + "0" * (len(value_bits) - rightmost_one_index - 1)
                else:
                    # If there is no '1' in the string, the value bits string is all '0's
                    new_value_bits = "0" * len(value_bits)
                faulty_weight = sign_bit + new_value_bits
                self.print_log_info(
                    fault_free_binary=fault_free_weight,
                    normal_faulty_binary=normal_faulty_weight,
                    faulty_binary=faulty_weight,
                    bit_index=bit_index,
                )
                return faulty_weight
            elif count_of_ones == 0:
                faulty_weight = "0" * len(normal_faulty_weight)
                self.print_log_info(
                    fault_free_binary=fault_free_weight,
                    normal_faulty_binary=normal_faulty_weight,
                    faulty_binary=faulty_weight,
                    bit_index=bit_index,
                )
                return faulty_weight
            else:
                self.print_log_info(
                    fault_free_binary=fault_free_weight,
                    normal_faulty_binary=normal_faulty_weight,
                    faulty_binary=normal_faulty_weight,
                    bit_index=bit_index,
                )
                return normal_faulty_weight

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
            # ----------------------------------------------------------------------------------------
            # Flip the selected bit
            # new_bit = "0" if binary_str[bit_idx] == "1" else "1"
            # binary_array[array_idx] = binary_str[:bit_idx] + new_bit + binary_str[bit_idx + 1 :]
            # ----------------------------------------------------------------------------------------
            faulty_weight = self.inject_fault_to_weight(fault_free_weight=binary_str, bit_index=bit_idx)
            binary_array[array_idx] = faulty_weight
        return binary_array

    def _inject_multi_bit_faults(self, binary_array: np.ndarray, num_bits_to_flip: int):
        for _ in range(num_bits_to_flip):
            array_idx = random.randint(0, len(binary_array) - 1)
            bit_idx = random.randint(0, self.converter.total_bits - 1)
            binary_str = binary_array[array_idx]
            # -------------------------------------------------------------------------------------------
            # Flip the selected bit
            # new_bit = "0" if binary_str[bit_idx] == "1" else "1"
            # binary_array[array_idx] = binary_str[:bit_idx] + new_bit + binary_str[bit_idx + 1 :]
            # ----------------------------------------------------------------------------------------------
            faulty_weight = self.inject_fault_to_weight(fault_free_weight=binary_str, bit_index=bit_idx)
            binary_array[array_idx] = faulty_weight
        return binary_array

    def inject_faults(self, weights: np.ndarray):

        fixed_point_array = self.converter.array_to_fixed_point(array=weights, format_type=self.fix_point_format)
        faulty_array = np.copy(fixed_point_array)

        num_bits_to_flip = self.get_num_bits_to_flip(array=fixed_point_array)

        print(f"INFO: num bits to flip: {num_bits_to_flip}")

        flat_array = fixed_point_array.flatten()

        binary_array = np.array([b.decode("utf-8") for b in flat_array])

        if self.injection_type == "single_bit_flip":
            binary_array = self._inject_single_bit_faults(binary_array=binary_array, num_bits_to_flip=num_bits_to_flip)
        elif self.injection_type == "multi_bit_flip":
            binary_array = self._inject_multi_bit_faults(binary_array=binary_array, num_bits_to_flip=num_bits_to_flip)
        else:
            raise ValueError("Invalid mode. Choose 'single_bit_flip' or 'multi_bit_flip'.")

        faulty_array = np.array([b.encode("utf-8") for b in binary_array]).reshape(fixed_point_array.shape)
        float_faulty_array = self.converter.fixed_point_array_to_float(
            array=faulty_array, format_type=self.fix_point_format
        )
        return float_faulty_array
