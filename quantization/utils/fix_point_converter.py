import numpy as np


class FixedPointConverter:
    def __init__(self, int_bits, frac_bits):
        self.int_bits = int_bits
        self.frac_bits = frac_bits
        self.total_bits = int_bits + frac_bits

    def float_to_twos_complement(self, value: float) -> str:
        if not (-(2**self.int_bits) <= value < 2**self.int_bits):
            raise ValueError(f"Value {value} is out of range for the fixed-point format (two's complement)")

        # Scale the value to fixed-point representation
        scaled_value = int(round(value * (2**self.frac_bits)))

        # Mask to ensure the value fits within the specified bits
        mask = (1 << self.total_bits) - 1

        # Convert to 2's complement binary representation
        if scaled_value < 0:
            scaled_value = (scaled_value + (1 << self.total_bits)) & mask

        # Format as binary string
        binary_str = format(scaled_value, f"0{self.total_bits}b")

        return binary_str

    def float_to_sign_magnitude(self, value: float) -> str:
        if not (-(2 ** (self.int_bits - 1)) < value < 2 ** (self.int_bits - 1)):
            raise ValueError(f"Value {value} is out of range for the fixed-point format (sign magnitude)")

        # Scale the value to fixed-point representation
        scaled_value = int(round(abs(value) * (2**self.frac_bits)))

        # Handle sign bit
        sign_bit = "1" if value < 0 else "0"

        # Format magnitude part as binary string
        magnitude_binary = format(scaled_value, f"0{self.total_bits - 1}b")

        # Combine sign bit and magnitude
        binary_str = sign_bit + magnitude_binary

        return binary_str

    def twos_complement_to_float(self, binary_str: str) -> float:
        if len(binary_str) != self.total_bits:
            raise ValueError(f"Binary string length {len(binary_str)} does not match total bits {self.total_bits}")

        # Convert binary string to integer
        value = int(binary_str, 2)

        # Check if the value is negative (2's complement)
        if binary_str[0] == "1":
            value -= 1 << self.total_bits

        # Convert back to float
        float_value = value / (2**self.frac_bits)

        return float_value

    def sign_magnitude_to_float(self, binary_str: str) -> float:
        if len(binary_str) != self.total_bits:
            raise ValueError(f"Binary string length {len(binary_str)} does not match total bits {self.total_bits}")

        # Extract sign bit and magnitude
        sign_bit = binary_str[0]
        magnitude_binary = binary_str[1:]

        # Convert magnitude to integer
        magnitude_value = int(magnitude_binary, 2)

        # Determine sign
        sign = -1 if sign_bit == "1" else 1

        # Convert to float
        float_value = sign * magnitude_value / (2**self.frac_bits)

        return float_value

    def array_to_fixed_point(self, array: np.ndarray, format_type: str = "twos_complement"):
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a NumPy array")

        # Prepare an empty array to store the fixed-point representations
        shape = array.shape
        dtype = f"S{self.total_bits}"  # We are converting to binary strings
        fixed_point_array = np.empty(shape, dtype=dtype)

        # Iterate over each element and convert to fixed-point format
        for idx in np.ndindex(array.shape):
            value = array[idx]
            if format_type == "twos_complement":
                fixed_point_array[idx] = self.float_to_twos_complement(value).encode("utf-8")
            elif format_type == "sign_magnitude":
                fixed_point_array[idx] = self.float_to_sign_magnitude(value).encode("utf-8")
            else:
                raise ValueError("Invalid format_type. Choose 'twos_complement' or 'sign_magnitude'.")

        return fixed_point_array

    def fixed_point_array_to_float(self, array: np.ndarray, format_type: str = "twos_complement"):
        if not isinstance(array, np.ndarray):
            raise ValueError("Input must be a NumPy array")

        # Prepare an empty array to store the fixed-point representations
        shape = array.shape
        dtype = float  # We are converting to binary strings
        float_array = np.empty(shape, dtype=dtype)

        # Iterate over each element and convert to fixed-point format
        for idx in np.ndindex(array.shape):
            value = array[idx].decode("utf-8")
            if format_type == "twos_complement":
                float_array[idx] = self.twos_complement_to_float(binary_str=value)
            elif format_type == "sign_magnitude":
                float_array[idx] = self.sign_magnitude_to_float(binary_str=value)
            else:
                raise ValueError("Invalid format_type. Choose 'twos_complement' or 'sign_magnitude'.")

        return float_array

    def apply_distance_check(self, original_array: np.ndarray, converted_array: np.ndarray):
        if not isinstance(original_array, np.ndarray) or not isinstance(converted_array, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        if original_array.shape != converted_array.shape:
            raise ValueError(
                f"Arrays must be have the same shape. original shape: {original_array.shape}, converted shape: {converted_array.shape}"
            )

        distance = 2 ** (-self.frac_bits)

        for idx in np.ndindex(original_array.shape):
            original_value = original_array[idx]
            converted_value = converted_array[idx]
            actual_distance = abs(original_value - converted_value)
            if actual_distance > distance:
                raise ValueError(
                    f"Distance ERROR: Maximum distance must be {distance} but the actual distance is {actual_distance}. original value: {original_value}, converted value: {converted_value}"
                )

    def float_to_fix(self, array: np.ndarray, format_type: str = "twos_complement"):
        fixed_point_array = self.array_to_fixed_point(array=array, format_type=format_type)
        float_array = self.fixed_point_array_to_float(array=fixed_point_array, format_type=format_type)
        self.apply_distance_check(original_array=array, converted_array=float_array)
        return float_array


# Example usage
if __name__ == "__main__":
    int_bits = 1
    frac_bits = 7
    converter = FixedPointConverter(int_bits, frac_bits)

    value = -0.4356789
    twos_complement = converter.float_to_twos_complement(value)
    sign_magnitude = converter.float_to_sign_magnitude(value)
    print(f"2's Complement of {value}: {twos_complement}")
    print(f"Sign-Magnitude of {value}: {sign_magnitude}")
    float1 = converter.twos_complement_to_float(binary_str=twos_complement)
    float2 = converter.sign_magnitude_to_float(binary_str=sign_magnitude)
    print(f" float from 2's Complement of {value}: {float1}")
    print(f" float from Sign-Magnitude of {value}: {float2}")
