import os
import zipfile


def zip_directories(directories, output_zipfile):
    """
    Zips multiple directories into a single ZIP file.

    :param directories: List of directory paths to include in the ZIP file.
    :param output_zipfile: The output ZIP file name.
    """
    with zipfile.ZipFile(output_zipfile, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dir_path in directories:
            # Walk through each directory
            print(f"INFO: zipping directory {dir_path}...")
            for root, _, files in os.walk(dir_path):
                for file in files:
                    # Create a complete file path
                    file_path = os.path.join(root, file)
                    # Add file to zip, preserving directory structure
                    zipf.write(file_path, os.path.relpath(file_path, start=os.path.dirname(dir_path)))


if __name__ == "__main__":
    base_directory = "results"
    selected_bit = 3
    type = f"specific_bit#{selected_bit}"
    int_bits = 2
    fraction_bits = 6
    format = "sign_magnitude"
    directories_to_zip = [
        os.path.join(
            base_directory,
            f"{type}-{int_bits}-{fraction_bits}_fault_injection_ratio{0.0001}-int{int_bits}-fraction{fraction_bits}-format-{format}",
        ),
        os.path.join(
            base_directory,
            f"{type}-{int_bits}-{fraction_bits}_fault_injection_ratio{0.001}-int{int_bits}-fraction{fraction_bits}-format-{format}",
        ),
        os.path.join(
            base_directory,
            f"{type}-{int_bits}-{fraction_bits}_fault_injection_ratio{0.01}-int{int_bits}-fraction{fraction_bits}-format-{format}",
        ),
        os.path.join(
            base_directory,
            f"{type}-{int_bits}-{fraction_bits}_fault_injection_ratio{0.1}-int{int_bits}-fraction{fraction_bits}-format-{format}",
        ),
        os.path.join(
            base_directory,
            f"{type}-{int_bits}-{fraction_bits}_fault_injection_ratio{1e-05}-int{int_bits}-fraction{fraction_bits}-format-{format}",
        ),
        os.path.join(
            base_directory,
            f"{type}-{int_bits}-{fraction_bits}_fault_injection_ratio{1e-06}-int{int_bits}-fraction{fraction_bits}-format-{format}",
        ),
    ]

    # Output zip file name
    output_zipfile_name = os.path.join(
        base_directory, f"output_specific_bit#{selected_bit}_{int_bits}_{fraction_bits}.zip"
    )

    # Zip the directories
    zip_directories(directories_to_zip, output_zipfile_name)

    print(f"Successfully created {output_zipfile_name} contains {len(directories_to_zip)} directories")
