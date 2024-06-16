import os
from PIL import Image
import pyheif


def heic_to_jpg(heic_file_path, jpg_file_path):
    # Read HEIC file
    heif_file = pyheif.read(heic_file_path)

    # Convert to a Pillow Image
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    # Save as JPG
    image.save(jpg_file_path, "JPEG")


def convert_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".heic"):
            heic_file_path = os.path.join(directory_path, filename)
            jpg_file_path = os.path.splitext(heic_file_path)[0] + ".jpg"
            heic_to_jpg(heic_file_path, jpg_file_path)
            print(f"Converted {heic_file_path} to {jpg_file_path}")


# Example usage
input_directory = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\Straight Images"  # Replace with the path to your HEIC images directory
convert_directory(input_directory)
