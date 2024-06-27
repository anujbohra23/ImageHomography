import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the previously saved homography matrix
homography_matrix_path = "anuj_homography_matrix_left.npy"
homography_anuj = np.load(homography_matrix_path)


def warp_image(image, homography_anuj):
    """
    Apply the homography matrix to warp the input image.
    """
    warped_image = cv2.warpPerspective(
        image, homography_anuj, (image.shape[1], image.shape[0])
    )
    return warped_image


def process_image(angled_image_path, output_directory):
    """
    Process an angled image:
    - Apply homography transformation.
    - Save the original angled and warped images in the output_directory.
    """
    # Load the angled image
    new_image_angled = cv2.imread(angled_image_path, cv2.IMREAD_GRAYSCALE)

    if new_image_angled is None:
        print(f"Error: Unable to load image {angled_image_path}")
        return

    # Warp the angled image using the homography matrix
    warped_image = warp_image(new_image_angled, homography_anuj)

    # Generate and save the figure
    plt.figure(figsize=(10, 5))

    # Angled image
    plt.subplot(1, 2, 1)
    plt.imshow(new_image_angled, cmap="gray")
    plt.title("Angled Image")
    plt.axis("off")

    # Warped image
    plt.subplot(1, 2, 2)
    plt.imshow(warped_image, cmap="gray")
    plt.title("Warped Image")
    plt.axis("off")

    # Save the figure
    base_name = os.path.splitext(os.path.basename(angled_image_path))[0]
    output_file = os.path.join(output_directory, f"{base_name}_warped.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Images saved to {output_file}")


def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    angled_images_path = os.path.join(input_directory, "AngledImages")

    for filename in os.listdir(angled_images_path):
        if filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
            angled_image_path = os.path.join(angled_images_path, filename)
            process_image(angled_image_path, output_directory)


if __name__ == "__main__":
    input_directory = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset"
    output_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\warpedImagesCoffeeMatrix(left)"
    )

    main(input_directory, output_directory)
