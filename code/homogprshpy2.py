import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the previously saved homography matrix
homography_matrix_path = "anuj_homography_matrix_pears.npy"
homography_anuj = np.load(homography_matrix_path)


def warp_image(image, homography_anuj):
    """
    Apply the homography matrix to warp the input image.
    """
    warped_image = cv2.warpPerspective(
        image, homography_anuj, (image.shape[1], image.shape[0])
    )
    return warped_image


def load_images_from_folder(folder_path):
    """
    Load all images from the specified folder and return them.
    """
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, 0)  # Load as grayscale
        if img is not None:
            images.append(img)
        else:
            print(f"Error loading image: {img_path}")
    return images


def main():
    INPUT_PATH = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset"
    angled_images_path = os.path.join(INPUT_PATH, "AngledImages")
    straight_images_path = os.path.join(INPUT_PATH, "StraightImages")

    # Load all angled and straight images
    angled_images = load_images_from_folder(angled_images_path)
    straight_images = load_images_from_folder(straight_images_path)

    if not angled_images or not straight_images:
        print("Error: No images found in the specified folders.")
        return None, None, None

    # Warp the first angled image using the homography matrix
    warped_image = warp_image(angled_images[0], homography_anuj)

    return angled_images[0], straight_images[0], warped_image


if __name__ == "__main__":
    # Call the main function
    new_image_angled, new_image_straight, warped_image = main()

    if (
        new_image_angled is not None
        and new_image_straight is not None
        and warped_image is not None
    ):
        # Define the output directory
        output_directory = (
            r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\warpedImages"
        )

        # File name for the saved figure
        output_file = os.path.join(output_directory, "warped_image_sweet_new.png")

        # Generate and display the images
        plt.figure(figsize=(10, 5))
        plt.subplot(131), plt.imshow(new_image_angled, cmap="gray"), plt.title(
            "Angled Image"
        ), plt.axis("off")
        plt.subplot(132), plt.imshow(new_image_straight, cmap="gray"), plt.title(
            "Straight Image"
        ), plt.axis("off")
        plt.subplot(133), plt.imshow(warped_image, cmap="gray"), plt.title(
            "Warped Image"
        ), plt.axis("off")

        # Save the figure
        plt.savefig(output_file, bbox_inches="tight")
        plt.show()

        print(f"Image saved to {output_file}")
    else:
        print("Error: One or more images were not loaded correctly.")
