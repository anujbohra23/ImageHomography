import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


def apply_gaussian_blur(image, kernel_size=5, sigma=1):
    """
    Apply Gaussian blur to the input image.
    """
    blurred_image = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image


def apply_sobel_gradient(image):
    """
    Apply Sobel gradient to the input image to detect edges.
    """
    grad_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv.magnitude(grad_x, grad_y)
    grad_magnitude = np.uint8(grad_magnitude)
    return grad_magnitude


def detect_lines(image):
    """
    Detect lines in the input image using Hough Line Transform.
    """
    lines = cv.HoughLines(image, 1, np.pi / 180, 150)
    return lines


def calculate_angles(lines):
    """
    Calculate angles of the detected lines.
    """
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.rad2deg(theta)  # Convert angle from radians to degrees
            angles.append(angle)
    return angles


def process_image(image_path, output_path):
    """
    Process an image by applying Gaussian blur, Sobel gradient, and Hough Line Transform,
    and then calculate and display the angles of the detected lines.
    """
    # Load the image
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(image)

    # Apply Sobel gradient
    gradient_image = apply_sobel_gradient(blurred_image)

    # Detect lines using Hough Line Transform
    lines = detect_lines(gradient_image)

    # Calculate angles of the detected lines
    angles = calculate_angles(lines)

    # Display the images and angles
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(image, cmap="gray"), plt.title(
        "Original Image"
    ), plt.axis("off")
    plt.subplot(132), plt.imshow(blurred_image, cmap="gray"), plt.title(
        "Gaussian Blurred"
    ), plt.axis("off")
    plt.subplot(133), plt.imshow(gradient_image, cmap="gray"), plt.title(
        "Sobel Gradient"
    ), plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    # Print the angles
    if angles:
        print(f"Detected angles (in degrees): {angles}")
    else:
        print("No lines were detected.")


if __name__ == "__main__":
    image_path = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\AngledImages\A_Coffee2.jpg"
    output_path = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\frames"
    process_image(image_path, output_path)
