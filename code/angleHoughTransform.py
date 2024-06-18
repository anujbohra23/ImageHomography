import os
import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path, output_directory):
    """
    Process a single image, apply Hough transform, detect angles between lines, and save the output.
    """
    src = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Error opening image {image_path}!")
        return

    dst = cv.Canny(src, 50, 200, None, 3)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    angles = []

    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

            # Calculate the angle in degrees
            angle = np.degrees(theta)
            angles.append(angle)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    # Display angles on the image
    if angles:
        angles = np.sort(angles)
        for i in range(len(angles)):
            cv.putText(
                cdst,
                f"{angles[i]:.2f}",
                (50, 50 + i * 30),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    # Using matplotlib to display images
    plt.figure(figsize=(15, 10))
    plt.subplot(131), plt.imshow(src, cmap="gray"), plt.title("Source Image"), plt.axis(
        "off"
    )
    plt.subplot(132), plt.imshow(cdst), plt.title(
        "Detected Lines (Standard Hough)"
    ), plt.axis("off")
    plt.subplot(133), plt.imshow(cdstP), plt.title(
        "Detected Lines (Probabilistic Hough)"
    ), plt.axis("off")

    # Save the figure
    base_name = os.path.basename(image_path)
    output_file = os.path.join(
        output_directory, f"{os.path.splitext(base_name)[0]}_hough.png"
    )
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()


def main(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
            image_path = os.path.join(input_directory, filename)
            process_image(image_path, output_directory)


if __name__ == "__main__":
    input_directory = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\Dataset\IIT Patna Dataset\AngledImages"
    )
    output_directory = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\HoughImages"
    main(input_directory, output_directory)
