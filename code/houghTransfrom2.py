import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main(argv):

    default_file = (
        r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\input\Angled\A_Coffee.jpg"
    )
    filename = argv[0] if len(argv) > 0 else default_file

    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    if src is None:
        print("Error opening image!")
        print("Usage: hough_lines.py [image_name -- default " + default_file + "] \n")
        return -1

    dst = cv.Canny(src, 50, 200, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

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
    plt.show()

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
