import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the previously saved homography matrix
homography_matrix_path = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\code\HomographyMatrix\anuj_homography_matrix_left.npy"
homography_anuj = np.load(homography_matrix_path)


def warp_image(image, homography_anuj):
    """
    Apply the homography matrix to warp the input image.
    """
    warped_image = cv2.warpPerspective(
        image, homography_anuj, (image.shape[1], image.shape[0])
    )
    return warped_image


def calculate_inliers(image1, image2, homography):
    """
    Calculate the inlier matches between two images using feature matching and a given homography matrix.
    Returns inlier matches.
    """
    orb = cv2.ORB_create()

    # Find keypoints and descriptors for both images
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # Use Brute Force matcher to find matches between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Apply perspective transform to keypoints of image2 using the homography matrix
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2_transformed = cv2.perspectiveTransform(pts2, homography)

    # Calculate distance between keypoints after transformation
    dist = np.sqrt(np.sum((pts1 - pts2_transformed) ** 2, axis=2))

    # Find inliers (considering a threshold of 10 pixels)
    inliers = [m for i, m in enumerate(matches) if dist[i] < 10]

    return inliers, kp1, kp2


def process_image(angled_image_path, output_directory):
    """
    Process an angled image:
    - Apply homography transformation.
    - Save the original angled and warped images with inlier matches in the output_directory.
    """
    # Load the angled image
    new_image_angled = cv2.imread(angled_image_path, cv2.IMREAD_GRAYSCALE)

    if new_image_angled is None:
        print(f"Error: Unable to load image {angled_image_path}")
        return

    # Warp the angled image using the homography matrix
    warped_image = warp_image(new_image_angled, homography_anuj)

    # Calculate inlier matches between original and warped image
    inliers, kp1, kp2 = calculate_inliers(
        new_image_angled, warped_image, homography_anuj
    )
    print(f"Number of inlier matches: {len(inliers)}")

    # Draw inlier matches
    matched_image = cv2.drawMatches(
        new_image_angled,
        kp1,
        warped_image,
        kp2,
        inliers,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Save the figure with inlier matches
    plt.figure(figsize=(15, 7))

    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title("Inlier Matches")
    plt.axis("off")

    # Save the figure
    base_name = os.path.splitext(os.path.basename(angled_image_path))[0]
    output_file = os.path.join(output_directory, f"{base_name}_inlier_matches.png")
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"Inlier matches saved to {output_file}")


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
    output_directory = r"C:\Users\Anuj Bohra\Desktop\IIT_Patna\anujwarp\InlierImages"

    main(input_directory, output_directory)
