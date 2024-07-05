# workds fine on sklearn dataset
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as sk_image
import numpy as np

# Sample Image
img = load_sample_image("flower.jpg")[:15, :20, 0]
plt.imshow(img)

# Size of each patch 10x10
patch_size = (10, 10)
patches = sk_image.extract_patches_2d(img, patch_size=patch_size)

# Plotting the patches
fig, axs = plt.subplots(6, 11, figsize=(11, 6), tight_layout=True)
for i, ax in enumerate(axs.reshape(-1)):
    ax.imshow(patches[i], vmin=img.min(), vmax=img.max())
    ax.axis("off")

# Recreate the meshgrid for the patch positions
im_H, im_W = img.shape[:2]
p_H, p_W = patch_size

y = np.arange(0, im_H - (p_H - 1))
x = np.arange(0, im_W - (p_W - 1))

X, Y = np.meshgrid(x, y)
positions = np.dstack((X, Y)).reshape(-1, 2)
plt.imshow(img)
plt.scatter(*positions.T, s=10, c="red")
plt.show()
