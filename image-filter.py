"""
image_processing.py
-------------------
Demonstrates basic image processing using NumPy array operations.
Transformations applied:
  - Grayscale conversion (ITU-R BT.709 luminance weights)
  - Brightness increase (factor = 1.5)
  - Brightness decrease (factor = 0.5)
"""

import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random 
import skimage as sk
from skimage import io

image = plt.imread('/Users/admin/Desktop/Programming/Numpy-Pandas/Numpy/Exercies /4677b0ec8305d2d62feb1c1baf714ff6.jpg')

# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

def to_Grayscale(image):
    """
    Convert an RGB image to grayscale using ITU-R BT.709 luminance weights.
 
    The weighted sum reflects human eye sensitivity across channels:
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
 
    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3) and dtype uint8.
 
    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W) and dtype uint8.
    """
    # BT.709 standard luminance weights for R, G, B
    weights = np.array([0.2126, 0.7152, 0.0722])
    
    # Dot product collapses the 3 channels into a single luminance value per pixel
    gray = np.dot(image , weights)
    
    return gray.astype(np.uint8)
to_Grayscale_image = to_Grayscale(image)

def brightness_control(image, factor = 1.5):
    """
    Scale pixel brightness by a given factor.
 
    Parameters
    ----------
    image : np.ndarray
        Image array of shape (H, W, 3) and dtype uint8.
    factor : float
        Multiplier for pixel values.
        > 1.0 brightens, < 1.0 darkens, 1.0 leaves unchanged.
 
    Returns
    -------
    np.ndarray
        Brightness-adjusted image of the same shape and dtype uint8.
    """
    # Cast to float64 to prevent uint8 overflow during multiplication
    # (e.g. 200 * 1.5 = 44 in uint8 due to silent wrap-around)
    bright = image.astype(np.float64) * factor

    # Clamp values to [0, 255] before converting back to uint8
    bright = np.clip(bright, 0,255)
  
    return bright.astype(np.uint8)

fifty_percent_brightness = brightness_control(image)
fifty_percent_dark = brightness_control(image, factor=0.5)

 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Plot all four versions side by side
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
# cmap='gray' required for 2D arrays — without it Matplotlib applies a false-color 
axes[0].imshow(image);axes[0].set_title("Original")
axes[1].imshow(to_Grayscale_image,   cmap='gray'); axes[1].set_title("Grayscale")
axes[2].imshow(fifty_percent_brightness); axes[2].set_title("Brightness x1.5")
axes[3].imshow(fifty_percent_dark); axes[3].set_title("Brightness x0.5")
for ax in axes: ax.axis('off')
plt.tight_layout()
plt.show()

