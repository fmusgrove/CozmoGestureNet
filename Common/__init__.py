import numpy as np


def np_color_to_gray(image: np.ndarray):
    gray_image = image.dot([0.07, 0.72, 0.21])
    gray_image = np.min(gray_image, 255).astype(np.uint8)
    return gray_image
