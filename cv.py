import numpy as np
import cv2


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def denoise(img):
    show_img = cv2.fastNlMeansDenoisingColored(img, templateWindowSize=5, searchWindowSize=21, h=8, hColor=10)
    show_img = unsharp_mask(show_img, kernel_size=(5, 5), sigma=1.5, amount=1.2, threshold=0)

    return show_img
