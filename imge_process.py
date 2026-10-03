
import numpy as np
import cv2


def enhance_contrast_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Basic image processing pipeline: Contrast enhancement (CLAHE) + Denoising."""
    enhanced = enhance_contrast_clahe(img)
    denoised = denoise_image(enhanced, method='gaussian')
    return denoised


def denoise_image(img: np.ndarray, method: str = 'gaussian', ksize: int = 5) -> np.ndarray:
    if method == 'gaussian':
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    elif method == 'median':
        return cv2.medianBlur(img, ksize)
    elif method == 'bilateral':
        return cv2.bilateralFilter(img, d=ksize, sigmaColor=75, sigmaSpace=75)
    else:
        raise ValueError(f"Unknown denoise method '{method}'. Choose 'gaussian', 'median', or 'bilateral'.")



if __name__ == '__main__':
    dummy_img = np.random.randint(0, 256, (80, 112), dtype=np.uint8)
    enhanced = enhance_contrast_clahe(dummy_img)
    denoised = denoise_image(dummy_img, method='gaussian')