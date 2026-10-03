
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


def apply_morphology(mask: np.ndarray, op: str = 'opening', kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if op == 'opening':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif op == 'closing':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif op == 'erosion':
        return cv2.erode(mask, kernel, iterations=1)
    elif op == 'dilation':
        return cv2.dilate(mask, kernel, iterations=1)
    else:
        raise ValueError(f"Unknown morphology operation '{op}'.")


def binarize_otsu(img: np.ndarray) -> np.ndarray:
    """Binarize grayscale image using Otsu's thresholding."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0.0, 1.0] float32."""
    img_f = img.astype(np.float32)
    min_val, max_val = img_f.min(), img_f.max()
    if max_val - min_val > 1e-6:
        return (img_f - min_val) / (max_val - min_val)
    return img_f


if __name__ == '__main__':
    dummy_img = np.random.randint(0, 256, (80, 112), dtype=np.uint8)
    enhanced = enhance_contrast_clahe(dummy_img)
    denoised = denoise_image(dummy_img, method='gaussian')
    binary = binarize_otsu(dummy_img)
    cleaned = apply_morphology(binary, op='opening')
    norm = normalize_image(dummy_img)