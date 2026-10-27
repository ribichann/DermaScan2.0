"""OCR pipeline for skincare ingredient labels."""
from __future__ import annotations

import re
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    if Path(TESSERACT_PATH).exists():
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
except (OSError, AttributeError):
    pass


def _open_pil(image_path: str) -> Image.Image:
    with Image.open(image_path) as source:
        return source.convert("RGB")


def _border_mean(gray_image) -> float:
    """Return mean border luminance from a grayscale NumPy image."""
    import numpy as np
    height, width = gray_image.shape[:2]
    thickness = max(1, min(12, height // 30, width // 30))
    border = np.concatenate([
        gray_image[:thickness, :].ravel(),
        gray_image[-thickness:, :].ravel(),
        gray_image[:, :thickness].ravel(),
        gray_image[:, -thickness:].ravel(),
    ])
    return float(border.mean()) if border.size else 255.0


def _cv_preprocess(image_path: str):
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        # cv2.imread may fail on some Windows paths / Unicode paths.
        pil = _open_pil(image_path)
        rgb = np.array(pil)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert when the border is dark, commonly indicating white text on a dark bottle.
    if _border_mean(gray) < 140:
        gray = cv2.bitwise_not(gray)

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    return Image.fromarray(adaptive), Image.fromarray(gray)


def _tidy(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(image_path: str) -> str:
    """Extract label text with OpenCV preprocessing and Tesseract fallback."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError("Uploaded image was not found.")

    preprocessed = None
    normalized = None
    try:
        preprocessed, normalized = _cv_preprocess(str(path))
    except Exception:
        # PIL path remains available when OpenCV is missing or image decoding fails.
        try:
            original = _open_pil(str(path))
            original = ImageOps.grayscale(original)
            original = ImageOps.autocontrast(original)
            original = ImageEnhance.Contrast(original).enhance(2.0)
            original = original.resize((original.width * 2, original.height * 2), Image.Resampling.BICUBIC)
            normalized = original.filter(ImageFilter.SHARPEN)
        except Exception as exc:
            raise ValueError("The image could not be opened. Upload a JPG, PNG, or WEBP image.") from exc

    configs = ["--oem 3 --psm 6", "--oem 3 --psm 3"]
    candidates = []
    for image in (preprocessed, normalized):
        if image is None:
            continue
        for config in configs:
            try:
                candidates.append(_tidy(pytesseract.image_to_string(image, config=config)))
            except Exception:
                continue

    # Last safe fallback: OCR the original with PIL without OpenCV processing.
    if not any(len(candidate) >= 12 for candidate in candidates):
        try:
            original = _open_pil(str(path))
            candidates.append(_tidy(pytesseract.image_to_string(original, config="--oem 3 --psm 3")))
        except Exception:
            pass

    best = max(candidates, key=lambda value: (len(re.sub(r"\W", "", value)), len(value)), default="")
    if not best:
        raise RuntimeError(
            "No text could be extracted. Ensure Tesseract OCR is installed and the ingredient label is clear."
        )
    return best
