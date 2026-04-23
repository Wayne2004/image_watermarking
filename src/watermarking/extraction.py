"""
===================================================================
BMDS2133 Image Processing — Group Assignment
Module 3 (Role C): Watermark Extraction
===================================================================
Algorithm : Hybrid DWT-DCT Differential Blind Watermark Extraction
"""

import cv2
import numpy as np
import math
from pathlib import Path

from .watermarking import (
    preprocess_image,
    apply_dwt,
    apply_dct_blocks,
    _extract_soft_from_band,
    BLOCK_SIZE,
    ALPHA,
)
from .arnold import (
    inverse_arnold_transform,
    arnold_descramble_bits,
    arnold_period,
)


def extract_watermark_raw(
    watermarked_image: np.ndarray,
    n_bits: int,
    watermark_shape: tuple,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Core extraction using Differential Consensus logic.
    """
    # Preprocess
    Y_norm, _, _, _ = preprocess_image(watermarked_image)

    # DWT
    _, _, LH, HL, _ = apply_dwt(Y_norm)

    # DCT on sub-bands
    dct_LH = apply_dct_blocks(LH)
    dct_HL = apply_dct_blocks(HL)

    # 1. Extract vote scores (-N to +N)
    s_lh = _extract_soft_from_band(dct_LH, n_bits, alpha)
    s_hl = _extract_soft_from_band(dct_HL, n_bits, alpha)
    
    # 2. Consensus: Sum of votes from both sub-bands
    # If the sum is positive, the majority of blocks across both bands said '1'
    voted_bits = ((s_lh + s_hl) >= 0).astype(np.uint8)

    # 3. Reshape back to logo
    target_size = watermark_shape[0] * watermark_shape[1]
    if len(voted_bits) < target_size:
        voted_bits = np.pad(voted_bits, (0, target_size - len(voted_bits)))

    wm_2d = voted_bits[:target_size].reshape(watermark_shape)
    wm_img = (wm_2d * 255).astype(np.uint8)
    return wm_img


def descramble_watermark(
    scrambled_watermark: np.ndarray,
    iterations: int = 5,
    grid_shape: tuple = None,
    original_length: int = None,
    padding: int = 0,
    original_shape: tuple = None,
) -> np.ndarray:
    if grid_shape is not None:
        bits = scrambled_watermark.flatten()
        bits = (bits > 128).astype(np.uint8)
        descrambled = arnold_descramble_bits(
            bits, grid_shape, iterations, original_length, padding
        )
        
        # Use provided shape or infer it
        if original_shape is not None:
            rows, cols = original_shape
            n = rows * cols
        elif original_length is not None:
            n = original_length
            rows = int(math.ceil(math.sqrt(n)))
            cols = int(math.ceil(n / rows))
        else:
            n = len(descrambled)
            rows = int(math.ceil(math.sqrt(n)))
            cols = int(math.ceil(n / rows))
            
        return (descrambled[:n].reshape((rows, cols)) * 255).astype(np.uint8)
    return scrambled_watermark


def extract_watermark_robust(
    watermarked_image: np.ndarray,
    n_bits: int,
    watermark_shape: tuple,
    alpha: float = ALPHA,
    arnold_iterations: int = 0,
    grid_shape: tuple = None,
    original_length: int = None,
    padding: int = 0,
    original_shape: tuple = None,
) -> np.ndarray:
    h, w = watermarked_image.shape[:2]
    align = BLOCK_SIZE * 2
    if h % align != 0 or w % align != 0:
        new_h, new_w = (h // align) * align, (w // align) * align
        watermarked_image = cv2.resize(watermarked_image, (new_w, new_h))

    extracted = extract_watermark_raw(watermarked_image, n_bits, watermark_shape, alpha)

    if arnold_iterations > 0:
        extracted = descramble_watermark(
            extracted, arnold_iterations, grid_shape, original_length, padding, original_shape
        )
    return extracted


def run_extraction_pipeline(
    image_path: str,
    n_bits: int,
    watermark_shape: tuple,
    alpha: float = ALPHA,
    arnold_iterations: int = 0,
    grid_shape: tuple = None,
    original_length: int = None,
    padding: int = 0,
    output_path: str = None,
    original_shape: tuple = None,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None: raise FileNotFoundError(f"Cannot read: {image_path}")

    extracted = extract_watermark_robust(
        image, n_bits, watermark_shape, alpha,
        arnold_iterations, grid_shape, original_length, padding, original_shape
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), extracted)

    return {
        "extracted_watermark": extracted,
        "n_bits_extracted": n_bits,
        "watermark_shape": watermark_shape,
        "output_path": str(output_path) if output_path else None,
        "arnold_descrambled": arnold_iterations > 0,
    }

def extract_watermark_batch(image_paths, n_bits, shape, **kwargs):
    return {Path(p).stem: extract_watermark_robust(cv2.imread(p), n_bits, shape, **kwargs) for p in image_paths}
