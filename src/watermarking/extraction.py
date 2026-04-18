"""
===================================================================
BMDS2133 Image Processing — Group Assignment
Module 3 (Role C): Watermark Extraction
===================================================================
Algorithm : Hybrid DWT-DCT Blind Watermark Extraction (Enhanced)

This module implements the inverse process of the embedding pipeline:
  - Blind extraction from attacked / unattacked watermarked images
  - Support for Arnold Transform descrambling
  - Robust handling of geometric attacks (cropping, scaling)
  - Batch extraction for robustness evaluation
  - End-to-end extraction pipeline

Extraction pipeline (blind — no original image needed)
───────────────────────────────────────────────────────
Watermarked image  →  Y  →  DWT  →  LH, HL  →  DCT blocks
  →  read sign of coeff (3,4) in each  →  majority vote  →  bit
  →  inverse Arnold descramble (if scrambling was used)  →  output

Key features
────────────
  • Blind extraction: original host image is NOT required
  • Dual sub-band majority voting (LH + HL)
  • Adaptive alpha per-block reconstruction
  • Arnold Transform descrambling with configurable key
  • Automatic size recovery after cropping / scaling attacks
  • Batch processing for multiple attacked images
"""

import cv2
import numpy as np
import math
from pathlib import Path

from .watermarking import (
    preprocess_image,
    apply_dwt,
    apply_dct_blocks,
    _extract_bits_from_band,
    BLOCK_SIZE,
    ALPHA,
)
from .arnold import (
    inverse_arnold_transform,
    arnold_descramble_bits,
    arnold_period,
)


# ─────────────────────────────────────────────────────────────────
# CORE EXTRACTION (low-level, mirrors embed_watermark)
# ─────────────────────────────────────────────────────────────────

def extract_watermark_raw(
    watermarked_image: np.ndarray,
    n_bits: int,
    watermark_shape: tuple,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Extract the embedded watermark from a watermarked image.

    This is the core blind extraction function that mirrors the
    embedding process.  It extracts bits from both LH and HL sub-bands
    and uses majority voting to decide each bit.

    Parameters
    ----------
    watermarked_image : BGR uint8 numpy array
    n_bits            : int  number of bits to extract
    watermark_shape   : tuple (rows, cols) for reshaping output
    alpha             : float  base embedding strength (must match embedding)

    Returns
    -------
    np.ndarray  2-D uint8 (0 or 255) of shape watermark_shape
    """
    # Preprocess: BGR → YCbCr → Y normalised
    Y_norm, _, _, _ = preprocess_image(watermarked_image)

    # DWT decomposition
    _, LL, LH, HL, HH = apply_dwt(Y_norm)

    # DCT on both sub-bands
    dct_LH = apply_dct_blocks(LH)
    dct_HL = apply_dct_blocks(HL)

    # Extract bits from both sub-bands
    bits_LH = _extract_bits_from_band(dct_LH, n_bits, alpha)
    bits_HL = _extract_bits_from_band(dct_HL, n_bits, alpha)

    # Innovation II: majority vote between LH and HL
    # Agree → use agreed value.  Disagree → fall back to LH (tie-break).
    min_len = min(len(bits_LH), len(bits_HL))
    voted_bits = np.where(
        bits_LH[:min_len] == bits_HL[:min_len],
        bits_LH[:min_len],
        bits_LH[:min_len]   # tie-break: LH wins
    )

    # Pad if short, reshape to 2-D
    target_size = watermark_shape[0] * watermark_shape[1]
    if len(voted_bits) < target_size:
        voted_bits = np.pad(voted_bits, (0, target_size - len(voted_bits)))

    wm_2d = voted_bits[:target_size].reshape(watermark_shape)
    wm_img = (wm_2d * 255).astype(np.uint8)

    print(f"[extract_watermark_raw] Extracted {min_len} bits  "
          f"shape={watermark_shape}  (LH + HL majority vote)")
    return wm_img


# ─────────────────────────────────────────────────────────────────
# ARNOLD TRANSFORM DESC RAMBLING
# ─────────────────────────────────────────────────────────────────

def descramble_watermark(
    scrambled_watermark: np.ndarray,
    iterations: int = 5,
    grid_shape: tuple = None,
    original_length: int = None,
    padding: int = 0,
    original_shape: tuple = None,
) -> np.ndarray:
    """
    Apply Inverse Arnold Transform to descramble an extracted watermark.

    Parameters
    ----------
    scrambled_watermark : 2-D uint8 (image) or 1-D uint8 (bit array)
    iterations          : int  Arnold iterations used during scrambling
    grid_shape          : tuple (side, side) or None
    original_length     : int  original bit count (for trimming padding)
    padding             : int  number of padding bits to remove
    original_shape      : tuple (rows, cols) of the target rectangle

    Returns
    -------
    np.ndarray  descrambled watermark
    """
    if grid_shape is not None:
        # 1-D bit array mode
        bits = scrambled_watermark.flatten()
        descrambled = arnold_descramble_bits(
            bits, grid_shape, iterations, original_length, padding
        )
        
        # Determine the final shape
        if original_shape is not None:
            # Use specific rectangular shape (Role C request)
            rows, cols = original_shape
            n = rows * cols
        elif original_length is not None:
            # Infer square-ish shape from length
            n = original_length
            rows = int(math.ceil(math.sqrt(n)))
            cols = int(math.ceil(n / rows))
        else:
            # Use the full descrambled length
            n = len(descrambled)
            rows = int(math.ceil(math.sqrt(n)))
            cols = int(math.ceil(n / rows))
            
        return (descrambled[:n].reshape((rows, cols)) * 255).astype(np.uint8)
    else:
        # 2-D image mode (must be square)
        if scrambled_watermark.shape[0] != scrambled_watermark.shape[1]:
            raise ValueError(
                f"Direct 2-D descrambling requires square image, "
                f"got {scrambled_watermark.shape}"
            )
        descrambled_grid = inverse_arnold_transform(
            scrambled_watermark, iterations=iterations
        )
        print(f"[descramble_watermark] Descrambled with {iterations} iterations  "
              f"shape={descrambled_grid.shape}")
        return descrambled_grid


# ─────────────────────────────────────────────────────────────────
# ROBUST EXTRACTION (handles attacked images)
# ─────────────────────────────────────────────────────────────────

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
    """
    Extract watermark from a watermarked (possibly attacked) image.

    This is the robust version that handles:
      - Geometric attack recovery (cropping, scaling)
      - Optional Arnold Transform descrambling

    For attacked images that have different dimensions than the
    original embedding dimensions, this function automatically
    resizes the image to the nearest aligned size before extraction.

    Parameters
    ----------
    watermarked_image : BGR uint8 numpy array
    n_bits            : int  number of embedded watermark bits
    watermark_shape   : tuple (rows, cols) original watermark grid shape
    alpha             : float  base embedding strength
    arnold_iterations : int  Arnold scramble iterations (0 = no scrambling)
    grid_shape        : tuple (side, side) for Arnold descrambling
    original_length   : int  original bit count (for Arnold padding trim)
    padding           : int  Arnold padding count
    original_shape    : tuple (rows, cols) of the target rectangle

    Returns
    -------
    np.ndarray  2-D uint8 extracted (and optionally descrambled) watermark
    """
    # ── Size recovery for geometric attacks ──────────────────────
    h, w = watermarked_image.shape[:2]
    align = BLOCK_SIZE * 2  # 16

    # Check if dimensions need alignment recovery
    target_h = watermark_shape[0] * BLOCK_SIZE * 2  # approximate
    target_w = watermark_shape[1] * BLOCK_SIZE * 2

    # If image was cropped (smaller than expected), resize up
    # to the original embedding dimensions for correct DWT alignment
    if h % align != 0 or w % align != 0:
        new_h = (h // align) * align
        new_w = (w // align) * align
        if new_h == 0:
            new_h = align
        if new_w == 0:
            new_w = align
        if new_h != h or new_w != w:
            print(f"[extract_watermark_robust] Aligning image from "
                  f"({h},{w}) → ({new_h},{new_w})")
            watermarked_image = cv2.resize(
                watermarked_image, (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )

    # ── Core extraction ──────────────────────────────────────────
    extracted = extract_watermark_raw(
        watermarked_image, n_bits, watermark_shape, alpha
    )

    # ── Arnold descrambling (if used during embedding) ───────────
    if arnold_iterations > 0:
        extracted = descramble_watermark(
            extracted, arnold_iterations, grid_shape, original_length, padding, original_shape
        )

    return extracted


# ─────────────────────────────────────────────────────────────────
# BATCH EXTRACTION
# ─────────────────────────────────────────────────────────────────

def extract_watermark_batch(
    image_paths: list,
    n_bits: int,
    watermark_shape: tuple,
    alpha: float = ALPHA,
    arnold_iterations: int = 0,
    grid_shape: tuple = None,
    original_length: int = None,
    padding: int = 0,
    output_dir: str = None,
) -> dict:
    """
    Extract watermarks from multiple images (e.g., after various attacks).

    Useful for batch robustness evaluation across different attack types.

    Parameters
    ----------
    image_paths       : list of str  paths to attacked/watermarked images
    n_bits            : int  number of embedded watermark bits
    watermark_shape   : tuple (rows, cols) original watermark grid shape
    alpha             : float  base embedding strength
    arnold_iterations : int  Arnold scramble iterations (0 = none)
    grid_shape        : tuple (side, side) for Arnold descrambling
    original_length   : int  original bit count
    padding           : int  Arnold padding count
    output_dir        : str  directory to save extracted watermarks (or None)

    Returns
    -------
    dict  {image_name: extracted_watermark_array}
    """
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for img_path in image_paths:
        img_path = str(img_path)
        name = Path(img_path).stem
        print(f"\n[extract_watermark_batch] Processing: {name}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"[extract_watermark_batch] SKIPPED (cannot read): {img_path}")
            continue

        extracted = extract_watermark_robust(
            image, n_bits, watermark_shape, alpha,
            arnold_iterations, grid_shape, original_length, padding
        )

        results[name] = extracted

        if output_dir is not None:
            out_path = Path(output_dir) / f"extracted_{name}.png"
            cv2.imwrite(str(out_path), extracted)
            print(f"[extract_watermark_batch] Saved → {out_path}")

    print(f"\n[extract_watermark_batch] Processed {len(results)} images")
    return results


# ─────────────────────────────────────────────────────────────────
# END-TO-END EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────

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
    """
    End-to-end watermark extraction pipeline (Module 3).

    Loads image → recovers alignment → extracts watermark →
    optionally descrambles → saves result.

    Parameters
    ----------
    image_path        : str  path to watermarked (possibly attacked) image
    n_bits            : int  number of embedded watermark bits
    watermark_shape   : tuple (rows, cols) original watermark grid shape
    alpha             : float  base embedding strength
    arnold_iterations : int  Arnold scramble iterations (0 = none)
    grid_shape        : tuple (side, side) for Arnold descrambling
    original_length   : int  original bit count
    padding           : int  Arnold padding count
    output_path       : str  where to save the extracted watermark
    original_shape    : tuple (rows, cols) of the target rectangle

    Returns
    -------
    dict with keys:
        extracted_watermark : 2-D uint8 numpy array
        n_bits_extracted    : int  actual bits extracted
        watermark_shape     : tuple
        output_path         : str  saved file path (or None)
        arnold_descrambled  : bool  whether Arnold descrambling was applied
    """
    print("\n" + "=" * 60)
    print("  DWT-DCT Watermark Extraction Pipeline  (Module 3)")
    print("=" * 60)

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    print(f"\n[Module 3] Loading image: {Path(image_path).name}  shape={image.shape}")

    # Extract
    extracted = extract_watermark_robust(
        image, n_bits, watermark_shape, alpha,
        arnold_iterations, grid_shape, original_length, padding, original_shape
    )

    # Save
    saved_path = None
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), extracted)
        saved_path = str(out)
        print(f"[Module 3] Extracted watermark saved → {saved_path}")

    print("\n" + "=" * 60)
    print("  Extraction Complete")
    print("=" * 60)
    print(f"  Bits extracted : {n_bits}")
    print(f"  Output shape   : {extracted.shape}")
    print(f"  Arnold         : {'Yes (iter={})'.format(arnold_iterations) if arnold_iterations > 0 else 'No'}")
    if saved_path:
        print(f"  Saved to       : {saved_path}")
    print("=" * 60 + "\n")

    return {
        "extracted_watermark": extracted,
        "n_bits_extracted": n_bits,
        "watermark_shape": watermark_shape,
        "output_path": saved_path,
        "arnold_descrambled": arnold_iterations > 0,
    }
