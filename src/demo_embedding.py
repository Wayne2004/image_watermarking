"""
===================================================================
demo_embedding.py  —  BMDS2133 Image Processing Group Assignment
===================================================================
Self-contained demonstration of Module 1 + Module 2.

Usage
─────
    cd image_watermarking
    python demo_embedding.py

What it does
────────────
1.  Loads mushroom.png (USC-SIPI-style host image).
2.  Generates a synthetic binary watermark if none is found.
3.  Runs the full DWT-DCT embedding pipeline.
4.  Saves:
        assets/watermarked_images/watermarked_mushroom.png
        assets/watermarks/generated_watermark.png          (if auto-generated)
        results/extracted_watermarks/extracted_watermark.png
5.  Prints a metrics report (PSNR, embedding stats).
"""

import sys
import os
from pathlib import Path

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.dirname(__file__))

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import cv2
import numpy as np

from watermarking import (
    run_embedding_pipeline,
    load_image,
    preprocess_image,
    apply_dwt,
    extract_watermark,
    calculate_psnr,
    ALPHA,
    BLOCK_SIZE,
)


# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
INPUT_DIR         = PROJECT_ROOT / "assets" / "input_images"
WATERMARKS_DIR    = PROJECT_ROOT / "assets" / "watermarks"
WATERMARKED_DIR   = PROJECT_ROOT / "assets" / "watermarked_images"
RESULTS_DIR       = PROJECT_ROOT / "results" / "extracted_watermarks"

HOST_IMAGE        = INPUT_DIR      / "mushroom.png"
WATERMARK_IMAGE   = WATERMARKS_DIR / "generated_watermark.png"
WATERMARKED_OUT   = WATERMARKED_DIR / "watermarked_mushroom.png"
EXTRACTED_OUT     = RESULTS_DIR    / "extracted_watermark.png"


def generate_synthetic_watermark(capacity: int, save_path: Path) -> np.ndarray:
    """
    Create a simple binary watermark pattern (checkerboard + text area)
    and save it so the team can replace it with their own logo later.
    """
    # Use roughly 25 % of capacity
    n_bits   = capacity // 4
    side     = int(np.sqrt(n_bits))
    wm       = np.zeros((side, side), dtype=np.uint8)

    # Checkerboard pattern
    for r in range(side):
        for c in range(side):
            wm[r, c] = 255 if (r + c) % 2 == 0 else 0

    # Draw a simple border frame
    wm[0, :]  = 255
    wm[-1, :] = 255
    wm[:, 0]  = 255
    wm[:, -1] = 255

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), wm)
    print(f"[demo] Synthetic watermark generated → {save_path}  ({side}×{side} px)")
    return wm


def print_report(original, watermarked, extracted_wm, wm_bits, capacity, watermark_shape):
    """Pretty-print a metrics summary."""
    separator = "─" * 58

    print()
    print("╔" + "═" * 56 + "╗")
    print("║   EMBEDDING REPORT — Module 1 + Module 2              ║")
    print("╠" + "═" * 56 + "╣")

    print(f"║  Host image shape     : {str(original.shape):<30}║")
    print(f"║  Watermarked shape    : {str(watermarked.shape):<30}║")
    print(separator)

    print(f"║  Embedding capacity   : {capacity:<30}║")
    print(f"║  Bits embedded        : {len(wm_bits):<30}║")
    print(f"║  Utilisation          : {len(wm_bits)/capacity*100:5.1f} %                          ║")
    print(f"║  Watermark grid       : {str(watermark_shape):<30}║")
    print(f"║  Alpha (strength)     : {ALPHA:<30.4f}║")
    print(f"║  Block size           : {BLOCK_SIZE}×{BLOCK_SIZE}                              ║")
    print(f"║  Wavelet              : Haar                          ║")
    print(f"║  Embedding sub-band   : LH (horizontal detail)       ║")

    print("╚" + "═" * 56 + "╝")
    print()


def main():
    print("\n" + "=" * 60)
    print("  DWT-DCT Watermarking Demo  (Module 1 + Module 2)")
    print("=" * 60)

    # ── Determine capacity before choosing watermark ───────────────
    original = load_image(str(HOST_IMAGE))
    Y_norm, Cb, Cr, _ = preprocess_image(original)
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    h_b, w_b = LH.shape
    capacity = (h_b // BLOCK_SIZE) * (w_b // BLOCK_SIZE)

    # ── Select or generate watermark ──────────────────────────────
    if WATERMARK_IMAGE.exists():
        wm_input = str(WATERMARK_IMAGE)
        print(f"[demo] Using existing watermark: {WATERMARK_IMAGE.name}")
    else:
        wm_img   = generate_synthetic_watermark(capacity, WATERMARK_IMAGE)
        wm_input = wm_img    # pass array directly

    # ── Run embedding pipeline ─────────────────────────────────────
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)
    result = run_embedding_pipeline(
        image_path     = str(HOST_IMAGE),
        watermark_input= wm_input,
        output_path    = str(WATERMARKED_OUT),
        alpha          = ALPHA,
    )

    watermarked_image = result["watermarked_image"]
    watermark_shape   = result["watermark_shape"]
    n_bits            = result["n_bits"]

    # ── Extract watermark to verify ───────────────────────────────
    print("[demo] Running extraction (blind — no original needed)…")
    extracted_wm = extract_watermark(watermarked_image, n_bits, watermark_shape, ALPHA)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(EXTRACTED_OUT), extracted_wm)
    print(f"[demo] Extracted watermark saved → {EXTRACTED_OUT}")

    # ── Print summary report ───────────────────────────────────────
    # Reload to get aligned version
    original_final = load_image(str(HOST_IMAGE))
    h_wm = watermarked_image.shape[0]
    w_wm = watermarked_image.shape[1]
    original_final = original_final[:h_wm, :w_wm]

    # Dummy bits array for report (just need length)
    dummy_bits = np.zeros(n_bits, dtype=np.uint8)

    print_report(
        original_final,
        watermarked_image,
        extracted_wm,
        dummy_bits,
        capacity,
        watermark_shape,
    )

    print(f"  Watermarked image → {WATERMARKED_OUT}")
    print(f"  Extracted watermark → {EXTRACTED_OUT}\n")


if __name__ == "__main__":
    main()