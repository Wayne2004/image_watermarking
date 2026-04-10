"""
===================================================================
demo_extraction.py  —  BMDS2133 Image Processing Group Assignment
Self-contained demonstration of Module 3 (Watermark Extraction)
===================================================================

Usage
─────
    cd image_watermarking
    python demo_extraction.py

What it does
────────────
1.  Runs the embedding pipeline (Module 1 + 2) to create a watermarked image.
2.  Applies various attacks (JPEG compression, cropping, scaling, noise, blur).
3.  Extracts the watermark from each attacked image using Module 3.
4.  Demonstrates Arnold Transform scrambling/descrambling.
5.  Computes BER and NCC metrics for each extraction.
6.  Prints a comprehensive extraction report.

Dependencies
────────────
    Requires Module 1 + 2 (embedding) and attacks/evaluation modules.
    All are imported from the same src/ package.
"""

import sys
import os
from pathlib import Path

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

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
from watermarking.extraction import (
    extract_watermark_robust,
    run_extraction_pipeline,
    extract_watermark_batch,
    descramble_watermark,
)
from watermarking.arnold import (
    arnold_transform,
    inverse_arnold_transform,
    arnold_scramble_bits,
    arnold_descramble_bits,
    arnold_period,
)
from attacks import (
    jpeg_compression,
    gaussian_noise,
    blurring,
    cropping,
    scaling,
)
from evaluation import (
    calculate_ber,
    calculate_ssim,
)


# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
INPUT_DIR         = PROJECT_ROOT / "assets" / "input_images"
WATERMARKS_DIR    = PROJECT_ROOT / "assets" / "watermarks"
WATERMARKED_DIR   = PROJECT_ROOT / "assets" / "watermarked_images"
RESULTS_DIR       = PROJECT_ROOT / "results" / "demo_extraction"

HOST_IMAGE        = INPUT_DIR      / "mushroom.png"
WATERMARK_IMAGE   = WATERMARKS_DIR / "watermark.png"
WATERMARKED_OUT   = WATERMARKED_DIR / "watermarked_mushroom.png"
EXTRACT_DIR       = RESULTS_DIR    / "extracted"
ATTACK_DIR        = RESULTS_DIR    / "attacked"


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def calculate_ncc(original: np.ndarray, extracted: np.ndarray) -> float:
    """Normalized Cross-Correlation."""
    if original.shape != extracted.shape:
        extracted = cv2.resize(extracted, (original.shape[1], original.shape[0]))
    orig_f = original.astype(np.float64)
    extr_f = extracted.astype(np.float64)
    numerator   = np.sum(orig_f * extr_f)
    denominator = np.sqrt(np.sum(orig_f ** 2) * np.sum(extr_f ** 2))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


# ─────────────────────────────────────────────────────────────────
# Attack suite
# ─────────────────────────────────────────────────────────────────

def _scale_roundtrip(image, factor):
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * factor), int(h * factor)),
                       interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


ATTACKS = [
    ("clean",           lambda img: img.copy()),
    ("jpeg_q75",        lambda img: jpeg_compression(img, quality=75)),
    ("jpeg_q50",        lambda img: jpeg_compression(img, quality=50)),
    ("jpeg_q30",        lambda img: jpeg_compression(img, quality=30)),
    ("crop_10pct",      lambda img: cropping(img, percentage=0.10)),
    ("crop_25pct",      lambda img: cropping(img, percentage=0.25)),
    ("scale_075",       lambda img: _scale_roundtrip(img, 0.75)),
    ("scale_050",       lambda img: _scale_roundtrip(img, 0.50)),
    ("noise_sigma10",   lambda img: gaussian_noise(img, sigma=10)),
    ("noise_sigma20",   lambda img: gaussian_noise(img, sigma=20)),
    ("blur_k5",         lambda img: blurring(img, kernel_size=5)),
    ("blur_k9",         lambda img: blurring(img, kernel_size=9)),
]


# ─────────────────────────────────────────────────────────────────
# Arnold Transform Demo
# ─────────────────────────────────────────────────────────────────

def demo_arnold_transform():
    """Demonstrate Arnold Transform scrambling and descrambling."""
    print("\n" + "=" * 60)
    print("  DEMO 1: Arnold Transform Scrambling")
    print("=" * 60)

    # Create a small test pattern
    test = np.array([
        [255, 255, 0,   0,   0],
        [255, 255, 0,   0,   0],
        [0,   0,   255, 255, 255],
        [0,   0,   255, 255, 255],
        [0,   0,   255, 255, 255],
    ], dtype=np.uint8)

    N = test.shape[0]
    period = arnold_period(N)
    print(f"\n  Test image: {N}x{N}, period = {period}")

    # Scramble
    iterations = 3
    scrambled = arnold_transform(test, iterations=iterations)
    print(f"\n  Original ({iterations} iterations → Scrambled):")
    print(f"  Original:\n{test}")
    print(f"\n  Scrambled:\n{scrambled}")

    # Descramble
    descrambled = inverse_arnold_transform(scrambled, iterations=iterations)
    match = np.array_equal(test, descrambled)
    print(f"\n  Descrambled (matches original: {match}):")
    print(f"  {descrambled}")

    # Verify period
    full_cycle = arnold_transform(test, iterations=period)
    period_match = np.array_equal(test, full_cycle)
    print(f"\n  Period verification ({period} iterations → original: {period_match})")


# ─────────────────────────────────────────────────────────────────
# Main Extraction Demo
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  DWT-DCT Watermark Extraction Demo  (Module 3)")
    print("=" * 60)

    # ── Step 1: Embed watermark ──────────────────────────────────
    print("\n[Step 1] Embedding watermark into host image…")
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)

    if not HOST_IMAGE.exists():
        print(f"  ERROR: Host image not found: {HOST_IMAGE}")
        print(f"  Place an image in assets/input_images/")
        return

    if not WATERMARK_IMAGE.exists():
        print(f"  ERROR: Watermark image not found: {WATERMARK_IMAGE}")
        print(f"  Place an image in assets/watermarks/")
        return

    result = run_embedding_pipeline(
        image_path=str(HOST_IMAGE),
        watermark_input=str(WATERMARK_IMAGE),
        output_path=str(WATERMARKED_OUT),
        alpha=ALPHA,
    )

    watermarked_image = result["watermarked_image"]
    watermark_shape   = result["watermark_shape"]
    n_bits            = result["n_bits"]
    psnr_val          = result["psnr"]
    wm_bits           = result["watermark_bits"]

    # Create reference watermark from bits
    rows, cols = watermark_shape
    # n_bits may be < rows*cols; pad for reshape
    ref_bits = np.zeros(rows * cols, dtype=np.uint8)
    ref_bits[:n_bits] = wm_bits[:n_bits]
    ref_wm = ref_bits.reshape((rows, cols))
    ref_wm_img = (ref_wm * 255).astype(np.uint8)

    # ── Step 2: Arnold Transform Demo ────────────────────────────
    demo_arnold_transform()

    # ── Step 3: Apply attacks ────────────────────────────────────
    print("\n[Step 2] Applying attacks…")
    ATTACK_DIR.mkdir(parents=True, exist_ok=True)

    attacked_images = {}
    for name, attack_func in ATTACKS:
        attacked = attack_func(watermarked_image.copy())
        attacked_images[name] = attacked
        save_path = ATTACK_DIR / f"{name}.png"
        cv2.imwrite(str(save_path), attacked)
        print(f"  {name:<20} shape={attacked.shape}")

    # ── Step 4: Extract watermarks ───────────────────────────────
    print("\n[Step 3] Extracting watermarks from attacked images…")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    extracted_watermarks = {}
    metrics = []

    for name, attacked_img in attacked_images.items():
        print(f"\n  Extracting from: {name}")

        extracted = extract_watermark_robust(
            attacked_img, n_bits, watermark_shape, ALPHA
        )

        extracted_watermarks[name] = extracted
        save_path = EXTRACT_DIR / f"extracted_{name}.png"
        cv2.imwrite(str(save_path), extracted)

        # Compute metrics
        ber = calculate_ber(ref_wm_img, extracted)
        ncc = calculate_ncc(ref_wm_img, extracted)

        metrics.append({
            "attack": name,
            "ber": ber,
            "ncc": ncc,
        })

        ber_pct = f"{ber * 100:.2f}%"
        ncc_str = f"{ncc:.4f}"
        print(f"    BER = {ber_pct:<10}  NCC = {ncc_str}")

    # ── Step 5: Print comprehensive report ───────────────────────
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  MODULE 3: EXTRACTION REPORT".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  Host image         : {HOST_IMAGE.name:<45}║")
    print(f"║  Watermark          : {WATERMARK_IMAGE.name:<45}║")
    print(f"║  Embedding PSNR     : {psnr_val:6.2f} dB  {'✓ PASS (>38 dB)' if psnr_val >= 38 else '✗ FAIL'}" + " " * max(0, 27 - len('✓ PASS (>38 dB)')) + "║")
    print(f"║  Bits embedded      : {n_bits:<45}║")
    print(f"║  Watermark shape    : {str(watermark_shape):<45}║")
    print("╟" + "─" * 68 + "╢")
    print(f"║  {'Attack':<20} {'BER':>10} {'NCC':>10} {'Status':>14}  ║")
    print("╟" + "─" * 68 + "╢")

    for m in metrics:
        ber = m["ber"]
        ncc = m["ncc"]
        ber_str = f"{ber * 100:.2f}%"
        ncc_str = f"{ncc:.4f}"

        if ber < 0.05:
            status = "✓ Robust"
        elif ber < 0.10:
            status = "~ Moderate"
        else:
            status = "✗ Degraded"

        status_padded = status + " " * max(0, 14 - len(status))
        print(f"║  {m['attack']:<20} {ber_str:>10} {ncc_str:>10} {status_padded:>14}  ║")

    print("╚" + "═" * 68 + "╝")

    # Summary
    robust_count = sum(1 for m in metrics if m["ber"] < 0.05)
    total_count = len(metrics)
    print(f"\n  Robustness summary: {robust_count}/{total_count} attacks passed BER < 5%")

    # Saved locations
    print(f"\n  Watermarked image   → {WATERMARKED_OUT}")
    print(f"  Attacked images     → {ATTACK_DIR}")
    print(f"  Extracted watermarks→ {EXTRACT_DIR}")
    print()


if __name__ == "__main__":
    main()
