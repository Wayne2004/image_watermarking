"""
===================================================================
demo_psnr.py  —  BMDS2133 Image Processing Group Assignment
Invisibility / Imperceptibility Demo (PSNR Analysis)
===================================================================

Usage
─────
    cd image_watermarking
    python src/demo_psnr.py

What it does
────────────
1.  Embeds watermarks into ALL images in assets/input_images/.
2.  Computes PSNR between each original and its watermarked version.
3.  Tests multiple alpha strengths to show the imperceptibility vs
    robustness trade-off.
4.  Prints a detailed ASCII report with pass/fail indicators.
5.  Saves watermarked images to assets/watermarked_images/.

Target
──────
    Project goal: PSNR > 38 dB for all images (imperceptible).
"""

import sys
import os
from pathlib import Path

# Make sure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

# Project root is the parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import cv2
import numpy as np

from watermarking import (
    load_image,
    preprocess_image,
    apply_dwt,
    prepare_watermark,
    embed_watermark,
    reconstruct_image,
    calculate_psnr,
    extract_watermark,
    ALPHA,
    BLOCK_SIZE,
)
from watermarking.arnold import arnold_transform, inverse_arnold_transform, arnold_period
from evaluation import calculate_ssim


# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────
INPUT_DIR       = PROJECT_ROOT / "assets" / "input_images"
WATERMARKS_DIR  = PROJECT_ROOT / "assets" / "watermarks"
WATERMARKED_DIR = PROJECT_ROOT / "assets" / "watermarked_images"

# PSNR target from project spec
PSNR_TARGET = 38.0

# Alpha values to test (showing trade-off)
ALPHA_VALUES = [0.02, 0.05, 0.08, 0.15, 0.25]


def generate_checkerboard(side: int = 32) -> np.ndarray:
    """Create a simple checkerboard binary watermark."""
    wm = np.zeros((side, side), dtype=np.uint8)
    for r in range(side):
        for c in range(side):
            wm[r, c] = 255 if (r + c) % 2 == 0 else 0
    return wm


def test_single_image(image_path: Path, watermark: np.ndarray,
                       alpha: float, output_dir: Path) -> dict:
    """
    Embed watermark into a single image and compute PSNR.

    Returns dict with metrics.
    """
    name = image_path.stem

    # Module 1: Data preparation
    original = load_image(str(image_path))
    Y_norm, Cb, Cr, _ = preprocess_image(original)
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    h_band, w_band = LH.shape
    capacity = (h_band // BLOCK_SIZE) * (w_band // BLOCK_SIZE)

    # Prepare watermark bits
    wm_bits = prepare_watermark(watermark, capacity)

    # Module 2: Embedding
    Y_wm, shape = embed_watermark(Y_norm, wm_bits, alpha)
    watermarked = reconstruct_image(Y_wm, Cb, Cr)

    # Save
    out_path = output_dir / f"wm_alpha{alpha:.2f}_{name}.png"
    cv2.imwrite(str(out_path), watermarked)

    # PSNR (align original to watermarked size)
    h_wm, w_wm = watermarked.shape[:2]
    orig_aligned = original[:h_wm, :w_wm]
    psnr_val = calculate_psnr(orig_aligned, watermarked)

    # SSIM
    ssim_val = calculate_ssim(orig_aligned, watermarked)

    # Extraction verification (blind)
    extracted = extract_watermark(watermarked, len(wm_bits), shape, alpha)
    ext_bits = (extracted.flatten()[:len(wm_bits)] > 127).astype(np.uint8)
    ber = np.sum(ext_bits != wm_bits) / len(wm_bits)

    return {
        "name": name,
        "shape": f"{watermarked.shape[1]}x{watermarked.shape[0]}",
        "psnr": psnr_val,
        "ssim": ssim_val,
        "ber": ber,
        "n_bits": len(wm_bits),
        "capacity": capacity,
        "alpha": alpha,
        "output": str(out_path),
    }


def demo_single_alpha():
    """
    Demo 1: Embed watermark into all images with default alpha.
    Show PSNR for each.
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: Invisibility Test — Default Alpha (α = {})".format(ALPHA))
    print("=" * 70)

    images = sorted(INPUT_DIR.glob("*"), key=lambda p: p.name.lower())
    images = [p for p in images if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]

    if not images:
        print(f"  No images found in {INPUT_DIR}")
        return

    watermark = generate_checkerboard(32)
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in images:
        r = test_single_image(img_path, watermark, ALPHA, WATERMARKED_DIR)
        results.append(r)

    # Report
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  PSNR IMPERCEPTIBILITY REPORT".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Image':<25} {'PSNR (dB)':>10} {'SSIM':>8} {'Status':>14}  ║")
    print("╟" + "─" * 68 + "╢")

    total_pass = 0
    for r in results:
        status = "✓ PASS" if r["psnr"] >= PSNR_TARGET else "✗ FAIL"
        if r["psnr"] >= PSNR_TARGET:
            total_pass += 1
        status_padded = status + " " * max(0, 14 - len(status))
        print(f"║  {r['name']:<25} {r['psnr']:>9.2f} {r['ssim']:>7.4f} {status_padded:>14}  ║")

    print("╟" + "─" * 68 + "╢")
    avg_psnr = np.mean([r["psnr"] for r in results])
    avg_ssim = np.mean([r["ssim"] for r in results])
    print(f"║  {'AVERAGE':<25} {avg_psnr:>9.2f} {avg_ssim:>7.4f} {'':<14}  ║")
    print("╟" + "─" * 68 + "╢")
    print(f"║  Result: {total_pass}/{len(results)} images passed PSNR >= {PSNR_TARGET} dB" + " " * (30 - len(str(total_pass)) - len(str(len(results)))) + "║")
    print("╚" + "═" * 68 + "╝")


def demo_alpha_tradeoff():
    """
    Demo 2: Test multiple alpha values on one image.
    Show PSNR vs embedding strength trade-off.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Alpha Trade-off — PSNR vs Embedding Strength")
    print("=" * 70)

    images = sorted(INPUT_DIR.glob("*"), key=lambda p: p.name.lower())
    images = [p for p in images if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]

    if not images:
        print(f"  No images found in {INPUT_DIR}")
        return

    # Use the first image
    img_path = images[0]
    watermark = generate_checkerboard(32)
    WATERMARKED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Host image: {img_path.name}")
    print(f"  Alpha values tested: {ALPHA_VALUES}")
    print(f"  PSNR target: {PSNR_TARGET} dB")

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  ALPHA TRADE-OFF ANALYSIS".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  {'Alpha (α)':>10} {'PSNR (dB)':>10} {'SSIM':>8} {'BER':>8} {'Status':>14}  ║")
    print("╟" + "─" * 68 + "╢")

    for alpha in ALPHA_VALUES:
        r = test_single_image(img_path, watermark, alpha, WATERMARKED_DIR)
        status = "✓ PASS" if r["psnr"] >= PSNR_TARGET else "✗ FAIL"
        status_padded = status + " " * max(0, 14 - len(status))

        # ASCII bar chart inline
        bar_len = min(int(r["psnr"] / 60 * 20), 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"║  {alpha:>10.2f} {r['psnr']:>9.2f} {r['ssim']:>7.4f} {r['ber']:>7.4f} {status_padded:>14}  ║")
        print(f"║  {'':>10} [{bar}] {r['psnr']:.1f} dB" + " " * 24 + "║")

    print("╟" + "─" * 68 + "╢")
    print(f"║  Legend: █ = PSNR progress (max 60 dB)                        ║")
    print(f"║  Target: PSNR >= {PSNR_TARGET} dB  |  BER < 0.05 for valid extraction  ║")
    print("╚" + "═" * 68 + "╝")


def demo_arnold_effect():
    """
    Demo 3: Show that Arnold scrambling does NOT affect PSNR.
    (Since scrambling only rearranges bits, the embedding strength
    is identical.)
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: Arnold Scrambling — No Effect on PSNR")
    print("=" * 70)

    images = sorted(INPUT_DIR.glob("*"), key=lambda p: p.name.lower())
    images = [p for p in images if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]

    if not images:
        print(f"  No images found in {INPUT_DIR}")
        return

    img_path = images[0]
    watermark = generate_checkerboard(32)

    original = load_image(str(img_path))
    Y_norm, Cb, Cr, _ = preprocess_image(original)
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    h_band, w_band = LH.shape
    capacity = (h_band // BLOCK_SIZE) * (w_band // BLOCK_SIZE)
    wm_bits = prepare_watermark(watermark, capacity)

    # Without scrambling
    Y_wm1, shape1 = embed_watermark(Y_norm, wm_bits, ALPHA)
    watermarked1 = reconstruct_image(Y_wm1, Cb, Cr)
    psnr1 = calculate_psnr(original[:Y_wm1.shape[0], :Y_wm1.shape[1]], watermarked1)

    # With scrambling
    from watermarking.arnold import arnold_scramble_bits
    scrambled, grid_shape, padding = arnold_scramble_bits(wm_bits, iterations=5)
    Y_wm2, shape2 = embed_watermark(Y_norm, scrambled, ALPHA)
    watermarked2 = reconstruct_image(Y_wm2, Cb, Cr)
    psnr2 = calculate_psnr(original[:Y_wm2.shape[0], :Y_wm2.shape[1]], watermarked2)

    print(f"\n  Image: {img_path.name}")
    print(f"  Without Arnold: PSNR = {psnr1:.2f} dB")
    print(f"  With Arnold (5 iter): PSNR = {psnr2:.2f} dB")
    print(f"  Difference: {abs(psnr1 - psnr2):.4f} dB")
    print(f"\n  Conclusion: Arnold scrambling only permutes bit positions.")
    print(f"  The same number of bits are embedded with the same alpha,")
    print(f"  so PSNR is identical (minor diff from bit distribution).")


def main():
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   PSNR IMPERCEPTIBILITY DEMO                                   #")
    print("#   Hybrid DWT-DCT Watermarking — BMDS2133                       #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    demo_single_alpha()
    demo_alpha_tradeoff()
    demo_arnold_effect()

    print("\n" + "#" * 70)
    print("#   DEMO COMPLETE                                                           #")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
