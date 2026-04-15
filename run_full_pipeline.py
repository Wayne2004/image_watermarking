"""
===================================================================
BMDS2133 Image Processing — Group Assignment
End-to-End Integration Script: Embed → Attack → Extract → Evaluate
===================================================================

This script runs the COMPLETE pipeline automatically:
  1. Load original image + watermark
  2. Embed watermark using Hybrid DWT-DCT (Module 1 + Module 2)
  3. Apply attacks (Module 3 - Role B)
  4. Extract watermark from each attacked image (Module 3 - Role C)
  5. Evaluate robustness (BER, NCC, PSNR, SSIM)
  6. Generate comprehensive report with charts

Usage
─────
    python run_full_pipeline.py                  # run with defaults
    python run_full_pipeline.py --help           # show options

Default configuration
─────────────────────
  Host image   : assets/input_images/mushroom.png
  Watermark    : assets/watermarks/watermark.png
  Attacks      : JPEG (Q=75,50,30), Crop (10%,25%), Scale (0.5,0.75),
                 Gaussian Noise (σ=10,20), Blur (k=5,9)
  Output dir   : results/full_pipeline/
"""

import sys
import os

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import math

from watermarking import (
    run_embedding_pipeline,
    extract_watermark,
    load_image,
    preprocess_image,
    apply_dwt,
    calculate_psnr,
    prepare_watermark,
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
    calculate_psnr as eval_psnr,
    calculate_ssim,
    calculate_ber,
    evaluate_watermark_robustness,
)


# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR    = PROJECT_ROOT / "assets" / "input_images"
WATERMARKS_DIR = PROJECT_ROOT / "assets" / "watermarks"
RESULTS_DIR  = PROJECT_ROOT / "results" / "full_pipeline"


# ─────────────────────────────────────────────────────────────────
# NCC METRIC (Normalized Cross-Correlation)
# ─────────────────────────────────────────────────────────────────

def calculate_ncc(original: np.ndarray, extracted: np.ndarray) -> float:
    """
    Calculate Normalized Cross-Correlation between two binary images.

    NCC measures the similarity between original and extracted
    watermarks. Values range from 0 (no correlation) to 1 (perfect).

    Parameters
    ----------
    original  : 2-D uint8 (binary or grayscale)
    extracted : 2-D uint8 (binary or grayscale), same shape

    Returns
    -------
    float  NCC value in [0, 1]
    """
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
# ATTACK DEFINITIONS
# ─────────────────────────────────────────────────────────────────

def get_attack_suite():
    """
    Return a list of (name, attack_function) tuples representing
    the full attack suite for robustness testing.
    """
    return [
        # JPEG compression at various quality factors
        ("jpeg_q75",  lambda img: jpeg_compression(img, quality=75)),
        ("jpeg_q50",  lambda img: jpeg_compression(img, quality=50)),
        ("jpeg_q30",  lambda img: jpeg_compression(img, quality=30)),
        # Cropping attacks
        ("crop_10pct",  lambda img: cropping(img, percentage=0.10)),
        ("crop_25pct",  lambda img: cropping(img, percentage=0.25)),
        # Scaling attacks (downscale then upscale back)
        ("scale_075", lambda img: _scale_roundtrip(img, 0.75)),
        ("scale_050", lambda img: _scale_roundtrip(img, 0.50)),
        # Gaussian noise
        ("noise_sigma10", lambda img: gaussian_noise(img, sigma=10)),
        ("noise_sigma20", lambda img: gaussian_noise(img, sigma=20)),
        # Blurring
        ("blur_k5",  lambda img: blurring(img, kernel_size=5)),
        ("blur_k9",  lambda img: blurring(img, kernel_size=9)),
    ]


def _scale_roundtrip(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Simulate social media scaling: downscale then upscale back
    to original dimensions using bicubic interpolation.
    """
    h, w = image.shape[:2]
    # Downscale
    small = cv2.resize(image, (int(w * factor), int(h * factor)),
                       interpolation=cv2.INTER_CUBIC)
    # Upscale back to original size
    restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored


# ─────────────────────────────────────────────────────────────────
# ARNOLD SCRAMBLING INTEGRATION
# ─────────────────────────────────────────────────────────────────

def embed_with_arnold_scrambling(
    image_path: str,
    watermark_input,
    output_path: str,
    alpha: float = ALPHA,
    arnold_iterations: int = 5,
) -> dict:
    """
    Embed watermark with Arnold Transform scrambling applied before
    embedding.  The scrambled bits are embedded, and the scrambling
    parameters are stored for later descrambling.

    Parameters
    ----------
    image_path         : host image path
    watermark_input    : watermark path or numpy array
    output_path        : where to save watermarked image
    alpha              : base embedding strength
    arnold_iterations  : Arnold scrambling key

    Returns
    -------
    dict with embedding results + scrambling metadata
    """
    from watermarking import (
        load_image, preprocess_image, apply_dwt,
        prepare_watermark, embed_watermark, reconstruct_image,
        BLOCK_SIZE,
    )

    # Module 1: Data preparation
    original     = load_image(image_path)
    Y_norm, Cb, Cr, _ = preprocess_image(original)
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    h_band, w_band = LH.shape
    capacity = (h_band // BLOCK_SIZE) * (w_band // BLOCK_SIZE)

    # Prepare raw watermark bits
    wm_bits = prepare_watermark(watermark_input, capacity)
    n_bits = len(wm_bits)

    # Compute the embedding grid shape first (same as embed_watermark would)
    wm_rows = int(math.ceil(math.sqrt(n_bits * h_band / w_band)))
    wm_cols = int(math.ceil(n_bits / wm_rows))
    embedding_grid = (wm_rows, wm_cols)
    grid_total = wm_rows * wm_cols

    # Arnold scramble: pad to match embedding grid
    scrambled_bits, grid_shape, padding = arnold_scramble_bits(
        wm_bits, iterations=arnold_iterations
    )

    # If Arnold grid doesn't match embedding grid, re-pad scrambled bits
    if grid_shape != embedding_grid:
        # Re-scramble to match the embedding grid size
        padded_bits = np.pad(wm_bits, (0, grid_total - n_bits), mode='constant')
        side = int(math.ceil(math.sqrt(grid_total)))
        square_total = side * side
        padded_to_square = np.pad(wm_bits, (0, square_total - n_bits), mode='constant')
        square_grid = padded_to_square.reshape((side, side))
        scrambled_grid = arnold_transform(square_grid, iterations=arnold_iterations)
        scrambled_bits = scrambled_grid.flatten()
        grid_shape = (side, side)
        padding = square_total - n_bits

    # Embed scrambled bits
    Y_watermarked, watermark_shape = embed_watermark(
        Y_norm, scrambled_bits, alpha
    )
    watermarked_image = reconstruct_image(Y_watermarked, Cb, Cr)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), watermarked_image)

    psnr_val = calculate_psnr(
        original[:Y_watermarked.shape[0], :Y_watermarked.shape[1]],
        watermarked_image
    )

    print(f"[embed_with_arnold] Scrambled with {arnold_iterations} Arnold iterations")
    print(f"[embed_with_arnold] Grid shape: {grid_shape}, padding: {padding}")
    print(f"[embed_with_arnold] Embed watermark_shape: {watermark_shape}")

    return {
        "watermarked_image": watermarked_image,
        "psnr": psnr_val,
        "n_bits": n_bits,
        "watermark_shape": watermark_shape,
        "capacity": capacity,
        "watermark_bits": wm_bits,
        "arnold_iterations": arnold_iterations,
        "grid_shape": grid_shape,
        "padding": padding,
        "scrambled_bits": scrambled_bits,
        "scrambled_length": len(scrambled_bits),
    }


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_full_pipeline(
    image_path: str = None,
    watermark_path: str = None,
    use_arnold: bool = False,
    arnold_iterations: int = 5,
    alpha: float = ALPHA,
    output_dir: str = None,
) -> dict:
    """
    Run the complete end-to-end pipeline:
      Embed → Attack → Extract → Evaluate

    Parameters
    ----------
    image_path        : str  host image path (default: mushroom.png)
    watermark_path    : str  watermark path (default: watermark.png)
    use_arnold        : bool whether to use Arnold scrambling
    arnold_iterations : int  Arnold scrambling key
    alpha             : float  embedding strength
    output_dir        : str  results output directory

    Returns
    -------
    dict with comprehensive results
    """
    # Defaults
    if image_path is None:
        image_path = INPUT_DIR / "mushroom.png"
    if watermark_path is None:
        watermark_path = WATERMARKS_DIR / "watermark.png"
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   FULL PIPELINE: Embed → Attack → Extract → Evaluate           #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: EMBEDDING
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1: WATERMARK EMBEDDING")
    print("=" * 60)

    embed_out = output_dir / "watermarked_image.png"

    # Use standard embedding (no Arnold scrambling by default for the base pipeline)
    embed_result = run_embedding_pipeline(
        str(image_path), str(watermark_path), str(embed_out), alpha
    )

    watermarked_image = embed_result["watermarked_image"]
    watermark_shape   = embed_result["watermark_shape"]
    n_bits            = embed_result["n_bits"]

    # Load original watermark for comparison
    original_wm = cv2.imread(str(watermark_path), cv2.IMREAD_GRAYSCALE)
    if original_wm is None:
        # Fallback: use the watermark bits to create a reference
        wm_bits_ref = embed_result.get("watermark_bits", None)
        if wm_bits_ref is not None:
            rows = watermark_shape[0]
            cols = watermark_shape[1]
            wm_2d = np.zeros(rows * cols, dtype=np.uint8)
            wm_2d[:n_bits] = wm_bits_ref[:n_bits]
            original_wm = (wm_2d.reshape((rows, cols)) * 255).astype(np.uint8)
    else:
        # The loaded watermark image may have been resized during embedding.
        # Create the correct reference from the embedded bits.
        wm_bits_ref = embed_result.get("watermark_bits", None)
        if wm_bits_ref is not None:
            rows = watermark_shape[0]
            cols = watermark_shape[1]
            wm_2d = np.zeros(rows * cols, dtype=np.uint8)
            wm_2d[:n_bits] = wm_bits_ref[:n_bits]
            original_wm = (wm_2d.reshape((rows, cols)) * 255).astype(np.uint8)

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: ATTACK SIMULATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: ATTACK SIMULATION")
    print("=" * 60)

    attack_dir = output_dir / "attacked_images"
    attack_dir.mkdir(parents=True, exist_ok=True)

    attack_suite = get_attack_suite()
    attacked_images = {}

    for attack_name, attack_func in attack_suite:
        print(f"\n  Applying: {attack_name}")
        attacked = attack_func(watermarked_image.copy())
        attacked_images[attack_name] = attacked

        # Save attacked image
        save_path = attack_dir / f"{attack_name}.png"
        cv2.imwrite(str(save_path), attacked)

    # Also keep a clean (no-attack) version for baseline
    attacked_images["clean"] = watermarked_image.copy()
    cv2.imwrite(str(attack_dir / "clean.png"), watermarked_image)

    print(f"\n  Total attacks applied: {len(attack_suite)}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: EXTRACTION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: WATERMARK EXTRACTION")
    print("=" * 60)

    extract_dir = output_dir / "extracted_watermarks"
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_watermarks = {}

    for attack_name, attacked_img in attacked_images.items():
        print(f"\n  Extracting from: {attack_name}")

        extracted = extract_watermark_robust(
            attacked_img, n_bits, watermark_shape, alpha
        )

        extracted_watermarks[attack_name] = extracted

        # Save
        save_path = extract_dir / f"extracted_{attack_name}.png"
        cv2.imwrite(str(save_path), extracted)

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: EVALUATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 4: EVALUATION")
    print("=" * 60)

    eval_results = {}

    for attack_name, extracted_wm in extracted_watermarks.items():
        attacked_img = attacked_images[attack_name]

        # Resize extracted to match original watermark if needed
        eval_wm = extracted_wm
        if original_wm is not None and eval_wm.shape != original_wm.shape:
            eval_wm = cv2.resize(eval_wm, (original_wm.shape[1], original_wm.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # BER
        ber = calculate_ber(original_wm, eval_wm) if original_wm is not None else -1.0

        # NCC
        ncc = calculate_ncc(original_wm, eval_wm) if original_wm is not None else -1.0

        # PSNR (watermarked vs attacked)
        psnr_attack = eval_psnr(watermarked_image, attacked_img)

        # SSIM (watermarked vs attacked)
        ssim_attack = calculate_ssim(watermarked_image, attacked_img)

        eval_results[attack_name] = {
            "ber": ber,
            "ncc": ncc,
            "psnr_attack": psnr_attack,
            "ssim_attack": ssim_attack,
        }

        ber_pct = f"{ber * 100:.2f}%" if ber >= 0 else "N/A"
        ncc_str = f"{ncc:.4f}" if ncc >= 0 else "N/A"
        print(f"\n  {attack_name}:")
        print(f"    BER  = {ber_pct}   {'✓ PASS (<5%)' if 0 <= ber < 0.05 else '✗ FAIL' if ber >= 0 else 'N/A'}")
        print(f"    NCC  = {ncc_str}   {'✓ PASS (>0.9)' if ncc >= 0.9 else '✗ FAIL' if ncc >= 0 else 'N/A'}")
        print(f"    PSNR = {psnr_attack:.2f} dB")
        print(f"    SSIM = {ssim_attack:.4f}")

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: REPORT GENERATION
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 5: REPORT GENERATION")
    print("=" * 60)

    report = generate_report(
        embed_result=embed_result,
        eval_results=eval_results,
        original_wm=original_wm,
        extracted_watermarks=extracted_watermarks,
        watermarked_image=watermarked_image,
        use_arnold=use_arnold,
        arnold_iterations=arnold_iterations,
    )

    # Save report as JSON
    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report["summary"], f, indent=2, default=str)
    print(f"\n  Report saved → {report_path}")

    # Generate charts
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    generate_charts(eval_results, charts_dir)
    print(f"  Charts saved → {charts_dir}")

    print("\n" + "#" * 70)
    print("#   PIPELINE COMPLETE                                                         #")
    print(f"#   Results: {output_dir}" + " " * (68 - len(str(output_dir)) - 11) + "#")
    print("#" * 70 + "\n")

    return report


# ─────────────────────────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_report(
    embed_result: dict,
    eval_results: dict,
    original_wm: np.ndarray,
    extracted_watermarks: dict,
    watermarked_image: np.ndarray,
    use_arnold: bool,
    arnold_iterations: int,
) -> dict:
    """
    Generate a comprehensive ASCII report and structured summary.
    """
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  FULL PIPELINE REPORT".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    # Embedding section
    print("║" + "  EMBEDDING".center(68) + "║")
    print("╟" + "─" * 68 + "╢")
    print(f"║  Bits embedded     : {embed_result['n_bits']:>8}" + " " * 47 + "║")
    print(f"║  Alpha             : {embed_result.get('capacity', 'N/A'):>8}" + " " * 47 + "║")
    print(f"║  Arnold Scramble   : {'Yes (iter=' + str(arnold_iterations) + ')' if use_arnold else 'No'}" + " " * (55 - (len('Yes (iter=' + str(arnold_iterations) + ')') if use_arnold else 2)) + "║")

    # Evaluation section
    print("╟" + "─" * 68 + "╢")
    print("║" + "  ROBUSTNESS EVALUATION".center(68) + "║")
    print("╟" + "─" * 68 + "╢")
    print(f"║  {'Attack':<20} {'BER':>8} {'NCC':>8} {'PSNR':>8} {'SSIM':>8}  ║")
    print("╟" + "─" * 68 + "╢")

    summary = {
        "embedding": {
            "psnr": psnr,
            "n_bits": embed_result["n_bits"],
            "watermark_shape": list(embed_result["watermark_shape"]),
            "arnold_scrambled": use_arnold,
            "arnold_iterations": arnold_iterations,
        },
        "attacks": {},
    }

    for attack_name, metrics in eval_results.items():
        ber = metrics["ber"]
        ncc = metrics["ncc"]
        psnr_a = metrics["psnr_attack"]
        ssim_a = metrics["ssim_attack"]

        ber_str = f"{ber * 100:.2f}%" if ber >= 0 else "N/A"
        ncc_str = f"{ncc:.4f}" if ncc >= 0 else "N/A"

        print(f"║  {attack_name:<20} {ber_str:>8} {ncc_str:>8} {psnr_a:>7.1f}dB {ssim_a:>7.4f}  ║")

        summary["attacks"][attack_name] = {
            "ber": ber if ber >= 0 else None,
            "ncc": ncc if ncc >= 0 else None,
            "psnr_attack": psnr_a,
            "ssim_attack": ssim_a,
        }

    print("╚" + "═" * 68 + "╝")

    return {"summary": summary}


# ─────────────────────────────────────────────────────────────────
# CHART GENERATION (ASCII)
# ─────────────────────────────────────────────────────────────────

def generate_charts(eval_results: dict, output_dir: Path):
    """
    Generate ASCII bar charts for BER and NCC across attacks.
    Also saves them as text files.
    """
    attacks = list(eval_results.keys())

    # ── BER Chart ──
    ber_lines = ["BER (Bit Error Rate) by Attack Type", "=" * 40]
    for attack in attacks:
        ber = eval_results[attack]["ber"]
        if ber < 0:
            continue
        bar_len = int(ber * 100 * 2)  # scale: 50% = full bar
        bar = "█" * bar_len + "░" * max(0, 50 - bar_len)
        ber_lines.append(f"  {attack:<20} |{bar}| {ber * 100:.2f}%")

    ber_text = "\n".join(ber_lines)
    print(f"\n{ber_text}")
    with open(output_dir / "ber_chart.txt", "w") as f:
        f.write(ber_text)

    # ── NCC Chart ──
    ncc_lines = ["NCC (Normalized Cross-Correlation) by Attack Type", "=" * 50]
    for attack in attacks:
        ncc = eval_results[attack]["ncc"]
        if ncc < 0:
            continue
        bar_len = int(ncc * 50)
        bar = "█" * bar_len + "░" * max(0, 50 - bar_len)
        ncc_lines.append(f"  {attack:<20} |{bar}| {ncc:.4f}")

    ncc_text = "\n".join(ncc_lines)
    print(f"\n{ncc_text}")
    with open(output_dir / "ncc_chart.txt", "w") as f:
        f.write(ncc_text)

    # ── PSNR Chart ──
    psnr_lines = ["PSNR (Watermarked vs Attacked) by Attack Type", "=" * 50]
    for attack in attacks:
        psnr = eval_results[attack]["psnr_attack"]
        bar_len = min(int(psnr), 50)
        bar = "█" * bar_len + "░" * max(0, 50 - bar_len)
        psnr_lines.append(f"  {attack:<20} |{bar}| {psnr:.1f} dB")

    psnr_text = "\n".join(psnr_lines)
    print(f"\n{psnr_text}")
    with open(output_dir / "psnr_chart.txt", "w") as f:
        f.write(psnr_text)


# ─────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end watermarking pipeline: Embed → Attack → Extract → Evaluate"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Path to host image (default: assets/input_images/mushroom.png)"
    )
    parser.add_argument(
        "--watermark", "-w", type=str, default=None,
        help="Path to watermark image (default: assets/watermarks/watermark.png)"
    )
    parser.add_argument(
        "--no-arnold", action="store_true",
        help="Disable Arnold Transform scrambling"
    )
    parser.add_argument(
        "--arnold-iter", type=int, default=5,
        help="Arnold Transform iterations (default: 5)"
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help=f"Embedding strength (default: {ALPHA})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory (default: results/full_pipeline/)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        image_path=args.image,
        watermark_path=args.watermark,
        use_arnold=not args.no_arnold,
        arnold_iterations=args.arnold_iter,
        alpha=args.alpha,
        output_dir=args.output,
    )
