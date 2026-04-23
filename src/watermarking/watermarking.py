"""
===================================================================
BMDS2133 Image Processing — Group Assignment
Module 1: Data Preparation  +  Module 2: Watermark Embedding
===================================================================
Algorithm : Hybrid DWT-DCT Differential Blind Watermarking
"""

import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from pathlib import Path
import math


from .arnold import (
    arnold_scramble_bits,
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

BLOCK_SIZE  = 8          
ALPHA       = 0.20       # Role C: Increased for heavy JPEG resistance
HVS_GAIN    = 3.0        
# We now use TWO coefficients for differential embedding
COEFF_A     = (3, 4)     
COEFF_B     = (4, 3)     
WAVELET     = "haar"     
DWT_LEVEL   = 1          
CALIBRATION_SIZE = 0     # Differential is self-syncing, no header needed

ALPHA_BASE  = ALPHA


# ─────────────────────────────────────────────────────────────────
# MODULE 1: DATA PREPARATION
# ─────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img

def save_image(image: np.ndarray, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), image)

def preprocess_image(image: np.ndarray) -> tuple:
    ycbcr     = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)
    align = BLOCK_SIZE * 2
    h, w  = Y.shape
    new_h = (h // align) * align
    new_w = (w // align) * align
    if new_h != h or new_w != w:
        Y = Y[:new_h, :new_w]
        Cb = Cb[:new_h, :new_w]
        Cr = Cr[:new_h, :new_w]
        ycbcr = ycbcr[:new_h, :new_w]
    Y_norm = Y.astype(np.float64) / 255.0
    return Y_norm, Cb, Cr, ycbcr

def prepare_watermark(watermark_input, embed_capacity: int) -> tuple:
    """
    Prepare a watermark for embedding and save a reference baseline.
    """
    if isinstance(watermark_input, (str, Path)):
        path = Path(str(watermark_input))
        if not path.exists():
            import PIL.Image as PILImage, PIL.ImageDraw as PILDraw, PIL.ImageFont as PILFont
            font_size = 72
            try:
                font = PILFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = PILFont.load_default()
            temp_img = PILImage.new('L', (1000, 200))
            draw = PILDraw.Draw(temp_img)
            bbox = draw.textbbox((0, 0), str(watermark_input), font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            img_pil = PILImage.new('L', (tw + 4, th + 4), color=0)
            draw = PILDraw.Draw(img_pil)
            draw.text((2, 2), str(watermark_input), font=font, fill=255)
            wm = np.array(img_pil)
        else:
            wm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if wm is None:
                raise ValueError(f"Could not read watermark image: {watermark_input}")
    else:
        wm = watermark_input
        if len(wm.shape) == 3:
            wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)

    # Initial binarization
    _, wm_bin = cv2.threshold(wm, 128, 1, cv2.THRESH_BINARY)
    
    # Resize to fit capacity if necessary
    total = wm_bin.shape[0] * wm_bin.shape[1]
    if total > embed_capacity:
        scale = math.sqrt(embed_capacity / total)
        nh, nw = max(1, int(wm_bin.shape[0]*scale)), max(1, int(wm_bin.shape[1]*scale))
        # Use INTER_AREA for structural integrity during shrinking
        wm_bin = cv2.resize(wm_bin, (nw, nh), interpolation=cv2.INTER_AREA)
        _, wm_bin = cv2.threshold(wm_bin, 0.5, 1, cv2.THRESH_BINARY)

    # Role C Fix: Save the "Gold Standard" reference for evaluation
    # This matches the pixel-grid of the extraction exactly.
    ref_img = (wm_bin * 255).astype(np.uint8)
    ref_path = Path("assets/watermarks/reference_baseline.png")
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref_path), ref_img)
    print(f"[prepare_watermark] Reference baseline saved to {ref_path}")
    
    return wm_bin.flatten().astype(np.uint8), wm_bin.shape

# ─────────────────────────────────────────────────────────────────
# MODULE 2: EMBEDDING
# ─────────────────────────────────────────────────────────────────

def apply_dwt(channel: np.ndarray) -> tuple:
    coeffs = pywt.dwt2(channel, WAVELET)
    LL, (LH, HL, HH) = coeffs
    return coeffs, LL, LH, HL, HH

def apply_dct_blocks(subband: np.ndarray) -> np.ndarray:
    h, w = subband.shape
    dct_img = np.zeros_like(subband)
    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            block = subband[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
            dct_img[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = dct(dct(block.T, norm='ortho').T, norm='ortho')
    return dct_img

def apply_idct_blocks(dct_subband: np.ndarray) -> np.ndarray:
    h, w = dct_subband.shape
    spatial = np.zeros_like(dct_subband)
    for r in range(0, h, BLOCK_SIZE):
        for c in range(0, w, BLOCK_SIZE):
            block = dct_subband[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE]
            spatial[r:r+BLOCK_SIZE, c:c+BLOCK_SIZE] = idct(idct(block, norm='ortho', axis=1), norm='ortho', axis=0)
    return spatial

def _compute_adaptive_alpha(dct_block: np.ndarray, alpha_base: float) -> float:
    # Use block energy but cap it for stability
    mid_energy = float(np.mean(np.abs(dct_block)))
    return alpha_base * (1.0 + min(HVS_GAIN * mid_energy, 2.0))

def _embed_bit_into_band(dct_band: np.ndarray, watermark_bits: np.ndarray, alpha_base: float) -> np.ndarray:
    """
    Differential Embedding: Compares two coefficients.
    Supports Redundancy by repeating bits across all available blocks.
    """
    h, w = dct_band.shape
    n_blocks_row, n_blocks_col = h // BLOCK_SIZE, w // BLOCK_SIZE
    dct_modified = dct_band.copy()
    n_bits = len(watermark_bits)
    
    count = 0
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            bit = int(watermark_bits[count % n_bits])
            br, bc = r * BLOCK_SIZE, c * BLOCK_SIZE
            
            block = dct_modified[br:br+BLOCK_SIZE, bc:bc+BLOCK_SIZE]
            alpha_l = _compute_adaptive_alpha(block, alpha_base)
            
            # Differential Logic
            v1 = block[COEFF_A[0], COEFF_A[1]]
            v2 = block[COEFF_B[0], COEFF_B[1]]
            
            avg = (v1 + v2) / 2.0
            if bit == 1:
                # Make v1 significantly larger than v2
                v1 = avg + alpha_l / 2.0
                v2 = avg - alpha_l / 2.0
            else:
                # Make v2 significantly larger than v1
                v1 = avg - alpha_l / 2.0
                v2 = avg + alpha_l / 2.0
                
            dct_modified[br+COEFF_A[0], bc+COEFF_A[1]] = v1
            dct_modified[br+COEFF_B[0], bc+COEFF_B[1]] = v2
            count += 1
    return dct_modified

def _extract_soft_from_band(dct_band: np.ndarray, n_bits: int, alpha_base: float) -> np.ndarray:
    """
    Differential Extraction: Uses hard-voting (Sign-based).
    Every block gets exactly one 'vote' (+1 or -1).
    This ensures Gaussian noise in one block can't overwhelm others.
    """
    h, w = dct_band.shape
    n_blocks_row, n_blocks_col = h // BLOCK_SIZE, w // BLOCK_SIZE
    # scores will now store the sum of 'votes'
    scores = np.zeros(n_bits, dtype=np.float64)
    
    count = 0
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            br, bc = r * BLOCK_SIZE, c * BLOCK_SIZE
            v1 = dct_band[br+COEFF_A[0], bc+COEFF_A[1]]
            v2 = dct_band[br+COEFF_B[0], bc+COEFF_B[1]]
            
            # Vote +1 if v1 > v2, else -1
            vote = 1.0 if v1 > v2 else -1.0
            scores[count % n_bits] += vote
            count += 1
    return scores

def embed_watermark(Y_norm, watermark_bits, alpha=ALPHA):
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    dct_LH_mod = _embed_bit_into_band(apply_dct_blocks(LH), watermark_bits, alpha)
    dct_HL_mod = _embed_bit_into_band(apply_dct_blocks(HL), watermark_bits, alpha)
    new_coeffs = (LL, (apply_idct_blocks(dct_LH_mod), apply_idct_blocks(dct_HL_mod), HH))
    return np.clip(pywt.idwt2(new_coeffs, WAVELET), 0.0, 1.0), None

def reconstruct_image(Y_wm, Cb, Cr):
    Y_8 = np.clip(Y_wm * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([Y_8, Cr, Cb]), cv2.COLOR_YCrCb2BGR)

def extract_watermark(image, n_bits, shape, alpha=ALPHA):
    Y_norm, _, _, _ = preprocess_image(image)
    _, _, LH, HL, _ = apply_dwt(Y_norm)
    
    s_lh = _extract_soft_from_band(apply_dct_blocks(LH), n_bits, alpha)
    s_hl = _extract_soft_from_band(apply_dct_blocks(HL), n_bits, alpha)
    
    # Simple consensus: if diff > 0, bit 1
    voted = ((s_lh + s_hl) >= 0).astype(np.uint8)
    return (voted[:shape[0]*shape[1]].reshape(shape) * 255).astype(np.uint8)

def calculate_psnr(orig, proc):
    if orig.shape != proc.shape: proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
    mse = np.mean((orig.astype(np.float64) - proc.astype(np.float64))**2)
    return float('inf') if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))

def run_embedding_pipeline(image_path, watermark_input, output_path, alpha=ALPHA, arnold_iterations=0):
    original = load_image(image_path)
    Y_norm, Cb, Cr, _ = preprocess_image(original)
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    capacity = (LH.shape[0] // BLOCK_SIZE) * (LH.shape[1] // BLOCK_SIZE)
    
    bits, shape_orig = prepare_watermark(watermark_input, capacity)
    is_text = isinstance(watermark_input, str) and not Path(watermark_input).exists()
    
    actual_iters, wm_shape_embed, actual_n_bits = 0, shape_orig, len(bits)
    if arnold_iterations > 0:
        bits, grid, pad = arnold_scramble_bits(bits, iterations=arnold_iterations)
        actual_iters, wm_shape_embed, actual_n_bits = arnold_iterations, grid, len(bits)

    Y_wm, _ = embed_watermark(Y_norm, bits, alpha)
    wm_img = reconstruct_image(Y_wm, Cb, Cr)
    save_image(wm_img, output_path)
    
    return {
        "watermarked_image": wm_img,
        "n_bits": actual_n_bits,
        "watermark_shape": wm_shape_embed,
        "original_shape": shape_orig,
        "arnold_iterations": actual_iters,
        "is_text": is_text,
        "capacity": capacity,
        "watermark_bits": bits.copy(),
    }
