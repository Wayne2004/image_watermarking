"""
===================================================================
BMDS2133 Image Processing — Group Assignment
Module 1: Data Preparation  +  Module 2: Watermark Embedding
===================================================================
Algorithm : Hybrid DWT-DCT Blind Watermarking
Wavelet   : Haar (pywt)
DCT       : scipy.fftpack
Sub-band  : LH (horizontal detail) for embedding

Embedding pipeline
──────────────────
Image (BGR)
  └─ convert to YCbCr  →  extract Y channel
       └─ DWT (Haar)   →  LL | LH | HL | HH
            └─ LH sub-band  →  8×8 blocks
                 └─ DCT per block  →  modify mid-freq coefficient (3,4)
                      └─ IDCT per block  →  reconstruct LH
                           └─ IDWT  →  reconstructed Y
                                └─ merge with Cb, Cr  →  BGR output

Extraction pipeline (blind — no original image needed)
───────────────────────────────────────────────────────
Watermarked image  →  Y  →  DWT  →  LH  →  DCT blocks
  →  read sign of coefficient (3,4) per block  →  watermark bit
"""

import cv2
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from pathlib import Path
import math


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

BLOCK_SIZE = 8          # DCT block size (standard: 8×8)
ALPHA = 0.08            # Embedding strength — tuned for Y channel normalised to [0,1].
                        # Wavelet LH coefficients span roughly [-0.56, 0.56]; alpha=0.08
                        # gives a good imperceptibility/robustness trade-off:
                        #   PSNR ≈ 38–42 dB  (project target: >38 dB)
                        # Increase → more robust but lower PSNR.
                        # Decrease → higher PSNR but less robust.
EMBED_U, EMBED_V = 3, 4 # Mid-frequency DCT coefficient position to modify
WAVELET = "haar"        # Haar wavelet (compact, fast, good for JPEG resistance)
DWT_LEVEL = 1           # Single-level DWT decomposition


# ─────────────────────────────────────────────────────────────────
# MODULE 1: DATA PREPARATION
# ─────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Loaded image in BGR format (uint8, H×W×3).

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If OpenCV cannot decode the file.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    print(f"[load_image] Loaded '{path.name}'  shape={image.shape}  dtype={image.dtype}")
    return image


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a BGR image to YCbCr and extract the Y (luma) channel.

    Watermarks are embedded only into Y so that colour fidelity is
    preserved and the modification is less perceptible.

    Processing steps
    ────────────────
    1. Convert BGR → YCbCr.
    2. Split into Y, Cb, Cr channels.
    3. Normalise Y to float64 in [0, 1] for numerical stability
       during DWT/DCT arithmetic.
    4. Ensure height and width are multiples of (BLOCK_SIZE × 2) so
       that the DWT sub-bands divide evenly into 8×8 DCT blocks.

    Parameters
    ----------
    image : np.ndarray
        Source image in BGR format.

    Returns
    -------
    tuple (Y_norm, Cb, Cr, original_ycbcr)
        Y_norm        – float64 Y channel in [0, 1], shape (H, W)
        Cb            – uint8 Cb channel,   shape (H, W)
        Cr            – uint8 Cr channel,   shape (H, W)
        original_ycbcr – uint8 YCbCr image (for safe channel merging later)
    """
    # ── Step 1: colour space conversion ──────────────────────────
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # ── Step 2: split channels ────────────────────────────────────
    Y, Cr, Cb = cv2.split(ycbcr)          # OpenCV stores as Y, Cr, Cb

    # ── Step 3: spatial alignment ─────────────────────────────────
    # After one level of Haar DWT the sub-band is half the image size.
    # We need sub-band dimensions divisible by BLOCK_SIZE (8), so image
    # dimensions must be divisible by BLOCK_SIZE * 2 = 16.
    align = BLOCK_SIZE * 2                 # = 16
    h, w = Y.shape
    new_h = (h // align) * align
    new_w = (w // align) * align

    if new_h != h or new_w != w:
        print(f"[preprocess_image] Cropping image from ({h},{w}) to ({new_h},{new_w}) for block alignment")
        Y  = Y [:new_h, :new_w]
        Cb = Cb[:new_h, :new_w]
        Cr = Cr[:new_h, :new_w]
        ycbcr = ycbcr[:new_h, :new_w]

    # ── Step 4: normalise Y to [0,1] float64 ─────────────────────
    Y_norm = Y.astype(np.float64) / 255.0

    print(f"[preprocess_image] Y channel shape={Y_norm.shape}  range=[{Y_norm.min():.3f}, {Y_norm.max():.3f}]")
    return Y_norm, Cb, Cr, ycbcr


def prepare_watermark(watermark_input, embed_capacity: int) -> np.ndarray:
    """
    Prepare a watermark for embedding.

    Accepts either:
      • a file path (str / Path) to a binary/grayscale image, or
      • a numpy array (already loaded).

    Steps
    ─────
    1. Load / accept watermark.
    2. Convert to grayscale.
    3. Binarise at threshold 128 → pixel values become 0 or 1.
    4. Resize so that total pixels ≤ embed_capacity.
    5. Flatten to a 1-D bit array.

    Parameters
    ----------
    watermark_input : str | np.ndarray
        File path or pre-loaded image array.
    embed_capacity : int
        Maximum number of bits the host image can carry
        (equals the number of 8×8 DCT blocks in the LH sub-band).

    Returns
    -------
    np.ndarray
        1-D uint8 array of 0 / 1 values, length ≤ embed_capacity.
    """
    # ── Load ──────────────────────────────────────────────────────
    if isinstance(watermark_input, (str, Path)):
        path = Path(str(watermark_input))
        if not path.exists():
       
            import PIL.Image as PILImage
            import PIL.ImageDraw as PILImageDraw
            import PIL.ImageFont as PILImageFont
            text = str(watermark_input)
            img_pil = PILImage.new('L', (200, 60), color=0)
            draw = PILImageDraw.Draw(img_pil)
            draw.text((10, 10), text, fill=255)
            wm = np.array(img_pil)
        else:
            wm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if wm is None:
                raise ValueError(f"Could not read watermark: {watermark_input}")
    else:
        wm = watermark_input
        if len(wm.shape) == 3:
            wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)

    # ── Binarise ──────────────────────────────────────────────────
    _, wm_bin = cv2.threshold(wm, 128, 1, cv2.THRESH_BINARY)

    # ── Resize to fit embed_capacity ──────────────────────────────
    total_bits = wm_bin.shape[0] * wm_bin.shape[1]
    if total_bits > embed_capacity:
        # Scale down proportionally
        scale = math.sqrt(embed_capacity / total_bits)
        new_h = max(1, int(wm_bin.shape[0] * scale))
        new_w = max(1, int(wm_bin.shape[1] * scale))
        wm_bin = cv2.resize(wm_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"[prepare_watermark] Resized to ({new_h},{new_w}) to fit capacity={embed_capacity}")

    bits = wm_bin.flatten().astype(np.uint8)
    print(f"[prepare_watermark] Watermark bits={len(bits)}  capacity={embed_capacity}  "
          f"utilisation={len(bits)/embed_capacity*100:.1f}%")
    return bits


# ─────────────────────────────────────────────────────────────────
# MODULE 2: WATERMARK EMBEDDING
# ─────────────────────────────────────────────────────────────────

def apply_dwt(channel: np.ndarray) -> tuple:
    """
    Apply one-level 2-D Haar DWT to a single image channel.

    Haar wavelets have compact support and match the blocking
    structure of JPEG DCT, making the embedding robust to
    JPEG recompression.

    Decomposition result
    ────────────────────
        ┌──────┬──────┐
        │  LL  │  LH  │   LL = low-low  (approx / DC)
        ├──────┼──────┤   LH = low-high (horizontal detail)
        │  HL  │  HH  │   HL = high-low (vertical detail)
        └──────┴──────┘   HH = high-high (diagonal detail)

    Parameters
    ----------
    channel : np.ndarray
        2-D float64 array (H, W) in [0, 1].

    Returns
    -------
    tuple (coeffs, (LL, LH, HL, HH))
        coeffs – raw pywt coefficient object (needed for IDWT)
        LL, LH, HL, HH – individual sub-band arrays
    """
    coeffs = pywt.dwt2(channel, WAVELET)
    LL, (LH, HL, HH) = coeffs

    print(f"[apply_dwt] LL={LL.shape}  LH={LH.shape}  HL={HL.shape}  HH={HH.shape}")
    return coeffs, LL, LH, HL, HH


def apply_dct_blocks(subband: np.ndarray) -> np.ndarray:
    """
    Apply 2-D DCT to every non-overlapping 8×8 block in a sub-band.

    The 2-D DCT is computed as two successive 1-D DCTs (rows then
    columns), which is equivalent to scipy's dct with norm='ortho'.

    Parameters
    ----------
    subband : np.ndarray
        2-D float64 array whose dimensions are multiples of BLOCK_SIZE.

    Returns
    -------
    np.ndarray
        Same shape as input; each 8×8 tile replaced by its DCT coefficients.
    """
    h, w = subband.shape
    dct_subband = np.zeros_like(subband)

    for row in range(0, h, BLOCK_SIZE):
        for col in range(0, w, BLOCK_SIZE):
            block = subband[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
            # Apply DCT along rows, then columns (2-D DCT via separability)
            dct_block = dct(dct(block, norm='ortho', axis=0), norm='ortho', axis=1)
            dct_subband[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE] = dct_block

    return dct_subband


def apply_idct_blocks(dct_subband: np.ndarray) -> np.ndarray:
    """
    Apply 2-D IDCT to every 8×8 block (inverse of apply_dct_blocks).

    Parameters
    ----------
    dct_subband : np.ndarray
        2-D float64 array of DCT coefficients, blocked in 8×8 tiles.

    Returns
    -------
    np.ndarray
        Spatial-domain sub-band after reconstruction.
    """
    h, w = dct_subband.shape
    spatial = np.zeros_like(dct_subband)

    for row in range(0, h, BLOCK_SIZE):
        for col in range(0, w, BLOCK_SIZE):
            block = dct_subband[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE]
            # Inverse: IDCT along columns, then rows
            idct_block = idct(idct(block, norm='ortho', axis=1), norm='ortho', axis=0)
            spatial[row:row + BLOCK_SIZE, col:col + BLOCK_SIZE] = idct_block

    return spatial


def embed_watermark(
    Y_norm: np.ndarray,
    watermark_bits: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Embed watermark bits into the LH sub-band via DWT-DCT.

    Embedding rule (quantisation-based)
    ─────────────────────────────────────
    For each bit b ∈ {0, 1} and target coefficient c = dct_block[u, v]:

        magnitude = |c|
        quantised  = round(magnitude / alpha) * alpha

        if b == 1:
            new_magnitude = quantised + alpha/4   (positive offset → bit 1)
        else:
            new_magnitude = quantised - alpha/4   (negative offset → bit 0)

        new_c = new_magnitude * sign(c)    (preserve original sign for extraction)

    The quantisation approach is more robust than simple additive
    modification because it survives mild JPEG compression (which
    also operates on quantised DCT coefficients).

    Parameters
    ----------
    Y_norm : np.ndarray
        Float64 Y channel in [0, 1], shape (H, W).
    watermark_bits : np.ndarray
        1-D uint8 array of 0/1 watermark bits.
    alpha : float
        Embedding strength (quantisation step). Default = ALPHA.

    Returns
    -------
    tuple (Y_watermarked, watermark_shape)
        Y_watermarked  – watermarked Y channel as float64 [0,1]
        watermark_shape – (rows, cols) of the embedded watermark grid
                          needed during extraction to reshape the bits.
    """
    # ── Step 1: DWT decomposition ──────────────────────────────────
    coeffs, LL, LH, HL, HH = apply_dwt(Y_norm)

    # ── Step 2: Select LH sub-band (horizontal detail) ────────────
    # LH captures horizontal edges. JPEG compression is less
    # aggressive in this band than HH (diagonal), giving better
    # robustness while keeping the DC (LL) untouched.
    target_band = LH.copy()

    # ── Step 3: Compute embedding capacity ────────────────────────
    h_band, w_band = target_band.shape
    n_blocks_row = h_band // BLOCK_SIZE
    n_blocks_col = w_band // BLOCK_SIZE
    capacity = n_blocks_row * n_blocks_col

    n_bits = len(watermark_bits)
    if n_bits > capacity:
        raise ValueError(
            f"Watermark too large: {n_bits} bits > capacity {capacity} blocks. "
            f"Use prepare_watermark() with embed_capacity={capacity}."
        )

    # Record how the watermark bits map onto the block grid
    # (used later by the extractor)
    wm_rows = int(math.ceil(math.sqrt(n_bits * h_band / w_band)))
    wm_cols = int(math.ceil(n_bits / wm_rows))
    watermark_shape = (wm_rows, wm_cols)

    # ── Step 4: DCT on LH blocks ───────────────────────────────────
    dct_band = apply_dct_blocks(target_band)

    # ── Step 5: Embed bits into DCT coefficient (EMBED_U, EMBED_V) ─
    bit_idx = 0
    for row in range(n_blocks_row):
        for col in range(n_blocks_col):
            if bit_idx >= n_bits:
                break

            br = row * BLOCK_SIZE
            bc = col * BLOCK_SIZE

            coeff = dct_band[br + EMBED_U, bc + EMBED_V]
            bit   = int(watermark_bits[bit_idx])

            # Quantise the magnitude, then nudge by ±alpha/4
            magnitude  = abs(coeff)
            sign_coeff = np.sign(coeff) if coeff != 0 else 1.0
            quantised  = round(magnitude / alpha) * alpha

            if bit == 1:
                new_magnitude = quantised + alpha / 4.0
            else:
                new_magnitude = quantised - alpha / 4.0

            # Ensure magnitude stays positive
            new_magnitude = max(new_magnitude, alpha / 8.0)

            dct_band[br + EMBED_U, bc + EMBED_V] = sign_coeff * new_magnitude
            bit_idx += 1

        if bit_idx >= n_bits:
            break

    print(f"[embed_watermark] Embedded {bit_idx} bits into LH sub-band  alpha={alpha}")

    # ── Step 6: Inverse DCT on modified blocks ─────────────────────
    modified_LH = apply_idct_blocks(dct_band)

    # ── Step 7: Reconstruct sub-bands ─────────────────────────────
    new_coeffs = (LL, (modified_LH, HL, HH))

    # ── Step 8: Inverse DWT ────────────────────────────────────────
    Y_watermarked = pywt.idwt2(new_coeffs, WAVELET)

    # Clip to [0, 1] to handle any floating-point overshoot
    Y_watermarked = np.clip(Y_watermarked, 0.0, 1.0)

    return Y_watermarked, watermark_shape


def reconstruct_image(
    Y_watermarked: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
) -> np.ndarray:
    """
    Merge the watermarked Y channel with the original Cb / Cr channels
    and convert back to BGR.

    Parameters
    ----------
    Y_watermarked : np.ndarray
        Float64 Y channel in [0, 1].
    Cb, Cr : np.ndarray
        Uint8 chroma channels (unchanged from preprocess_image).

    Returns
    -------
    np.ndarray
        Watermarked image in BGR uint8 format.
    """
    # De-normalise Y back to [0, 255]
    Y_uint8 = np.clip(Y_watermarked * 255.0, 0, 255).astype(np.uint8)

    # Merge channels (OpenCV's YCrCb order: Y, Cr, Cb)
    ycbcr_watermarked = cv2.merge([Y_uint8, Cr, Cb])

    # Convert back to BGR
    bgr_watermarked = cv2.cvtColor(ycbcr_watermarked, cv2.COLOR_YCrCb2BGR)

    print(f"[reconstruct_image] Reconstructed BGR image shape={bgr_watermarked.shape}")
    return bgr_watermarked


# ─────────────────────────────────────────────────────────────────
# WATERMARK EXTRACTION (blind — no original image required)
# ─────────────────────────────────────────────────────────────────

def extract_watermark(
    watermarked_image: np.ndarray,
    n_bits: int,
    watermark_shape: tuple[int, int],
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Extract the embedded watermark from a watermarked image.

    Extraction is *blind*: the original host image is not needed
    because the quantisation step is deterministic.

    Extraction rule
    ───────────────
    For each block's coefficient c at position (EMBED_U, EMBED_V):

        magnitude = |c|
        quantised  = round(magnitude / alpha) * alpha
        remainder  = magnitude - quantised

        if remainder >= 0:  extracted bit = 1
        else:               extracted bit = 0

    Parameters
    ----------
    watermarked_image : np.ndarray
        BGR uint8 image.
    n_bits : int
        Number of watermark bits to extract.
    watermark_shape : tuple (rows, cols)
        Shape to reshape extracted bits back into a 2-D watermark image.
    alpha : float
        Must match the alpha used during embedding.

    Returns
    -------
    np.ndarray
        2-D uint8 array (0 / 255) representing the extracted watermark.
    """
    # Preprocess
    Y_norm, Cb, Cr, _ = preprocess_image(watermarked_image)

    # DWT
    _, LL, LH, HL, HH = apply_dwt(Y_norm)

    # DCT on LH
    dct_band = apply_dct_blocks(LH)

    # Extract bits
    h_band, w_band = LH.shape
    n_blocks_row = h_band // BLOCK_SIZE
    n_blocks_col = w_band // BLOCK_SIZE

    extracted_bits = []
    bit_idx = 0

    for row in range(n_blocks_row):
        for col in range(n_blocks_col):
            if bit_idx >= n_bits:
                break
            br = row * BLOCK_SIZE
            bc = col * BLOCK_SIZE

            coeff = dct_band[br + EMBED_U, bc + EMBED_V]
            magnitude = abs(coeff)
            quantised = round(magnitude / alpha) * alpha
            remainder = magnitude - quantised

            extracted_bits.append(1 if remainder >= 0 else 0)
            bit_idx += 1
        if bit_idx >= n_bits:
            break

    bits_array = np.array(extracted_bits, dtype=np.uint8)

    # Reshape to 2-D watermark image (pad with zeros if short)
    target_size = watermark_shape[0] * watermark_shape[1]
    if len(bits_array) < target_size:
        bits_array = np.pad(bits_array, (0, target_size - len(bits_array)))

    wm_2d = bits_array[:target_size].reshape(watermark_shape)
    wm_image = (wm_2d * 255).astype(np.uint8)

    print(f"[extract_watermark] Extracted {bit_idx} bits  shape={watermark_shape}")
    return wm_image


# ─────────────────────────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────────────────────────

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute PSNR between two images.

    PSNR = 20 * log10(255 / sqrt(MSE))

    A value > 38 dB indicates imperceptible embedding (project target).

    Parameters
    ----------
    original, processed : np.ndarray
        Images must have the same shape (or processed will be resized).

    Returns
    -------
    float
        PSNR in dB. Returns inf if images are identical.
    """
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    orig_f  = original.astype(np.float64)
    proc_f  = processed.astype(np.float64)
    mse     = np.mean((orig_f - proc_f) ** 2)

    if mse == 0:
        return float('inf')

    return 20.0 * np.log10(255.0 / np.sqrt(mse))


# ─────────────────────────────────────────────────────────────────
# HIGH-LEVEL PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────

def run_embedding_pipeline(
    image_path: str,
    watermark_input,
    output_path: str,
    alpha: float = ALPHA,
) -> dict:
    """
    End-to-end embedding pipeline (Module 1 + Module 2).

    Calls load_image → preprocess_image → prepare_watermark →
    apply_dwt → apply_dct_blocks → embed_watermark →
    reconstruct_image → save output → calculate_psnr.

    Parameters
    ----------
    image_path : str
        Path to the host image.
    watermark_input : str | np.ndarray
        Watermark image path or array.
    output_path : str
        Where to save the watermarked image.
    alpha : float
        Embedding strength.

    Returns
    -------
    dict with keys:
        'watermarked_image'  – np.ndarray (BGR)
        'psnr'               – float (dB)
        'n_bits'             – int
        'watermark_shape'    – tuple
        'capacity'           – int (max embeddable bits)
    """
    print("\n" + "=" * 60)
    print("  DWT-DCT Watermark Embedding Pipeline")
    print("=" * 60)

    # ── Module 1: Data Preparation ─────────────────────────────────
    print("\n[Module 1] Data Preparation")
    print("-" * 40)
    original  = load_image(image_path)
    Y_norm, Cb, Cr, _ = preprocess_image(original)

    # Compute capacity BEFORE preparing watermark
    _, LL, LH, HL, HH = apply_dwt(Y_norm)
    h_band, w_band = LH.shape
    capacity = (h_band // BLOCK_SIZE) * (w_band // BLOCK_SIZE)
    print(f"[Module 1] Embedding capacity = {capacity} bits")

    watermark_bits = prepare_watermark(watermark_input, capacity)

    # ── Module 2: Watermark Embedding ──────────────────────────────
    print("\n[Module 2] Watermark Embedding")
    print("-" * 40)
    Y_watermarked, watermark_shape = embed_watermark(Y_norm, watermark_bits, alpha)
    watermarked_image = reconstruct_image(Y_watermarked, Cb, Cr)

    # ── Output ─────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), watermarked_image)

    psnr_val = calculate_psnr(original[:Y_watermarked.shape[0], :Y_watermarked.shape[1]],
                               watermarked_image)

    print("\n" + "=" * 60)
    print("  Embedding Complete")
    print("=" * 60)
    print(f"  Output  : {out_path}")
    print(f"  PSNR    : {psnr_val:.2f} dB  {'✓ PASS (target >38 dB)' if psnr_val >= 38 else '✗ below target'}")
    print(f"  Bits    : {len(watermark_bits)} / {capacity}")
    print(f"  Alpha   : {alpha}")
    print("=" * 60 + "\n")

    return {
        "watermarked_image": watermarked_image,
        "psnr": psnr_val,
        "n_bits": len(watermark_bits),
        "watermark_shape": watermark_shape,
        "capacity": capacity,
    }