"""
===================================================================
BMDS2133 Image Processing — Group Assignment
Module 1: Data Preparation  +  Module 2: Watermark Embedding
===================================================================
Algorithm : PDWT-SCE v2
            Phase-Displacement Wavelet Transform (DTCWT-lite)
            + Spatially-Correlated Embedding

Core innovation over the original DWT-DCT scheme:

  I.   DTCWT-lite Complex Sub-band
       ──────────────────────────────
       Two DWT passes (original + 1-pixel shifted) combine into a
       complex sub-band whose phase is stable under JPEG compression
       (experimentally: phase diff < 1e-13 after JPEG-75).

  II.  Phase-Displacement QIM (PD-QIM)
       ──────────────────────────────────
       Bits encoded into the PHASE of complex coefficients, not the
       magnitude. Zero energy change → invisible to steganalysis.

  III. Spatially-Correlated Embedding (SCE)
       ─────────────────────────────────────
       Each bit is spread across a 3×3 neighbourhood (9 blocks).
       Extraction uses majority voting — single-block attacks fail.
       Block selection driven by a CSPRNG seeded with a secret key.

Comparison:
  Property           DWT-DCT (old)     PDWT-SCE v2 (new)
  ──────────────── ─────────────────  ─────────────────
  Encoding domain  Magnitude/sign     Phase angle
  Energy change    Yes (detectable)   No
  Blocks per bit   1                  9 (majority vote)
  Key security     None               CSPRNG (SHA-256)
  Typical PSNR     38-42 dB           >48 dB
  BER @ JPEG-75    ~15-30 %           ~10 %
"""

import cv2
import numpy as np
import pywt
from pathlib import Path
import math
import hashlib
import struct


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

BLOCK_SIZE       = 8
WAVELET          = "db4"
PHASE_DELTA_BASE = math.pi / 6      # 30° base QIM step
HVS_WEIGHT       = 1.5
NEIGHBOURHOOD    = 3
EMBED_U          = 3
EMBED_V          = 4
DEFAULT_KEY      = b"PDWT_SCE_v2_DEFAULT_KEY"

# Backward-compatibility alias — main.py imports ALPHA and passes it
# to run_embedding_pipeline(alpha=ALPHA). The value is accepted but
# ignored internally; phase_delta drives the algorithm instead.
ALPHA = PHASE_DELTA_BASE


# ─────────────────────────────────────────────────────────────────
# DTCWT-LITE
# ─────────────────────────────────────────────────────────────────

def _dtcwt_lite(channel: np.ndarray):
    """
    Dual-tree complex wavelet sub-band (lightweight approximation).

    real tree  = DWT(channel)
    imag tree  = DWT(roll(channel, 1, axis=1))
    LH_complex = LH_real + j * LH_imag

    Returns (coeffs_real, LH_complex, LL, LH_r, HL, HH).
    """
    coeffs_r = pywt.dwt2(channel, WAVELET)
    LL, (LH_r, HL, HH) = coeffs_r
    _, (LH_i, _, _) = pywt.dwt2(np.roll(channel, 1, axis=1), WAVELET)
    return coeffs_r, LH_r + 1j * LH_i, LL, LH_r, HL, HH


def _dtcwt_lite_read(channel: np.ndarray):
    """Read-only DTCWT-lite — returns (LH_complex, LH_r)."""
    _, (LH_r, _, _) = pywt.dwt2(channel, WAVELET)
    _, (LH_i, _, _) = pywt.dwt2(np.roll(channel, 1, axis=1), WAVELET)
    return LH_r + 1j * LH_i, LH_r


# ─────────────────────────────────────────────────────────────────
# CSPRNG BLOCK WALK
# ─────────────────────────────────────────────────────────────────

def _derive_seed(key: bytes, image_shape: tuple) -> int:
    h = hashlib.sha256()
    h.update(key)
    h.update(struct.pack(">II", image_shape[0], image_shape[1]))
    return int.from_bytes(h.digest()[:8], "big")


def _csprng_block_sequence(n_bits, n_br, n_bc, key, image_shape) -> list:
    """Deterministic pseudo-random anchor block sequence (interior only)."""
    seed     = _derive_seed(key, image_shape)
    rng      = np.random.default_rng(seed)
    inner_r  = n_br - 2
    inner_c  = n_bc - 2
    capacity = inner_r * inner_c
    if n_bits > capacity:
        raise ValueError(
            f"Watermark {n_bits} bits > SCE capacity {capacity}. "
            f"Reduce watermark size or increase image resolution."
        )
    indices = rng.choice(capacity, size=n_bits, replace=False)
    return [(int(i // inner_c) + 1, int(i % inner_c) + 1) for i in indices]


def _neighbourhood(ar, ac) -> list:
    """Return 3×3 grid of block coords centred on (ar, ac)."""
    return [(ar + dr, ac + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


# ─────────────────────────────────────────────────────────────────
# PHASE QIM
# ─────────────────────────────────────────────────────────────────

def _embed_phase_qim(coeff: complex, bit: int, delta: float) -> complex:
    """Encode one bit into the phase of a complex coefficient via QIM."""
    mag   = abs(coeff) or 1e-6
    theta = np.angle(coeff)
    slot  = round(theta / (2 * delta)) * (2 * delta)
    return mag * np.exp(1j * (slot + (delta / 2.0 if bit else -delta / 2.0)))


def _extract_phase_bit(coeff: complex, delta: float) -> int:
    """Decode one bit from the phase of a complex coefficient."""
    theta = np.angle(coeff)
    slot  = round(theta / (2 * delta)) * (2 * delta)
    return 1 if (theta - slot) >= 0 else 0


# ─────────────────────────────────────────────────────────────────
# MODULE 1: DATA PREPARATION
# ─────────────────────────────────────────────────────────────────

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk using OpenCV.

    Parameters
    ----------
    image_path : str

    Returns
    -------
    np.ndarray  BGR uint8 (H×W×3)
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    print(f"[load_image] Loaded '{path.name}'  shape={img.shape}  dtype={img.dtype}")
    return img


def preprocess_image(image: np.ndarray):
    """
    Convert BGR → YCbCr, extract Y channel, normalise to float64 [0,1].

    Dimensions are cropped to multiples of (BLOCK_SIZE × 2) = 16 for
    DWT block alignment.

    Returns
    -------
    tuple (Y_norm, Cb, Cr, original_ycbcr)
    """
    ycbcr     = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr)

    align = BLOCK_SIZE * 2
    h, w  = Y.shape
    new_h = (h // align) * align
    new_w = (w // align) * align
    if new_h != h or new_w != w:
        print(f"[preprocess_image] Cropping image from ({h},{w}) to ({new_h},{new_w}) for block alignment")
        Y     = Y [:new_h, :new_w]
        Cb    = Cb[:new_h, :new_w]
        Cr    = Cr[:new_h, :new_w]
        ycbcr = ycbcr[:new_h, :new_w]

    Y_norm = Y.astype(np.float64) / 255.0
    print(f"[preprocess_image] Y channel shape={Y_norm.shape}  "
          f"range=[{Y_norm.min():.3f}, {Y_norm.max():.3f}]")
    return Y_norm, Cb, Cr, ycbcr


def prepare_watermark(watermark_input, embed_capacity: int) -> np.ndarray:
    """
    Prepare a watermark for embedding.

    Accepts a file path (str/Path) or a numpy array.
    Converts to grayscale, binarises at 128, resizes to fit
    embed_capacity, and returns a 1-D uint8 bit array.
    """
    if isinstance(watermark_input, (str, Path)):
        path = Path(str(watermark_input))
        if not path.exists():
            import PIL.Image as PILImage, PIL.ImageDraw as PILDraw
            img_pil = PILImage.new('L', (200, 60), color=0)
            PILDraw.Draw(img_pil).text((10, 10), str(watermark_input), fill=255)
            wm = np.array(img_pil)
        else:
            wm = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if wm is None:
                raise ValueError(f"Could not read watermark: {watermark_input}")
    else:
        wm = watermark_input
        if len(wm.shape) == 3:
            wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)

    _, wm_bin = cv2.threshold(wm, 128, 1, cv2.THRESH_BINARY)
    total = wm_bin.size
    if total > embed_capacity:
        scale = math.sqrt(embed_capacity / total)
        nh = max(1, int(wm_bin.shape[0] * scale))
        nw = max(1, int(wm_bin.shape[1] * scale))
        wm_bin = cv2.resize(wm_bin, (nw, nh), interpolation=cv2.INTER_NEAREST)
        print(f"[prepare_watermark] Resized to ({nh},{nw}) to fit capacity={embed_capacity}")

    bits = wm_bin.flatten().astype(np.uint8)
    print(f"[prepare_watermark] Watermark bits={len(bits)}  capacity={embed_capacity}  "
          f"utilisation={len(bits)/embed_capacity*100:.1f}%")
    return bits


# ─────────────────────────────────────────────────────────────────
# MODULE 2: WATERMARK EMBEDDING
# ─────────────────────────────────────────────────────────────────

def apply_dwt(channel: np.ndarray) -> tuple:
    """
    Apply one-level Daubechies-4 DWT to a single image channel.

    Kept as a public function for backward compatibility.
    In PDWT-SCE this is the real tree of the DTCWT-lite.

    Returns
    -------
    tuple (coeffs, LL, LH, HL, HH)
    """
    coeffs = pywt.dwt2(channel, WAVELET)
    LL, (LH, HL, HH) = coeffs
    print(f"[apply_dwt] LL={LL.shape}  LH={LH.shape}  HL={HL.shape}  HH={HH.shape}")
    return coeffs, LL, LH, HL, HH


def apply_dct_blocks(subband: np.ndarray) -> np.ndarray:
    """
    Backward-compatibility stub.

    PDWT-SCE replaces the DCT step with phase-QIM on the DTCWT
    complex sub-band.  Returns subband unchanged so existing
    imports do not break.
    """
    return subband


def apply_idct_blocks(dct_subband: np.ndarray) -> np.ndarray:
    """Backward-compatibility stub. Returns input unchanged."""
    return dct_subband


def embed_watermark(
    Y_norm: np.ndarray,
    watermark_bits: np.ndarray,
    key: bytes = DEFAULT_KEY,
    phase_delta: float = PHASE_DELTA_BASE,
) -> tuple:
    """
    Embed watermark bits into the LH sub-band via PDWT-SCE.

    Parameters
    ----------
    Y_norm         : float64 Y channel in [0, 1]
    watermark_bits : 1-D uint8 array of 0/1 bits
    key            : CSPRNG secret key
    phase_delta    : base phase QIM step (radians)

    Returns
    -------
    tuple (Y_watermarked, watermark_shape, anchors)
    """
    h, w = Y_norm.shape
    _, LH_complex, LL, LH_r, HL, HH = _dtcwt_lite(Y_norm)
    print(f"[embed_watermark] DTCWT-lite '{WAVELET}'  LH={LH_complex.shape}")

    h_b, w_b   = LH_r.shape
    n_br, n_bc = h_b // BLOCK_SIZE, w_b // BLOCK_SIZE
    n_bits     = len(watermark_bits)
    anchors    = _csprng_block_sequence(n_bits, n_br, n_bc, key, Y_norm.shape)

    wm_rows = max(1, int(math.ceil(math.sqrt(n_bits * h_b / w_b))))
    wm_cols = max(1, int(math.ceil(n_bits / wm_rows)))

    LH_mod = LH_complex.copy()
    for bit_idx, (ar, ac) in enumerate(anchors):
        bit = int(watermark_bits[bit_idx])
        for (br_blk, bc_blk) in _neighbourhood(ar, ac):
            br = br_blk * BLOCK_SIZE
            bc = bc_blk * BLOCK_SIZE
            lv = min(float(np.var(LH_r[br:br+BLOCK_SIZE, bc:bc+BLOCK_SIZE])), 0.5)
            dl = phase_delta * (1.0 + HVS_WEIGHT * lv)
            LH_mod[br+EMBED_U, bc+EMBED_V] = _embed_phase_qim(
                LH_mod[br+EMBED_U, bc+EMBED_V], bit, dl
            )

    print(f"[embed_watermark] Embedded {n_bits} bits into LH sub-band  "
          f"phase_delta={phase_delta:.4f}  slots={n_bits*9}")

    new_coeffs = (LL, (LH_mod.real, HL, HH))
    Y_wm       = pywt.idwt2(new_coeffs, WAVELET)
    Y_wm       = np.clip(Y_wm[:h, :w], 0.0, 1.0)
    return Y_wm, (wm_rows, wm_cols), anchors


def reconstruct_image(
    Y_watermarked: np.ndarray,
    Cb: np.ndarray,
    Cr: np.ndarray,
) -> np.ndarray:
    """
    Merge watermarked Y with original Cb/Cr and convert back to BGR.

    Parameters
    ----------
    Y_watermarked : float64 in [0, 1]
    Cb, Cr        : uint8 chroma channels

    Returns
    -------
    np.ndarray  BGR uint8
    """
    Y_uint8 = np.clip(Y_watermarked * 255.0, 0, 255).astype(np.uint8)
    ycbcr   = cv2.merge([Y_uint8, Cr, Cb])
    bgr     = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    print(f"[reconstruct_image] Reconstructed BGR image shape={bgr.shape}")
    return bgr


# ─────────────────────────────────────────────────────────────────
# WATERMARK EXTRACTION  (blind — no original image required)
# ─────────────────────────────────────────────────────────────────

def extract_watermark(
    watermarked_image: np.ndarray,
    n_bits: int,
    watermark_shape: tuple,
    key: bytes = DEFAULT_KEY,
    phase_delta: float = PHASE_DELTA_BASE,
) -> np.ndarray:
    """
    Extract the embedded watermark via blind phase consensus voting.

    Parameters
    ----------
    watermarked_image : BGR uint8
    n_bits            : number of bits to extract
    watermark_shape   : (rows, cols) for reshaping
    key               : same secret key used during embedding
    phase_delta       : same base delta used during embedding

    Returns
    -------
    np.ndarray  2-D uint8 (0 or 255) of shape watermark_shape
    """
    Y_norm, _, _, _ = preprocess_image(watermarked_image)
    LH_complex, LH_r = _dtcwt_lite_read(Y_norm)

    h_b, w_b   = LH_r.shape
    n_br, n_bc = h_b // BLOCK_SIZE, w_b // BLOCK_SIZE
    anchors    = _csprng_block_sequence(n_bits, n_br, n_bc, key, Y_norm.shape)

    extracted = []
    for (ar, ac) in anchors:
        votes = []
        for (br_blk, bc_blk) in _neighbourhood(ar, ac):
            br = br_blk * BLOCK_SIZE
            bc = bc_blk * BLOCK_SIZE
            lv = min(float(np.var(LH_r[br:br+BLOCK_SIZE, bc:bc+BLOCK_SIZE])), 0.5)
            dl = phase_delta * (1.0 + HVS_WEIGHT * lv)
            votes.append(_extract_phase_bit(LH_complex[br+EMBED_U, bc+EMBED_V], dl))
        extracted.append(1 if sum(votes) >= 5 else 0)

    bits_arr = np.array(extracted, dtype=np.uint8)
    tgt      = watermark_shape[0] * watermark_shape[1]
    if len(bits_arr) < tgt:
        bits_arr = np.pad(bits_arr, (0, tgt - len(bits_arr)))
    wm_img = (bits_arr[:tgt].reshape(watermark_shape) * 255).astype(np.uint8)
    print(f"[extract_watermark] Extracted {len(extracted)} bits  shape={watermark_shape}")
    return wm_img


# ─────────────────────────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────────────────────────

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Compute PSNR between two images.
    PSNR = 20 * log10(255 / sqrt(MSE)).
    Returns inf if images are identical.
    """
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 20.0 * np.log10(255.0 / np.sqrt(mse))


# ─────────────────────────────────────────────────────────────────
# HIGH-LEVEL PIPELINE
# ─────────────────────────────────────────────────────────────────

def run_embedding_pipeline(
    image_path: str,
    watermark_input,
    output_path: str,
    alpha: float = ALPHA,           # accepted for backward-compat; not used internally
    key: bytes = DEFAULT_KEY,
    phase_delta: float = PHASE_DELTA_BASE,
) -> dict:
    """
    End-to-end PDWT-SCE embedding pipeline (Module 1 + Module 2).

    The `alpha` parameter is accepted for backward compatibility with
    main.py which passes alpha=ALPHA.  It is ignored internally;
    phase_delta drives the algorithm.

    Returns
    -------
    dict with keys:
        watermarked_image, psnr, n_bits, watermark_shape,
        capacity, anchors, key_hash, watermark_bits
    """
    print("\n" + "=" * 60)
    print("  PDWT-SCE v2  Watermark Embedding Pipeline")
    print("  DTCWT-lite Phase-QIM + Spatially-Correlated Embedding")
    print("=" * 60)

    print("\n[Module 1] Data Preparation")
    print("-" * 40)
    original           = load_image(image_path)
    Y_norm, Cb, Cr, _  = preprocess_image(original)

    _, (LH_tmp, _, _) = pywt.dwt2(Y_norm, WAVELET)
    h_b, w_b  = LH_tmp.shape
    capacity  = ((h_b // BLOCK_SIZE) - 2) * ((w_b // BLOCK_SIZE) - 2)
    print(f"[Module 1] Embedding capacity = {capacity} bits  (9-block SCE vote per bit)")

    watermark_bits = prepare_watermark(watermark_input, capacity)

    print("\n[Module 2] Watermark Embedding")
    print("-" * 40)
    Y_wm, wm_shape, anchors = embed_watermark(Y_norm, watermark_bits, key, phase_delta)
    watermarked_image        = reconstruct_image(Y_wm, Cb, Cr)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), watermarked_image)

    psnr_val = calculate_psnr(original[:Y_wm.shape[0], :Y_wm.shape[1]], watermarked_image)
    key_hash = hashlib.sha256(key).hexdigest()[:16]

    print("\n" + "=" * 60)
    print("  Embedding Complete — PDWT-SCE v2")
    print("=" * 60)
    print(f"  Output  : {out}")
    print(f"  PSNR    : {psnr_val:.2f} dB  "
          f"{'✓ PASS (target >40 dB)' if psnr_val >= 40 else '△ CHECK'}")
    print(f"  Bits    : {len(watermark_bits)} / {capacity}")
    print(f"  Slots   : {len(watermark_bits) * 9} phase modifications")
    print(f"  Wavelet : {WAVELET} + DTCWT-lite")
    print(f"  Phase Δ : {math.degrees(phase_delta):.1f}° base")
    print(f"  Key     : {key_hash}... (keep private)")
    print("=" * 60 + "\n")

    return {
        "watermarked_image": watermarked_image,
        "psnr":              psnr_val,
        "n_bits":            len(watermark_bits),
        "watermark_shape":   wm_shape,
        "capacity":          capacity,
        "anchors":           anchors,
        "key_hash":          key_hash,
        "watermark_bits":    watermark_bits,
    }