"""
Watermarking package — Module 1 (Data Preparation) + Module 2 (Embedding)
Algorithm: Hybrid DWT-DCT with Adaptive Alpha + Dual Sub-band Embedding
"""

from .watermarking import (
    # Module 1 — Data Preparation
    load_image,
    preprocess_image,
    prepare_watermark,
    # Module 2 — Watermark Embedding
    apply_dwt,
    apply_dct_blocks,
    apply_idct_blocks,
    embed_watermark,
    reconstruct_image,
    # Module 3 — Watermark Extraction
    extract_watermark,
    # Metrics
    calculate_psnr,
    # High-level pipeline
    run_embedding_pipeline,
    # Constants
    ALPHA,
    BLOCK_SIZE,
    WAVELET,
)

# Module 3: Extraction sub-module
from .extraction import (
    extract_watermark_raw,
    extract_watermark_robust,
    extract_watermark_batch,
    run_extraction_pipeline,
    descramble_watermark,
)

# Module 3: Arnold Transform
from .arnold import (
    arnold_transform,
    inverse_arnold_transform,
    arnold_period,
    arnold_scramble_bits,
    arnold_descramble_bits,
)

__all__ = [
    # Module 1
    "load_image",
    "preprocess_image",
    "prepare_watermark",
    # Module 2
    "apply_dwt",
    "apply_dct_blocks",
    "apply_idct_blocks",
    "embed_watermark",
    "reconstruct_image",
    # Module 3 — Extraction
    "extract_watermark",
    "extract_watermark_raw",
    "extract_watermark_robust",
    "extract_watermark_batch",
    "run_extraction_pipeline",
    "descramble_watermark",
    # Module 3 — Arnold Transform
    "arnold_transform",
    "inverse_arnold_transform",
    "arnold_period",
    "arnold_scramble_bits",
    "arnold_descramble_bits",
    # Metrics
    "calculate_psnr",
    # High-level pipeline
    "run_embedding_pipeline",
    # Constants
    "ALPHA",
    "BLOCK_SIZE",
    "WAVELET",
]