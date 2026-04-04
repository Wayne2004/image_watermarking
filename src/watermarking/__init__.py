"""
Watermarking package — Module 1 (Data Preparation) + Module 2 (Embedding)
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
    # Extraction (bonus)
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

__all__ = [
    "load_image",
    "preprocess_image",
    "prepare_watermark",
    "apply_dwt",
    "apply_dct_blocks",
    "apply_idct_blocks",
    "embed_watermark",
    "reconstruct_image",
    "extract_watermark",
    "calculate_psnr",
    "run_embedding_pipeline",
    "ALPHA",
    "BLOCK_SIZE",
    "WAVELET",
]