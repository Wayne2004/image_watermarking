"""
Automated Evaluation Framework
Functions to measure watermark robustness and image quality
"""

from .evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    calculate_ncc,
    evaluate_watermark_robustness
)

__all__ = [
    'calculate_psnr',
    'calculate_ssim',
    'calculate_ber',
    'calculate_ncc',
    'evaluate_watermark_robustness'
]
