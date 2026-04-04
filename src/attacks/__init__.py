"""
Attack Simulation Framework
Simulates real-world distortions to test watermark robustness
"""

from .attacks import (
    jpeg_compression,
    gaussian_noise,
    blurring,
    cropping,
    scaling
)

__all__ = [
    'jpeg_compression',
    'gaussian_noise',
    'blurring',
    'cropping',
    'scaling'
]
