import os
import sys
import cv2
import numpy as np
import math
from pathlib import Path

# Add src to path so we can import our modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# define output directory for saving extracted watermark in
out_dir = PROJECT_ROOT / "test"
out_dir.mkdir(parents=True, exist_ok=True)  # create if missing
# out_path = out_dir / "extracted-watermark.png"

from watermarking.watermarking import prepare_watermark
from watermarking.arnold import (
    arnold_scramble_bits, 
    arnold_descramble_bits,
    _make_square_size
)

test_image_path = PROJECT_ROOT / "assets" / "input_images" / "mushroom.png"
test_image = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
# cv2.imshow('test image', test_image)
cv2.imwrite(str(out_dir / "test.png"), test_image)

