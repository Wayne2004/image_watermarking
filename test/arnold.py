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
out_path = out_dir / "extracted-watermark.png"


from watermarking.watermarking import prepare_watermark
from watermarking.arnold import (
    arnold_scramble_bits, 
    arnold_descramble_bits,
    _make_square_size
)
from watermarking.extraction import *

def run_arnold_test(image_path, iterations=5):
    print(f"\n{'='*60}")
    print(f" ARNOLD SCRAMBLING UNIT TEST ")
    print(f"{'='*60}")

    # 1. Input: Load and prepare raw bits
    # We use a large capacity to ensure it doesn't resize the image too much
    # and we can see the original dimensions.
    raw_bits, original_shape = prepare_watermark(image_path, embed_capacity=999999)
    original_length = len(raw_bits)
    
    print(f"[Step 1] Image Loaded: {Path(image_path).name}")
    print(f"         Original Shape : {{row: {original_shape[0]}, col: {original_shape[1]}}}")
    print(f"         Total Bits     : {original_length}")

    # 2. Padding: Calculate square size and add padding
    side, padding_count = _make_square_size(raw_bits)
    embedded_shape = (side, side)
    
    # Explicitly create padded bits for inspection
    padded_bits = np.pad(raw_bits, (0, padding_count), mode='constant')
    
    print(f"\n[Step 2] Padding Added")
    print(f"         Embedded Shape : {{row: {side}, col: {side}}}")
    print(f"         Padding Bits   : {padding_count}")
    print(f"         New Total Bits : {len(padded_bits)}")

    # 3. Scrambling: Carry out Arnold Scrambling
    # We use the library function which also returns these values
    scrambled_bits, grid_shape, actual_padding = arnold_scramble_bits(raw_bits, iterations=iterations)
    
    print(f"\n[Step 3] Arnold Scrambling Complete")
    print(f"         Key (Iterations): {iterations}")
    print(f"         Scrambled Grid : {grid_shape}")

    # 4. Inverse: Carry out Inverse Arnold Scrambling
    # This recovers the full square (including padding)
    descrambled_square_bits = arnold_descramble_bits(
        scrambled_bits, 
        grid_shape=grid_shape, 
        iterations=iterations,
        original_length=None # Don't trim yet so we can see the padded result
    )
    
    print(f"\n[Step 4] Inverse Arnold Complete")
    print(f"         Square recovered (with padding)")

    # 5. Trimming: Remove padding and restore original shape
    final_trimmed_bits = descrambled_square_bits[:original_length]
    final_image = final_trimmed_bits.reshape(original_shape)
    
    print(f"\n[Step 5] Padding Removed & Reshaped")
    print(f"         Final Shape    : {final_image.shape}")
    
    # Verification
    match = np.array_equal(raw_bits, final_trimmed_bits)
    status = "SUCCESS" if match else "FAILED"
    print(f"\n{'='*60}")
    print(f" TEST RESULT: {status}")
    print(f"{'='*60}")

    # 6. debinarize: change 0 and 1 to 0 and 255 (black n white)
    final_image_debin = (final_image * 255).astype(np.uint8)

    # 6. Convert 2-D bit array into an image
    written = cv2.imwrite(str(out_path), final_image_debin)
    print('written extracted-watermark.png:',written)
    

    # 
    raw_image = raw_bits.reshape(original_shape)
    print('raw bits:',raw_bits)
    print('raw image:',raw_image)
    cv2.imwrite(str(out_dir / 'raw_watermark.png'), raw_image)


    # Save to variables for your inspection as requested
    results = {
        "raw_bits": raw_bits,
        "padded_bits": padded_bits,
        "scrambled_bits": scrambled_bits,
        "descrambled_square_bits": descrambled_square_bits,
        "final_trimmed_bits": final_trimmed_bits
    }
    return results

if __name__ == "__main__":
    # Use the mushroom image from assets as a test case
    test_image = PROJECT_ROOT / "assets" / "input_images" / "mushroom.png"
    
    if not test_image.exists():
        # Fallback to any image in input_images
        images = list((PROJECT_ROOT / "assets" / "input_images").glob("*.*"))
        if images:
            test_image = images[0]
        else:
            print("Error: No test images found in assets/input_images/")
            sys.exit(1)
            
    variables = run_arnold_test(str(test_image))

    print(variables["raw_bits"] == variables["final_trimmed_bits"])
    
    # You can now inspect 'variables["raw_bits"]', etc. if running interactively
    print("\nScript finished. All intermediate bit arrays are stored in the 'variables' dictionary.")
