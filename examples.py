"""
Example Usage: Attack Simulation & Evaluation Framework
This script demonstrates how to use the attack and evaluation modules together
"""

import cv2
import numpy as np
from src.attacks import (
    jpeg_compression,
    gaussian_noise,
    blurring,
    cropping,
    scaling
)
from src.evaluation import (
    calculate_psnr,
    calculate_ssim,
    calculate_ber,
    evaluate_watermark_robustness
)


def example_attack_simulation():
    """
    Example 1: Demonstrate attack simulation
    """
    print("=" * 60)
    print("EXAMPLE 1: Attack Simulation")
    print("=" * 60)
    
    # Load a watermarked image
    # watermarked_img = cv2.imread('results/watermarked_image.jpg')
    # For testing, create a dummy image
    watermarked_img = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    print("\n1. JPEG Compression Attack")
    print("-" * 40)
    jpeg_attacked = jpeg_compression(watermarked_img, quality=60)
    print(f"   Input shape: {watermarked_img.shape}")
    print(f"   Output shape: {jpeg_attacked.shape}")
    print(f"   Quality factor: 60")
    
    print("\n2. Gaussian Noise Attack")
    print("-" * 40)
    noise_attacked = gaussian_noise(watermarked_img, sigma=15)
    print(f"   Noise std dev (sigma): 15")
    print(f"   Output shape: {noise_attacked.shape}")
    
    print("\n3. Blurring Attack")
    print("-" * 40)
    blur_attacked = blurring(watermarked_img, kernel_size=7, blur_type='gaussian')
    print(f"   Kernel size: 7x7")
    print(f"   Blur type: gaussian")
    print(f"   Output shape: {blur_attacked.shape}")
    
    print("\n4. Cropping Attack")
    print("-" * 40)
    crop_attacked = cropping(watermarked_img, percentage=0.1)
    print(f"   Crop percentage: 10% from edges")
    print(f"   Original shape: {watermarked_img.shape}")
    print(f"   Cropped shape: {crop_attacked.shape}")
    
    print("\n5. Scaling Attack")
    print("-" * 40)
    scale_attacked = scaling(watermarked_img, scale_factor=0.5)
    print(f"   Scale factor: 0.5 (50%)")
    print(f"   Original shape: {watermarked_img.shape}")
    print(f"   Scaled shape: {scale_attacked.shape}")


def example_evaluation_metrics():
    """
    Example 2: Demonstrate evaluation metrics
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Evaluation Metrics")
    print("=" * 60)
    
    # Create test images
    original_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    watermarked_img = original_img.copy()
    
    # Add some slight degradation to simulate watermarking
    watermarked_img = cv2.GaussianBlur(watermarked_img, (3, 3), 0)
    
    # Simulate attack
    attacked_img = jpeg_compression(watermarked_img, quality=70)
    
    print("\n1. PSNR (Peak Signal-to-Noise Ratio)")
    print("-" * 40)
    psnr_value = calculate_psnr(original_img, watermarked_img)
    print(f"   PSNR(Original vs Watermarked): {psnr_value:.2f} dB")
    
    psnr_attack = calculate_psnr(watermarked_img, attacked_img)
    print(f"   PSNR(Watermarked vs Attacked): {psnr_attack:.2f} dB")
    print(f"   Interpretation:")
    print(f"   - High PSNR (>30dB) = imperceptible changes")
    print(f"   - Low PSNR (<20dB) = visible degradation")
    
    print("\n2. SSIM (Structural Similarity Index)")
    print("-" * 40)
    ssim_value = calculate_ssim(original_img, watermarked_img)
    print(f"   SSIM(Original vs Watermarked): {ssim_value:.4f}")
    
    ssim_attack = calculate_ssim(watermarked_img, attacked_img)
    print(f"   SSIM(Watermarked vs Attacked): {ssim_attack:.4f}")
    print(f"   Interpretation:")
    print(f"   - SSIM = 1.0: Identical")
    print(f"   - SSIM > 0.9: High similarity")
    print(f"   - SSIM < 0.5: Low similarity")
    
    print("\n3. BER (Bit Error Rate)")
    print("-" * 40)
    # Create test watermarks
    original_watermark = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    extracted_watermark = original_watermark.copy()
    
    # Add some corruption to extracted watermark
    corruption = np.random.randint(0, 30, (64, 64), dtype=np.uint8)
    extracted_watermark = np.clip(extracted_watermark.astype(np.int32) - corruption, 0, 255).astype(np.uint8)
    
    ber_value = calculate_ber(original_watermark, extracted_watermark)
    print(f"   BER: {ber_value:.4f} ({ber_value*100:.2f}%)")
    print(f"   Interpretation:")
    print(f"   - BER = 0.0: Perfect extraction")
    print(f"   - BER < 0.05: Excellent robustness (0-5% errors)")
    print(f"   - BER > 0.1: Poor robustness (>10% errors)")


def example_comprehensive_evaluation():
    """
    Example 3: Comprehensive robustness evaluation
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Comprehensive Robustness Evaluation")
    print("=" * 60)
    
    # Create test data
    original_img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    watermarked_img = cv2.GaussianBlur(original_img, (3, 3), 0)
    attacked_img = jpeg_compression(watermarked_img, quality=65)
    
    original_wm = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    extracted_wm = original_wm.copy()
    noise = np.random.randint(0, 20, (128, 128), dtype=np.uint8)
    extracted_wm = np.clip(extracted_wm.astype(np.int32) - noise, 0, 255).astype(np.uint8)
    
    # Evaluate
    results = evaluate_watermark_robustness(
        original_img,
        watermarked_img,
        attacked_img,
        original_wm,
        extracted_wm
    )
    
    print("\nRobustness Report:")
    print("-" * 40)
    print(f"PSNR (Embedding):      {results['psnr_embedding']:.2f} dB")
    print(f"SSIM (Embedding):      {results['ssim_embedding']:.4f}")
    print(f"PSNR (After Attack):   {results['psnr_attack']:.2f} dB")
    print(f"SSIM (After Attack):   {results['ssim_attack']:.4f}")
    print(f"BER (Watermark):       {results['ber']:.4f} ({results['ber']*100:.2f}%)")
    
    print("\nInterpretation:")
    print("-" * 40)
    if results['psnr_embedding'] > 35:
        print("✓ Watermarking is imperceptible")
    else:
        print("✗ Watermarking causes noticeable degradation")
    
    if results['ber'] < 0.1:
        print("✓ Watermark extraction is robust to attack")
    else:
        print("✗ Watermark extraction is degraded by attack")


def example_attack_robustness_testing():
    """
    Example 4: Test robustness across multiple attack types
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Robustness Across Multiple Attacks")
    print("=" * 60)
    
    watermarked_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    attacks = [
        ("JPEG (Q=50)", lambda img: jpeg_compression(img, quality=50)),
        ("JPEG (Q=70)", lambda img: jpeg_compression(img, quality=70)),
        ("Gaussian Noise (σ=10)", lambda img: gaussian_noise(img, sigma=10)),
        ("Gaussian Noise (σ=20)", lambda img: gaussian_noise(img, sigma=20)),
        ("Blur (k=5)", lambda img: blurring(img, kernel_size=5)),
        ("Blur (k=9)", lambda img: blurring(img, kernel_size=9)),
        ("Crop (10%)", lambda img: cropping(img, percentage=0.1)),
        ("Crop (20%)", lambda img: cropping(img, percentage=0.2)),
        ("Scale (0.75)", lambda img: scaling(img, scale_factor=0.75)),
        ("Scale (0.5)", lambda img: scaling(img, scale_factor=0.5)),
    ]
    
    print("\nPSNR After Each Attack:")
    print("-" * 40)
    print(f"{'Attack':<25} {'PSNR (dB)':<12} {'Quality'}")
    print("-" * 40)
    
    for attack_name, attack_func in attacks:
        attacked_img = attack_func(watermarked_img)
        
        # Resize if needed for PSNR comparison
        if attacked_img.shape != watermarked_img.shape:
            attacked_img_resized = cv2.resize(attacked_img, 
                                             (watermarked_img.shape[1], watermarked_img.shape[0]))
            psnr_val = calculate_psnr(watermarked_img, attacked_img_resized)
        else:
            psnr_val = calculate_psnr(watermarked_img, attacked_img)
        
        # Quality assessment
        if psnr_val > 30:
            quality = "High"
        elif psnr_val > 20:
            quality = "Medium"
        else:
            quality = "Low"
        
        print(f"{attack_name:<25} {psnr_val:<12.2f} {quality}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WATERMARK ATTACK SIMULATION & EVALUATION FRAMEWORK")
    print("=" * 60)
    
    # Run all examples
    example_attack_simulation()
    example_evaluation_metrics()
    example_comprehensive_evaluation()
    example_attack_robustness_testing()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
