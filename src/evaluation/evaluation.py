"""
Evaluation Functions
Functions to calculate image quality and watermark robustness metrics
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def calculate_psnr(original_image, attacked_image):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Measures the quality of the attacked image compared to the original.
    Higher PSNR indicates less degradation.
    
    Formula: PSNR = 20 * log10(MAX_I / sqrt(MSE))
    where MAX_I = 255 for 8-bit images
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original/reference image (before attack)
    attacked_image : numpy.ndarray
        Attacked/degraded image
    
    Returns:
    --------
    float
        PSNR value in dB
        - > 50 dB: Imperceptible difference
        - 30-50 dB: Very good quality
        - 20-30 dB: Acceptable quality
        - < 20 dB: Poor quality
    
    Example:
    --------
    >>> import cv2
    >>> original = cv2.imread('original.jpg')
    >>> attacked = cv2.imread('attacked.jpg')
    >>> psnr_value = calculate_psnr(original, attacked)
    >>> print(f"PSNR: {psnr_value:.2f} dB")
    """
    # Convert to same shape if needed
    if original_image.shape != attacked_image.shape:
        attacked_image = cv2.resize(attacked_image, 
                                   (original_image.shape[1], original_image.shape[0]))
    
    # Convert to float for calculation
    original = original_image.astype(np.float64)
    attacked = attacked_image.astype(np.float64)
    
    # Calculate Mean Squared Error
    mse = np.mean((original - attacked) ** 2)
    
    # Handle edge case where images are identical
    if mse == 0:
        return float('inf')
    
    # MAX_I for 8-bit image is 255
    max_pixel_value = 255.0
    
    # Calculate PSNR
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr_value


def calculate_ssim(original_image, attacked_image, data_range=255):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Measures the structural similarity between original and attacked images.
    Considers luminance, contrast, and structure similarity.
    
    Formula: SSIM = (2*μ_x*μ_y + c1)(2*σ_xy + c2) / ((μ_x²+μ_y²+c1)(σ_x²+σ_y²+c2))
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original/reference image
    attacked_image : numpy.ndarray
        Attacked/degraded image
    data_range : int, optional
        Data range of the input image (default=255 for 8-bit)
    
    Returns:
    --------
    float
        SSIM value (-1 to 1)
        - 1.0: Identical images
        - 0.5-1.0: High similarity
        - 0.0-0.5: Moderate to low similarity
        - Negative: Very dissimilar
    
    Example:
    --------
    >>> import cv2
    >>> original = cv2.imread('original.jpg')
    >>> attacked = cv2.imread('attacked.jpg')
    >>> ssim_value = calculate_ssim(original, attacked)
    >>> print(f"SSIM: {ssim_value:.4f}")
    """
    # Convert to same shape if needed
    if original_image.shape != attacked_image.shape:
        attacked_image = cv2.resize(attacked_image,
                                   (original_image.shape[1], original_image.shape[0]))
    
    # Handle multi-channel images (e.g., BGR)
    if len(original_image.shape) == 3:
        # Calculate SSIM for each channel and average
        ssim_values = []
        for channel in range(original_image.shape[2]):
            ssim_val = ssim(original_image[:, :, channel],
                          attacked_image[:, :, channel],
                          data_range=data_range)
            ssim_values.append(ssim_val)
        ssim_value = np.mean(ssim_values)
    else:
        # Grayscale image
        ssim_value = ssim(original_image, attacked_image, data_range=data_range)
    
    return ssim_value


def calculate_ber(original_watermark, extracted_watermark):
    """
    Calculate Bit Error Rate (BER).
    
    Measures the accuracy of watermark extraction after attacks.
    Lower BER indicates better robustness.
    
    Formula: BER = (number of bit errors) / (total number of bits)
    
    Parameters:
    -----------
    original_watermark : numpy.ndarray
        Original watermark (binary or grayscale)
    extracted_watermark : numpy.ndarray
        Extracted watermark after attack
    
    Returns:
    --------
    float
        BER value (0 to 1)
        - 0.0: Perfect extraction (no errors)
        - 0.01-0.05: Excellent robustness (1-5% errors)
        - 0.05-0.1: Good robustness (5-10% errors)
        - > 0.1: Poor robustness
    
    Example:
    --------
    >>> import cv2
    >>> original_wm = cv2.imread('watermark.jpg', 0)
    >>> extracted_wm = cv2.imread('extracted_watermark.jpg', 0)
    >>> ber = calculate_ber(original_wm, extracted_wm)
    >>> print(f"BER: {ber:.4f} ({ber*100:.2f}%)")
    """
    # Ensure same shape
    if original_watermark.shape != extracted_watermark.shape:
        extracted_watermark = cv2.resize(extracted_watermark,
                                        (original_watermark.shape[1], original_watermark.shape[0]))
    
    # Convert to binary (threshold at 128)
    _, original_binary = cv2.threshold(original_watermark, 128, 1, cv2.THRESH_BINARY)
    _, extracted_binary = cv2.threshold(extracted_watermark, 128, 1, cv2.THRESH_BINARY)
    
    # Flatten arrays for comparison
    original_bits = original_binary.flatten()
    extracted_bits = extracted_binary.flatten()
    
    # Count bit errors
    bit_errors = np.sum(original_bits != extracted_bits)
    total_bits = original_bits.size
    
    # Calculate BER
    ber = bit_errors / total_bits
    
    return ber


def evaluate_watermark_robustness(original_image, 
                                  watermarked_image,
                                  attacked_image,
                                  original_watermark,
                                  extracted_watermark):
    """
    Comprehensive evaluation of watermark robustness.
    
    Calculates PSNR, SSIM, and BER in a single call and returns
    a dictionary with all metrics.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original image before watermarking
    watermarked_image : numpy.ndarray
        Image after watermark embedding
    attacked_image : numpy.ndarray
        Image after attack simulation
    original_watermark : numpy.ndarray
        Original watermark
    extracted_watermark : numpy.ndarray
        Extracted watermark from attacked image
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'psnr_embedding': PSNR between original and watermarked images
        - 'ssim_embedding': SSIM between original and watermarked images
        - 'psnr_attack': PSNR between watermarked and attacked images
        - 'ssim_attack': SSIM between watermarked and attacked images
        - 'ber': Bit error rate of extracted watermark
    
    Example:
    --------
    >>> results = evaluate_watermark_robustness(
    ...     original_img, watermarked_img, attacked_img,
    ...     original_wm, extracted_wm
    ... )
    >>> print(f"Embedding PSNR: {results['psnr_embedding']:.2f} dB")
    >>> print(f"BER after attack: {results['ber']:.4f}")
    """
    results = {
        'psnr_embedding': calculate_psnr(original_image, watermarked_image),
        'ssim_embedding': calculate_ssim(original_image, watermarked_image),
        'psnr_attack': calculate_psnr(watermarked_image, attacked_image),
        'ssim_attack': calculate_ssim(watermarked_image, attacked_image),
        'ber': calculate_ber(original_watermark, extracted_watermark)
    }
    
    return results
