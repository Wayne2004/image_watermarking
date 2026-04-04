"""
Attack Simulation Functions
Functions to simulate real-world distortions on watermarked images
"""

import cv2
import numpy as np
from PIL import Image
import io


def jpeg_compression(image, quality=75):
    """
    Simulate JPEG compression attack (e.g., WhatsApp compression).
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR format for OpenCV, or grayscale)
    quality : int, optional
        JPEG quality factor (0-100, default=75)
        - Higher values = better quality, less compression
        - Lower values = more compression, more artifacts
    
    Returns:
    --------
    numpy.ndarray
        JPEG-compressed image in the same format as input
    
    Example:
    --------
    >>> import cv2
    >>> img = cv2.imread('watermarked_image.jpg')
    >>> attacked_img = jpeg_compression(img, quality=60)
    """
    # Ensure quality is within valid range
    quality = max(0, min(100, quality))
    
    # Encode image as JPEG with specified quality
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    # Decode back to numpy array
    attacked_image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    
    return attacked_image


def gaussian_noise(image, mean=0, sigma=10):
    """
    Add Gaussian noise to the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR, RGB, or grayscale)
    mean : float, optional
        Mean of the Gaussian noise (default=0)
    sigma : float, optional
        Standard deviation of the noise (default=10)
        - Typical range: 5-50 for visible corruption
        - sigma=10 adds ~3-5% noise
    
    Returns:
    --------
    numpy.ndarray
        Noisy image (clipped to valid range [0, 255])
    
    Example:
    --------
    >>> import cv2
    >>> img = cv2.imread('watermarked_image.jpg')
    >>> attacked_img = gaussian_noise(img, sigma=15)
    """
    # Generate Gaussian noise with same shape as image
    noise = np.random.normal(mean, sigma, image.shape)
    
    # Add noise to image
    noisy_image = image.astype(np.float32) + noise
    
    # Clip values to valid range [0, 255]
    attacked_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return attacked_image


def blurring(image, kernel_size=5, blur_type='gaussian'):
    """
    Apply blurring/filtering attack on the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR, RGB, or grayscale)
    kernel_size : int, optional
        Size of the blur kernel (must be odd, default=5)
        - Larger values = more blurring effect
        - Typical range: 3, 5, 7, 9, 11, ...
    blur_type : str, optional
        Type of blur filter (default='gaussian')
        - 'gaussian': Gaussian blur (most common)
        - 'median': Median filter (good for noise)
        - 'bilateral': Bilateral filter (preserves edges)
    
    Returns:
    --------
    numpy.ndarray
        Blurred image
    
    Example:
    --------
    >>> import cv2
    >>> img = cv2.imread('watermarked_image.jpg')
    >>> attacked_img = blurring(img, kernel_size=7, blur_type='gaussian')
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    if blur_type == 'gaussian':
        attacked_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif blur_type == 'median':
        attacked_image = cv2.medianBlur(image, kernel_size)
    
    elif blur_type == 'bilateral':
        attacked_image = cv2.bilateralFilter(image, kernel_size, 75, 75)
    
    else:
        raise ValueError(f"Unknown blur_type: {blur_type}. Choose from 'gaussian', 'median', 'bilateral'")
    
    return attacked_image


def cropping(image, crop_box=None, percentage=0.1):
    """
    Crop portions of the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR, RGB, or grayscale)
    crop_box : tuple, optional
        Crop region as (x_start, y_start, x_end, y_end)
        If None, crops from a corner based on percentage
    percentage : float, optional
        Percentage of image to crop from each edge (0-1, default=0.1)
        - 0.1 = 10% cropped from each side
        - Only used if crop_box is None
    
    Returns:
    --------
    numpy.ndarray
        Cropped image
    
    Example:
    --------
    >>> import cv2
    >>> img = cv2.imread('watermarked_image.jpg')
    >>> # Crop 10% from edges
    >>> attacked_img = cropping(img, percentage=0.1)
    >>> # Or specify exact crop region
    >>> attacked_img = cropping(img, crop_box=(10, 10, 200, 200))
    """
    height, width = image.shape[:2]
    
    if crop_box is None:
        # Calculate crop region based on percentage
        percentage = max(0, min(0.5, percentage))  # Limit to 0-50%
        x_crop = int(width * percentage)
        y_crop = int(height * percentage)
        
        # Crop from all sides
        attacked_image = image[y_crop:height-y_crop, x_crop:width-x_crop]
    else:
        # Use specified crop box
        x_start, y_start, x_end, y_end = crop_box
        x_start = max(0, min(x_start, width))
        x_end = max(x_start, min(x_end, width))
        y_start = max(0, min(y_start, height))
        y_end = max(y_start, min(y_end, height))
        
        attacked_image = image[y_start:y_end, x_start:x_end]
    
    return attacked_image


def scaling(image, scale_factor=0.5, interpolation='linear'):
    """
    Resize/scale the image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image (BGR, RGB, or grayscale)
    scale_factor : float, optional
        Scale factor (default=0.5)
        - < 1.0: downscaling (e.g., 0.5 = half size)
        - > 1.0: upscaling (e.g., 2.0 = double size)
    interpolation : str, optional
        Interpolation method (default='linear')
        - 'nearest': Nearest neighbor (fast, blocky)
        - 'linear': Bilinear interpolation
        - 'cubic': Bicubic interpolation (better quality)
        - 'lanczos': Lanczos interpolation (best quality)
    
    Returns:
    --------
    numpy.ndarray
        Scaled image
    
    Example:
    --------
    >>> import cv2
    >>> img = cv2.imread('watermarked_image.jpg')
    >>> # Downscale to 50%
    >>> attacked_img = scaling(img, scale_factor=0.5)
    >>> # Upscale to 150%
    >>> attacked_img = scaling(img, scale_factor=1.5)
    """
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Map interpolation type
    interp_methods = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    interp = interp_methods.get(interpolation, cv2.INTER_LINEAR)
    
    attacked_image = cv2.resize(image, (new_width, new_height), interpolation=interp)
    
    return attacked_image
