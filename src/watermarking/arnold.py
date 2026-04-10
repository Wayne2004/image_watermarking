"""
Arnold Transform (Cat Map) for Watermark Scrambling/Encryption

The Arnold Transform permutes pixel positions using the Cat Map:
    [x']   [1     p] [x]   [mod N]
    [y'] = [q   pq+1] [y] = [mod N]

Standard form (p=1, q=1):
    x' = (x + y)         mod N
    y' = (x + 2y)        mod N

Properties:
    - Periodic: after T iterations the image returns to its original state
    - The period T depends on N (image dimension)
    - Used as lightweight scrambling: even if extracted, watermark is
      meaningless noise without the correct number of iterations (the key)
"""

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────
# ARNOLD TRANSFORM HELPERS
# ─────────────────────────────────────────────────────────────────

def arnold_period(N: int, max_iter: int = 384) -> int:
    """
    Compute the Arnold Transform period for an N×N image.

    The period is the number of iterations after which the image
    returns to its original configuration.  We find it by iterating
    a single point until it cycles back.

    Parameters
    ----------
    N          : int   image dimension (assumed square)
    max_iter   : int   safety cap to prevent infinite loop

    Returns
    -------
    int        period T  (or max_iter if not found)
    """
    x, y = 1, 1  # track a non-origin point
    for t in range(1, max_iter + 1):
        x, y = (x + y) % N, (x + 2 * y) % N
        if x == 1 and y == 1:
            return t
    return max_iter


def arnold_transform(image: np.ndarray, iterations: int = 1) -> np.ndarray:
    """
    Apply the Arnold Cat Map to scramble a 2-D binary/grayscale image.

    Parameters
    ----------
    image       : 2-D np.ndarray  (square, uint8 or binary)
    iterations  : int   number of Cat Map iterations (the encryption key)

    Returns
    -------
    np.ndarray  scrambled image of same shape and dtype
    """
    if len(image.shape) != 2:
        raise ValueError(f"Arnold transform requires 2-D input, got {len(image.shape)}-D")

    N = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Arnold transform requires square input, got {image.shape}")
    if N < 2:
        return image.copy()

    scrambled = np.zeros_like(image)
    for x in range(N):
        for y in range(N):
            # Forward Cat Map: iterate 'iterations' times
            nx, ny = x, y
            for _ in range(iterations):
                nx, ny = (nx + ny) % N, (nx + 2 * ny) % N
            scrambled[nx, ny] = image[x, y]

    return scrambled


def inverse_arnold_transform(image: np.ndarray, iterations: int = 1,
                              period: int = None) -> np.ndarray:
    """
    Reverse the Arnold Cat Map to descramble a previously scrambled image.

    The inverse of T iterations is (period - T) forward iterations,
    or equivalently we can apply the inverse map directly:
        x = (2*x' - y')  mod N
        y = (-x' + y')   mod N

    Parameters
    ----------
    image       : 2-D np.ndarray  (square, uint8 or binary)
    iterations  : int   number of forward iterations that were used to scramble
    period      : int   pre-computed period (computed automatically if None)

    Returns
    -------
    np.ndarray  descrambled (original) image
    """
    if len(image.shape) != 2:
        raise ValueError(f"Inverse Arnold transform requires 2-D input, got {len(image.shape)}-D")

    N = image.shape[0]
    if image.shape[0] != image.shape[1]:
        raise ValueError(f"Inverse Arnold transform requires square input, got {image.shape}")
    if N < 2:
        return image.copy()

    if period is None:
        period = arnold_period(N)

    # Inverse = forward with (period - iterations) steps
    inverse_iters = (period - (iterations % period)) % period
    return arnold_transform(image, iterations=inverse_iters)


# ─────────────────────────────────────────────────────────────────
# NON-SQUARE IMAGE SUPPORT
# ─────────────────────────────────────────────────────────────────

def _make_square_size(bits: np.ndarray) -> tuple:
    """
    Compute the smallest square size that can hold the given bits,
    and the padding amount needed.

    Returns
    -------
    (side, padding)  where padding is number of zeros to append
    """
    n = len(bits)
    side = int(math.ceil(math.sqrt(n)))
    total = side * side
    padding = total - n
    return side, padding


def arnold_scramble_bits(watermark_bits: np.ndarray,
                          iterations: int = 5) -> tuple:
    """
    Scramble a 1-D watermark bit array using Arnold Transform.

    The bit array is reshaped into the smallest square grid, then
    the Arnold Cat Map is applied.

    Parameters
    ----------
    watermark_bits : 1-D np.ndarray of 0/1
    iterations     : int  scrambling key

    Returns
    -------
    (scrambled_1d, grid_shape, padding)
        scrambled_1d : 1-D scrambled bit array
        grid_shape   : (side, side) of the square grid
        padding      : number of trailing padding bits added
    """
    side, padding = _make_square_size(watermark_bits)

    # Pad to square
    if padding > 0:
        padded = np.pad(watermark_bits, (0, padding), mode='constant')
    else:
        padded = watermark_bits.copy()

    grid = padded.reshape((side, side))
    scrambled_grid = arnold_transform(grid, iterations=iterations)
    scrambled_1d = scrambled_grid.flatten()

    return scrambled_1d, (side, side), padding


def arnold_descramble_bits(scrambled_bits: np.ndarray,
                            grid_shape: tuple,
                            iterations: int = 5,
                            original_length: int = None,
                            padding: int = 0) -> np.ndarray:
    """
    Descramble a 1-D watermark bit array using Inverse Arnold Transform.

    Parameters
    ----------
    scrambled_bits : 1-D np.ndarray of 0/1 (scrambled)
    grid_shape     : (side, side) of the square grid used during scrambling
    iterations     : int  same key used during scrambling
    original_length: int  original bit count before padding (for trimming)
    padding        : int  number of padding bits to remove

    Returns
    -------
    np.ndarray  descrambled 1-D bit array
    """
    grid = scrambled_bits.reshape(grid_shape)
    descrambled_grid = inverse_arnold_transform(grid, iterations=iterations)
    descrambled_1d = descrambled_grid.flatten()

    if original_length is not None:
        return descrambled_1d[:original_length]
    elif padding > 0:
        return descrambled_1d[:-padding]
    return descrambled_1d
