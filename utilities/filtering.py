# filtering.py
# Contains functions for advanced image pre-processing.

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, gaussian_laplace

def apply_filter(image, method='Gaussian', **kwargs):
    """
    Applies a filter to the input image.

    Args:
        image (np.ndarray): The input image.
        method (str): The filtering method to use.
        **kwargs: Filter-specific parameters (e.g., sigma).

    Returns:
        np.ndarray: The filtered image.
    """
    img_float = image.astype(np.float32)
    
    if method == 'Gaussian':
        sigma = kwargs.get('sigma', 1.0)
        # print(f"Applying Gaussian filter with sigma={sigma}...")
        return gaussian_filter(img_float, sigma=sigma)
        
    elif method == 'Mean':
        size = kwargs.get('size', 3)
        # print(f"Applying Mean filter with size={size}...")
        return uniform_filter(img_float, size=size)
        
    elif method == 'Laplacian of Gaussian (LoG)':
        sigma = kwargs.get('sigma', 1.0)
        # print(f"Applying LoG filter with sigma={sigma}...")
        # LoG enhances blob-like structures. We take the absolute value.
        return np.abs(gaussian_laplace(img_float, sigma=sigma))
        
    elif method == 'Difference of Gaussians (DoG)':
        low_sigma = kwargs.get('low_sigma', 1.0)
        high_sigma = kwargs.get('high_sigma', 2.5 * low_sigma)
        # print(f"Applying DoG filter with sigmas=({low_sigma}, {high_sigma})...")
        diff = gaussian_filter(img_float, low_sigma) - gaussian_filter(img_float, high_sigma)
        if diff.min() < 0:
            diff -= diff.min()
        return diff
        
    else:
        # print("No filter applied.")
        return img_float

