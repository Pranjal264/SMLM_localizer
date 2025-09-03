# thresholding.py
# Contains functions for image thresholding.


import numpy as np
from skimage.filters import threshold_otsu

def custom_threshold_optimized(image, alpha=0.01):
    """
    Calculate the optimal threshold using a vectorized custom entropy-based method.

    Parameters:
    - image (numpy.ndarray): Input grayscale image. Must be of an integer type.
    - alpha (float): Tolerance parameter for finding the threshold.

    Returns:
    - optimal_threshold (int): Optimal threshold value for image segmentation.
    """
    # Ensure the image is not empty
    if image.size == 0:
        return 0

    # 1. Use np.bincount for a much faster histogram calculation.
    image = image.astype(np.int32)  # Ensure integer type for bincount
    max_val = int(np.max(image))
    hist = np.bincount(image.ravel(), minlength=max_val + 1)

    # Normalize histogram to get probabilities
    hist_norm = hist / hist.sum()

    # Vectorize the entropy calculation
    # Calculate the cumulative distribution function (CDF)
    P1 = np.cumsum(hist_norm)

    # We calculate entropy for thresholds from 0 to max_val-1
    threshold_values = np.arange(max_val)
    P1 = P1[threshold_values]
    
    # Probability of the second class (foreground)
    # Using np.maximum to avoid floating point inaccuracies close to 1.0
    P2 = np.maximum(1.0 - P1, 0)

    # Simplify the custom entropy formula and compute it for all thresholds at once.
    log_arg1 = np.where(P1 > 0, P1, 1)
    log_arg2 = np.where(P2 > 0, P2, 1)
    
    entropy_values = -P1 * np.log(log_arg1) - P2 * np.log(log_arg2)
    
    if entropy_values.size == 0:
        return 0 # Handles images with max value of 0

    # Find the optimal threshold from the entropy values.
    mean_entropy = np.mean(entropy_values)
    
    # np.flatnonzero is often slightly faster than np.where(...)[0]
    candidate_indices = np.flatnonzero(np.abs(entropy_values - mean_entropy) < alpha)

    # Add a fallback for cases where no threshold meets the criteria.
    if candidate_indices.size == 0:
        return np.argmin(np.abs(entropy_values - mean_entropy))

    # Return the last threshold that satisfies the condition, as in the original code.
    return candidate_indices[-1]

def apply_threshold(image, method='Otsu', **kwargs):
    """
    Applies a threshold to create a binary mask.

    Args:
        image (np.ndarray): The input (filtered) image.
        method (str): The thresholding method to use.
        **kwargs: Method-specific parameters.

    Returns:
        np.ndarray: A binary (boolean) image mask.
    """
    if method == 'Otsu':
        # print("Applying Otsu's global threshold...")
        # Otsu can fail on images with no variance, so we add a check
        if image.min() == image.max():
            return np.zeros(image.shape, dtype=bool)
        thresh = threshold_otsu(image)
        return image > thresh
        
    elif method == 'Adaptive':
        # print("Applying adaptive local threshold...")
        thresh = custom_threshold_optimized(image, alpha=0.01)
        return image > thresh
        
    elif method == 'Manual':
        threshold_value = kwargs.get('threshold_value', 128)
        # print(f"Applying manual threshold at value={threshold_value}...")
        return image > threshold_value
        
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

