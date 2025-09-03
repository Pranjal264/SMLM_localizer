# crude_localization.py
# Contains functions for finding approximate integer-pixel locations.

import numpy as np
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max, blob_log
from scipy.ndimage import (maximum_filter, minimum_filter, label as ndimage_label, center_of_mass)


def find_crude_localizations(image, mask, method='Peak Local Max', **kwargs):
    """
    Finds crude localizations from a filtered image and a binary mask.

    Args:
        image (np.ndarray): The filtered grayscale image.
        mask (np.ndarray): The binary mask from the thresholding step.
        method (str): The localization method to use.

    Returns:
        np.ndarray: An array of shape (N, 2) with the (row, col) coordinates.
    """
    if method == 'Peak Local Max':
        min_distance = kwargs.get('min_distance', 5)
        # print("Finding crude localizations with Peak Local Max...")
        # We use the mask to search for peaks only in the regions of interest
        coords = peak_local_max(image, min_distance=min_distance, labels=mask)
        # print(f"Found {len(coords)} crude localizations.")
        return coords
        
    # elif method == 'Center of Mass':
    #     # print("Finding crude localizations with Center of Mass...")
    #     labeled_mask = label(mask)
    #     props = regionprops(labeled_mask)
    #     coords = np.array([prop.centroid for prop in props]).astype(int)
    #     # print(f"Found {len(coords)} crude localizations.")
    #     return coords
    elif method == 'Center of Mass':

        # Get the neighborhood_size from kwargs, with a sensible default
        neighborhood_size = kwargs.get('neighborhood_size', 3)
        data_max = maximum_filter(image, neighborhood_size)
        maxima = (image == data_max)
        
        valid_maxima = maxima & mask 

        labeled, num_objects = ndimage_label(valid_maxima)
        if num_objects == 0:
            return np.array([])

        coords = np.array(center_of_mass(
            image, labeled, range(1, num_objects + 1)))

        return coords.astype(int)
        
    elif method == 'Blob Detection (LoG)':
        # This method doesn't use the mask, it works on the filtered image directly
        min_sigma = kwargs.get('min_sigma', 1)
        max_sigma = kwargs.get('max_sigma', 3)
        threshold = kwargs.get('threshold', 5) 
        # print("Finding crude localizations with Blob Detection...")
        blobs = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        if blobs.shape[0] > 0:
            coords = blobs[:, :2].astype(int)
            # print(f"Found {len(coords)} crude localizations.")
            return coords
        else:
            return np.array([])
            
    else:
        raise ValueError(f"Unknown crude localization method: {method}")

