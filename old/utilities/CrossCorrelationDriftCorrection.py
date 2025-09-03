import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from multiprocessing import Pool


def minmax_8bit(image):
    return ((image - image.min()) / (image.max() - image.min()) * 255.).astype(np.uint8)


def white_tophat_opencv(image, kernel_size=(15, 15)):
    # Create a structuring element (you can adjust the size and shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply the white top-hat transformation using OpenCV
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    return tophat


def gaussian_2d(xy, amp, xo, yo, sigma_x, sigma_y, offset):
    x, y = xy
    return (amp * np.exp(-(((x - xo)**2 / (2 * sigma_x**2)) + ((y - yo)**2 / (2 * sigma_y**2)))) + offset).ravel()


def fit_peak(cross_corr, window_size):
    window_size = 2*window_size
    shape = cross_corr.shape
    window = cross_corr[shape[0]//2 - window_size:shape[0] //
                        2 + window_size, shape[1]//2 - window_size:shape[1]//2 + window_size]
    y, x = np.indices(window.shape)

    x_data = np.vstack((x.ravel(), y.ravel()))
    y_data = window.ravel()

    initial_guess = (np.max(window),
                     window.shape[1] // 2, window.shape[0] // 2, 5, 5, 0)

    try:
        # Perform curve fitting with the initial guess
        popt, _ = curve_fit(gaussian_2d, x_data, y_data, p0=initial_guess)
        _, xo, yo, _, _, _ = popt  # Extract sub-pixel accurate peak positions
    except RuntimeError:
        # If fitting fails, fall back to the peak maximum
        print("Fitting failed, using peak maximum.")
        xo, yo = np.unravel_index(np.argmax(window), window.shape)

    return (yo + shape[0]//2 - window_size, xo + shape[1]//2 - window_size)


def binned_stack(image_stack, group_size):
    number_of_images = image_stack.shape[0]
    number_of_groups = number_of_images // group_size

    # Compute projections in groups
    image_pairs = []
    for group_idx in range(0, number_of_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        group_projection = np.max(image_stack[group_start:group_end], axis=0)
        image_pairs.append(group_projection)
    return np.array(image_pairs)



def compute_drift(image_pair):

    image1_norm, image2, window_size = image_pair
    # image_2_bg_sub = image2 - np.mean(image2)

    image_2_bg_sub = white_tophat_opencv(image2, (window_size, window_size))
    image_2_norm = image_2_bg_sub / np.max(image_2_bg_sub)

    conv = fftconvolve(image1_norm, image_2_norm[::-1, ::-1], mode='same')
    peak_loc = fit_peak(conv, window_size)
    drift = [peak_loc[1] - conv.shape[1] //
             2, peak_loc[0] - conv.shape[0] // 2]

    return -np.array(drift)


def process_images(image_stack, image1_norm, interval, window_size):
    number_of_images = image_stack.shape[0]
    image_pairs = [(image1_norm, image_stack[i], window_size)
                   for i in np.arange(0, number_of_images, interval)]

    # Use multiprocessing to compute drift in parallel
    with Pool() as pool:
        drifts = list(
            tqdm(pool.imap(compute_drift, image_pairs), total=len(image_pairs)))

    return drifts


def apply_drift_correction(args):
    image, drift_1, drift_2 = args
    transform_matrix = np.array(
        [[1, 0, -drift_1], [0, 1, -drift_2]])
    corrected_image = cv2.warpAffine(
        image, transform_matrix, (image.shape[1], image.shape[0]))
    return corrected_image


def correct_images_multiprocessing(image_stack, drifts):
    # Prepare arguments for parallel drift correction
    args = [(image_stack[i], drifts[0][i], drifts[1][i])
            for i in range(image_stack.shape[0])]

    # Use multiprocessing to apply drift correction
    with Pool() as pool:
        corrected_images = list(
            tqdm(pool.imap(apply_drift_correction, args), total=len(args)))

    return corrected_images


def process_images_group_projection(image_stack, image1_norm, interval, window_size, group_size):
    number_of_images = image_stack.shape[0]
    number_of_groups = number_of_images // group_size
    
    # Compute projections in groups
    image_pairs = []
    for group_idx in range(0, number_of_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        group_projection = np.mean(image_stack[group_start:group_end], axis=0)
        image_pairs.append((image1_norm, group_projection, window_size))

    # Use multiprocessing to compute drift for group projections
    with Pool() as pool:
        drifts = list(tqdm(pool.imap(compute_drift, image_pairs), total=len(image_pairs)))

    return drifts
