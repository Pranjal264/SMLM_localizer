# BSD 3-Clause License

# Copyright (c) 2024, Pranjal Choudhury

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

######################################################################################################################################
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.morphology import white_tophat, disk, black_tophat


def normalize_8bit(image):
    return np.uint8(((image - image.min())/(image.max()-image.min()))*255.)


def bandpass_filter(image, low_sigma=1, high_sigma=5):
    """
    Perform bandpass filtering on the image.

    Arguments:
    - image: input image array
    - low_sigma: lower bound for Gaussian filtering
    - high_sigma: upper bound for Gaussian filtering

    Returns:
    - filtered_image: bandpass-filtered image
    """
    low_pass = gaussian_filter(image, sigma=low_sigma)
    high_pass = image - gaussian_filter(image, sigma=high_sigma)
    return high_pass - low_pass


def white_tophat_opencv(image, kernel_size=(15, 15)):
    # Create a structuring element (you can adjust the size and shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Apply the white top-hat transformation using OpenCV
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    return tophat


def custom_threshold(image, alpha=0.1):
    """
    Calculate the optimal threshold using custom entropy-based method.

    Parameters:
    - image (numpy.ndarray): Input image for threshold calculation.

    Returns:
    - optimal_threshold (int): Optimal threshold value for image segmentation.
    """
    hist = ndimage.histogram(image, min=0, max=np.max(
        image), bins=int(np.max(image)))  # Calculate histogram
    hist_norm = hist.ravel() / hist.sum()  # Normalize histogram

    # Calculate entropies of foreground and background for all possible thresholds
    entropy_values = np.zeros(int(np.max(image)))
    for t in range(int(np.max(image))):
        p1 = hist_norm[:t+1].sum()
        p2 = hist_norm[t+1:].sum()

        entropy_values[t] = -p1 * \
            np.log(int(1/(1+p1))+p1) - p2 * np.log(int(1/(1+p2))+p2)

    # Find the optimal threshold based on mean entropy
    optimal_threshold = np.where(
        np.abs(entropy_values - np.mean(entropy_values)) < alpha)[0][-1]
    # optimal_threshold = np.argmax(entropy_values) #kapur's method

    return optimal_threshold


def detection(image, threshold, PSF_size):
    '''
    Detect probable PSF locations with preprocessing and adaptive thresholding.

    Arguments:
    - image: input image array
    - PSF_size: side of the PSF in pixels (experimental parameter)
    - method: thresholding method ('mean', 'otsu', 'adaptive')
    - alpha: parameter for custom thresholding method (default is 0.1)

    Returns: coordinates of the detected PSFs (x, y)
    '''

    # Step 4: Thresholding (choose between different methods)
    if threshold == '1 - mean thresholding':
        image = normalize_8bit(image)
        enhanced_image = white_tophat_opencv(image, (PSF_size, PSF_size))
        enhanced_image = enhanced_image / np.max(enhanced_image)
        # threshold_ = 5*np.std(enhanced_image)
        threshold_ = 0.3
        coordinates = peak_local_max(
            enhanced_image, min_distance=PSF_size, threshold_abs=threshold_)
    elif threshold == "2 - adaptive thresholding":
        image = gaussian(image)
        image = image / image.max() * 255.
        threshold_ = custom_threshold(image, alpha=0.1)
        coordinates = peak_local_max(
            image, min_distance=PSF_size, threshold_abs=threshold_)
    elif threshold == "3 - Otsu's thresholding":
        enhanced_image = gaussian(image)
        # image = image / image.max() * 255.
        # enhanced_image = white_tophat(image, disk(PSF_size))
        enhanced_image = enhanced_image / np.max(enhanced_image)
        threshold_ = threshold_otsu(enhanced_image)
        coordinates = peak_local_max(
            enhanced_image, min_distance=PSF_size, threshold_abs=threshold_)

    detection_x = []
    detection_y = []
    # Step 5: Detect PSFs using peak_local_max

    # Extract x and y coordinates of detected PSFs
    y, x = coordinates.T
    detection_x.append(x)
    detection_y.append(y)

    return (detection_x, detection_y)


#  This functions implements Algorithm 1  ##


# def detection(image, threshold, PSF_size):
#     '''
#     detects probable PSF locations using skimage.feature.peak_local_max()

#     Arguments:
#     image: input image array
#     threshold: a number >0
#     PSf_size: side of the PSF in pixels (experimental parameter)

#     Returns: coordinates of the detected PSFs
#     '''
#     image = image / image.max() * 255.
#     image = gaussian(image)
#     if threshold == "1 - mean thresholding":
#         threshold_ = np.mean(image)
#     if threshold == "2 - adaptive thresholding":
#         threshold_ = custom_threshold(image)
#     if threshold == "3 - Otsu's thresholding":
#         threshold_ = threshold_otsu(image)
#     detection_x = []
#     detection_y = []
#     coordinates = peak_local_max(
#         image, min_distance=PSF_size, threshold_abs=threshold_)
#     y, x = coordinates.T
#     detection_x.append(x)
#     detection_y.append(y)

#     return (detection_x, detection_y)
