# Copyright (c) 2023 Pranjal Choudhury, Bosanta Ranjan Boruah

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

######################################################################################################################################

# importing the dependencies
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from scipy import ndimage
from skimage.filters import gaussian, threshold_otsu


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

# defining the PSF detector


def rough_position_estimator(data, threshold, neighborhood_size):
    '''
    Arguments:
    data: input image array
    threshold: a number >0
    neighbourhood size: neighouring pixels 4 or 9 or any number

    Returns: coordinates of the detected PSFs
    '''

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = data == data_max
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = (data_max - data_min) > threshold

    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    coordinates_xy = np.array(ndimage.center_of_mass(
        data, labeled, range(1, num_objects+1)))

    return coordinates_xy

##  This functions implements Algorithm 2  ##
# defining a function to detect the PSFs using COM


def detection(image, threshold, neighborhood_size):
    '''
    Arguments:
    data: input image array
    threshold: a number >0
    neighbourhood size: neighouring pixels 4 or 9 or any number

    Returns: coordinates of the detected PSFs
    '''
    image = gaussian(image)
    image = image / image.max() * 255.
    if threshold == "1 - mean thresholding":
        threshold_ = np.mean(image)
    if threshold == "2 - adaptive thresholding":
        threshold_ = custom_threshold(image)
    if threshold == "3 - Otsu's thresholding":
        threshold_ = threshold_otsu(image)

    # images = image.astype(np.uint8)

    params = np.zeros(2)
    molecule_x = []
    molecule_y = []

    coordinates = rough_position_estimator(
        image, threshold_, neighborhood_size)

    # if coordinates are detected, append the x and y to respective lists
    if coordinates.size > 0:
        y, x = coordinates.T
        molecule_x.append(x)
        molecule_y.append(y)

    params = molecule_x, molecule_y

    return params
