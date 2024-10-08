# Copyright (c) 2023 Pranjal Choudhury, Bosanta Ranjan Boruah

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

######################################################################################################################################

# importing the dependencies
import numpy as np
import utilities.RemoveIncorrectLocalizations

# plotting a binary image of all the detections


def scatter_plot(image_shape, mol_x, mol_y, mag=10):
    '''
    Plots a binary image according to the x and y localizations

    Arguments:
    image_shape : input image shape
    mol_x : array of detections (x-coordinate)
    mol_y : array of detections (y-coordinate)
    mag : magnification required for the super resolved image (default =10 )

    Returns: binary image with all the detections
    '''

    # activating the pixels where the index corresponding to the localizations
    mol_x, mol_y = utilities.RemoveIncorrectLocalizations.remove_outliers(
        mol_x, mol_y, image_shape)

    binary_image = np.zeros([int(image_shape[0]*mag), int(image_shape[1]*mag)])
    for i, j in zip(mol_x, mol_y):
        binary_image[int(j*mag), int(i*mag)] = 1
    return binary_image

# plotting a 2D histogramof all the detections, binning all detections acccurate upo one
# decial point (or based on the value of the parameter mag) in pne pixel


def histogram(image_shape, mol_x, mol_y, mag=10):
    '''
    Plots a 2D histogram according to the x and y localizations and bins

    Arguments:
    image_shape : input image shape
    mol_x : array of detections (x-coordinate)
    mol_y : array of detections (y-coordinate)
    mag : magnification required for the super resolved image (default =10 )

    Returns: binary image with all the detections
    '''
    mol_x, mol_y = utilities.RemoveIncorrectLocalizations.remove_outliers(
        mol_x, mol_y, image_shape)
    # determining the extents of the histogram image by putting dummy coordinates at the edges
    start = np.array([0])
    end_x = np.array([image_shape[1]])
    end_y = np.array([image_shape[0]])
    mol_x1 = np.hstack((start, mol_x, end_x))
    mol_y1 = np.hstack((start, mol_y, end_y))

    # histogram
    hist, _, _ = np.histogram2d(
        mol_y1, mol_x1, bins=[int(image_shape[0]*mag), int(image_shape[1]*mag)])
    #extent = [y_edges[0], y_edges[-1], x_edges[0], x_edges[-1]]

    # removing  the values created by the dummy coordnates coordinates
    hist[0, 0] = hist[0, 0]-1
    hist[0, int(image_shape[1]*mag-1)] = hist[0, int(image_shape[1]*mag-1)]-1
    hist[int(image_shape[0]*mag-1), 0] = hist[int(image_shape[0]*mag-1), 0]-1
    hist[int(image_shape[0]*mag-1), int(image_shape[1]*mag-1)
         ] = hist[int(image_shape[0]*mag-1), int(image_shape[1]*mag-1)]-1

    return hist

# plotting 2D averaged shifted histograms by plotting 2D histograms and then shifting
# them in both x and y direcions and then averaging them


def averaged_shifted_histogram(image_shape, mol_x, mol_y, mag=10):
    '''
    Plots a 2D averaged shifted histogram according to the x and y localizations and bins 
    and the number of shifts in each axis

    Arguments:
    image_shape : input image shape
    mol_x : array of detections (x-coordinate)
    mol_y : array of detections (y-coordinate)
    mag : magnification required for the super resolved image (default =10 )

    Returns: binary image with all the detections
    '''
    max_shift_x = 1
    max_shift_y = 1
    num_shifts_x = 2 * max_shift_x + 1
    num_shifts_y = 2 * max_shift_y + 1

    # Shift the histogram in x-direction
    shifted_hists_x = [np.roll(histogram(image_shape, mol_x, mol_y, mag), shift, axis=0)
                       for shift in range(-max_shift_x, max_shift_x+1)]

    # Shift the histogram in y-direction
    shifted_hists_xy = []
    for shift_y in range(-max_shift_y, max_shift_y+1):
        for shifted_hist_x in shifted_hists_x:
            shifted_hist_xy = np.roll(shifted_hist_x, shift_y, axis=1)
            shifted_hists_xy.append(shifted_hist_xy)

    avg_hist = np.sum(shifted_hists_xy, axis=0) / (num_shifts_x * num_shifts_y)

    # Set values below the mean to zero to remove noise
    avg_hist[avg_hist < np.mean(avg_hist)] = 0

    return avg_hist
