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
