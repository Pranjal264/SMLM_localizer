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

import numpy as np
from scipy.optimize import curve_fit
import numba


@numba.jit
def two_d_gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    '''
    Defines a 2D gaussian in a grid and then reshapes it in 1D, used in fitting

    Arguments
        xdata_tuple : 1D array, meshgrid of x and y axis
        amplitude : strength of the gaussian
        xo : center of the gaussian (x coordinate)
        yo : center of the gaussian (y coordinate)
        sigma_x : standard deviation of the gaussian (along x axis)
        sigma_y : standard deviation of the gaussian (along y axis)
        theta : rotation of the gaissian from the x axis
        offset : z-ofset of the gaussian

        Returns: 1D array with values corresponding to a gaussian
    '''
    (x, y) = xdata_tuple
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude * \
        np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

##  This functions implements Algorithm 4 (2D fitting)  ##


def localizer(coordinates, images, window_size, initial_guess):
    '''
    Localizes the PSF by doing 2D fitting of the pixel data with the model gaussian

    Arguments:
    coordinates : crude localized coordinates (x and y)
    images : input image
    window_size : number of pixels around each crude detection for creating the sub-image
    initial_guess : initial guess of the various parameters for fitting
    method : string, used to unpack the crude detections

    Returns: fitted arguments
    '''
    # Collect the x, y, and intensity values of all the molecules
    molecule_intensity = []
    localized_molecule_x = []
    localized_molecule_y = []
    sigma_x = []
    sigma_y = []

    # Initialize the fitting parameters
    params_fit = [[0]]*5

    if len(coordinates) > 0:
        # Extract x and y coordinates from the input coordinate array
        # if method == 'BlobDetection':
        #     x, y = coordinates[0], coordinates[1]
        # else:
        #     x, y = coordinates
        x, y = coordinates

        if x.shape[0] > 0:
            for j in range(len(x[0])):
                x1, y1 = int(x[0][j]), int(y[0][j])

                # Extract the sub-image
                sub_image = images[y1 - window_size: y1 +
                                   window_size, x1 - window_size: x1 + window_size]
                sub_image_flat = sub_image.ravel()

                # Generate coordinate grids for fitting the Gaussian curves
                xx, yy = np.meshgrid(
                    np.arange(sub_image.shape[1]), np.arange(sub_image.shape[0]))
                try:
                    # Fit a 2D Gaussian to the PSF data
                    popt, pcov = curve_fit(
                        two_d_gaussian, (xx, yy), sub_image_flat, p0=initial_guess)

                    # Collect the fitted parameters
                    molecule_intensity.append(popt[0])
                    localized_molecule_x.append((x1-window_size)+popt[1])
                    localized_molecule_y.append((y1-window_size)+popt[2])
                    sigma_x.append(popt[3])
                    sigma_y.append(popt[4])

                    # Store the collected fitted parameters
                    params_fit = molecule_intensity, localized_molecule_x, localized_molecule_y, sigma_x, sigma_y

                except:
                    pass

    return params_fit
