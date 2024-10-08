# Copyright (c) 2023 Pranjal Choudhury, Bosanta Ranjan Boruah

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

######################################################################################################################################

# importing the dependencies
import numpy as np
from scipy.optimize import curve_fit
import numba

# defining the gaussian in one dimentions


@numba.jit
def one_d_gaussian(xdata, amplitude, xo, sigma_x, theta, offset):
    '''
    Defines a 1D Gaussian function

    Arguments:
    xdata : 1D array, x axis for the gaussian
    amplitude : strength of the gaussian
    xo : center of the gaussian
    sigma_x : standard deviation of the gaussian
    theta : rotation of the gaissian from the x axis (not used in this case)
    offset : y-ofset of the gaussian 

    Returns: 1D array with values corresponding to a gaussian
    '''
    x = xdata
    xo = float(xo)
    g = offset + amplitude*np.exp(-(x-xo)**2/(2*sigma_x**2))
    return g

##  This functions implements Algorithm 4 (1D fitting)  ##


def localizer(coordinates, images, window_size, initial_guess):
    '''
    Localizes the PSF by doing successive 1D fitting of the pixel data with the model gaussian 

    Arguments:
    coordinates : crude localized coordinates (x and y)
    images : input image
    window_size : number of pixels around each crude detection for creating the sub-image
    initial_guess : initial guess of the various parameters for fitting
    method : string, used to unpack the crude detections

    Returns: fitted arguements
    '''

    # Collect the x, y, and intensity values of all the molecules
    molecule_intensity = []
    localized_molecule_x = []
    localized_molecule_y = []
    sigma_x = []
    sigma_y = []

    # Initialize the fitting parameters
    init_params = initial_guess
    params_fit = [[0]]*5

    # Get the size of the input image
    image_size = images.shape

    # Extract initial guess values for x and y coordinates
    initial_guess_x = init_params[0], init_params[1], init_params[3], init_params[5], init_params[6]
    initial_guess_y = init_params[0], init_params[2], init_params[4], init_params[5], init_params[6]

    if len(coordinates) > 0:

        # Extract x and y coordinates from the input coordinate array
        # if method == 'BlobDetection':
        #     x, y = coordinates[0], coordinates[1]
        # else:
        #     x, y = coordinates
        x, y = coordinates

        for j in range(len(x[0])):

            x1, y1 = int(x[0][j]), int(y[0][j])

            # Calculate the boundary coordinates for the sub-image
            if y1-window_size < 0:
                aa = 0
            else:
                aa = y1-window_size
            if y1+window_size > image_size[1]:
                bb = image_size[1]
            else:
                bb = y1+window_size
            if x1-window_size < 0:
                cc = 0
            else:
                cc = x1-window_size
            if x1+window_size > image_size[1]:
                dd = image_size[0]
            else:
                dd = x1+window_size

                # Extract the sub-image
            sub_image = images[int(aa):int(bb), int(cc):int(dd)]

            # Find the coordinates of the maximum pixel value in the sub-image
            xa, ya = np.where(sub_image == np.max(sub_image))

            # Generate coordinate grids for fitting the Gaussian curves
            xx, yy = np.arange(sub_image.shape[1]), np.arange(
                sub_image.shape[0])
            try:
                # Fit a 1D Gaussian curve to the x-axis profile
                popt1, pcov1 = curve_fit(
                    one_d_gaussian, xx, sub_image[:, xa[0]], p0=initial_guess_x)

                try:
                    # Fit a 1D Gaussian curve to the y-axis profile
                    popt2, pcov2 = curve_fit(
                        one_d_gaussian, yy, sub_image[int(popt1[1]), :], p0=initial_guess_y)

                    # Collect the fitted parameters
                    molecule_intensity.append(popt2[0])
                    localized_molecule_y.append(aa+popt1[1])
                    sigma_y.append(popt1[3])
                    localized_molecule_x.append(cc+popt2[1])
                    sigma_x.append(popt2[4])

                except:
                    pass
            except:
                pass
                # Store the collected fitted parameters
        params_fit = molecule_intensity, localized_molecule_x, localized_molecule_y, sigma_x, sigma_y

    return params_fit
