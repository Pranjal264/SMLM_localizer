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
from scipy.optimize import minimize
import numba
from joblib import Parallel, delayed

# Define the 2D Gaussian model


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
    return g

# Define the negative log-likelihood function (to be minimized)


def likelihood(params, image):
    """
    defines the likelihood function for estimation

    Arguments: 
    params : amplitude, xo, yo, sigma_x, sigma_y, theta, offset (defined in the function two_d_gaussian)
    image : 2D array with pixel values

    Returns : the negative log likelihood.
    """
    regularization = 1e-10				# a very small value, so that the log operation doesnot face any error

    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = params

    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

    expected_image = two_d_gaussian(
        (x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset)

    # Apply regularization to avoid zero values
    expected_image = np.maximum(expected_image, regularization)

    log_likelihood = np.sum(image * np.log(expected_image) - expected_image)
    return -log_likelihood


##  This functions implements Algorithm 5  ##

# MLE
def localizer(coordinates, images, window_size, initial_guess):
    '''
    Localizes the PSF by doing 2D MLE of the pixel data with the model gaussian 

    Arguments:
    coordinates : crude localized coordinates (x and y)
    images : input image
    window_size : number of pixels around each crude detection for creating the sub-image
    initial_guess : initial guess of the various parameters for estimation
    method : string, used to unpack the crude detections

    Returns: fitted arguments
    '''

    num_cores = 4            # Number of CPU cores to use for parallel processing

    # Collect the x, y, and intensity values of all the molecules

    molecule_intensity = []
    localized_molecule_x = []
    localized_molecule_y = []
    sigma_x = []
    sigma_y = []

    params_fit = [[0]]*5

    if len(coordinates) > 0:
        # if method == 'BlobDetection':
        #     x, y = coordinates[0], coordinates[1]
        # else:
        #     x, y = coordinates
        x, y = coordinates

        def process_molecule(j):
            x1, y1 = int(x[0][j]), int(y[0][j])

            # Extract a sub-image around the molecule
            sub_image = images[y1-window_size:y1 +
                               window_size, x1-window_size:x1+window_size]

            # bounds=((0,255),(0,sub_image[1]),(0,sub_image[0]),(0,np.inf),(0,np.inf),(0,100),(0,2*np.pi))
            #bounds = [(0,255), (0,sub_image[1]), (0,sub_image[0]),(0,window_size), (0,window_size), (0,2*np.pi), (0,100)]
            try:
                # Perform minimization of negative log-likelihood to estimate parameters
                popt = minimize(likelihood, initial_guess, args=(
                    sub_image), method='Nelder-Mead', tol=1e-2)
                return [popt.x[0], (x1-window_size)+popt.x[1], (y1-window_size)+popt.x[2], popt.x[3], popt.x[4]]
            except:
                return None

        # Process molecules in parallel
        results = Parallel(n_jobs=num_cores)(
            delayed(process_molecule)(j) for j in range(len(x)))

        for result in results:
            if result is not None:
                # Collect the estimated parameters of the molecules
                molecule_intensity.append(result[0])
                localized_molecule_x.append(result[1])
                localized_molecule_y.append(result[2])
                sigma_x.append(result[3])
                sigma_y.append(result[4])

        params_fit = molecule_intensity, localized_molecule_x, localized_molecule_y, sigma_x, sigma_y

    return params_fit
