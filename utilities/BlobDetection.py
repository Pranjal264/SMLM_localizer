# Copyright (c) 2023 Pranjal Choudhury, Bosanta Ranjan Boruah

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

######################################################################################################################################

# importing the dependencies
import numpy as np
from skimage.feature import blob_log


##  This functions implements Algorithm 3  ##

# defining a function to detect the PSFs using blob_LOG
def detection(image, threshold, PSF_size):
    '''
    Argumentss:
    image: input image array
    threshold: a number >0
    PSf_size: side of the PSF in pixels (experimental parameter)

    Returns: coordinates of the detected PSFs
    '''

    #image = image.astype(np.uint8)
    image_info = image.shape								# extracting the shape of the image
    image_size = np.zeros(2)
    image_size[0], image_size[1] = image_info
    image = image / image.max() * 255.
    threshold = 70
    # params=np.zeros(3)
    params = [[0]]*3									# initializing the parameters
    molecule_x = []
    molecule_y = []
    # sigma_mol = []

    blobs = blob_log(image, min_sigma=2, max_sigma=PSF_size,
                     num_sigma=10, threshold=threshold/10.)

    # if blobs are detected, append the x, y, and sigma values to respective lists
    if blobs.size > 0:
        y, x, s = blobs.T
        molecule_x.append(x)
        molecule_y.append(y)
        # sigma_mol.append(s)

    params = molecule_x, molecule_y  # , sigma_mol

    return params


# # importing the dependencies
# import numpy as np
# from skimage.feature import blob_log
# from multiprocessing import Pool, cpu_count

# # Defining a helper function for processing a single image with blob_LOG
# def process_image(image_and_params):
#     image, threshold, PSF_size = image_and_params

#     # Detecting PSFs in the image
#     blobs = blob_log(image, min_sigma=2, max_sigma=PSF_size, num_sigma=10, threshold=threshold / 10.)

#     molecule_x, molecule_y, sigma_mol = [], [], []
#     # If blobs are detected, append the x, y, and sigma values to respective lists
#     if blobs.size > 0:
#         y, x, s = blobs.T
#         molecule_x.append(x)
#         molecule_y.append(y)
#         sigma_mol.append(s)

#     return molecule_x, molecule_y, sigma_mol


# # Defining a multiprocessing function to detect PSFs using blob_LOG
# def detection(images, threshold, PSF_size):
#     '''
#     Arguments:
#     images: list of input image arrays
#     threshold: a number > 0
#     PSF_size: side of the PSF in pixels (experimental parameter)

#     Returns: a list of tuples with coordinates of the detected PSFs (for each image)
#     '''

#     # Prepare the input parameters for each image
#     inputs = [(image, threshold, PSF_size) for image in images]

#     # Create a pool of worker processes, using all available CPU cores
#     with Pool(cpu_count()) as pool:
#         results = pool.map(process_image, inputs)

#     return results
