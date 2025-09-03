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
