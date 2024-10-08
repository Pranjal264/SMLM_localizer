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
from scipy.fft import fft2


def localizer(coordinates, images, window_size, initial_guess):
    # Collect the x, y values of all the molecules
    molecule_intensity = []
    localized_molecule_x = []
    localized_molecule_y = []
    sigma_x = []
    sigma_y = []
    # Initialize the fitting parameters
    params_fit = [[0]]*5
    image_size = images.shape
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

                # # Extract the sub-image
                # sub_image = images[y1 - window_size: y1 +
                #                    window_size, x1 - window_size: x1 + window_size]
                try:
                    fft_values = fft2(sub_image)
                    window_pixel_size = sub_image.shape[0]

                    angX = np.angle(fft_values[0, 1])
                    if (angX > 0):
                        angX = angX-2*np.pi

                    PositionX = (abs(angX)/(2*np.pi/window_pixel_size))

                    angY = np.angle(fft_values[1, 0])
                    if (angY > 0):
                        angY = angY-2*np.pi

                    PositionY = (abs(angY)/(2*np.pi/window_pixel_size))
                    localized_molecule_x.append((x1-window_size)+PositionX)
                    localized_molecule_y.append((y1-window_size)+PositionY)

                    MagnitudeX = abs(fft_values[0, 1])
                    MagnitudeY = abs(fft_values[1, 0])
                    sigma_x.append(MagnitudeX)
                    sigma_y.append(MagnitudeY)
                    molecule_intensity.append(np.max(sub_image))
                    params_fit = molecule_intensity, localized_molecule_x, localized_molecule_y, sigma_x, sigma_y
                except:
                    pass

    return params_fit


# # Import necessary modules
# import numpy as np
# from scipy.fft import fft2
# from multiprocessing import Pool, cpu_count

# # Helper function to process each molecule localization


# def process_localization(params):
#     x1, y1, images, window_size = params

#     # Extract the sub-image
#     sub_image = images[y1 - window_size: y1 +
#                        window_size, x1 - window_size: x1 + window_size]

#     # Compute FFT of the sub-image
#     fft_values = fft2(sub_image)
#     window_pixel_size = sub_image.shape[0]

#     # Calculate x position using phase information
#     angX = np.angle(fft_values[0, 1])
#     if angX > 0:
#         angX = angX - 2 * np.pi
#     PositionX = abs(angX) / (2 * np.pi / window_pixel_size)

#     # Calculate y position using phase information
#     angY = np.angle(fft_values[1, 0])
#     if angY > 0:
#         angY = angY - 2 * np.pi
#     PositionY = abs(angY) / (2 * np.pi / window_pixel_size)

#     # Return the localized coordinates
#     return (x1 - window_size + PositionX, y1 - window_size + PositionY)

# # Multiprocessing-based localizer function


# def localizer(coordinates, images, window_size, initial_guess, method):
#     # Collect the x, y values of all the molecules
#     molecule_intensity = []
#     localized_molecule_x = []
#     localized_molecule_y = []
#     sigma_x = []
#     sigma_y = []
#     params_fit = [[0]]*2

#     if len(coordinates) > 0:
#         # Extract x and y coordinates from the input coordinate array
#         if method == 'BlobDetection':
#             x, y = coordinates[0], coordinates[1]
#         else:
#             x, y = coordinates

#         if x.shape[0] > 0:
#             # Prepare input for multiprocessing
#             inputs = [(int(x[0][j]), int(y[0][j]), images, window_size)
#                       for j in range(len(x[0]))]

#             # Create a pool of worker processes
#             with Pool(cpu_count()) as pool:
#                 results = pool.map(process_localization, inputs)

#             # Append results to the output lists
#             for res in results:
#                 localized_molecule_x.append(res[0])
#                 localized_molecule_y.append(res[1])

#             # Set params_fit for output
#             params_fit = molecule_intensity, localized_molecule_x, localized_molecule_y, sigma_x, sigma_y

#     return params_fit
