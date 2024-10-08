# Copyright (c) 2023 Pranjal Choudhury, Bosanta Ranjan Boruah

# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

######################################################################################################################################


import numpy as np


def remove_outliers(loc_x, loc_y, image_shape):
    '''
    Removes localizations which are outside the original image dimensions 

    Arguments:
    loc_x : array of detections (x-coordinate)
    loc_y : array of detections (y-coordinate)
    image_shape : shape of the input image

    Returns : array of detection with outliers removed
    '''

    loc_x1 = []
    loc_y1 = []
    for i, j in zip(loc_x, loc_y):
        if (i < image_shape[1]) and (i > 0):
            if (j < image_shape[0]) and (j > 0):
                loc_x1.append(i)
                loc_y1.append(j)
    mol_x, mol_y = np.array(loc_x1), np.array(loc_y1)
    return (mol_x, mol_y)
