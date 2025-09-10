import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def render(coords, ax, image_shape, method='Gaussian', **kwargs):
    """
    Renders the localizations using the specified method.
    """
    ax.clear()
    rendered_image_data = None

    if coords.size == 0:
        print("No coordinates to render.")
        ax.set_facecolor('black')
        ax.set_title(f"{method} Rendering (No Data)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.canvas.draw()
        return

    magnification = kwargs.get('pixel_size', 10.0)

    if method == 'Scatter Plot':
        print("Rendering as scatter plot...")
        ax.scatter(coords[:, 0], coords[:, 1], s=1, c='red', marker='.')
        ax.set_facecolor('black')
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0) 
        ax.set_title("Scatter Plot Rendering")
        rendered_image_data = None

    elif method == '2D Histogram':
        print("Rendering as 2D Histogram...")
        # Define the bins based on the magnified output image size
        bins_x = int(image_shape[1] * magnification)
        bins_y = int(image_shape[0] * magnification)
        
        # Define the range of the histogram based on the original image dimensions
        range_x = [0, image_shape[1]]
        range_y = [0, image_shape[0]]
        hist_ = np.histogram2d(coords[:, 1], coords[:, 0], bins=[bins_x, bins_y], range=[range_x, range_y])[0]
        ax.imshow(hist_, cmap='hot')
        ax.set_facecolor('black')
        ax.set_aspect('equal', 'box')
        # ax.invert_yaxis() # Match image coordinate system
        ax.set_title(f"2D Histogram Rendering with {magnification}X Magnification")
        rendered_image_data = hist_

    elif method == 'ASH':
        print("Rendering as Averaged Shifted Histogram...")
        # Define the bins based on the magnified output image size
        bins_x = int(image_shape[1] * magnification)
        bins_y = int(image_shape[0] * magnification)
        
        # Define the range of the histogram based on the original image dimensions
        range_x = [0, image_shape[1]]
        range_y = [0, image_shape[0]]

        hist_ = np.histogram2d(coords[:, 1], coords[:, 0], bins=[bins_x, bins_y], range=[range_x, range_y])[0]
        hist =  np.flipud(hist_)
        max_shift_x = 1
        max_shift_y = 1
        num_shifts_x = 2 * max_shift_x + 1
        num_shifts_y = 2 * max_shift_y + 1

        # Shift the histogram in x-direction
        shifted_hists_x = [np.roll(hist, shift, axis=0)
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
        avg_hist = avg_hist[::-1, :]
        ax.imshow(avg_hist, cmap='hot', origin='upper', aspect='equal')
        ax.set_facecolor('black')
        ax.set_aspect('equal', 'box')
        ax.set_title(f"ASH Rendering with {magnification}X magnification")
        rendered_image_data = avg_hist

    elif method == 'Gaussian':
        sigma = kwargs.get('sigma', 1.5)
        print("Rendering with Gaussian blobs...")
        
        h, w = image_shape
        h_render = int(h * magnification)
        w_render = int(w * magnification)
        rendered_image = np.zeros((h_render, w_render), dtype=np.float32)
        
        # Scale coordinates to the magnified canvas
        scaled_coords = coords * magnification
        
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0: kernel_size += 1
        half_k = kernel_size // 2
        y_k, x_k = np.mgrid[-half_k:half_k+1, -half_k:half_k+1]
        kernel = np.exp(-(x_k**2 + y_k**2) / (2 * sigma**2))

        for x_c, y_c in tqdm(scaled_coords, desc="Rendering Points"):
            ix, iy = int(round(x_c)), int(round(y_c))
            
            y_start, y_end = iy - half_k, iy + half_k + 1
            x_start, x_end = ix - half_k, ix + half_k + 1
            
            if y_start < 0 or y_end > h_render or x_start < 0 or x_end > w_render:
                continue
            
            rendered_image[y_start:y_end, x_start:x_end] += kernel
        
        # Use origin='upper' for correct image orientation
        ax.imshow(rendered_image, cmap='hot', origin='upper', aspect='equal')
        ax.set_title(f"Gaussian Rendering with {magnification}X Magnification")# \n Image dimensions: {image_shape[1]*magnification} X {image_shape[0]*magnification}")
        rendered_image_data = rendered_image

    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.canvas.draw()

    return rendered_image_data