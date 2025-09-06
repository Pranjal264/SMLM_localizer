import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.fft import fft2

# --- Helper function for Gaussian models ---
def _gaussian_2d(coords, A, x0, y0, sigma, C):
    """A symmetric 2D Gaussian function."""
    x, y = coords
    # Assuming symmetric sigma for stability
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) + C

def localize(image, crude_coords, method='Gaussian Fit (LS)', **kwargs):
    """
    Refines crude localizations to sub-pixel precision.

    Args:
        image (np.ndarray): The original (unfiltered) image.
        crude_coords (np.ndarray): Array of (row, col) crude coordinates.
        method (str): The sub-pixel localization method.

    Returns:
        np.ndarray: Array of refined (x, y) coordinates.
    """
    box_size = kwargs.get('box_size', 7)
    if box_size % 2 == 0: box_size += 1 # Ensure odd
    half_box = box_size // 2
    
    refined_coords = []
    additional_info = [] # To store amplitude and FWHM if needed

    y_grid, x_grid = np.mgrid[:box_size, :box_size]

    for r, c in crude_coords:
        if (r - half_box < 0 or r + half_box + 1 > image.shape[0] or
            c - half_box < 0 or c + half_box + 1 > image.shape[1]):
            continue

        roi = image[r - half_box : r + half_box + 1, c - half_box : c + half_box + 1].astype(np.float64)
        
        try:
            if method == 'Gaussian Fit (LS)':
                initial_guess = [roi.max() - roi.min(), half_box, half_box, 1.0, roi.min()]
                popt, _ = curve_fit(_gaussian_2d, (x_grid.ravel(), y_grid.ravel()), roi.ravel(), p0=initial_guess)
                amp, fit_y, fit_x, fwhm, _ = popt

            elif method == 'Gaussian Fit (MLE)':
                def neg_log_likelihood(params, data):
                    model = _gaussian_2d((y_grid.ravel(), x_grid.ravel()), *params)
                    model[model <= 0] = 1e-9 # Avoid log(0)
                    return -np.sum(data * np.log(model) - model)
                
                initial_guess = [roi.max() - roi.min(), half_box, half_box, 1.0, roi.min()]
                
                # Add bounds for stability
                bounds = [(0, None), (0, box_size), (0, box_size), (0.1, half_box), (0, None)]
                
                result = minimize(neg_log_likelihood, initial_guess, args=(roi.ravel(),), method='L-BFGS-B', bounds=bounds)
                if not result.success: continue
                popt = result.x
                # fit_y, fit_x = popt[1], popt[2]
                amp, fit_y, fit_x, fwhm, _ = popt

            elif method == 'Phasor':
                try:
                    fft_values = fft2(roi)
                    window_pixel_size = roi.shape[0]
                    angX = np.angle(fft_values[0, 1])
                    if (angX > 0):
                        angX = angX-2*np.pi

                    PositionX = (abs(angX)/(2*np.pi/window_pixel_size))

                    angY = np.angle(fft_values[1, 0])
                    if (angY > 0):
                        angY = angY-2*np.pi
                    PositionY = (abs(angY)/(2*np.pi/window_pixel_size))

                    fit_x = PositionX
                    fit_y = PositionY
                    amp = roi.max() - roi.min()
                    fwhm = 1.0 # Placeholder, not used in Phasor

                except: 
                    pass

            
            else:
                raise ValueError(f"Unknown subpixel method: {method}")

            # Quality control: check if the fit is near the center
            if abs(fit_x - half_box) < 1.5 and abs(fit_y - half_box) < 1.5:
                # Convert from ROI coordinates to global image coordinates
                global_x = c - half_box + fit_x
                global_y = r - half_box + fit_y
                refined_coords.append((global_y, global_x))
                additional_info.append((amp, fwhm))

        except (RuntimeError, ValueError) as e:
            # print(f"Skipping point due to error: {e}") # Uncomment for debugging
            continue # Skip if fitting fails

    # print(f"Refined {len(refined_coords)} localizations with {method}.")
    return (np.array(refined_coords), np.array(additional_info))