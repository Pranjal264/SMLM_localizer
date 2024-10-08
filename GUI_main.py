import os
import time
import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
# from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, ttk, Scrollbar
from tkinter import messagebox

# Importing custom utility modules
import utilities.PeakLocalMaxima
import utilities.CentreOfMass
import utilities.BlobDetection
import utilities.OneDLSFitting
import utilities.TwoDLSFitting
import utilities.TwoDMLE
import utilities.PhasorLocalization
import utilities.ImageVisualization
# import utilities.RemoveIncorrectLocalizations

# Function to process images (unchanged)


def process_image(image, image_count, processing_parameters):
    CRUDE_LOCALIZATION_TYPE, RECONSTRUCTION_TYPE, SUB_PIXEL_LOCALIZATION_METHOD, THRESHOLD, PSF_RADIUS, NEIGHBOURHOOD_SIZE, WINDOW_SIZE, INIT_PARAMS, BACKGROUND = processing_parameters

    frame = image_count
    crude_coordinates = []
    params = []

    # detection using peak local maxima
    if CRUDE_LOCALIZATION_TYPE == "1 - Peak Local Maxima":
        # METHOD = 'PeakLocalMaxima'
        crude_coordinates = utilities.PeakLocalMaxima.detection(
            image=image, threshold=THRESHOLD, PSF_size=PSF_RADIUS)

    # detection using Centre of Mass
    if CRUDE_LOCALIZATION_TYPE == "2 - Centre of Mass":
        # METHOD = 'COM'
        crude_coordinates = utilities.CentreOfMass.detection(
            image=image, threshold=THRESHOLD, neighborhood_size=NEIGHBOURHOOD_SIZE)

    # detection using Blob LoG
    if CRUDE_LOCALIZATION_TYPE == "3 - Blob Detection":
        # METHOD = 'BlobDetection'
        crude_coordinates = utilities.BlobDetection.detection(
            image=image, threshold=THRESHOLD, PSF_size=PSF_RADIUS * 2)

    ### if we want only crude localization, the following code will run ###
    if RECONSTRUCTION_TYPE == "1 - Crude Localization":
        return crude_coordinates, frame*np.ones(len(crude_coordinates[0][0]), dtype='int')

    ### if we want sub pixel localization, the following code will run ###
    if RECONSTRUCTION_TYPE == "2 - Sub Pixel Localization":
        ######################################################################################################################################################################################
        # sub pixel localization ;  Input: {crude coordinates, image, window size, initial guess, method}  Output: {params (PSF intensity, x_position, y_position, sigma_x, sigma_y)}
        ######################################################################################################################################################################################

        # sub pixel localization using 1D Least Square fitting
        if SUB_PIXEL_LOCALIZATION_METHOD == "1 - 1D LS Fitting":
            params = utilities.OneDLSFitting.localizer(coordinates=np.array(
                crude_coordinates), images=image, window_size=WINDOW_SIZE, initial_guess=INIT_PARAMS)

        # sub pixel localization using 2D Least Square fitting
        if SUB_PIXEL_LOCALIZATION_METHOD == "2 - 2D LS Fitting":
            params = utilities.TwoDLSFitting.localizer(coordinates=np.array(
                crude_coordinates), images=image, window_size=WINDOW_SIZE, initial_guess=INIT_PARAMS)

        # sub pixel localization using 2D MLE
        if SUB_PIXEL_LOCALIZATION_METHOD == "3 - MLE":
            params = utilities.TwoDMLE.localizer(coordinates=np.array(
                crude_coordinates), images=image, window_size=WINDOW_SIZE, initial_guess=INIT_PARAMS)

        # sub pixel localization using phasor localization
        if SUB_PIXEL_LOCALIZATION_METHOD == "4 - Phasor":
            params = utilities.PhasorLocalization.localizer(coordinates=np.array(
                crude_coordinates), images=image, window_size=WINDOW_SIZE, initial_guess=INIT_PARAMS)
    return crude_coordinates, params, frame*np.ones(len(params[0]), dtype='int')


def run_processing():

    global df, image_av

    start_time = time.time()

    # Getting parameters from the GUI
    input_file = input_file_entry.get()
    IMAGE_STACK_FILE = input_file

    image_stack = io.imread(IMAGE_STACK_FILE)
    BACKGROUND = np.mean(image_stack, axis=0)

    NUMBER_OF_IMAGES, DIMENSION_1, DIMENSION_2 = image_stack.shape
    IMAGE_SHAPE = DIMENSION_1, DIMENSION_2

    RECONSTRUCTION_TYPE = reconstruction_type_combo.get()
    CRUDE_LOCALIZATION_TYPE = crude_localization_type_combo.get()
    SUB_PIXEL_LOCALIZATION_METHOD = sub_pixel_localization_method_combo.get()

    THRESHOLD = threshold_combo.get()
    PSF_RADIUS = psf_radius_slider.get()
    NEIGHBOURHOOD_SIZE = neighbourhood_size_slider.get()
    WINDOW_SIZE = window_size_slider.get()
    # Keeping this fixed for simplicity
    INIT_PARAMS = [255, 5, 5, 1.6, 1.6, 0, 0]

    processing_parameters = CRUDE_LOCALIZATION_TYPE, RECONSTRUCTION_TYPE, SUB_PIXEL_LOCALIZATION_METHOD, THRESHOLD, PSF_RADIUS, NEIGHBOURHOOD_SIZE, WINDOW_SIZE, INIT_PARAMS, BACKGROUND

    # Process images using multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(process_image, [(
            image_stack[i], i, processing_parameters) for i in range(NUMBER_OF_IMAGES)])

    # Collect results
    frames = []
    crude_localized_molecule_x = []
    crude_localized_molecule_y = []
    molecule_intensity = []
    localized_molecule_x = []
    localized_molecule_y = []
    sigma_x = []
    sigma_y = []

    for result in results:
        if RECONSTRUCTION_TYPE == "1 - Crude Localization":
            crude_coordinates, frame = result
            frames.append(frame)
            crude_localized_molecule_x.append(crude_coordinates[0][0])
            crude_localized_molecule_y.append(crude_coordinates[1][0])

        if RECONSTRUCTION_TYPE == "2 - Sub Pixel Localization":
            crude_coordinates, params, frame = result
            frames.append(frame)
            molecule_intensity.append(params[0])
            localized_molecule_x.append(params[1])
            localized_molecule_y.append(params[2])
            sigma_x.append(params[3])
            sigma_y.append(params[4])

    if RECONSTRUCTION_TYPE == "2 - Sub Pixel Localization":
        frame_no = np.concatenate(frames, axis=0)
        intensity = np.concatenate(molecule_intensity, axis=0)
        detection_x = np.concatenate(localized_molecule_x, axis=0)
        detection_y = np.concatenate(localized_molecule_y, axis=0)
        sigma_1 = np.concatenate(sigma_x, axis=0)
        sigma_2 = np.concatenate(sigma_y, axis=0)
    else:
        frame_no = np.concatenate(frames, axis=0)
        detection_x = np.concatenate(crude_localized_molecule_x, axis=0)
        detection_y = np.concatenate(crude_localized_molecule_y, axis=0)

    # detection_x, detection_y = utilities.RemoveIncorrectLocalizations.remove_outliers(
    #     detection_x, detection_y, IMAGE_SHAPE)
    end_time = time.time()
    print(f'total processing time: {end_time - start_time}s')
    messagebox.showinfo("Processing Complete",
                        f'processed {NUMBER_OF_IMAGES} images in {end_time - start_time:.2f}s ')

    if RECONSTRUCTION_TYPE == "1 - Crude Localization":
        df = pd.DataFrame({'frame_no': frame_no,  'detection_x': detection_x,
                          'detection_y': detection_y})
    if RECONSTRUCTION_TYPE == "2 - Sub Pixel Localization":
        df = pd.DataFrame({'frame_no': frame_no, 'intensity': intensity, 'detection_x': detection_x,
                          'detection_y': detection_y, 'sigma_x': np.abs(sigma_1), 'sigma_y': np.abs(sigma_2)})

    image_sc = utilities.ImageVisualization.scatter_plot(
        IMAGE_SHAPE, detection_x, detection_y, 10)
    image_hs = utilities.ImageVisualization.histogram(
        IMAGE_SHAPE, detection_x, detection_y, 10)
    image_av = utilities.ImageVisualization.averaged_shifted_histogram(
        IMAGE_SHAPE, detection_x, detection_y, 10)

    # Display the final images
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(image_sc, cmap='gray')
    plt.title('Scatter Plot')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image_hs, cmap='gray', vmin=0, vmax=5 * np.std(image_av))
    plt.title('Histogram')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_av, cmap='gray', vmax=5 * np.std(image_av))
    plt.title('Averaged Shifted Histogram')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # end_time = time.time()
    # print(f'total processing time: {end_time - start_time}s')
    # messagebox.showinfo("Processing Complete",
    # f'Total processing time: {end_time - start_time:.2f}s')

# Function to browse for an input image


def browse_file():
    filename = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.tif;*.tiff;*.png;*.jpg;*.jpeg")])
    input_file_entry.delete(0, tk.END)
    input_file_entry.insert(0, filename)


def save_dataframe():
    # Open a file dialog to select where to save the CSV file
    file_path = filedialog.asksaveasfilename(
        defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df.to_csv(file_path, index=False)
        messagebox.showinfo(
            "Save DataFrame", f"DataFrame saved successfully at {file_path}")

# Function to save the image as a PNG file


def save_image():
    # Open a file dialog to select where to save the image
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[(
        "PNG files", "*.png"), ("TIFF files", "*.tif"), ("JPEG files", "*.jpg")])
    if file_path:
        plt.imsave(file_path, image_av, cmap='gray')
        messagebox.showinfo(
            "Save Image", f"Image saved successfully at {file_path}")


# Setting up the GUI
if __name__ == '__main__':
    root = tk.Tk()
    root.title("Image Processing GUI")

    # Configure the grid layout
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    # Input file selection
    input_file_label = tk.Label(root, text="Select Input Image:")
    input_file_label.grid(row=0, column=0, pady=5, padx=5, sticky="W")

    input_file_entry = tk.Entry(root, width=50)
    input_file_entry.grid(row=0, column=1, pady=5, padx=5)

    browse_button = tk.Button(root, text="Browse", command=browse_file)
    browse_button.grid(row=0, column=2, pady=5, padx=5)

    # First row for dropdown menus
    threshold_label = tk.Label(root, text="Threshold:")
    threshold_label.grid(row=1, column=0, pady=5, padx=5, sticky="W")

    threshold_combo = ttk.Combobox(root, values=[
                                   "1 - mean thresholding", "2 - adaptive thresholding", "3 - Otsu's thresholding"])
    threshold_combo.grid(row=1, column=1, pady=5, padx=5)
    threshold_combo.current(0)

    crude_localization_type_label = tk.Label(
        root, text="Select Crude Localization Type:")
    crude_localization_type_label.grid(
        row=2, column=0, pady=5, padx=5, sticky="W")

    crude_localization_type_combo = ttk.Combobox(
        root, values=["1 - Peak Local Maxima", "2 - Centre of Mass", "3 - Blob Detection"])
    crude_localization_type_combo.grid(row=2, column=1, pady=5, padx=5)
    crude_localization_type_combo.current(0)

    reconstruction_type_label = tk.Label(
        root, text="Select Reconstruction Type:")
    reconstruction_type_label.grid(row=3, column=0, pady=5, padx=5, sticky="W")

    reconstruction_type_combo = ttk.Combobox(
        root, values=["1 - Crude Localization", "2 - Sub Pixel Localization"])
    reconstruction_type_combo.grid(row=3, column=1, pady=5, padx=5)
    reconstruction_type_combo.current(1)

    sub_pixel_localization_method_label = tk.Label(
        root, text="Select Sub Pixel Localization Method:")
    sub_pixel_localization_method_label.grid(
        row=4, column=0, pady=5, padx=5, sticky="W")

    sub_pixel_localization_method_combo = ttk.Combobox(
        root, values=["1 - 1D LS Fitting", "2 - 2D LS Fitting", "3 - MLE", "4 - Phasor"])
    sub_pixel_localization_method_combo.grid(row=4, column=1, pady=5, padx=5)
    sub_pixel_localization_method_combo.current(1)

    # Sliders
    psf_radius_label = tk.Label(root, text="PSF Radius:")
    psf_radius_label.grid(row=5, column=0, pady=5, padx=5, sticky="W")

    psf_radius_slider = tk.Scale(root, from_=1, to=20, orient=tk.HORIZONTAL)
    psf_radius_slider.grid(row=5, column=1, pady=5, padx=5)
    psf_radius_slider.set(5)

    neighbourhood_size_label = tk.Label(root, text="Neighbourhood Size:")
    neighbourhood_size_label.grid(row=6, column=0, pady=5, padx=5, sticky="W")

    neighbourhood_size_slider = tk.Scale(
        root, from_=1, to=20, orient=tk.HORIZONTAL)
    neighbourhood_size_slider.grid(row=6, column=1, pady=5, padx=5)
    neighbourhood_size_slider.set(5)

    window_size_label = tk.Label(root, text="Window Size:")
    window_size_label.grid(row=7, column=0, pady=5, padx=5, sticky="W")

    window_size_slider = tk.Scale(root, from_=1, to=20, orient=tk.HORIZONTAL)
    window_size_slider.grid(row=7, column=1, pady=5, padx=5)
    window_size_slider.set(5)

    # Buttons row
    process_button = tk.Button(
        root, text="Process Images", command=run_processing)
    process_button.grid(row=8, column=0, pady=20, padx=5)

    save_dataframe_button = tk.Button(
        root, text="Save DataFrame", command=save_dataframe)
    save_dataframe_button.grid(row=8, column=1, pady=5, padx=5)

    save_image_button = tk.Button(root, text="Save Image", command=save_image)
    save_image_button.grid(row=8, column=2, pady=5, padx=5)

    # Start the GUI main loop
    root.mainloop()


