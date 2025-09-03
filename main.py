# main_gui.py
# The main application file with a comprehensive GUI for SMLM analysis.


import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import multiprocessing
from functools import partial
import time
import pandas as pd

# Import our custom modules
from utilities import filtering
from utilities import thresholding
from utilities import crude_localization
from utilities import subpixel_localization
from utilities import rendering

def process_frame(frame_data, params):
    """Worker function to process a single frame of the image stack."""
    frame_index, frame_image = frame_data
    try:
        filtered_image = filtering.apply_filter(frame_image, **params['filter'])
        mask = thresholding.apply_threshold(filtered_image, **params['threshold'])
        crude_coords = crude_localization.find_crude_localizations(filtered_image, mask, **params['crude'])
        if crude_coords.size == 0: return np.array([])
        refined_coords, additional_info = subpixel_localization.localize(frame_image, crude_coords, **params['subpixel'])
        if refined_coords.size == 0: return np.array([])

        frame_col = np.full((len(refined_coords), 1), frame_index)
        
        full_data = np.hstack((frame_col, refined_coords[:, ::-1], additional_info))
        
        return full_data
        # return refined_coords
    except Exception as e:
        print(f"Error processing frame {frame_index}: {e}")
        return np.array([])

class SMLMAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SMLM Analyzer")
        self.root.geometry("1366x900")

        #  FONT & STYLE CONFIGURATION 
        # 1. Define the fonts you want to use. You can easily change these values.
        DEFAULT_FONT_SIZE = 12
        TITLE_FONT_SIZE = 14
        FONT_FAMILY = "Helvetica"

        # 2. Configure the ttk styles for the GUI elements.
        style = ttk.Style()
        style.theme_use('vista') # Or 'clam', 'alt', 'default', 'classic'

        # Configure all default ttk widgets
        default_font = (FONT_FAMILY, DEFAULT_FONT_SIZE)
        style.configure('.', font=default_font, padding=3)

        # Configure specific ttk widgets for more control
        style.configure('TLabel', font=default_font)
        style.configure('TButton', font=default_font)
        style.configure('TLabelFrame.Label', font=(FONT_FAMILY, TITLE_FONT_SIZE, 'bold')) # For the title of labelframes
        style.configure('Accent.TButton', font=(FONT_FAMILY, 14, 'bold')) # Update your custom style

        # 3. Configure Matplotlib's default font sizes for all plots.
        plt.rcParams.update({
            'font.size': DEFAULT_FONT_SIZE,
            'axes.titlesize': TITLE_FONT_SIZE,
            'axes.labelsize': DEFAULT_FONT_SIZE,
            'xtick.labelsize': DEFAULT_FONT_SIZE - 2,
            'ytick.labelsize': DEFAULT_FONT_SIZE - 2,
            'legend.fontsize': DEFAULT_FONT_SIZE,
        })
        #  END OF FONT & STYLE CONFIGURATION 

        self.image_stack = None
        self.final_coords = None
        self.image_shape = None
        self.num_frames = None
        self.rendered_image_data = None

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_panel = tk.Frame(main_frame, width=400, relief=tk.RIDGE, bd=2)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self._create_control_widgets(control_panel)

        display_panel = tk.Frame(main_frame)
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(display_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Matplotlib figures will now use the rcParams we set above
        self.fig_raw, self.ax_raw = plt.subplots()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.notebook)
        self.notebook.add(self.canvas_raw.get_tk_widget(), text='Raw Image')

        self.fig_preview, self.ax_preview = plt.subplots()
        self.canvas_preview = FigureCanvasTkAgg(self.fig_preview, master=self.notebook)
        self.notebook.add(self.canvas_preview.get_tk_widget(), text='Threshold Preview')

        self.fig_render, self.ax_render = plt.subplots()
        self.canvas_render = FigureCanvasTkAgg(self.fig_render, master=self.notebook)
        self.notebook.add(self.canvas_render.get_tk_widget(), text='Rendered Result')


    def _create_control_widgets(self, parent):
        """Creates all the widgets for the control panel."""
        
        #  File Loading 
        file_frame = ttk.LabelFrame(parent, text="1. Load Image Stack")
        file_frame.pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(file_frame, text="Browse for TIF Stack", command=self.load_image).pack(pady=5, padx=10, fill=tk.X)
        self.file_label = ttk.Label(file_frame, text="No file loaded.", wraplength=350)
        self.file_label.pack(pady=5, padx=10)

        #  Pipeline Configuration 
        pipe_frame = ttk.LabelFrame(parent, text="2. Analysis Pipeline")
        pipe_frame.pack(padx=10, pady=10, fill=tk.X)

        self.params = {}
        
        #  Filtering 
        ttk.Label(pipe_frame, text="Filtering:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['filter_method'] = tk.StringVar(value='Gaussian')
        ttk.OptionMenu(pipe_frame, self.params['filter_method'], 'Gaussian', 'Gaussian', 'Mean', 'Laplacian of Gaussian (LoG)', 'Difference of Gaussians (DoG)', 'None').grid(row=0, column=1, columnspan=2, sticky=tk.EW)

        #  Thresholding 
        ttk.Label(pipe_frame, text="Thresholding:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['threshold_method'] = tk.StringVar(value='Adaptive')
        ttk.OptionMenu(pipe_frame, self.params['threshold_method'], 'Adaptive', 'Adaptive', 'Otsu', 'Manual').grid(row=1, column=1, columnspan=2, sticky=tk.EW)
        
        #  Manual Threshold Slider (initially hidden) 
        self.manual_thresh_frame = ttk.Frame(pipe_frame)
        self.manual_thresh_label = ttk.Label(self.manual_thresh_frame, text="Threshold:")
        self.manual_thresh_label.pack(side=tk.LEFT, padx=5)
        self.manual_thresh_value = tk.DoubleVar()
        self.manual_thresh_slider = ttk.Scale(self.manual_thresh_frame, from_=0, to=65535, variable=self.manual_thresh_value, orient=tk.HORIZONTAL, command=lambda e: self.preview_threshold())
        self.manual_thresh_slider.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.manual_thresh_display = ttk.Label(self.manual_thresh_frame, textvariable=self.manual_thresh_value, width=6, font=('Helvetica', 14, 'bold'))
        self.manual_thresh_display.pack(side=tk.LEFT, padx=5)
        
        #  Other Parameters 
        ttk.Label(pipe_frame, text="Crude Localization:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['crude_method'] = tk.StringVar(value='Peak Local Max')
        ttk.OptionMenu(pipe_frame, self.params['crude_method'], 'Peak Local Max', 'Peak Local Max', 'Center of Mass', 'Blob Detection (LoG)').grid(row=3, column=1, columnspan=2, sticky=tk.EW)

        ttk.Label(pipe_frame, text="Sub-pixel Localization:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['subpixel_method'] = tk.StringVar(value='Gaussian Fit (LS)')
        ttk.OptionMenu(pipe_frame, self.params['subpixel_method'], 'Gaussian Fit (LS)', 'Gaussian Fit (MLE)', 'Gaussian Fit (LS)', 'Phasor').grid(row=4, column=1, columnspan=2, sticky=tk.EW)
        
        ttk.Label(pipe_frame, text="Window Size:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['box_size_entry'] = ttk.Entry(pipe_frame, width=10)
        self.params['box_size_entry'].insert(0, "7")
        self.params['box_size_entry'].grid(row=5, column=1, sticky=tk.W)

        ttk.Label(pipe_frame, text="Rendering:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['render_method'] = tk.StringVar(value='Gaussian')
        ttk.OptionMenu(pipe_frame, self.params['render_method'], 'Gaussian', 'Gaussian', '2D Histogram', 'ASH','Scatter Plot').grid(row=6, column=1, columnspan=2, sticky=tk.EW)

        ttk.Label(pipe_frame, text="Magnification:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['magnification_entry'] = ttk.Entry(pipe_frame, width=10)
        self.params['magnification_entry'].insert(0, "10.0")
        self.params['magnification_entry'].grid(row=7, column=1, sticky=tk.W)
        
        ttk.Button(pipe_frame, text="Preview Threshold", command=self.preview_threshold).grid(row=8, column=0, columnspan=3, pady=10, sticky=tk.EW)

        #  Run Control 
        run_frame = ttk.LabelFrame(parent, text="3. Execute")
        run_frame.pack(padx=10, pady=10, fill=tk.X)
        ttk.Button(run_frame, text="RUN FULL ANALYSIS", command=self.run_analysis, style='Accent.TButton').pack(padx=10, pady=10, fill=tk.X, ipady=10)
        
        self.progress_label = ttk.Label(run_frame, text="Status: Idle")
        self.progress_label.pack(pady=5)
        self.progress_bar = ttk.Progressbar(run_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(pady=5, padx=10, fill=tk.X)

        #  Export Control 
        export_frame = ttk.LabelFrame(parent, text="4. Export Results")
        export_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)
        # export_frame.pack(padx=10, pady=10, fill=tk.X, expand=True)
        ttk.Button(export_frame, text="Save Localizations (CSV)", command=self.save_localizations).pack(side=tk.LEFT, expand=True, padx=5, pady=5)
        ttk.Button(export_frame, text="Save Rendered Image Data", command=self.save_rendered_image).pack(side=tk.RIGHT, expand=True, padx=5, pady=5)
        
        # This trace must be set AFTER the manual_thresh_frame is created
        self.params['threshold_method'].trace('w', self._on_threshold_method_change)
        self._on_threshold_method_change() # Call once to set initial state

    def _on_threshold_method_change(self, *args):
        """Callback to show/hide the manual threshold slider."""
        if self.params['threshold_method'].get() == 'Manual':
            self.manual_thresh_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        else:
            self.manual_thresh_frame.grid_forget()

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("TIF files", "*.tif;*.tiff")])
        if not filepath: return
        
        try:
            self.image_stack = tifffile.imread(filepath)
            if self.image_stack.ndim == 2: self.image_stack = self.image_stack[np.newaxis, :, :]
            
            self.image_shape = self.image_stack[0].shape
            self.num_frames = len(self.image_stack)
            self.file_label.config(text=f"{filepath.split('/')[-1]} ({self.num_frames} frames)")
            
            img_min = self.image_stack.min()
            img_max = self.image_stack.max()
            self.manual_thresh_slider.config(from_=img_min, to=img_max)
            self.manual_thresh_value.set((img_min + img_max) / 4)
            
            self.ax_raw.clear()
            self.ax_raw.imshow(np.sum(self.image_stack, axis = 0), cmap='gray')
            self.ax_raw.set_title(f"Summed image of {self.num_frames} frames \n Image dimensions: {self.image_shape[1]} X {self.image_shape[0]} ")
            self.ax_raw.axis('off')
            self.fig_raw.tight_layout() 
            self.canvas_raw.draw()
            self.notebook.select(0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image stack: {e}")

    def preview_threshold(self):
        """Applies the selected filter and threshold to the first frame and displays it."""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        try:
            preview_image = self.image_stack[self.num_frames//2]
            filter_params = {'method': self.params['filter_method'].get(), 'sigma': 1.0}
            filtered_image = filtering.apply_filter(preview_image, **filter_params)
            
            thresh_method = self.params['threshold_method'].get()
            thresh_params = {'method': thresh_method}
            if thresh_method == 'Manual':
                thresh_params['threshold_value'] = self.manual_thresh_value.get()
            
            mask = thresholding.apply_threshold(filtered_image, **thresh_params)
            
            self.ax_preview.clear()
            self.ax_preview.imshow(mask, cmap='gray')
            self.ax_preview.axis('off')
            self.ax_preview.set_title(f"Threshold Preview ({thresh_method})")
            self.fig_preview.tight_layout() 
            self.canvas_preview.draw()
            self.notebook.select(1) 
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate preview: {e}")

    def run_analysis(self):
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load an image stack first.")
            return

        try:
            start_time = time.time()
            num_frames = len(self.image_stack)
            self.progress_bar['maximum'] = num_frames
            
            analysis_params = {
                'filter': {'method': self.params['filter_method'].get(), 'sigma': 1.0},
                'threshold': {'method': self.params['threshold_method'].get()},
                'crude': {'method': self.params['crude_method'].get()},
                'subpixel': {'method': self.params['subpixel_method'].get(), 'box_size': int(self.params['box_size_entry'].get())}
            }
            
            # box_size = int(self.params['box_size_entry'].get())
            
            # filtering kwargs
            analysis_params['filter']['sigma'] = 1.5
            analysis_params['filter']['size'] = 3
            analysis_params['filter']['low_sigma'] = 1
            analysis_params['filter']['high_sigma'] = 2.5

            # crude localization kwargs   
            box_size = int(self.params['box_size_entry'].get())        
            psf_size = (box_size - 1) // 2

            # Calculate the desired PSF size for crude localization
            # PLM
            analysis_params['crude']['min_distance'] = psf_size
            # COM
            analysis_params['crude']['neighborhood_size'] = psf_size

            # Blob
            analysis_params['crude']['max_sigma'] = psf_size
            analysis_params['crude']['threshold'] = 1 * np.mean(self.image_stack)


            if analysis_params['threshold']['method'] == 'Manual':
                analysis_params['threshold']['threshold_value'] = self.manual_thresh_value.get()
            
            worker_func = partial(process_frame, params=analysis_params)
            
            num_cores = multiprocessing.cpu_count()
            print(f"Starting analysis on {num_cores} cores...")
            with multiprocessing.Pool(processes=num_cores) as pool:
                results_iterator = pool.imap_unordered(worker_func, enumerate(self.image_stack))
                
                all_data_list = []
                for i, result in enumerate(results_iterator):
                    if result.size > 0: 
                        all_data_list.append(result)
                    self.progress_bar['value'] = i + 1
                    self.progress_label.config(text=f"Processing: {i + 1}/{num_frames}")
                    self.root.update_idletasks()

            if not all_data_list:
                messagebox.showinfo("Info", "Analysis complete, but no localizations found.")
                self.progress_label.config(text="Status: Idle")
                return

            self.final_coords = np.vstack(all_data_list)

            self.progress_label.config(text="Rendering final image...")
            self.root.update_idletasks()
            render_params = {
                'method': self.params['render_method'].get(),
                'pixel_size': float(self.params['magnification_entry'].get())
            }
            render_coords = self.final_coords[:, 1:3]
            self.ax_render.clear() 
            self.rendered_image_data = rendering.render(render_coords, self.ax_render, self.image_shape, **render_params)
            self.fig_render.tight_layout() 
            self.canvas_render.draw()
            self.notebook.select(2) 
            
            end_time = time.time()
            self.progress_label.config(text="Status: Done!")
            messagebox.showinfo("Success", f"Analysis complete in {end_time - start_time:.2f}s!\nFound {len(self.final_coords)} total localizations.")

        except Exception as e:
            self.progress_label.config(text="Status: Error!")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def save_localizations(self):
        if self.final_coords is None or self.final_coords.size == 0:
            messagebox.showwarning("Warning", "No localization data to save.")
            return
        
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save Localizations As")
        if not filepath: return
        try:
            df = pd.DataFrame(self.final_coords, columns=['frame', 'x_pos', 'y_pos', 'amplitude', 'sigma'])
            df['frame'] = df['frame'] + 1
            df_sorted = df.sort_values(by='frame').reset_index(drop=True)
            df_sorted.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Successfully saved sorted localizations to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

    def save_rendered_image(self):
        if self.rendered_image_data is None:
            messagebox.showwarning("Warning", "No rendered image data to save.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("TIFF Image Data", "*.tif;*.tiff")], title="Save Rendered Image Data As")
        if not filepath: return
        try:
            tifffile.imwrite(filepath, self.rendered_image_data.astype(np.float32))
            messagebox.showinfo("Success", f"Successfully saved rendered image data to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image data: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = SMLMAnalyzerApp(root)
    root.mainloop()
