# main_gui.py
# The main application file with a comprehensive GUI for SMLM analysis.


import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

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

# function to process each frame
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
    except Exception as e:
        print(f"Error processing frame {frame_index}: {e}")
        return np.array([])

class SMLMAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SMLM Analyzer")
        self.root.geometry("1200x900")

        #  customtkinter theme and appearence
        ctk.set_appearance_mode("Dark") ## Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("dark-blue") ## Options: "blue", "green", "dark-blue"

        #  Font configuration 
        self.TITLE_FONT = ("Helvetica", 16, "bold")

        dark_bg = "#424242"  
        text_color = "#DCE4EE" 

        # Matplotlib global style configuration
        plt.rcParams.update({
            'figure.facecolor': dark_bg,    
            'axes.facecolor':   dark_bg,    
            'axes.edgecolor':   text_color, 
            'axes.labelcolor':  text_color, 
            'xtick.color':      text_color, 
            'ytick.color':      text_color, 
            'text.color':       text_color, 
            'figure.titlesize': self.TITLE_FONT[1], 
            'axes.titlecolor':  text_color  
        })

        # Application state variables
        self.image_stack = None
        self.final_coords = None
        self.image_shape = None
        self.num_frames = None
        self.rendered_image_data = None

        main_frame = ctk.CTkFrame(root, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_panel = ctk.CTkFrame(main_frame, width=400)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self._create_control_widgets(control_panel)

        display_panel = ctk.CTkFrame(main_frame, fg_color="transparent")
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ctk.CTkTabview(display_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.add('Raw Image')
        self.notebook.add('Threshold Preview')
        self.notebook.add('Rendered Result')
        
        self.fig_raw, self.ax_raw = plt.subplots()
        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.notebook.tab('Raw Image'))
        self.canvas_raw.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_preview, self.ax_preview = plt.subplots()
        self.canvas_preview = FigureCanvasTkAgg(self.fig_preview, master=self.notebook.tab('Threshold Preview'))
        self.canvas_preview.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_render, self.ax_render = plt.subplots()
        self.canvas_render = FigureCanvasTkAgg(self.fig_render, master=self.notebook.tab('Rendered Result'))
        self.canvas_render.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_control_widgets(self, parent):
        """Creates all the widgets for the control panel."""
        
        #  File Loading 
        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(padx=10, pady=10, fill=tk.X)
        ctk.CTkLabel(file_frame, text="Load Image Stack", font=self.TITLE_FONT).pack(pady=(5,10), padx=10, anchor="w")
        ctk.CTkButton(file_frame, text="Browse for TIF Stack", command=self.load_image).pack(pady=5, padx=10, fill=tk.X)
        self.file_label = ctk.CTkLabel(file_frame, text="No file loaded.", wraplength=350)
        self.file_label.pack(pady=5, padx=10)

        #  Pipeline Configuration 
        pipe_frame = ctk.CTkFrame(parent)
        pipe_frame.pack(padx=10, pady=10, fill=tk.X)
        pipe_frame.grid_columnconfigure((1,2), weight=1)
        ctk.CTkLabel(pipe_frame, text="Analysis Pipeline", font=self.TITLE_FONT).grid(row=0, column=0, columnspan=3, pady=(5,10), padx=10, sticky="w")
        self.params = {}
        
        ctk.CTkLabel(pipe_frame, text="Filtering:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['filter_method'] = tk.StringVar(value='Gaussian')
        ctk.CTkOptionMenu(pipe_frame, variable=self.params['filter_method'], values=['Gaussian', 'Mean', 'LoG', 'DoG', 'None']).grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        ctk.CTkLabel(pipe_frame, text="Thresholding:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['threshold_method'] = tk.StringVar(value='Adaptive')
        ctk.CTkOptionMenu(pipe_frame, variable=self.params['threshold_method'], values=['Adaptive', 'Otsu', 'Manual']).grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        self.manual_thresh_frame = ctk.CTkFrame(pipe_frame, fg_color="transparent")
        self.manual_thresh_value = tk.DoubleVar()
        self.manual_thresh_slider = ctk.CTkSlider(self.manual_thresh_frame, from_=0, to=65535, variable=self.manual_thresh_value, command=lambda e: self.preview_threshold())
        self.manual_thresh_slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.manual_thresh_display = ctk.CTkLabel(self.manual_thresh_frame, textvariable=self.manual_thresh_value, width=40)
        self.manual_thresh_display.pack(side=tk.LEFT, padx=5)
        
        ctk.CTkLabel(pipe_frame, text="Crude Localization:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['crude_method'] = tk.StringVar(value='Peak Local Max')
        ctk.CTkOptionMenu(pipe_frame, variable=self.params['crude_method'], values=['Peak Local Max', 'Center of Mass', 'Blob Detection (LoG)']).grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)

        ctk.CTkLabel(pipe_frame, text="Sub-pixel Localization:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['subpixel_method'] = tk.StringVar(value='Gaussian Fit (LS)')
        ctk.CTkOptionMenu(pipe_frame, variable=self.params['subpixel_method'], values=['Gaussian Fit (LS)', 'Gaussian Fit (MLE)', 'Phasor']).grid(row=5, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        ctk.CTkLabel(pipe_frame, text="Window Size:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['box_size_entry'] = ctk.CTkEntry(pipe_frame, width=120)
        self.params['box_size_entry'].insert(0, "7")
        self.params['box_size_entry'].grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        ctk.CTkButton(pipe_frame, text="Preview Threshold", command=self.preview_threshold).grid(row=7, column=0, columnspan=3, pady=10, sticky=tk.EW, padx=5)

        #  Run Control 
        run_frame = ctk.CTkFrame(parent)
        run_frame.pack(padx=10, pady=10, fill=tk.X)
        ctk.CTkLabel(run_frame, text="Execute", font=self.TITLE_FONT).pack(pady=(5,10), padx=10, anchor="w")
        ctk.CTkButton(run_frame, text="RUN FULL ANALYSIS", command=self.run_analysis, height=40, font=("Helvetica", 15, 'bold')).pack(padx=10, pady=10, fill=tk.X)
        
        self.progress_label = ctk.CTkLabel(run_frame, text="Status: Idle")
        self.progress_label.pack(pady=5)
        self.progress_bar = ctk.CTkProgressBar(run_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=5, padx=10, fill=tk.X)

        #  Render and Export Control Frame 
        self.render_export_frame = ctk.CTkFrame(parent)
        self.render_export_frame.pack(side=tk.BOTTOM, padx=10, pady=10, fill=tk.X)
        self.render_export_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.render_export_frame, text="Render & Export", font=self.TITLE_FONT).grid(row=0, column=0, columnspan=2, pady=(5,10), padx=10, sticky="w")

        ctk.CTkLabel(self.render_export_frame, text="Rendering:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['render_method'] = tk.StringVar(value='Gaussian')
        self.render_menu = ctk.CTkOptionMenu(self.render_export_frame, variable=self.params['render_method'], values=['Gaussian', '2D Histogram', 'ASH', 'Scatter Plot'])
        self.render_menu.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)

        ctk.CTkLabel(self.render_export_frame, text="Magnification:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.params['magnification_entry'] = ctk.CTkEntry(self.render_export_frame)
        self.params['magnification_entry'].insert(0, "10.0")
        self.params['magnification_entry'].grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        self.rerender_button = ctk.CTkButton(self.render_export_frame, text="Rerender Image", command=self.rerender_image)
        self.rerender_button.grid(row=3, column=0, columnspan=2, pady=10, padx=5, sticky=tk.EW)
        
        self.save_loc_button = ctk.CTkButton(self.render_export_frame, text="Save Localizations (CSV)", command=self.save_localizations)
        self.save_loc_button.grid(row=4, column=0, padx=5, pady=5, sticky=tk.EW)
        
        self.save_img_button = ctk.CTkButton(self.render_export_frame, text="Save Rendered Image Data", command=self.save_rendered_image)
        self.save_img_button.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)

        self.params['threshold_method'].trace_add('write', self._on_threshold_method_change)
        self._on_threshold_method_change()
        self._toggle_post_analysis_controls('disabled') # Initially disable render/export

    def _toggle_post_analysis_controls(self, state='disabled'):
        """ ## NEW ## Helper to enable/disable rendering and export widgets."""
        for widget in [self.render_menu, self.params['magnification_entry'], self.rerender_button, self.save_loc_button, self.save_img_button]:
            widget.configure(state=state)

    def _on_threshold_method_change(self, *args):
        """Callback to show/hide the manual threshold slider."""
        if self.params['threshold_method'].get() == 'Manual':
            self.manual_thresh_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=2)
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
            self.file_label.configure(text=f"{filepath.split('/')[-1]} ({self.num_frames} frames)")
            
            img_min, img_max = self.image_stack.min(), self.image_stack.max()
            self.manual_thresh_slider.configure(from_=img_min, to=img_max)
            self.manual_thresh_value.set((img_min + img_max) / 4)
            
            self.ax_raw.clear()
            self.ax_raw.imshow(np.sum(self.image_stack, axis = 0), cmap='gray')
            self.ax_raw.set_title(f"Summed image of {self.num_frames} frames \n Image dimensions: {self.image_shape[1]} X {self.image_shape[0]} ")
            self.ax_raw.axis('off')
            self.fig_raw.tight_layout() 
            self.canvas_raw.draw()
            self.notebook.set('Raw Image')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image stack: {e}")

    def preview_threshold(self):
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
            self.ax_preview.imshow(mask*preview_image, cmap='gray')
            self.ax_preview.axis('off')
            self.ax_preview.set_title(f"Threshold Preview ({thresh_method})")
            self.fig_preview.tight_layout() 
            self.canvas_preview.draw()
            self.notebook.set('Threshold Preview') 
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate preview: {e}")

    def run_analysis(self):
        """ ## MODIFIED ## Runs only localization, then triggers one initial render."""
        if self.image_stack is None:
            messagebox.showwarning("Warning", "Please load an image stack first.")
            return

        try:
            start_time = time.time()
            self.progress_bar.set(0)
            self.final_coords = None # Reset previous results
            self._toggle_post_analysis_controls('disabled') # Disable buttons during run
            
            analysis_params = {
                'filter': {'method': self.params['filter_method'].get()},
                'threshold': {'method': self.params['threshold_method'].get()},
                'crude': {'method': self.params['crude_method'].get()},
                'subpixel': {'method': self.params['subpixel_method'].get(), 'box_size': int(self.params['box_size_entry'].get())}
            }
            
            analysis_params['filter'].update({'sigma': 1.5, 'size': 3, 'low_sigma': 1, 'high_sigma': 2.5})
            box_size = int(self.params['box_size_entry'].get()) 
            psf_size = (box_size - 1) // 2
            analysis_params['crude'].update({'min_distance': psf_size, 'neighborhood_size': psf_size, 'max_sigma': psf_size, 'threshold': 1 * np.mean(self.image_stack)})

            if analysis_params['threshold']['method'] == 'Manual':
                analysis_params['threshold']['threshold_value'] = self.manual_thresh_value.get()
            
            worker_func = partial(process_frame, params=analysis_params)
            
            num_cores = multiprocessing.cpu_count()
            print(f"Starting analysis on {num_cores} cores...")
            with multiprocessing.Pool(processes=num_cores) as pool:
                all_data_list = []
                for i, result in enumerate(pool.imap_unordered(worker_func, enumerate(self.image_stack))):
                    if result.size > 0: 
                        all_data_list.append(result)
                    progress_value = (i + 1) / self.num_frames
                    self.progress_bar.set(progress_value)
                    self.progress_label.configure(text=f"Processing: {i + 1}/{self.num_frames}")
                    self.root.update_idletasks()

            if not all_data_list:
                messagebox.showinfo("Info", "Analysis complete, but no localizations found.")
                self.progress_label.configure(text="Status: Idle")
                return

            self.final_coords = np.vstack(all_data_list)
            
            end_time = time.time()
            self.progress_label.configure(text="Status: Done!")
            messagebox.showinfo("Success", f"Analysis complete in {end_time - start_time:.2f}s!\nFound {len(self.final_coords)} total localizations.")

            #  Trigger initial render and enable controls 
            self.rerender_image()
            self._toggle_post_analysis_controls('normal')

        except Exception as e:
            self.progress_label.configure(text="Status: Error!")
            self._toggle_post_analysis_controls('disabled')
            messagebox.showerror("Error", f"An error occurred: {e}")

    def rerender_image(self):
        """ ## NEW ## Renders the image using current GUI parameters without re-analyzing."""
        if self.final_coords is None:
            messagebox.showwarning("Warning", "Please run analysis first to get localization data.")
            return
            
        try:
            self.progress_label.configure(text="Rendering image...")
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
            self.notebook.set('Rendered Result')
            
            self.progress_label.configure(text="Status: Done!")
            
        except Exception as e:
            self.progress_label.configure(text="Status: Render Error!")
            messagebox.showerror("Render Error", f"Could not render the image: {e}")

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
            messagebox.showinfo("Success", f"Successfully saved localizations to:\n{filepath}")
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
    root = ctk.CTk()
    app = SMLMAnalyzerApp(root)
    root.mainloop()