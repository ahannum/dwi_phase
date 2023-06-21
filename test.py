# Import utilities
from cmcrameri import cm
from helper_phase_utils import *
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go



# Load data
directory = '/Users/arielhannum/Documents/Stanford/CMR-Setsompop/Projects/Phase/Data/Brain' #'/Volumes/T7/phase_data/Brain'
color = cm.lajolla_r

import plotly.graph_objects as go
from plotly.subplots import make_subplots

timepoints = np.arange(8)
volunteers = np.arange(10)
# Set up the initial volunteer
initial_volunteer = 1

slice = 3
diffusion = -1

# Load the initial images
initial_image_m0, __, initial_mag_m0, initial_mask_m0  = load_image(0, 0, volunteers[initial_volunteer], diffusion, slice, directory)
initial_image_m1, __, initial_mag_m1, initial_mask_m1  = load_image(1, 0, volunteers[initial_volunteer], diffusion, slice, directory)
initial_image_m2, __, initial_mag_m2, initial_mask_m2  = load_image(2, 0, volunteers[initial_volunteer], diffusion, slice, directory)


import tkinter as tk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider,RadioButtons,CheckButtons



# Function to update the heatmap based on the slider value
def update_heatmap(val):
    global timepoint 
    timepoint = int(slider.val)
    # Load images for the timepoint 
    image_m0, __, mag_m0, mask_m0  = load_image(0, timepoint, volunteers[initial_volunteer], diffusion, slice, directory)
    image_m1, __, mag_m1, mask_m1  = load_image(1, timepoint, volunteers[initial_volunteer], diffusion, slice, directory)
    image_m2, __, mag_m2, mask_m2  = load_image(2, timepoint, volunteers[initial_volunteer], diffusion, slice, directory)

    #update the background image
    background[0].set_data(image_m0)
    background[1].set_data(image_m1)
    background[2].set_data(image_m2)

    #update the main image
    images[0].set_data(image_m0*mask_m0)
    images[1].set_data(image_m1*mask_m1)
    images[2].set_data(image_m2*mask_m2)

    #update the tracing 
    masks[0].set_data(get_edge(np.nan_to_num(initial_mask_m0)))
    masks[1].set_data(get_edge(np.nan_to_num(initial_mask_m1)))
    masks[2].set_data(get_edge(np.nan_to_num(initial_mask_m2)))

    # update the magnitude images
    mag[0].set_data(mag_m0)
    mag[1].set_data(mag_m1)
    mag[2].set_data(mag_m2)

    # Redraw the figure
    canvas.draw()


# Create a Tkinter window
window = tk.Tk()

# Create a figure and axis for the bar and slider
fig, ax = plt.subplots(2, 3, figsize=(5, 7), dpi = 200, sharey=True,sharex = True)
plt.subplots_adjust(bottom=0.25,wspace=0.03, hspace=0.01)

# Create a slider for the timepoints
slider_ax = plt.axes([0.2, 0.2, 0.6, 0.03], facecolor='lightgray')
slider = Slider(slider_ax, 'Timepoint', 0, 7, valinit=0, valstep=1)

# Plot the initial images for the first timepoint
# Initialize the plot elements
images = [None, None, None]
masks = [None, None, None]
background = [None, None, None]
mag = [None, None, None]

background[0] = ax[1,0].imshow(initial_image_m0,vmin = 0,vmax = math.pi/2,cmap =color,alpha = 0.5,interpolation = 'nearest')
background[1] = ax[1,1].imshow(initial_image_m1,vmin = 0,vmax = math.pi/2,cmap =color,alpha = 0.5,interpolation = 'nearest')
background[2] = ax[1,2].imshow(initial_image_m2,vmin = 0,vmax = math.pi/2,cmap =color,alpha = 0.5,interpolation = 'nearest')

images[0] = ax[1,0].imshow(initial_image_m0*initial_mask_m0,vmin = 0,vmax = math.pi/2,cmap =color,interpolation = 'nearest')
images[1] = ax[1,1].imshow(initial_image_m1*initial_mask_m1,vmin = 0,vmax = math.pi/2,cmap =color,interpolation = 'nearest')
images[2] = ax[1,2].imshow(initial_image_m2*initial_mask_m2,vmin = 0,vmax = math.pi/2,cmap =color,interpolation = 'nearest')

masks[0] = ax[1,0].imshow(get_edge(np.nan_to_num(initial_mask_m0)),vmin = 0,vmax = 1,cmap ='gray_r',interpolation = 'nearest')
masks[1] = ax[1,1].imshow(get_edge(np.nan_to_num(initial_mask_m1)),vmin = 0,vmax = 1,cmap ='gray_r',interpolation = 'nearest')
masks[2] = ax[1,2].imshow(get_edge(np.nan_to_num(initial_mask_m2)),vmin = 0,vmax = 1,cmap ='gray_r',interpolation = 'nearest')

mag[0] = ax[0,0].imshow(initial_mag_m0,vmin = 0,vmax = 500,cmap ='gray',interpolation = 'nearest')
mag[1] = ax[0,1].imshow(initial_mag_m1,vmin = 0,vmax = 500,cmap ='gray',interpolation = 'nearest')
mag[2] = ax[0,2].imshow(initial_mag_m2,vmin = 0,vmax = 500,cmap ='gray',interpolation = 'nearest')

# Set titles and labels 
ax[0,0].set_title('M$_0$',size = 20)
ax[0,1].set_title('M$_1$',size = 20)
ax[0,2].set_title('M$_2$',size = 20)

for jj in range(2):
    for ii in range(3):
        ax[jj,ii].get_xaxis().set_ticks([])
        ax[jj,ii].get_yaxis().set_ticks([])

ax[0, 0].set_ylabel('|M|',rotation = 0, labelpad=15,size = 15)
ax[1, 0].set_ylabel('$\sigma_{\phi}$',rotation = 0, labelpad=15,size = 15)


# Create colorbars for each row
cbar1 = fig.colorbar(mag[0], ax=ax[0, :],fraction=0.03, pad=0.03,aspect = 10,ticks = [0,250,500])
cbar2 = fig.colorbar(images[0], ax=ax[1, :],fraction=0.03, pad=0.03,aspect = 10,ticks = [0,math.pi/4,math.pi/2])

cbar1.ax.set_yticklabels([0,250,500],size =10) 
cbar2.ax.set_yticklabels(['0', '$\pi/4$', '$\pi/2$'],size =10) 

# Create a function to update the heatmap when the button is changed
def change_diffusion(label):
    global diffusion
    type = (label)
    diffusion = 0 if type == 'b=0' else 1 if type == 'G_x' else 2 if type == 'G_y' else 3
    update_heatmap(slider.val)

# Handle radio button
# Create the RadioButtons
radio_ax = plt.axes([0.2, 0.05, 0.3, 0.1], facecolor='lightgray')  # Adjust the position and size as needed
radio = RadioButtons(radio_ax, ('b=0', 'G_x', 'G_y', 'G_z'),active = 0)
# Set the function to be called when a button is clicked
#w= Tk.RadioButton(window, text="b=0", variable=diffusion, value=0,command=change_diffusion)
radio.on_clicked(change_diffusion)

# Attach the update_heatmap function to the slider's on_changed event
slider.on_changed(update_heatmap)


# Create a Tkinter canvas and add the figure to it
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack()


# Start the Tkinter event loop
window.mainloop()
