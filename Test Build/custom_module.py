# The code in this cell imports all necessary Python packages and sets Plots to be interactive in the notebook 
# Custom Functions for user creation that are just separated from the main module

import os
import numpy as np
import skimage.io as io
from scipy import optimize, stats
import pandas as pd
import math
import inspect
import time
from scipy import signal
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import colors
# %matplotlib notebook

from skimage import filters, measure, morphology, segmentation, util
from skimage.filters import try_all_threshold
from skimage.feature import peak_local_max, canny
from scipy import optimize, ndimage
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, watershed
from skimage.color import label2rgb, rgb2gray
from skimage.morphology import remove_small_objects, remove_small_holes

from ALAN_module import *

# Error with unspecified variable
def z_projection_with_cutoffs(df, image, **kwargs):
    section = kwargs.get('section', (False, False, False, False))
    invert = kwargs.get('invert', False)
    bottom, top, left, right = section
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    actin_channel = kwargs.get('actin_channel', 1)

    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    min_layer_height, max_layer_height, layer_height, actin_peak, norm_intensities_actin = layer_height_actin(df, image,
                                                                                                              actin_channel=actin_channel,
                                                                                                              invert=invert,
                                                                                                              section=section)
    df_cleared = clear_debris(df)

    density = len(df_cleared) / (x_max - x_min) ** 2 * 1000
    if density <= 2:
        bot_cutoff = 0.5
        top_cutoff = 0.6
    elif density >= 6:
        bot_cutoff = 0.2
        top_cutoff = 0.8
    else:
        bot_cutoff = 0.5 - 0.3 * (density - 2) / 4
        top_cutoff = 0.6 + 0.2 * (density - 2) / 4
    print(density, bot_cutoff, top_cutoff)
    min_actin_slice = np.argwhere(norm_intensities_actin >= bot_cutoff)[0][0]
    max_actin_slice = np.argwhere(norm_intensities_actin >= top_cutoff)[-1][0]
    actin_peak_slice = np.argwhere(norm_intensities_actin == 1)[0][0]

    im = image
    if plot == True and invert == True:
        x = np.arange(512)
        ymin = min_actin_slice * np.ones_like(x)
        ymax = max_actin_slice * np.ones_like(x)
        peak = actin_peak_slice * np.ones_like(x)
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.imshow(np.sum(im[:, actin_channel, :, :], axis=2)[::-1], origin='lower', cmap='Greys') # E1 im unspecified
        ax.plot(x, ymin, color='r')
        ax.plot(x, ymax, color='r')
        ax.plot(x, peak, color='r')
        ax.plot()
    else:
        if plot == True and invert != True:
            x = np.arange(512)
            mi, ma = z_projection_with_cutoffs(df, im) # ERROR1 im unspecified
            ymin = min_actin_slice * np.ones_like(x)
            ymax = max_actin_slice * np.ones_like(x)
            peak = actin_peak_slice * np.ones_like(x)
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.imshow(np.sum(im[:, actin_channel, :, :], axis=2), origin='lower', cmap='Greys') # ERROR1 im unspecified
            ax.plot(x, ymin, color='r')
            ax.plot(x, ymax, color='r')
            ax.plot(x, peak, color='r')
    if save_name != False:
        plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)
    print(min_actin_slice, max_actin_slice)

    return min_actin_slice, max_actin_slice

# Error at 'minlayerheight' parameter
def fake_layers(df, image, **kwargs):
    image = image_shuffle(image)
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    fake_type = kwargs.get('fake_type', False)
    np.random.seed(42)
    actin_channel = kwargs.get('actin_channel', 1)
    section = kwargs.get('section', (False, False, False, False))
    invert = kwargs.get('invert', False)
    bottom, top, left, right = section

    if top == False:
        xy_proj_actin = np.sum(np.sum(image, axis=3), axis=2)[:, actin_channel].copy()
    else:
        xy_proj_actin = np.sum(np.sum(image[:, :, bottom:top, left:right], axis=3), axis=2)[:, actin_channel].copy()

    if invert == True:
        xy_proj_actin = xy_proj_actin.copy()[::-1]

    norm_intensities_actin = (xy_proj_actin - np.min(xy_proj_actin)) / np.max(xy_proj_actin - np.min(xy_proj_actin))

    underlying_layer = np.random.normal(7, 1, 300)
    cells_on_top_double = np.random.normal(14, 1, 300)
    cells_on_top_ball = np.random.normal(24, 5, 700)
    cells_on_top_mountain = np.random.exponential(10, 700)+12
        
    double = np.concatenate([underlying_layer, cells_on_top_double])
    mountain = np.concatenate([underlying_layer, cells_on_top_mountain])
    ball = np.concatenate([underlying_layer, cells_on_top_ball])
    
    min_actin_slice = np.argwhere(norm_intensities_actin >= 0.3)[0][0]
    min_layer_height = slice_heights[min_actin_slice]
    
    if fake_type == False:
        return "That does not compute."
    elif fake_type == 'double':
        zs = double
        cells_on_top = cells_on_top_double
    elif fake_type == 'mountain':
        zs = mountain
        cells_on_top = cells_on_top_mountain
    elif fake_type == 'ball':
        zs = ball
        cells_on_top = cells_on_top_ball
    else:
        return "That does not compute."
    
    bins = np.linspace(0, 50, 51)
    bins_fit = np.linspace(0, 50, 1000)
    counts, bins_aux = np.histogram(zs, bins = bins)
    counts_to_fit = smooth_array(counts)
    
    peak_height_guess = np.max(counts_to_fit)
    peak_loc_guess = bins[np.argwhere(counts_to_fit==peak_height_guess)[0][0]]
            
            
    p0_double = [peak_height_guess, peak_loc_guess, 2, 30, 2*peak_loc_guess, 5]
    p0_single = [peak_height_guess, peak_loc_guess, 2]
    print(p0_double, p0_single)
    bounds_double = ([0, 1, 1, 0, 1, 1], [400, 80, 15, 400, 80, 15])
    bounds_single = ([0, 1, 1], [400, 80, 15])
    print(bounds_double, bounds_single)
    peak_loc = np.argwhere(counts_to_fit==np.max(counts_to_fit))[0][0]
    params_double, params_covariance_double = optimize.curve_fit(double_gaussian_fit, bins[:-1], counts_to_fit, p0_double, bounds = bounds_double)
    params_single, params_covariance_single = optimize.curve_fit(gaussian_fit, bins[:-1], counts_to_fit, p0_single, bounds = bounds_single)
    fit_single = gaussian_fit(bins_fit, params_single[0], params_single[1], params_single[2])
    fit_double = double_gaussian_fit(bins_fit, params_double[0], params_double[1], params_double[2], params_double[3], params_double[4], params_double[5])
    
    mean_deviation = np.sqrt(np.mean((fit_single - fit_double)**2))
    
    if mean_deviation >5:
        peaks = 2
        if params_double[0]>=params_double[3]:
            tall_peak = params_double[:3]
            short_peak = params_double[3:]
        else:
            tall_peak = params_double[3:] 
            short_peak = params_double[:3]
        
        if params_double[1]<=params_double[4]:
            left_peak = params_double[:3]
            right_peak = params_double[3:]
        else:
            left_peak = params_double[3:] 
            right_peak = params_double[:3]
        
        a1, b1, c1 = left_peak
        a2, b2, c2 = right_peak
    
        A = 1/c2**2-1/c1**2
        B = 2*b1/c1**2-2*b2/c2**2
        C = (b2/c2)**2-(b1/c1)**2+np.log(a1/a2)
    
        roots = np.roots([A,B,C])
        intersection = np.max(gaussian_fit(roots, a1,b1,c1))
        same_spot_value = intersection / short_peak[0]

        if same_spot_value <= 0.5:
            layer = 'organized'
            nuclear_peak = left_peak[1]
        else:
            layer = 'disorganized'
            nuclear_peak = left_peak[1]
    else:
        peaks = 1
        same_spot_value = 'undef'
        if params_single[2] > -0.372*(params_single[1]-min_layer_height) + 3.572: # ERROR1 no min_layer_height param?
            layer = 'disorganized'
            nuclear_peak = params_single[1]
        else:
            layer = 'organized'
            nuclear_peak = params_single[1]
    print(layer)
    
    counts_in, bins_in = np.histogram(underlying_layer, bins = bins)
    counts_on, bins_on = np.histogram(cells_on_top, bins = bins)
    
    if (plot == True or save_name != False):
        fig, ax = plt.subplots()
        
        if peaks == 1:
            ax.plot(bins_fit, fit_single, label = 'Single Gaussian Fit', linewidth = 3)
        elif peaks == 2:
            ax.plot(bins_fit, fit_double, label = 'Double Gaussian Fit', linewidth = 3)
        else:
            return 'peaks suck'
        ax.scatter(bins[:-1], counts_in, label = 'Artificial Layer', c = 'orange')
        ax.scatter(bins[:-1], counts_on, label = 'Artificial Cells on Top', c = 'g')
        
        ax.legend(loc = 'upper right')
        ax.set_ylabel('Number of nuclei')
        ax.set_xlabel('Z Position ($\mu$m)')
        ax.set_xlim(0, 50)
        fig.show
        
        if save_name != False:
            plt.savefig(save_name + '.pdf', bbox_inches = 'tight', pad_inches = 1)
