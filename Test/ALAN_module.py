# The code in this cell imports all necessary Python packages and sets Plots to be interactive in the notebook 

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


# Cell 4
# The code in this cell defines functions of ALAn.
# An explanation of these definitions can be found in the ALAn README.

def image_shuffle(image):
    a, b, c, d = image.shape
    if a == 512:
        xloc = 0
        if b == 512:
            yloc = 1
            if c > 4:
                zloc = 2
                colloc = 3
            else:
                zloc = 3
                colloc = 2
        elif c == 512:
            yloc = 2
            if b > 4:
                zloc = 1
                colloc = 3
            else:
                zloc = 3
                colloc = 1
        else:
            yloc = 3
            if b > 4:
                zloc = 1
                colloc = 2
            else:
                zloc = 2
                colloc = 1
    elif b == 512:
        xloc = 1
        if c == 512:
            yloc = 2
            if a > 4:
                zloc = 0
                colloc = 3
            else:
                zloc = 3
                colloc = 0
        else:
            yloc = 3
            if a > 4:
                zloc = 0
                colloc = 2
            else:
                zloc = 2
                colloc = 0
    else:
        xloc = 2
        yloc = 3
        if a > 4:
            zloc = 0
            colloc = 1
        else:
            zloc = 1
            colloc = 0
    return np.transpose(image, (zloc, colloc, xloc, yloc))


def get_layer_position(df, image):
    image = image_shuffle(image)
    num_z, num_c, num_x, num_y = image.shape
    df.columns = ['x', 'y', 'z', 'vol', 'positions']
    x_max = df['positions'][0]
    y_max = df['positions'][1]
    z_max = df['positions'][2]
    x_min = df['positions'][3]
    y_min = df['positions'][4]
    z_min = df['positions'][5]
    slice_heights = np.linspace(0, z_max - z_min, num=num_z)
    return (x_max, y_max, z_max, x_min, y_min, z_min, slice_heights)


def local_density(df, image, **kwargs):
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    box_radius = kwargs.get('box_radius', (x_max - x_min) / 2)
    df_cleared = clear_debris(df)
    xcent = (x_min + x_max) / 2
    ycent = (y_min + y_max) / 2
    xmin_box, xmax_box = xcent - box_radius, xcent + box_radius
    ymin_box, ymax_box = ycent - box_radius, ycent + box_radius
    df_box = df_cleared[(df_cleared['x'].values >= xmin_box) &
                        (df_cleared['x'].values <= xmax_box) &
                        (df_cleared['y'].values >= ymin_box) &
                        (df_cleared['y'].values <= ymax_box)]
    number_of_cells_in_box = len(df_box)
    box_density = (number_of_cells_in_box / (4 * box_radius ** 2)) * 1000
    return number_of_cells_in_box, box_density


def clear_debris(df, **kwargs):
    df.columns = ['x', 'y', 'z', 'vol', 'positions']
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    cutoff = kwargs.get('cutoff', 1.5)
    vols = df['vol'].values
    bottom_vol = np.mean(vols) - (cutoff * np.std(vols))
    df_cleared = df[df['vol'] >= bottom_vol]
    if plot == True:
        bins = np.linspace(np.min(vols), np.max(vols), num=100)
        fig, ax = plt.subplots()
        ax.hist(vols, bins=bins, alpha=1, label='Debris')
        ax.hist(df_cleared['vol'].values, bins=bins, alpha=1, label='Cells')
        ax.legend(loc='upper right')
        ax.set_ylabel('Number of cells')
        ax.set_xlabel('Nucleus volume ($\mu$m$^3$)')
        ax.axvline(x=bottom_vol, c='k', linestyle='--')
        fig.show
        if save_name != False:
            plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)
    return (df_cleared)


def layer_height_actin(df, image, **kwargs):
    image = image_shuffle(image)
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
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

    actin_peak = slice_heights[np.argwhere(norm_intensities_actin == np.max(norm_intensities_actin))][0][0]
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
    min_layer_height = slice_heights[min_actin_slice]
    max_layer_height = slice_heights[max_actin_slice]
    layer_height = max_layer_height - min_layer_height
    actin_intensity_top = norm_intensities_actin[max_actin_slice]

    if (plot == True):
        fig, ax = plt.subplots()
        ax.plot(slice_heights, norm_intensities_actin, c='cornflowerblue')
        ax.set_ylabel('Actin Intensity (A.U.)')
        ax.set_xlabel('Z-Position ($\mu$m)')
        ax.axvline(x=min_layer_height, c='k', linestyle='--', label='Layer Bounds')
        ax.axvline(x=max_layer_height, c='k', linestyle='--')
        ax.axvline(x=np.max(slice_heights), c='r', linestyle='--')
        ax.legend(loc='upper right')
        ax.set_ylim([0, 1.1])
        ax.set_xlim([0, 30])
        if save_name != False:
            plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)
    return (min_layer_height, max_layer_height, layer_height, actin_peak, norm_intensities_actin)


def smooth_array(array, **kwargs):
    number_to_smooth = kwargs.get('window_size', 3)
    window = np.ones(number_to_smooth) / number_to_smooth
    new_array = np.convolve(array, window, mode='same')
    return new_array


def find_shoulders(df, image, **kwargs):
    image = image_shuffle(image)
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    actin_channel = kwargs.get('actin_channel', 1)
    section = kwargs.get('section', (False, False, False, False))
    invert = kwargs.get('invert', False)
    bottom, top, left, right = section

    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    min_layer_height, max_layer_height, layer_height, actin_peak, norm_intensities_actin = layer_height_actin(df, image,
                                                                                                              actin_channel=actin_channel,
                                                                                                              invert=invert,
                                                                                                              section=section)

    if top == False:
        xy_proj_actin = np.sum(np.sum(image, axis=3), axis=2)[:, actin_channel].copy()
    else:
        xy_proj_actin = np.sum(np.sum(image[:, :, bottom:top, left:right], axis=3), axis=2)[:, actin_channel].copy()

    if invert == True:
        xy_proj_actin = xy_proj_actin.copy()[::-1]

    norm_intensities_actin = (xy_proj_actin - np.min(xy_proj_actin)) / np.max(xy_proj_actin - np.min(xy_proj_actin))
    actin_profile_derivative = smooth_array(np.diff(norm_intensities_actin), window_size=5)
    peaks = signal.find_peaks(actin_profile_derivative, prominence=0.008)

    if len(peaks[0]) >= 2:
        equivalence = peaks[1]['prominences'][1] / peaks[1]['prominences'][0]
    else:
        equivalence = 'undef'

    if plot == True or save_name != False:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        ax.plot(slice_heights, norm_intensities_actin, 'red', label='Actin Profile')
        ax.set_ylabel('Actin Intensity (A.U.)')
        ax.set_xlabel('Z-Position ($\mu$m)')
        ax2 = ax.twinx()
        ax.set_xlim(0, 30)
        ax2.plot(slice_heights[:-1] + (z_max - z_min) / (2 * len(slice_heights)), actin_profile_derivative, 'steelblue',
                 label='Actin Derivative')
        ax2.set_xlabel('Z-Position ($\mu$m)')
        ax2.set_ylabel('Actin Profile 1$^{st}$ Derivative (A.U. / $\mu$m)')
        ax.axvline(x=min_layer_height, c='k', linestyle='--', label='Layer Bounds')
        ax.axvline(x=max_layer_height, c='k', linestyle='--')
        fig.tight_layout()
        fig.show()
        if save_name != False:
            plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)

    return (equivalence, len(peaks[0]))


def gaussian_fit(x, A, B, C):
    return A * np.exp(-1 * ((x - B) / C) ** 2)


def double_gaussian_fit(x, A, B, C, D, E, F):
    return A * np.exp(-1 * ((x - B) / C) ** 2) + D * np.exp(-1 * ((x - E) / F) ** 2)


def nuclei_distribution(df, image, **kwargs):
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    section = kwargs.get('section', (False, False, False, False))
    invert = kwargs.get('invert', False)
    actin_channel = kwargs.get('actin_channel', 1)
    bottom, top, left, right = section
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    min_layer_height, max_layer_height, layer_height, actin_peak, norm_intensities_actin = layer_height_actin(df, image,
                                                                                                              actin_channel=actin_channel,
                                                                                                              invert=invert,
                                                                                                              section=section)

    if top != False:
        xlow = (x_max - x_min) * left / 512 + x_min
        xhigh = (x_max - x_min) * right / 512 + x_min
        ylow = (y_max - y_min) * bottom / 512 + y_min
        yhigh = (y_max - y_min) * top / 512 + y_min
    df_cleared = clear_debris(df)
    if top != False:
        df_cleared = df_cleared[(df_cleared['x'].values >= xlow) &
                                (df_cleared['x'].values <= xhigh) &
                                (df_cleared['y'].values >= ylow) &
                                (df_cleared['y'].values <= yhigh)]

    if invert == True:
        zs = z_max - df_cleared['z'].values
    else:
        zs = df_cleared['z'].values - z_min
    bins = np.linspace(0, 50, 51)
    bins_fit = np.linspace(0, 50, 1000)
    counts, bins_aux = np.histogram(zs, bins=bins)
    counts_to_fit = smooth_array(counts)

    peak_height_guess = np.max(counts_to_fit)
    peak_loc_guess = bins[np.argwhere(counts_to_fit == peak_height_guess)[0][0]]

    p0_double = [peak_height_guess, peak_loc_guess, 2, 30, 2 * peak_loc_guess, 5]
    p0_single = [peak_height_guess, peak_loc_guess, 2]
    print(p0_double, p0_single)
    bounds_double = ([0, 1, 1, 0, 1, 1], [400, 80, 15, 400, 80, 15])
    bounds_single = ([0, 1, 1], [400, 80, 15])
    print(bounds_double, bounds_single)
    peak_loc = np.argwhere(counts_to_fit == np.max(counts_to_fit))[0][0]
    params_double, params_covariance_double = optimize.curve_fit(double_gaussian_fit, bins[:-1], counts_to_fit,
                                                                 p0_double, bounds=bounds_double)
    params_single, params_covariance_single = optimize.curve_fit(gaussian_fit, bins[:-1], counts_to_fit, p0_single,
                                                                 bounds=bounds_single)
    fit_single = gaussian_fit(bins_fit, params_single[0], params_single[1], params_single[2])
    fit_double = double_gaussian_fit(bins_fit, params_double[0], params_double[1], params_double[2], params_double[3],
                                     params_double[4], params_double[5])

    mean_deviation = np.sqrt(np.mean((fit_single - fit_double) ** 2))

    if mean_deviation > 5:
        peaks = 2
        if params_double[0] >= params_double[3]:
            tall_peak = params_double[:3]
            short_peak = params_double[3:]
        else:
            tall_peak = params_double[3:]
            short_peak = params_double[:3]

        if params_double[1] <= params_double[4]:
            left_peak = params_double[:3]
            right_peak = params_double[3:]
        else:
            left_peak = params_double[3:]
            right_peak = params_double[:3]

        a1, b1, c1 = left_peak
        a2, b2, c2 = right_peak

        A = 1 / c2 ** 2 - 1 / c1 ** 2
        B = 2 * b1 / c1 ** 2 - 2 * b2 / c2 ** 2
        C = (b2 / c2) ** 2 - (b1 / c1) ** 2 + np.log(a1 / a2)

        roots = np.roots([A, B, C])
        intersection = np.max(gaussian_fit(roots, a1, b1, c1))
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
        if params_single[2] > -0.372 * (params_single[1] - min_layer_height) + 3.572:
            layer = 'disorganized'
            nuclear_peak = params_single[1]
        else:
            layer = 'organized'
            nuclear_peak = params_single[1]
    print(params_double, params_single, peak_loc_guess)

    if (plot == True or save_name != False):
        fig, ax = plt.subplots()
        ax.scatter(bins[:-1], counts_to_fit, label='Nuclear Position')
        if peaks == 1:
            ax.plot(bins_fit, fit_single, label='Single Gaussian Fit')
        elif peaks == 2:
            ax.plot(bins_fit, fit_double, label='Double Gaussian Fit')
        else:
            return 'peaks suck'

        ax.legend(loc='upper right')
        ax.set_ylabel('Number of nuclei')
        ax.set_xlabel('Z Position ($\mu$m)')
        ax.set_xlim(0, 30)
        ax2 = ax.twinx()
        ax2.plot(slice_heights, norm_intensities_actin, c='r')
        ax2.set_ylabel('Actin Intensity (A.U.)')
        ax2.axvline(x=min_layer_height, c='k', linestyle='--', label='Layer Bounds')
        ax2.axvline(x=max_layer_height, c='k', linestyle='--')
        ax2.axvline(x=actin_peak, c='k', linestyle='--')
        print(min_layer_height, max_layer_height)
        fig.show
        if save_name != False:
            plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)

    return nuclear_peak, layer


def terrain_map(df, image, **kwargs):
    color = kwargs.get('color', 'magma_r')
    save_name = kwargs.get('save_name', False)
    invert = kwargs.get('invert', False)
    df_cleared = clear_debris(df)
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    bottom, x1, x2, x3, x4 = layer_height_actin(df, image)

    xs = df_cleared['x'].values - x_min
    ys = df_cleared['y'].values - y_min
    zs = df_cleared['z'].values - z_min
    vols = df_cleared['vol'].values
    cross_sections = np.pi * (np.cbrt(vols * 3 / (4 * np.pi))) ** 2
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, -ys, s=cross_sections / 2, c=zs, cmap=color, vmin=bottom, vmax=bottom + 20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, x_max - x_min])
    ax.set_ylim([-(y_max - y_min), 0])
    if save_name != False:
        plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)
    return

# Error with unspecified variable
'''
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
'''

def layer_determination(df, image, **kwargs):
    section = kwargs.get('section', (False, False, False, False))
    invert = kwargs.get('invert', False)
    actin_channel = kwargs.get('actin_channel', 1)
    bottom, top, left, right = section

    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    min_layer_height, max_layer_height, layer_height, actin_peak, norm_intensities_actin = layer_height_actin(df, image,
                                                                                                              actin_channel=actin_channel,
                                                                                                              invert=invert,
                                                                                                              section=section)
    equivalence, num_peaks = find_shoulders(df, image, actin_channel=actin_channel, invert=invert, section=section)
    nuclear_peak, layer = nuclei_distribution(df, image, actin_channel=actin_channel, invert=invert, section=section)

    if top != False:
        xlow = (x_max - x_min) * left / 512 + x_min
        xhigh = (x_max - x_min) * right / 512 + x_min
        ylow = (y_max - y_min) * bottom / 512 + y_min
        yhigh = (y_max - y_min) * top / 512 + y_min
    df_cleared = clear_debris(df)
    if top != False:
        df_cleared = df_cleared[(df_cleared['x'].values >= xlow) &
                                (df_cleared['x'].values <= xhigh) &
                                (df_cleared['y'].values >= ylow) &
                                (df_cleared['y'].values <= yhigh)]

    actin_to_top = max_layer_height - actin_peak
    actin_to_bottom = actin_peak - min_layer_height

    peak_difference_rule = (actin_peak - nuclear_peak) >= 0
    actin_peak_position_rule = (actin_to_top - actin_to_bottom) < 0
    deriv_peak_rule = num_peaks == 1

    cells_above = np.sum(df_cleared['z'] - z_min >= max_layer_height)
    cells_inside = np.sum(df_cleared['z'] - z_min < max_layer_height)
    percentage_above = (cells_above / (cells_above + cells_inside) * 100)
    cell_density = (cells_inside + cells_above) / (x_max - x_min) ** 2 * 1000

    if layer == 'disorganized':
        layer_classification = 'Disorganized'

    else:
        if actin_peak_position_rule and peak_difference_rule:
            if deriv_peak_rule:
                layer_classification = 'Intermediate A'
            elif equivalence < 1:
                layer_classification = 'Intermediate B'
            else:
                layer_classification = 'Mature'
        else:
            layer_classification = 'Immature'
    print(layer_height)

    return layer_classification, cells_above, cells_inside, percentage_above, cell_density


def sub_classify(df, image, **kwargs):
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    invert = kwargs.get('invert', False)
    actin_channel = kwargs.get('actin_channel', 1)
    image = image_shuffle(image)
    sub_image = image[:, :, :-2, :-2].copy()
    sampling = kwargs.get('Sampling', 'Grid')
    df_cleared = clear_debris(df)
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    min_layer_height, max_layer_height, layer_height, actin_peak, norm_intensities_actin = layer_height_actin(df, image,
                                                                                                              actin_channel=actin_channel,
                                                                                                              invert=invert)
    layer_classification, cells_above, cells_inside, percentage_above, cell_density = layer_determination(df, image,
                                                                                                          actin_channel=actin_channel,
                                                                                                          invert=invert)
    full_x = (x_max - x_min) * 500 / 512
    full_y = (y_max - y_min) * 500 / 512
    if sampling == 'Grid':
        plot_to_make = np.zeros((500, 500))
        xs = np.linspace(x_min + full_x / 10, x_min + 9 * full_x / 10, 5)
        ys = np.linspace(y_min + full_y / 10, y_min + 9 * full_y / 10, 5)
        lts = np.linspace(0, 400, 5)
        rbs = np.linspace(0, 400, 5)
        a, b = np.meshgrid(xs, ys)
        centers = np.vstack((a.ravel(), b.ravel())).transpose()
        a, b = np.meshgrid(lts, rbs)
        bounds = np.vstack((a.ravel(), b.ravel())).transpose()
        all_sections_analyzed = []
        for i in range(len(centers)):
            a, b, c, d, e = layer_determination(df, image, actin_channel=actin_channel, invert=invert, section=(
            int(bounds[i, 0]), int(bounds[i, 0] + 100), int(bounds[i, 1]), int(bounds[i, 1] + 100)))
            f, g, h, j, k = layer_height_actin(df, image, actin_channel=actin_channel, invert=invert, section=(
            int(bounds[i, 0]), int(bounds[i, 0] + 100), int(bounds[i, 1]), int(bounds[i, 1] + 100)))
            all_sections_analyzed.append((a, b, c, h, layer_classification))
            if a == 'Disorganized':
                plot_to_make[int(bounds[i, 0]):int(bounds[i, 0] + 100),
                int(bounds[i, 1]):int(bounds[i, 1] + 100)] = 5 * np.ones((100, 100))
            elif a == 'Mature':
                plot_to_make[int(bounds[i, 0]):int(bounds[i, 0] + 100),
                int(bounds[i, 1]):int(bounds[i, 1] + 100)] = 4 * np.ones((100, 100))
            elif a == 'Intermediate B':
                plot_to_make[int(bounds[i, 0]):int(bounds[i, 0] + 100),
                int(bounds[i, 1]):int(bounds[i, 1] + 100)] = 3 * np.ones((100, 100))
            elif a == 'Intermediate A':
                plot_to_make[int(bounds[i, 0]):int(bounds[i, 0] + 100),
                int(bounds[i, 1]):int(bounds[i, 1] + 100)] = 2 * np.ones((100, 100))
            elif a == 'Immature':
                plot_to_make[int(bounds[i, 0]):int(bounds[i, 0] + 100),
                int(bounds[i, 1]):int(bounds[i, 1] + 100)] = np.ones((100, 100))
            else:
                return print('How did this mess up?')
        cmap = colors.ListedColormap(['lawngreen', 'forestgreen', 'darkturquoise', 'steelblue', 'orange'])
        if plot == True or save_name != False:
            fig, ax = plt.subplots()
            ax.imshow(plot_to_make, cmap=cmap, alpha=1, vmin=1, vmax=5)
            ax.set_xticks([])
            ax.set_yticks([])
            if save_name != False:
                plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)
    elif sampling == 'Random':
        return print('This method is not yet supported.')
    else:
        return print('This method is not yet supported.')
    return all_sections_analyzed


def nuclear_centroid_actin_overlay(df, image, **kwargs):
    image = image_shuffle(image)
    df_cleared = clear_debris(df)
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    actin_channel = kwargs.get('actin_channel', 1)
    save_name = kwargs.get('save_name', False)
    xy_proj_actin = np.sum(np.sum(image, axis=3), axis=2)[:, actin_channel].copy()
    norm_intensities_actin = (xy_proj_actin - np.min(xy_proj_actin)) / np.max(xy_proj_actin - np.min(xy_proj_actin))
    min_actin_slice = np.argwhere(norm_intensities_actin >= 0.3)[0][0]
    max_actin_slice = np.argwhere(norm_intensities_actin >= 0.5)[-1][0]
    sum_proj_z = np.sum(image[min_actin_slice:max_actin_slice, actin_channel, :, :], axis=0)
    df_clear_in_layer = df_cleared[df_cleared['z'] > (slice_heights[min_actin_slice] + z_min)]
    df_clear_in_layer = df_clear_in_layer[df_clear_in_layer['z'] < (slice_heights[max_actin_slice] + z_min)]
    real_xs = (df_clear_in_layer['x'] - x_min) * sum_proj_z.shape[0] / (x_max - x_min)
    real_ys = (df_clear_in_layer['y'] - y_min) * sum_proj_z.shape[1] / (y_max - y_min)
    all_xs = (df['x'] - x_min) * sum_proj_z.shape[0] / (x_max - x_min)
    all_ys = (df['y'] - y_min) * sum_proj_z.shape[1] / (y_max - y_min)
    print(np.max(sum_proj_z), np.min(sum_proj_z), np.mean(sum_proj_z))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sum_proj_z, cmap='Greys', vmin=np.min(sum_proj_z), vmax=np.mean(sum_proj_z) * 2, alpha=0.4)
    ax.scatter(all_xs, all_ys, c='r')
    ax.scatter(real_xs, real_ys, c='k')
    ax.set_xlim([0, sum_proj_z.shape[0]])
    ax.set_ylim([0, sum_proj_z.shape[1]])
    ax.set_xticks([])
    ax.set_yticks([])
    if save_name != False:
        plt.savefig(save_name + '.pdf', bbox_inches='tight', pad_inches=1)
    return


def xy_segmentation(df, image, **kwargs):
    image = image_shuffle(image)
    x_max, y_max, z_max, x_min, y_min, z_min, slice_heights = get_layer_position(df, image)
    scale = (x_max - x_min) / image.shape[2]
    df_cleared = clear_debris(df)
    density = len(df_cleared) / (x_max - x_min) ** 2 * 1000
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    actin_channel = kwargs.get('actin_channel', 1)
    dapi_channel = 0
    invert = kwargs.get('invert', False)

    if invert != False:
        xy_proj_actin = np.sum(np.sum(image, axis=3), axis=2)[:, actin_channel].copy()[::-1]
    else:
        xy_proj_actin = np.sum(np.sum(image, axis=3), axis=2)[:, actin_channel].copy()

    norm_intensities_actin = (xy_proj_actin - np.min(xy_proj_actin)) / np.max(xy_proj_actin - np.min(xy_proj_actin))
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

    start = np.argwhere(norm_intensities_actin >= bot_cutoff)[0][0]
    end = np.argwhere(norm_intensities_actin >= top_cutoff)[-1][0]

    truncated_stack = image[start + 3:end - 3, :, :, :].copy()

    sum_project = np.sum(truncated_stack, axis=0)
    DNA = sum_project[dapi_channel, :, :]
    Actin = sum_project[actin_channel, :, :]

    filtered = filters.gaussian(Actin)

    thresh = filters.threshold_local(filtered, block_size=55)
    binary = filtered > thresh

    clean_image = morphology.remove_small_objects(morphology.remove_small_objects(binary))
    dilate = morphology.dilation(morphology.dilation(morphology.dilation(clean_image)))
    skeleton = morphology.skeletonize(dilate)
    dilate_skeleton = morphology.dilation(morphology.dilation(morphology.dilation(morphology.dilation(skeleton))))
    erosion_skeleton = morphology.erosion(morphology.erosion(morphology.erosion(dilate_skeleton)))
    cell_mask = np.invert(erosion_skeleton)
    clean_mask = clear_border(cell_mask)
    distance_map = ndimage.distance_transform_edt(cell_mask)
    # local_peaks = peak_local_max(distance_map, min_distance = 10, threshold_abs = 2, indices=False) #BACH_EDIT2
    # BACH_EDIT2 BEGIN

    # PROBLEM: skimage.feature.peak_local_max(...) deprecated the indices parameter so the program just errors out
    """ Note from Documentation:
    Using a series of binarization, opening and closing, and other skimage features, the cells are segmented in XY. 
    Segmented cells are then measured using region_props.
    """
    # SOLUTION: get the bitmap(?)
    local_peaks = peak_local_max(distance_map, min_distance=10, threshold_abs=2)
    peaks_mask = np.zeros_like(distance_map, dtype=bool)  # array <- 0s
    peaks_mask[local_peaks] = True  # array <- 1s where there are entries for local_peaks
    local_peaks = peaks_mask  # array is now a Bitmap

    # B_EDIT2ACH END
    markers = measure.label(local_peaks)
    watershed_map = watershed(-distance_map, markers)
    labeled_image = watershed_map * clean_mask

    cell_props = regionprops(labeled_image)

    cell_ids = []
    cell_areas = []
    cell_centroid_rows = []
    cell_centroid_cols = []
    cell_perimeters = []
    cell_eccentricities = []
    cell_circularities = []

    for cell in cell_props:
        cell_ids.append(cell.label)
        cell_areas.append(cell.area * scale ** 2)
        row, col = cell.centroid
        cell_centroid_rows.append(row)
        cell_centroid_cols.append(col)
        cell_perimeters.append(cell.perimeter * scale)
        cell_eccentricities.append(cell.eccentricity)
        cell_circularities.append(4. * np.pi * cell.area / cell.perimeter ** 2)

    if save_name != False:
        data_dict = {'labels': cell_ids,
                     'areas': cell_areas,
                     'crows': cell_centroid_rows,
                     'ccols': cell_centroid_cols,
                     'perimeters': cell_perimeters,
                     'eccentricities': cell_eccentricities,
                     'circularities': cell_circularities}

        df = pd.DataFrame(data_dict)
        df.to_csv(save_name + '.csv')

    return np.mean(cell_areas), np.mean(cell_perimeters), np.mean(cell_circularities)


# Error at 'minlayerheight' parameter
'''
def fake_layers(**kwargs):
    plot = kwargs.get('plot', False)
    save_name = kwargs.get('save_name', False)
    fake_type = kwargs.get('fake_type', False)
    np.random.seed(42)

    underlying_layer = np.random.normal(7, 1, 300)
    cells_on_top_double = np.random.normal(14, 1, 300)
    cells_on_top_ball = np.random.normal(24, 5, 700)
    cells_on_top_mountain = np.random.exponential(10, 700)+12
        
    double = np.concatenate([underlying_layer, cells_on_top_double])
    mountain = np.concatenate([underlying_layer, cells_on_top_mountain])
    ball = np.concatenate([underlying_layer, cells_on_top_ball])
    

    
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
'''
