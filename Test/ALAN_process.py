import os
import numpy as np
import skimage.io as io
from scipy import optimize, stats
import pandas as pd
import math
import inspect
import time
from scipy import signal, optimize
from ALAN_module import image_shuffle, gaussian_fit, double_gaussian_fit, smooth_array, xy_segmentation


def parse(path):
    toc = time.time()
    files = os.listdir(path)
    spreadsheet_files = [f for f in files if f[-3:] == 'csv']
    images = [f for f in files if f[-3:] == 'tif']

    list_of_image_names = [sub[:-4] for sub in images]
    spreadsheet_files.sort()
    images.sort()
    list_of_spreadsheet_names = [sub[:-4] for sub in spreadsheet_files]
    unmatched_spreadsheets = [i for i in list_of_spreadsheet_names if i[:] not in list_of_image_names]
    unmatched_images = [i for i in list_of_image_names if i[:] not in list_of_spreadsheet_names]

    if len(unmatched_spreadsheets) > 0:
        for i in range(len(unmatched_spreadsheets)):
            list_of_spreadsheet_names.remove(unmatched_spreadsheets[i])
    spreadsheet_suffix = '.csv'
    aligned_spreadsheet_files = [sub + spreadsheet_suffix for sub in list_of_spreadsheet_names]
    aligned_spreadsheet_files.sort()

    if len(unmatched_images) > 0:
        for i in range(len(unmatched_images)):
            list_of_image_names.remove(unmatched_images[i])
    image_suffix = '.tif'
    aligned_images = [sub + image_suffix for sub in list_of_image_names]
    aligned_images.sort()

    list_of_dfs = []
    for item in aligned_spreadsheet_files:
        list_of_dfs.append(pd.read_csv(path + item))

    list_of_unshuffled_images = []

    for item in aligned_images:
        list_of_unshuffled_images.append(io.imread(path + item))

    tic = time.time()
    print('The time to read in all of the data in this folder is:')
    print(tic - toc)

    list_of_names = [f[:-4] for f in aligned_images]
    print('The name of the files read in are:')
    print(list_of_names)
    print('The dimensions of the 3 lists (list_of_names), (list_of_dfs) and (list_of_unshuffled_images) are:')
    print(len(list_of_names), len(list_of_dfs), len(list_of_unshuffled_images))

    if len(list_of_names) == len(list_of_dfs) == len(list_of_unshuffled_images):
        print('The dimensions of the lists are the same- you are okay to proceed!\n')
    else:
        print('The dimensions of the lists are not the same- something has gone wrong.\n')

    return list_of_names, list_of_dfs, list_of_unshuffled_images


def batch_process(list_of_dfs, list_of_unshuffled_images, list_of_names, **kwargs):
    save_name = kwargs.get('save_name', False)
    actin_channel = kwargs.get('actin_channel', 1)
    invert = kwargs.get('invert', False)

    list_of_cells_above = []
    list_of_percent_above = []
    list_of_cells_in = []
    list_of_layer_class = []
    list_of_cell_densities = []
    list_of_layer_heights = []
    list_of_bad_images = []
    list_of_cell_areas = []
    list_of_cell_perimeters = []
    list_of_cell_circularities = []

    toc = time.time()

    for i in range(len(list_of_dfs)):
        try:
            df = list_of_dfs[i]
            image = image_shuffle(list_of_unshuffled_images[i])

            num_z, num_c, num_x, num_y = image.shape
            df.columns = ['x', 'y', 'z', 'vol', 'positions']
            x_max = df['positions'][0]
            y_max = df['positions'][1]
            z_max = df['positions'][2]
            x_min = df['positions'][3]
            y_min = df['positions'][4]
            z_min = df['positions'][5]
            slice_heights = np.linspace(0, z_max - z_min, num=num_z)

            vols = df['vol'].values
            bottom_vol = np.mean(vols) - (1 * np.std(vols))
            df_cleared = df[df['vol'] >= bottom_vol]

            xy_proj_actin = np.sum(np.sum(image, axis=3), axis=2)[:, actin_channel]

            if invert:
                xy_proj_actin = xy_proj_actin.copy()[::-1]
            norm_intensities_actin = (xy_proj_actin - np.min(xy_proj_actin)) / np.max(
                xy_proj_actin - np.min(xy_proj_actin))
            actin_peak = slice_heights[np.argwhere(norm_intensities_actin == np.max(norm_intensities_actin))][0][0]
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
            min_actin_slice = np.argwhere(norm_intensities_actin >= bot_cutoff)[0][0]
            max_actin_slice = np.argwhere(norm_intensities_actin >= top_cutoff)[-1][0]
            min_layer_height = slice_heights[min_actin_slice]
            max_layer_height = slice_heights[max_actin_slice]
            layer_height = max_layer_height - min_layer_height
            actin_intensity_top = norm_intensities_actin[max_actin_slice]

            zs = df_cleared['z'].values - z_min
            bins = np.linspace(0, 50, 51)
            bins_fit = np.linspace(0, 50, 1000)
            counts, bins_aux = np.histogram(zs, bins=bins)
            counts_to_fit = smooth_array(counts)
            peak_height_guess = np.max(counts_to_fit)
            peak_loc_guess = bins[np.argwhere(counts_to_fit == peak_height_guess)[0][0]]
            p0_double = [peak_height_guess, peak_loc_guess, 2, 30, 2 * peak_loc_guess, 5]
            p0_single = [peak_height_guess, peak_loc_guess, 2]
            bounds_double = ([0, 1, 1, 0, 1, 1], [400, 80, 15, 400, 80, 15])
            bounds_single = ([0, 1, 1], [400, 80, 15])
            peak_loc = np.argwhere(counts_to_fit == np.max(counts_to_fit))[0][0]
            params_double, params_covariance_double = optimize.curve_fit(double_gaussian_fit, bins[:-1], counts_to_fit,
                                                                         p0_double, bounds=bounds_double)
            params_single, params_covariance_single = optimize.curve_fit(gaussian_fit, bins[:-1], counts_to_fit,
                                                                         p0_single, bounds=bounds_single)
            fit_single = gaussian_fit(bins_fit, params_single[0], params_single[1], params_single[2])
            fit_double = double_gaussian_fit(bins_fit, params_double[0], params_double[1], params_double[2],
                                             params_double[3], params_double[4], params_double[5])

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

            actin_profile_derivative = smooth_array(np.diff(norm_intensities_actin), window_size=5)
            peaks = signal.find_peaks(actin_profile_derivative, prominence=0.008)
            num_peaks = len(peaks[0])

            if len(peaks[0]) == 2:
                equivalence = peaks[1]['prominences'][1] / peaks[1]['prominences'][0]
            else:
                equivalence = 'undef'

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

            area, perimeter, circularity = xy_segmentation(df, image)

            list_of_cells_above.append(cells_above)
            list_of_percent_above.append(percentage_above)
            list_of_cells_in.append(cells_inside)
            list_of_layer_class.append(layer_classification)
            list_of_cell_densities.append(cell_density)
            list_of_layer_heights.append(layer_height)
            list_of_cell_areas.append(area)
            list_of_cell_perimeters.append(perimeter)
            list_of_cell_circularities.append(circularity)


        except Exception as e:
            print(e)
            list_of_bad_images.append(i)
            continue
    list_of_totals = []
    for i in range(len(list_of_cells_in)):
        list_of_totals.append(list_of_cells_in[i] + list_of_cells_above[i])
    if len(list_of_bad_images) > 0:
        print('The following images would not complete processing: \n', list_of_bad_images)
        dict_list_of_names = np.delete(list_of_names, list_of_bad_images)
    else:
        dict_list_of_names = list_of_names

    dict_data = {'names': dict_list_of_names,
                 'Is this a layer?': list_of_layer_class,
                 'cells in layer': list_of_cells_in,
                 'cells above layer': list_of_cells_above,
                 'total number of cells': list_of_totals,
                 'cell densities': list_of_cell_densities,
                 'layer height': list_of_layer_heights,
                 '% above': list_of_percent_above,
                 'average cell areas': list_of_cell_areas,
                 'average cell perimeters': list_of_cell_perimeters,
                 'average cell circularities': list_of_cell_circularities
                 }

    analyzed_df = pd.DataFrame(dict_data)

    tic = time.time()
    print('Time:', tic - toc)

    if save_name != False:
        analyzed_df.to_csv(save_name + '.csv')
        return analyzed_df
    else:
        return analyzed_df
