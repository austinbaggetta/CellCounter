import os
import cv2
import math
import pickle
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
from os.path import join as pjoin
from skimage.morphology import disk
from skimage.filters import threshold_otsu, threshold_mean


def choose_file(image_path, pattern):
    files = os.listdir(image_path)
    for file in files:
        if pattern in file:
            return pjoin(image_path, file)
        else:
            pass


def load_image(dpath, pattern, varr_name=None):
    image_path = choose_file(dpath, pattern)
    im = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    ## openCV requires images be 8-bit
    if im.dtype == np.uint16:
        im = (im/256).astype(np.uint8)
    darr = da.from_array(im, chunks=-1)
    varr = xr.DataArray(darr,
                        dims=['height', 'width'],
                        coords=dict(
                            height=np.arange(0, darr.shape[0]),
                            width=np.arange(0, darr.shape[1])
                        ))
    if varr_name is not None:
        varr = varr.rename(varr_name)
    else:
        varr = varr.rename('fluorescence')
    return varr


def denoise(image: xr.DataArray, method: str, **kwargs) -> xr.DataArray:
    if method == 'median':
        func = cv2.medianBlur
    elif method == 'gaussian':
        func = cv2.GaussianBlur
    else:
        raise NotImplementedError(f'Denoise method {method} not understood.')
    res = xr.apply_ufunc(
        func,
        image.load(),
        input_core_dims=[['height', 'width']],
        output_core_dims=[['height', 'width']],
        output_dtypes=[image.dtype],
        kwargs=kwargs
    )
    res = res.astype(image.dtype).rename(image.name + f'_{method}').chunk(chunks=-1)
    return res


def remove_background(image: xr.DataArray, method: str, wnd: int) -> xr.DataArray:
    """
    Args:
        method : str
            'tophat': This operation returns the bright spots of the image that are smaller
                      than the structuring element.
    """
    kernel = disk(wnd)
    if method == 'tophat':
        res = cv2.morphologyEx(image.values, cv2.MORPH_TOPHAT, kernel)
    elif method == 'opening':
        res = cv2.morphologyEx(image.values, cv2.MORPH_OPEN, kernel)

    res = xr.DataArray(res,
                        dims=['height', 'width'],
                        coords=dict(
                            height=np.arange(0, res.shape[0]),
                            width=np.arange(0, res.shape[1])
                        )).chunk(chunks=-1)
    return res.rename('background_subtracted')


def detect_cells(image, contours, contour_color=100, minimum_area=600, maximum_area=2500, average_cell_area=1000, connected_cell_area=600):
    if type(image) == xr.DataArray:
        vals = image.values
    else:
        vals = image
    
    cells = 0
    cell_contours = vals
    for c in contours:
        area = cv2.contourArea(c)
        if (area > minimum_area) & (area < maximum_area):
            cell_contours = cv2.drawContours(cell_contours, c, -1, contour_color, 3)
            if area > connected_cell_area:
                cells += math.ceil(area / average_cell_area)
            else:
                cells += 1
    return cells, cell_contours


def contour_selection(contour_ar=None, contour_color=100, visualize_only_cells=True, contour_path=None):
    if contour_ar is not None and contour_path is None:
        loaded_ar = contour_ar
    elif contour_ar is None and contour_path is not None:
        loaded_ar = xr.open_dataarray(contour_path)
    elif contour_ar is None and contour_path is None:
        raise Warning('Must supply a contour array or a contour path!')
    else:
        raise Warning('Cannot load a previous contour and supply a contour array!')

    num_contours = loaded_ar.attrs['num_cells']
    if visualize_only_cells:
        plot_data = loaded_ar.values == contour_color
    else:
        plot_data = loaded_ar.values

    return plot_data, num_contours


def calculate_threshold(im, thresh_type='otsu'):
    if type(im) == xr.DataArray:
        values = im.values
    else:
        values = im

    if thresh_type == 'otsu':
        func = threshold_otsu
    elif thresh_type == 'mean':
        func = threshold_mean
    return func(values)


def local_max(im, wnd_dist, diff):
    k = disk(wnd_dist)
    im_max = cv2.dilate(im, k)
    im_min = cv2.erode(im, k)
    im_diff = ((im_max - im_min) > diff).astype(np.uint8)
    im_ext = (im == im_max).astype(np.uint8)
    return cv2.bitwise_and(im_ext, im_diff).astype(np.uint8)


def single_seeds(lmax):
    """ 
    Creates a single seed at the median location for all the local maximum that are connected.
    """
    nlab, max_lab = cv2.connectedComponents(lmax)
    max_res = np.zeros_like(lmax)
    for lb in range(1, nlab):
        area = max_lab == lb
        if np.sum(area) > 1:
            crds = tuple(int(np.median(c)) for c in np.where(area))
            max_res[crds] = 1
        else:
            max_res[np.where(area)] = 1
    return max_res.astype(float)


def combine_seeds(max_res, dist_thresh=4):
    """ 
    Combines seeds that are within a certain pixel distance from each other.
    """
    seeds = np.where(max_res == 1)
    seed_df = pd.DataFrame({'seed': np.arange(0, seeds[0].shape[0]), 'x': seeds[0], 'y': seeds[1]})
    seeds_final = pd.DataFrame(columns=['seed', 'x', 'y'])
    duplicate_seeds = []
    for seed in np.arange(0, seeds[0].shape[0]):
        x_dif = seed_df['x'] - seed_df.loc[seed, 'x']
        y_dif = seed_df['y'] - seed_df.loc[seed, 'y']
        dist = np.hypot(x_dif, y_dif)
        if any((dist <= dist_thresh) & (dist > 0)):
            neighbor = np.where((dist <= dist_thresh) & (dist > 0))[0][0]
            same_seed = seed_df.loc[[seed, neighbor]]
            new_x, new_y = int(np.mean(same_seed['x'])), int(np.mean(same_seed['y']))
        else:
            new_x, new_y = seed_df.loc[seed, 'x'], seed_df.loc[seed, 'y']
            neighbor = np.nan
        
        if seed in duplicate_seeds:
            pass
        elif neighbor in duplicate_seeds:
            pass
        else:
            seeds_final.loc[seed, ['seed', 'x', 'y']] = [seed, new_x, new_y]
        
        if ~np.isnan(neighbor):
            duplicate_seeds.append(neighbor)
        
    cell_loc = np.zeros_like(max_res)
    for seed in seeds_final['seed']:
        cell_loc[seeds_final['x'][seeds_final['seed'] == seed].values[0], seeds_final['y'][seeds_final['seed'] == seed].values[0]] = 1
    return cell_loc, seeds_final


def save_params(median_params, bg_sub_params, local_max_params, combine_seeds_params, spath):
    all_params = {
        'median_params': median_params,
        'bg_sub_params': bg_sub_params,
        'local_max_params': local_max_params,
        'combine_seeds_params': combine_seeds_params
    }

    with open(pjoin(spath), 'wb') as file:
        pickle.dump(all_params, file, protocol=pickle.HIGHEST_PROTOCOL)