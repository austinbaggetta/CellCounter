import os
import cv2
import math
import numpy as np
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

def load_image(dpath, pattern, dtype=np.uint8, varr_name=None):
    image_path = choose_file(dpath, pattern)
    im = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH) ## may need to do cv2.IMREAD_GRAYSCALE
    darr = da.from_array(im, chunks=-1)
    varr = xr.DataArray(darr,
                        dims=['height', 'width'],
                        coords=dict(
                            height=np.arange(0, darr.shape[0]),
                            width=np.arange(0, darr.shape[1])
                        ))
    if dtype:
        varr = varr.astype(dtype)
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
    if method == 'tophat':
        kernel = disk(wnd)
        res = cv2.morphologyEx(image.values, cv2.MORPH_TOPHAT, kernel)
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