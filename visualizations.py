import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json 
from dash import Dash, dcc, callback, html, Output, Input, no_update


def custom_graph_template(x_title, y_title, template='simple_white', height=500, width=500, linewidth=1.5,
                          titles=[''], rows=1, columns=1, shared_y=False, shared_x=False, font_size=22, font_family='Arial', **kwargs):
    """
    Used to make a cohesive graph type. In most functions, these arguments are supplied through **kwargs.
    """
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles, shared_yaxes=shared_y, **kwargs)
    fig.update_yaxes(title=y_title, linewidth=linewidth)
    fig.update_xaxes(title=x_title, linewidth=linewidth)
    fig.update_layout(title={
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_annotations(font_size=font_size)
    fig.update_layout(template=template, height=height, width=width, font=dict(size=font_size), font_family=font_family, dragmode='pan')
    if shared_x:
        fig.update_xaxes(matches='x')
    if shared_y:
        fig.update_yaxes(matches='y')
    return fig


def extract_roi(relayoutdata, shape_numbers=[0], shape_type='rect'):
    if shape_type == 'rect':
        rois = {'roi': [], 'x0': [], 'x1': [], 'y0': [], 'y1': []}
    else:
        raise NotImplementedError(f'Shape type {shape_type} is not supported.')
    
    data = json.loads(relayoutdata)
    for shape in shape_numbers:
        if 'x0' in data[shape]:
            rois['roi'].append(shape)
            rois['x0'].append(round(data[shape]['x0']))
            rois['x1'].append(round(data[shape]['x1']))
            rois['y0'].append(round(data[shape]['y0']))
            rois['y1'].append(round(data[shape]['y1']))
        else:
            raise NotImplementedError('Shape output does not match shape_type argument.')
    return pd.DataFrame(rois)


def visualize_intensity_histogram(image, bins, range, density=True, marker_color='darkgrey', **kwargs):
    if type(image) == xr.DataArray:
        hist_data = image.values 
    else:
        hist_data = image

    if density:
        y_title = 'Probability'
    else:
        y_title = 'Count' 
    hist_bins, bin_edges = np.histogram(hist_data, bins=bins, range=range, density=density)
    fig = custom_graph_template(x_title='Grayscale Values', y_title=y_title, **kwargs)
    fig.add_trace(go.Bar(x=bin_edges, y=hist_bins, marker_color=marker_color, marker_line_width=2,
                         marker_line_color='black', opacity=0.8))
    return fig


def crop_image_to_roi(im, rois):
    if rois.shape[0] > 1:
        raise Warning('Too many ROIs to crop image!')
    
    if rois['x0'].values[0] < rois['x1'].values[0] and rois['y0'].values[0] < rois['y1'].values[0]:
        im = im[rois['y0'].values[0]:rois['y1'].values[0], rois['x0'].values[0]:rois['x1'].values[0]]
    elif rois['x0'].values[0] > rois['x1'].values[0] and rois['y0'].values[0] < rois['y1'].values[0]:
        im = im[rois['y0'].values[0]:rois['y1'].values[0], rois['x1'].values[0]:rois['x0'].values[0]]
    elif rois['x0'].values[0] < rois['x1'].values[0] and rois['y0'].values[0] > rois['y1'].values[0]:
        im = im[rois['y1'].values[0]:rois['y0'].values[0], rois['x0'].values[0]:rois['x1'].values[0]]
    else:
        im = im[rois['y1'].values[0]:rois['y0'].values[0], rois['x1'].values[0]:rois['x0'].values[0]]
    return im