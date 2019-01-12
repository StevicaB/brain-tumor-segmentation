# -*- coding: UTF-8 -*-
"""Module including basic functions for data normalisation.

All images are supposed to be supplied as numpy arrays.

"""
__version__ = '0.2'
__author__ = 'Esther Alberts'

import numpy as np

def window(data,
           mask=None,
           lower=0,
           upper=1,
           background=0,
           verbose=False):
    """Window `data` within the mask with `lower` and `upper` thresholds.
    The region outside the mask is set to `background`.
    
    Carefull, no rescaling is performed prior to windowing.
    """

    if mask is None:
        mask = np.ones_like(data)
    new_data = np.copy(data)

    # set background to lower
    new_data[mask <= 0] = background

    # set lower bound
    too_low_indices = np.logical_and(data < lower, mask > 0)
    new_data[too_low_indices] = lower

    # set upper bound
    too_high_indices = np.logical_and(data > upper, mask > 0)
    new_data[too_high_indices] = upper
    if verbose:
        low_nb = np.count_nonzero(too_low_indices)
        high_nb = np.count_nonzero(too_high_indices)
        print '%d (%d) voxels have been set to lower (upper) bounds' % (\
                                    low_nb, high_nb)

    return new_data

def window_percentage(data,
                      mask=None,
                      lower_percentage=2.5,
                      upper_percentage=2.5,
                      background=0):
    """ Set the lowest x procent of the data in the mask to the 
    x percentile and the highest x procent of the data in the mask 
    to the 1 - x percentile. """

    fg = data[mask > 0]
    lower_threshold = np.percentile(fg, lower_percentage)
    upper_threshold = np.percentile(fg, 100 - upper_percentage)

    new_data = window(data,
                      mask=mask,
                      lower=lower_threshold,
                      upper=upper_threshold,
                      background=background)

    return new_data

def rescale(data,
            mask=None,
            lower=0,
            upper=1,
            background=0):
    """Scale the mask region in data between `lower` and `upper`.

    The background is set to background.
    """

    # make a copy and make sure type allows for floats
    new_data = np.zeros_like(data, dtype=np.float)

    if mask is None:
        mask = np.ones_like(data)

    this_min = np.amin(data[mask > 0])
    this_max = np.amax(data[mask > 0])
    norm = float(this_max - this_min)
    if norm > np.finfo(np.double).eps:
        new_data[mask > 0] = ((data[mask > 0] - this_min) / norm) * \
                     np.float(upper - lower) + lower
    else:
        new_data[mask > 0] = (data[mask > 0] - this_min) + lower
    new_data[mask <= 0] = background

    if np.min(new_data) != lower or np.max(new_data) != upper:
        err = 'Rescaling lower and/or upper bound failed? Check datatype'
        raise ValueError(err)

    return new_data

def whiten(data, mask=None):
    """Whiten `data` to obtain zero means and unit std.

    Set the background to the lowest value in the whitened mask region.
    """

    if mask is None:
        mask = np.ones_like(data)

    norm = float(np.std(data[mask > 0]))
    new_data = np.zeros_like(data, dtype=np.float)
    if norm > np.finfo(np.double).eps:
        new_data[mask > 0] = (data[mask > 0] - np.mean(data[mask > 0])) / norm
    else:
        new_data[mask > 0] = data[mask > 0] - np.mean(data[mask > 0])
    new_data[mask <= 0] = np.amin(new_data)

    return new_data

def window_rescale(data, mask=None, window_std=4):
    """ Whiten data, window it at window_std standard deviations and
    rescale to [0,1].

    The returned data will be positive and rescaled to [0,1].
    """

    # whiten data
    white_data = whiten(data, mask=mask)

    # window data at three standard deviations
    windowed_data = window(white_data,
                           mask=mask,
                           lower=-window_std,
                           upper=window_std,
                           verbose=True)

    # set minimum to zero and max to 1
    pos_data = rescale(windowed_data,
                       mask=mask,
                       lower=0,
                       upper=1)

    return pos_data