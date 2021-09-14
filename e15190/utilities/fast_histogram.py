import functools

import fast_histogram as fh
import matplotlib.pyplot as plt
import numpy as np

@functools.wraps(fh.histogram1d)
def histo1d(x, range, bins, weights=None):
    """A wrapper function for fast_histogram.histogram1d()

    Parameters:
        x : array-like
            Input data.
        range : 2-tuple or 2-list
            The lower and upper range of the bins.
        bins : int
            The number of bins.
        weights : array-like or None, default None
            Weights for each value in `a`.

    Returns:
        An array of bin values.
    """
    return fh.histogram1d(x, range=range, bins=bins, weights=weights)

@functools.wraps(fh.histogram2d)
def histo2d(x, y, range, bins, weights=None):
    """A wrapper function for fast_histogram.histogram2d()

    Parameters:
        x : array-like, shape (N,)
            Input data x.
        y : array-like, shape (N,)
            Input data y.
        range : array-like, shape (2, 2)
            The lower and upper range of the bins: `[[xmin, xmax], [ymin, ymax]]`.
        bins : [int, int]
            The number of bins: `[x_bins, y_bins]`.
        weights : array-like of shape (N,) or None, default None
            Weights for each (x, y) pair.
        
    Returns:
        A two-dimensional array containing the bin values in shape `(x_bins, y_bins)`.
    """
    return fh.histogram2d(x, y, range=range, bins=bins, weights=weights)

@functools.wraps(plt.hist)
def plot_histo1d(hist_func, x, range, bins, **kwargs):
    weights = fh.histogram1d(x, range=range, bins=bins)
    x_centers = np.linspace(*range, bins + 1)
    x_centers = 0.5 * (x_centers[1:] + x_centers[:-1])
    return hist_func(
        x_centers,
        weights=weights,
        range=range,
        bins=bins,
        **kwargs,
    )

@functools.wraps(plt.hist2d)
def plot_histo2d(hist_func, x, y, range, bins, **kwargs):
    range, bins = map(np.array, (range, bins))
    weights = fh.histogram2d(x, y, range=range, bins=bins)
    x_centers = np.linspace(*range[0], bins[0] + 1)
    y_centers = np.linspace(*range[1], bins[1] + 1)
    x_centers = 0.5 * (x_centers[1:] + x_centers[:-1])
    y_centers = 0.5 * (y_centers[1:] + y_centers[:-1])
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)
    return hist_func(
        x_centers.flatten(),
        y_centers.flatten(),
        weights=weights.transpose().flatten(),
        range=range,
        bins=bins,
        **kwargs,
    )
