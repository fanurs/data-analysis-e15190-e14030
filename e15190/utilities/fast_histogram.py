"""A collection of wrapper functions for ``fast_histogram``.

The histogram functions provided by matplotlib, built on top of NumPy's
histogram functions, are versatile but slow for large datasets. On the other
hand, `fast_histogram <https://github.com/astrofrog/fast-histogram>`__ offers a
much faster implementation of histograms for regular binnings, which are much
more commonly used in many physics analysis, including this repository.

This module wraps around histogram functions from ``fast_histogram`` to provide
a more convenient interface.
"""
import fast_histogram as fh
import matplotlib.pyplot as plt
import numpy as np

def histo1d(x, range, bins, weights=None):
    """A wrapper function for ``fast_histogram.histogram1d()``

    Parameters
    ----------
    x : 1D array-like
        Input data.
    range : 2-tuple or 2-list
        The lower and upper range of the bins.
    bins : int
        The number of bins.
    weights : array-like or None, default None
        Weights for each value in `x`.

    Returns
    -------
    bin_contents : 1D array of shape (bins, )
        The bin contents.
    """
    return fh.histogram1d(x, range=range, bins=bins, weights=weights)

def histo2d(x, y, range, bins, weights=None):
    """A wrapper function for ``fast_histogram.histogram2d()``

    Parameters
    ----------
    x : array-like, shape (N, )
        Input data x.
    y : array-like, shape (N, )
        Input data y.
    range : array-like, shape (2, 2)
        The lower and upper range of the bins, ``[[xmin, xmax], [ymin, ymax]]``.
    bins : [int, int]
        The number of bins, ``[x_bins, y_bins]``.
    weights : array-like of shape (N, ), default None
        Weights for each (x, y) pair.
        
    Returns
    -------
    bin_contents : 2D array of shape (bins[0], bins[1])
        The bin contents.
    """
    return fh.histogram2d(x, y, range=range, bins=bins, weights=weights)

def plot_histo1d(hist_func, x, range, bins, **kwargs):
    """Plot a 1D histogram using ``fast_histogram``.

    Parameters
    ----------
    hist_func : plt.hist() or ax.hist()
        The histogram function from matplotlib.
    x : 1D array-like
        Data to be histogrammed.
    range : 2-tuple or 2-list
        The lower and upper range of the histogram.
    bins : int
        The number of bins.
    **kwargs
        Keyword arguments to be passed to ``hist_func``. See more at
        `matplotlib.pyplot.hist <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html>`__.
        
    Returns
    -------
    h : 1D array of shape (bins, )
        The bin contents.
    edges : 1D array of shape (bins + 1, )
        The bin edges.
    patches : histogram patches
        The histogram patches.
    """
    if 'weights' in kwargs:
        weights = fh.histogram1d(x, range=range, bins=bins, weights=kwargs['weights'])
        kwargs.pop('weights')
    else:
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

def plot_histo2d(hist_func, x, y, range, bins, **kwargs):
    """Plot a 2D histogram using ``fast_histogram``.

    Parameters
    ----------
    hist_func : plt.hist2d() or ax.hist2d()
        The histogram function from matplotlib.
    x : array-like, shape (N, )
        Data x to be histogrammed.
    y : array-like, shape (N, )
        Data y to be histogrammed.
    range : array-like, shape (2, 2)
        The x range and y range of the histogram,
        ``[[xmin, xmax], [ymin, ymax]]``.
    bins : [int, int]
        The number of bins, ``[x_bins, y_bins]``.
    **kwargs
        Keyword arguments to be passed to ``hist_func``. See more at
        `matplotlib.pyplot.hist2d <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist2d.html>`__.
    
    Returns
    -------
    h : 2D array of shape (bins[0], bins[1])
        The bin contents.
    x_edges : 1D array of shape (bins[0] + 1, )
        The bin edges along the x axis.
    y_edges : 1D array of shape (bins[1] + 1, )
        The bin edges along the y axis.
    patches : histogram patches
        The histogram patches.
    """
    range, bins = map(np.array, (range, bins))
    if 'weights' in kwargs:
        weights = fh.histogram2d(x, y, range=range, bins=bins, weights=kwargs['weights'])
        kwargs.pop('weights')
    else:
        weights = fh.histogram2d(x, y, range=range, bins=bins)
    weights = weights.transpose().flatten()
    x_centers = np.linspace(*range[0], bins[0] + 1)
    y_centers = np.linspace(*range[1], bins[1] + 1)
    x_centers = 0.5 * (x_centers[1:] + x_centers[:-1])
    y_centers = 0.5 * (y_centers[1:] + y_centers[:-1])
    x_centers, y_centers = np.meshgrid(x_centers, y_centers)
    return hist_func(
        x_centers.flatten(),
        y_centers.flatten(),
        weights=weights,
        range=range,
        bins=bins,
        **kwargs,
    )
