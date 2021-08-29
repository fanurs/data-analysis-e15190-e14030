import functools

import fast_histogram as fh
import matplotlib.pyplot as plt
import numpy as np

@functools.wraps(fh.histogram1d)
def histo1d(*args, **kwargs):
    return fh.histogram1d(*args, **kwargs)

@functools.wraps(fh.histogram2d)
def histo2d(*args, **kwargs):
    return fh.histogram2d(*args, **kwargs)

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
