import numpy as np

def create_ranges(low, upp, width, step=None):
    if step is None:
        step = width
    centers = np.arange(low + 0.5 * width, upp - 0.5 * width + 1e-3 * width, step)
    return np.vstack([centers - 0.5 * width, centers + 0.5 * width]).T