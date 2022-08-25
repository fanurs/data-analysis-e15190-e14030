from typing import Optional

import numpy as np

def create_ranges(
    low: float,
    upp: float,
    width: float,
    step: Optional[float] = None,
    n_steps: Optional[int] = None,
):
    """Create a list of ranges.

    Parameters
    ----------
    low : float
        The lower bound of the range.
    upp : float
        The upper bound of the range.
    width : float
        The width of each range.
    step : float, default None
        The distance between consecutive ranges, measuered at the range
        midpoints. Default is None, in which the step is equal to the width.
    n_steps : int, default None
        The number of steps or the number of ranges to generate. Default is
        None, in which it follows the behavior specified by the ``step``
        parameter. ``step`` is ignored if ``n_steps`` is specified.
    """
    if step is None:
        step = width
    if n_steps is not None:
        step = ((upp - low) - width) / (n_steps - 1)
    centers = np.arange(low + 0.5 * width, upp - 0.5 * width + 1e-3 * width, step)
    return np.vstack([centers - 0.5 * width, centers + 0.5 * width]).T