from typing import Tuple
import warnings

import numpy as np
from scipy import optimize

from e15190.utilities import fast_histogram as fh

class PeakFinderGaus1D:
    def __init__(self, x, hist_range, hist_bins):
        """

        Parameters
        ----------
        x : list-like of float
            x-values of the data
        hist_range : 2-tuple of float
            Range of the histogram.
        hist_bins : int
            Number of bins of the histogram.
        """
        self.x = np.array(x)
        self.mean = np.mean(x)
        self.std = np.std(x)
        self.hist_range = np.array(hist_range)
        self.hist_bins = hist_bins

    @staticmethod
    def gaus(t, amplt, mean, sigma):
        return amplt * np.exp(-0.5 * ((t - mean) / sigma) ** 2)
    
    @staticmethod
    def _get_histogram(t, hrange, hbins) -> Tuple[np.ndarray, np.ndarray]:
        y = fh.histo1d(t, hrange, hbins)
        x = np.linspace(*hrange, hbins + 1)
        x = 0.5 * (x[1:] + x[:-1])
        return x, y
    
    @staticmethod
    def _get_convoluted_y(y, kernel_range, ngrids, sigma=None) -> np.ndarray:
        """
        Parameters
        ----------
        y : np.ndarray
            The histogram counts to convolute.
        kernel_range : 2-tuple of floats
            The range of the kernel. This should match the range of the
            histogram.  Notice that the x of histogram is not an input of this
            function.
        ngrids : int
            The number of grids to use in the convolution. This should be
            less than or equal to the number of bins in the histogram.
        sigma : float, default None
            Standard deviation of the Gaussian kernel. If None, the bin width
            is used.
        
        Returns
        -------
        y_conv : np.ndarray
            The convoluted histogram counts.
        """
        kernel_width = kernel_range[1] - kernel_range[0]
        grid_width = kernel_width / ngrids
        kernel = PeakFinderGaus1D.gaus(
            np.linspace(-0.5 * kernel_width, 0.5 * kernel_width, ngrids),
            amplt=1, # unimportant
            mean=0, # always zero
            sigma=grid_width if sigma is None else sigma,
        )
        return np.convolve(y, kernel, mode='same') / kernel.sum()
    
    @staticmethod
    def _find_highest_peak(x, y, fit_range=None, kw_rough=None, kw_fine=None) -> np.ndarray:
        """Returns Gaussian parameters of the highest peak in the histogram.

        Parameters
        ----------
        x : np.ndarray
            The x-values of the histogram.
        y : np.ndarray
            The y-values of the histogram.
        fit_range : 2-tuple of float, default None
            The range of the histogram to fit. If None, the entire histogram is
            fit.
        kw_rough : dict, default None
            Keyword arguments for the rough fit supplied to
            ``scipy.optimize.curve_fit``.
        kw_fine : dict, default None
            Keyword arguments for the fine fit supplied to
            ``scipy.optimize.curve_fit``.
        
        Returns
        -------
        pars : np.ndarray of size 3
            The Gaussian parameters of the highest peak,
            ``[amplt, mean, sigma]``.
        """
        i = np.argmax(y)
        y_max = y[i]
        x_max = x[i]

        if fit_range is None:
            fit_range = [np.min(x), np.max(x)]
        mask = (x >= fit_range[0]) & (x <= fit_range[1])
        x_fit = x[mask]
        y_fit = y[mask]
        
        gaus = PeakFinderGaus1D.gaus

        # fit by varying only sigma
        try:
            sigma_init = 0.1 * np.std(x_fit)
            kw = dict(p0=[sigma_init])
            kw.update(kw_rough or dict())
            pars_rough, _ = optimize.curve_fit(
                lambda x, sigma: gaus(x, amplt=y_max, mean=x_max, sigma=sigma),
                x_fit, y_fit, **kw,
            )
        except RuntimeError:
            warnings.warn(f'Failed to fit peak at {x_max=}')
            return np.array([y_max, x_max, sigma_init])

        # fit by varying all parameters
        sigma_init = pars_rough[0]
        try:
            kw = dict(p0=[y_max, x_max, sigma_init])
            kw.update(kw_fine or dict())
            pars, _ = optimize.curve_fit(gaus, x_fit, y_fit, **kw)
        except RuntimeError:
            warnings.warn(f'Failed to fit peak at {x_max=}')
            return np.array([y_max, x_max, sigma_init])

        return pars
    
    def get_highest_peak(self, fit_range=None) -> np.ndarray:
        x = self.x.copy()
        x, y = self._get_histogram(x, self.hist_range, self.hist_bins)
        y_conv = self._get_convoluted_y(y, self.hist_range, self.hist_bins)
        pars = self._find_highest_peak(x, y_conv, fit_range=fit_range)
        return pars
