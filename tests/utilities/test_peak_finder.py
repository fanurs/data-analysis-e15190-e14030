import pytest

import numpy as np

from e15190.utilities.peak_finder import PeakFinderGaus1D

class TestPeakFinder:
    def test___init__(self):
        finder = PeakFinderGaus1D(
            np.array([1.0, 2.0, 3.0]),
            [0, 5],
            5,
        )
        assert np.allclose(finder.x, np.array([1.0, 2.0, 3.0]))
        assert finder.mean == pytest.approx(2.0)
        assert 0.5 < finder.std < 1.0
        assert np.allclose(finder.hist_range, [0, 5])
        assert finder.hist_bins == 5

    def test_gaus(self):
        gaus = PeakFinderGaus1D.gaus

        amplt = np.random.uniform(0.1, 2)
        assert gaus(0, amplt, 0, 1) == pytest.approx(amplt)
        assert gaus(100, amplt, 0, 1) == pytest.approx(0)
        assert gaus(-100, amplt, 0, 1) == pytest.approx(0)

        mean = np.random.uniform(-10, 10)
        assert gaus(mean, amplt, mean, 1) == pytest.approx(amplt)
        assert gaus(mean - 0.1, amplt, mean, 1) == pytest.approx(gaus(mean + 0.1, amplt, mean, 1))

        x = np.linspace(mean, mean + 2, 100)
        y = gaus(x, amplt, mean, 1)
        assert np.all(np.diff(y) < 0)
        x = np.linspace(mean, mean - 2, 100)
        y = gaus(x, amplt, mean, 1)
        assert np.all(np.diff(y) < 0)

        y1 = gaus(x, amplt, mean, 1)
        y2 = gaus(x, amplt, mean, 2)
        assert np.all(y2 >= y1)
    
    def test__get_histogram(self):
        hist = PeakFinderGaus1D._get_histogram

        data = [0.2, 0.3, 1.2, 3.9]
        hx, hy = hist(data, [0, 5], 5)
        assert len(hx) == len(hy) == 5
        assert np.allclose(hx, [0.5, 1.5, 2.5, 3.5, 4.5])
        assert np.allclose(hy, [2, 1, 0, 1, 0])
    
    def test__get_convoluted_y_sine(self):
        convolute = PeakFinderGaus1D._get_convoluted_y

        n = 200
        x = np.linspace(0, 10, n)
        y_exact = np.sin(x)
        y_data = y_exact + np.random.normal(0, 0.05, n)
        y_conv = convolute(y_data, kernel_range=[0, 10], ngrids=n)

        assert len(y_conv) == n
        assert np.allclose(y_exact, y_conv, atol=0.25)

        norm_conv = np.linalg.norm(y_conv - y_exact)
        norm_data = np.linalg.norm(y_data - y_exact)
        assert norm_conv < norm_data

    def test__get_convoluted_y_high_freq_sine(self):
        convolute = PeakFinderGaus1D._get_convoluted_y

        n = 300
        x = np.linspace(0, 5, n)
        y_exact = np.sin(5 * x)
        y_data = y_exact + np.random.normal(0, 0.05, n)
        y_conv = convolute(y_data, kernel_range=[0, 5], ngrids=200)

        assert len(y_conv) == n
        assert np.allclose(y_exact, y_conv, atol=0.25)

        norm_conv = np.linalg.norm(y_conv - y_exact)
        norm_data = np.linalg.norm(y_data - y_exact)
        assert norm_conv < norm_data
    
    @pytest.mark.filterwarnings('ignore:Failed to fit')
    def test__find_highest_peak_single_peak(self):
        gaus = PeakFinderGaus1D.gaus
        find_peak = PeakFinderGaus1D._find_highest_peak

        x = np.linspace(0, 5, 100)
        noise_close = []
        for _ in range(100):
            amplt = np.random.uniform(0.1, 10)
            mean = np.random.uniform(2, 3)
            sigma = np.random.uniform(0.05, 0.5)

            y = gaus(x, amplt, mean, sigma)
            pars = find_peak(x, y)
            assert np.allclose(pars, [amplt, mean, sigma], rtol=1e-5)

            y += np.random.normal(0, 0.05, len(y))
            pars = find_peak(x, y)
            assert not np.allclose(pars, [amplt, mean, sigma], rtol=1e-5)
            noise_close.append(np.allclose(pars, [amplt, mean, sigma], rtol=0.1))
        assert sum(noise_close) / len(noise_close) > 0.9
    
    @pytest.mark.skip(reason="Don't know how to always raise maxfev error")
    def test__find_highest_peak_not_a_peak(self):
        pass

    def test__find_highest_peak_two_peaks(self):
        gaus = PeakFinderGaus1D.gaus
        find_peak = PeakFinderGaus1D._find_highest_peak

        x = np.linspace(0, 5, 200)

        # far apart
        y0 = gaus(x, amplt=1, mean=0, sigma=0.1)
        y0 += np.random.normal(0, 0.02, len(y0))
        y1 = gaus(x, amplt=0.6, mean=2, sigma=0.3)
        y1 += np.random.normal(0, 0.02, len(y1))
        y = y0 + y1
        pars = find_peak(x, y)
        assert np.allclose(pars, [1, 0, 0.1], rtol=0.2, atol=0.05)

        # closer
        y0 = gaus(x, amplt=1, mean=0, sigma=0.1)
        y0 += np.random.normal(0, 0.02, len(y0))
        y1 = gaus(x, amplt=0.6, mean=1, sigma=0.3)
        y1 += np.random.normal(0, 0.02, len(y1))
        y = y0 + y1
        pars = find_peak(x, y)
        assert np.allclose(pars, [1, 0, 0.1], rtol=0.2, atol=0.05)

    def test_get_highest_peak(self):
        data = np.random.normal(loc=0, scale=0.05, size=100)
        data = np.hstack([data, np.random.normal(loc=1, scale=0.2, size=100)])

        finder = PeakFinderGaus1D(data, [-2, 3], 300)
        pars = finder.get_highest_peak()
        assert pars[1] == pytest.approx(0, abs=0.1)