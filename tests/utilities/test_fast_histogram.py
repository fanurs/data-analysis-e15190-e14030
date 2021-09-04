import pytest

import matplotlib.pyplot as plt
import numpy as np

from e15190.utilities import fast_histogram as fh

@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng()
    n_points = 100
    x = rng.uniform(0, 1, n_points)
    y = rng.uniform(0, 1, n_points)
    return {
        'x': x,
        'y': y,
    }

class TestGlobalFunctions:
    def test_histo1d(self, synthetic_data):
        ref_counts, _ = np.histogram(synthetic_data['x'], range=[0, 1], bins=10)
        counts = fh.histo1d(synthetic_data['x'], range=[0, 1], bins=10)
        assert np.allclose(ref_counts, counts)

    def test_histo2d(self, synthetic_data):
        ref_counts, _, _ = np.histogram2d(
            synthetic_data['x'], synthetic_data['y'],
            range=[[0, 1], [0, 1]],
            bins=[10, 10],
        )
        counts = fh.histo2d(
            synthetic_data['x'], synthetic_data['y'],
            range=[[0, 1], [0, 1]],
            bins=[10, 10],
        )
        assert np.allclose(ref_counts, counts)
    
    def test_plot_histo1d(self, synthetic_data):
        ref_counts, _ = np.histogram(synthetic_data['x'], range=[0, 1], bins=10)
        counts, _, _ = fh.plot_histo1d(
            plt.hist,
            synthetic_data['x'],
            range=[0, 1],
            bins=10,
        )
        assert np.allclose(ref_counts, counts)
    
    def test_plot_histo2d(self, synthetic_data):
        ref_counts, _, _ = np.histogram2d(
            synthetic_data['x'], synthetic_data['y'],
            range=[[0, 1], [0, 1]],
            bins=[10, 10],
        )
        counts, _, _, _ = fh.plot_histo2d(
            plt.hist2d,
            synthetic_data['x'],
            synthetic_data['y'],
            range=[[0, 1], [0, 1]],
            bins=[10, 10],
        )
        assert np.allclose(ref_counts, counts)
