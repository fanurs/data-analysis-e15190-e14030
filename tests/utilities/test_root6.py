import pytest

import numpy as np
from ROOT import TH1D, TH2D

from e15190.utilities import root6

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

def test_random_name():
    names = [root6.random_name() for _ in range(10)]
    assert len(set(names)) == 10

class TestHistogramConversion:
    def test_histo1d(self, synthetic_data):
        ref_counts, _ = np.histogram(synthetic_data['x'], range=[0, 1], bins=10)
        h1 = TH1D('h1', '', 10, 0, 1)
        for x in synthetic_data['x']:
            h1.Fill(x)
        counts = root6.histo_conversion.histo_to_dframe(h1)['y'].to_numpy()
        assert np.allclose(ref_counts, counts)
    
    def test_histo2d(self, synthetic_data):
        ref_counts, _, _ = np.histogram2d(
            synthetic_data['x'], synthetic_data['y'],
            range=[[0, 1], [0, 1]],
            bins=[10, 10],
        )
        h2 = TH2D('h2', '', 10, 0, 1, 10, 0, 1)
        for x, y in zip(synthetic_data['x'], synthetic_data['y']):
            h2.Fill(x, y)
        counts = root6.histo_conversion.histo_to_dframe(h2)['z'].to_numpy()
        assert np.allclose(ref_counts.flatten(), counts)
