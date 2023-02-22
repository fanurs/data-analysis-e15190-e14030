#%%
import pandas as pd

from e15190.physics import isoscaling

class TestIsoscaling:
    def test___init__(self):
        isc = isoscaling.Isoscaling()
        assert isc.ratios == dict()

    def test_add(self):
        isc = isoscaling.Isoscaling()
        isc.add(
            0, 1,
            pd.DataFrame([
                [1, 2.0, 0.2],
                [2, 4.0, 0.4],
                [3, 5.0, 0.6],
            ], columns=['x', 'y', 'yerr']),
            pd.DataFrame([
                [1, 3.0, 0.15],
                [2, 4, 0.2],
                [3, 3, 0.6],
            ], columns=['x', 'y', 'yerr']),
        )
        assert (0, 1) in isc.ratios
        df = isc.ratios[(0, 1)]
        assert df.x.tolist() == [1, 2, 3]
        assert df.y.tolist() == [3 / 2, 4 / 4, 3 / 5]
    
    def test_remove(self):
        isc = isoscaling.Isoscaling()
        isc.add(
            0, 1,
            pd.DataFrame([
                [1, 2.0, 0.2],
                [2, 4.0, 0.4],
                [3, 5.0, 0.6],
            ], columns=['x', 'y', 'yerr']),
            pd.DataFrame([
                [1, 3.0, 0.15],
                [2, 4, 0.2],
                [3, 3, 0.6],
            ], columns=['x', 'y', 'yerr']),
        )
        isc.remove(0, 1)
        assert isc.ratios == dict()
    
    
