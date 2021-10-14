import pytest

import numpy as np
import pandas as pd

from e15190.neutron_wall import position_calibration

class TestNWBPositionCalibrator:
    def test_smoke(self):
        calib = position_calibration.NWBPositionCalibrator()
        assert calib.read_run(4100, use_cache=True)
        assert calib.calibrate(save_params=False)
        calib_params = calib.calib_params
        for bar, params in calib_params.items():
            assert -100.0 < params[0] < 100.0
            assert 6.0 < params[1] < 8.5

class TestNWCalibrationReader:
    reader = position_calibration.NWCalibrationReader('B')

    def test_existing_runs(self):
        existing_runs = list(range(4134, 2870 + 1)) + list(range(4007, 4661 + 1))
        for run in existing_runs:
            df = self.reader(run)
            assert isinstance(df, pd.DataFrame)
            assert df['p0'].min() > -100.0
            assert df['p1'].min() < 100.0
            assert df['p1'].min() > 6.0
            assert df['p1'].max() < 8.5
    
    def test_nonexisting_runs(self):
        nonexisting_runs = [1999, 3001, 3999, 5000]

        for run in nonexisting_runs:
            with pytest.raises(ValueError) as err:
                self.reader(run)
    
    def test_extrapolation(self):
        df0 = self.reader(1998, extrapolate=True)
        df1 = self.reader(1999, extrapolate=True)
        assert np.allclose(df0.to_numpy(), df1.to_numpy())

        df0 = self.reader(3000, extrapolate=True)
        df1 = self.reader(3001, extrapolate=True)
        assert np.allclose(df0.to_numpy(), df1.to_numpy())

        df0 = self.reader(3998, extrapolate=True)
        df1 = self.reader(3999, extrapolate=True)
        assert np.allclose(df0.to_numpy(), df1.to_numpy())

        df0 = self.reader(5000, extrapolate=True)
        df1 = self.reader(5001, extrapolate=True)
        assert np.allclose(df0.to_numpy(), df1.to_numpy())