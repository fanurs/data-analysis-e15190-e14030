from typing import Any
import pytest

from os.path import expandvars
from pathlib import Path

import numpy as np
import pandas as pd

from e15190.neutron_wall import efficiency as eff

class Test_geNt4:
    def setup_class(self):
        self.df = eff.geNt4.get_intrinsic_efficiency_data()
        self.curve = eff.geNt4.get_intrinsic_efficiency()

    def test_geNt4_PATH(self):
        # Test if the PATH attribute is defined
        assert hasattr(eff.geNt4, 'PATH')

    def test_geNt4_PATH_exists(self):
        # Test if the PATH attribute points to an existing file
        assert Path(expandvars(eff.geNt4.PATH)).exists()

    def test_get_intrinsic_efficiency_data(self):
        # Test if it returns a pandas DataFrame
        assert isinstance(self.df, pd.DataFrame)

    def test_get_intrinisic_efficiency_data_columns(self):
        # Test if the DataFrame has the correct columns
        assert list(self.df.columns) == ['energy', 'efficiency']
    
    def test_get_intrinsic_efficiency_data_dtypes(self):
        # Test if the DataFrame has the correct dtypes
        assert list(self.df.dtypes) == [int, float]
    
    def test_get_intrinsic_efficiency_data_energy_monotonicity(self):
        # Test if energy column is monotonically increasing
        assert np.all(np.diff(self.df.energy) > 0)

    def test_get_intrinsic_efficiency_data_energy_range(self):
        # Test is the smallest energy is around zero (within 10 MeV)
        assert np.isclose(self.df.energy.min(), 0, atol=10)

        # Test if the largest energy is around 300 MeV (within 10 MeV)
        assert np.isclose(self.df.energy.max(), 300, atol=10)
    
    def test_get_intrinsic_efficiency_data_efficiency_monotonicity(self):
        idx_max = self.df.efficiency.idxmax()

        # Test if efficiency is first monotonically increasing
        assert np.all(np.diff(self.df.efficiency[:idx_max]) > 0)

        # Test if afterward monotonically decreasing within some tolerance (1e-3)
        assert np.all(np.diff(self.df.efficiency[idx_max:]) < 1e-3)

    def test_get_intrinsic_efficiency_data_efficiency_range(self):
        # Test if the largest efficiency is around 0.1 (within 0.01)
        assert np.isclose(self.df.efficiency.max(), 0.1, atol=0.01)

        # Test if the smallest efficiency above energy 50 MeV is around 0.03 (within 0.1)
        assert np.isclose(self.df.query('energy > 50').efficiency.min(), 0.03, atol=0.01)

    def test_get_intrinsic_efficiency(self):
        # Test if it returns a callable
        assert callable(self.curve)
    
    def test_get_intrinsic_efficiency_boundary(self):
        # Test if the boundary condition is zero at zero energy
        assert np.allclose(self.curve(np.array([0.0])), 0.0)
    
    def test_get_intrinsic_efficiency_interpolation(self):
        # Test if the interpolation is correct
        # Points evaluated at the spline knots should be equal to the original data points
        assert np.allclose(self.curve(self.df.energy), self.df.efficiency)
    
    def test_get_intrinsic_efficiency_extrapolation(self):
        # Points evaluated below zero or above the highest energy data point should be zero
        assert np.allclose(self.curve(np.array([0.0, 1e3])), 0.0)
    
    def test_get_intrinsic_efficiency_monotonicity(self):
        # Test if efficiency is monotonically decreasing after 50 MeV (within 1e-3)
        assert np.all(np.diff(self.curve(np.linspace(50, 300, 1000))) < 1e-3)

class TestScinfulQmd:
    def test_get_response(self):
        func = eff.ScinfulQmd.read_response

        df = func(20, 'scinful')
        assert list(df.columns) == ['light', 'resp', 'err']
        assert list(df.dtypes) == [float] * 3
        assert np.all(np.diff(df.light) > 0) # light is monotonically increasing
    
        df = func(150, 'qmd')
        assert list(df.columns) == ['light', 'resp', 'err']
        assert list(df.dtypes) == [float] * 3
        assert np.all(np.diff(df.light) > 0) # light is monotonically increasing

    def test_calculate_efficiency(self):
        func = eff.ScinfulQmd.calculate_efficiency

        df = eff.ScinfulQmd.read_response(65, 'scinful')
        assert func(df) == func(df, bias=3.0) # default bias is 3.0
        assert func(df, bias=1.0) > func(df, bias=3.0) # efficiency increases as bias decreases
        assert func(df, bias=0.0) < 1.0 # efficiency is less than 1.0
    
    def test_read_efficiency_curve(self):
        func = eff.ScinfulQmd.read_efficiency_curve

        df = func('scinful', (20, 100))
        assert list(df.columns) == ['energy', 'efficiency']
        assert list(df.dtypes) == [int, float]
        assert np.all(np.diff(df.energy) > 0)
        subdf = df.query('energy > 50')
        assert np.all(np.diff(subdf.efficiency) < 0) # efficiency is monotonically decreasing after 50 MeV

    def test_get_efficiency_curve(self):
        func = eff.ScinfulQmd.get_efficiency_curve

        assert not Path(expandvars(eff.ScinfulQmd.RESULT_PATH)).exists()
        df_from_raw = func(from_raw_output=True, save_txt=True)
        assert Path(expandvars(eff.ScinfulQmd.RESULT_PATH)).exists()

        df_from_txt = func(from_raw_output=False, save_txt=False)
        assert list(df_from_raw.columns) == list(df_from_txt.columns)
        assert np.array_equal(df_from_raw.energy, df_from_txt.energy)
        assert np.allclose(df_from_raw.efficiency, df_from_txt.efficiency)
        assert list(df_from_raw['mode']) == list(df_from_txt['mode'])
