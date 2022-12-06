import pytest

from os.path import expandvars
from pathlib import Path

import numpy as np

from e15190.neutron_wall import efficiency as eff

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
