#%%
import numpy as np
import pandas as pd

from e15190.hira import spectra

# %%
class TestLabPtransverseRapidity:
    def test__drop_spectrum_outliers(self):
        func = spectra.LabPtransverseRapidity._drop_spectrum_outliers
        df = pd.DataFrame([
            [1, 1e-4],
            [2, 1e-4],
            [3, 2e-4],
            [4, 3e-4],
            [5, 1e-2],
            [6, 2e-2],
            [7, 3e-2],
        ], columns=['x', 'y'])
        df_dropped = func(df, 10)
        return df_dropped

cls = TestLabPtransverseRapidity()
cls.test__drop_spectrum_outliers()
