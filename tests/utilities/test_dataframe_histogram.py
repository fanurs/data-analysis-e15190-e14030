import pytest

import numpy as np
import pandas as pd

from e15190.utilities import dataframe_histogram as dfh

@pytest.fixture
def dfs():
    return [
        pd.DataFrame({ # [0]
            'x': [1, 2, 3],
            'y': [3, 2, 1],
            'yerr': [0.3, 0.2, 0.1],
            'yferr': [0.1, 0.1, 0.1],
        }),
        pd.DataFrame({ # [1]
            'energy': [1, 2, 3],
            'multi': [3, 1, 0],
            'multi_err': [0.6, 0.1, 0],
            'multi_ferr': [0.2, 0.1, 0],
        }),
        pd.DataFrame({ # [2]
            'x': [1, 2, 3, 4],
            'y': [3, 1, 0, 0],
            'y_err': [0.6, 0.1, 0, 0],
            'y_ferr': [0.2, 0.1, 0, 0],
        }),
        pd.DataFrame({ # [3]
            'x': [1, 2, 3, 4],
            'y': [3, 1, 0, 0],
            'yerr': [0.6, 0.1, 0, 0],
        }),
        pd.DataFrame({ # [4]
            'x': [1, 2, 3, 4],
            'y': [3, 1, 0, 0],
            'y_ferr': [0.2, 0.1, 0, 0],
        }),
    ]

@pytest.fixture
def cols():
    return [
        ['x', 'y', 'yerr', 'yferr'], # [0]
        ['energy', 'multi', 'multi_err', 'multi_ferr'], # [1]
        ['x', 'pt', 'pterr', 'ptferr'], # [2]
        ['x', 'y', 'z', 'zerr', 'zferr'], # [3]
        ['p'], # [4]
        ['p', 'q'], # [5]
        ['p', 'q', 'qerr'], # [6]
        ['p', 'q', 'qferr'], # [7]
    ]

class TestIdentifier:
    def test___init__(self, dfs):
        for df in dfs:
            identifier = dfh.Identifier(df)
            assert isinstance(identifier.df, pd.DataFrame)
    
    def test__get_x(self, cols):
        func = dfh.Identifier._get_x
        assert func(cols[0]) == 'x'
        assert func(cols[1]) == 'energy'
        assert func(cols[2]) == 'x'
        with pytest.raises(ValueError) as excinfo:
            func(cols[3])
            assert 'too many' in str(excinfo.value).lower()
        with pytest.raises(ValueError) as excinfo:
            func(cols[4])
            assert 'too few' in str(excinfo.value).lower()
        assert func(cols[5]) == 'p'
        assert func(cols[6]) == 'p'
        assert func(cols[7]) == 'p'
    
    def test__get_y(self, cols):
        func = dfh.Identifier._get_y
        assert func(cols[0]) == 'y'
        assert func(cols[1]) == 'multi'
        assert func(cols[2]) == 'pt'
        with pytest.raises(ValueError) as excinfo:
            func(cols[3])
            assert 'too many' in str(excinfo.value).lower()
        with pytest.raises(ValueError) as excinfo:
            func(cols[4])
            assert 'too few' in str(excinfo.value).lower()
        assert func(cols[5]) == 'q'
        assert func(cols[6]) == 'q'
        assert func(cols[7]) == 'q'

    def test__get_yerr(self, cols):
        func = dfh.Identifier._get_yerr
        assert func(cols[0]) == 'yerr'
        assert func(cols[1]) == 'multi_err'
        assert func(cols[2]) == 'pterr'
        with pytest.raises(ValueError) as excinfo:
            func(cols[3])
            assert 'too many' in str(excinfo.value).lower()
        with pytest.raises(ValueError) as excinfo:
            func(cols[4])
            assert 'too few' in str(excinfo.value).lower()
        assert func(cols[5]) is None
        assert func(cols[6]) == 'qerr'
        assert func(cols[7]) is None
    
    def test__get_yferr(self, cols):
        func = dfh.Identifier._get_yferr
        assert func(cols[0]) == 'yferr'
        assert func(cols[1]) == 'multi_ferr'
        assert func(cols[2]) == 'ptferr'
        with pytest.raises(ValueError) as excinfo:
            func(cols[3])
            assert 'too many' in str(excinfo.value).lower()
        with pytest.raises(ValueError) as excinfo:
            func(cols[4])
            assert 'too few' in str(excinfo.value).lower()
        assert func(cols[5]) is None
        assert func(cols[6])  is None
        assert func(cols[7]) == 'qferr'

    def test_x_name(self, dfs):
        assert dfh.Identifier(dfs[0]).x_name == 'x'
    
    def test_y_name(self, dfs):
        assert dfh.Identifier(dfs[0]).y_name == 'y'

    def test_yerr_name(self, dfs):
        assert dfh.Identifier(dfs[0]).yerr_name == 'yerr'

    def test_yferr_name(self, dfs):
        assert dfh.Identifier(dfs[0]).yferr_name == 'yferr'

    def test_fill_errors(self, dfs):
        df3 = dfh.Identifier(dfs[3], fill_errors=False)
        assert df3.yerr_name == 'yerr'
        assert df3.yferr_name is None
        df3.fill_errors()
        assert df3.yerr_name == 'yerr'
        assert df3.yferr_name == 'yferr'

        df4 = dfh.Identifier(dfs[4], fill_errors=False)
        assert df4.yerr_name is None
        assert df4.yferr_name == 'y_ferr'
        df4.fill_errors()
        assert df4.yerr_name == 'y_err'
        assert df4.yferr_name == 'y_ferr'

def test_same_x_values(dfs):
    func = dfh.same_x_values
    assert func(dfs[0], dfs[1])
    with pytest.raises(ValueError) as excinfo:
        func(dfs[0], dfs[2])
        assert 'could not be broadcast' in str(excinfo.value).lower()

def test_add(dfs):
    df_add = dfh.add(dfs[0], dfs[1])
    assert np.allclose(df_add.y, dfs[0].y + dfs[1].multi)

def test_sub(dfs):
    df_sub = dfh.sub(dfs[0], dfs[1])
    assert np.allclose(df_sub.y, dfs[0].y - dfs[1].multi)

def test_mul(dfs):
    df_mult = dfh.mul(dfs[0], dfs[1])
    assert np.allclose(df_mult.y, dfs[0].y * dfs[1].multi)

def test_div(dfs):
    df_div = dfh.div(dfs[0], dfs[1])
    assert np.allclose(df_div.y, np.divide(
        dfs[0].y, dfs[1].multi,
        out=np.zeros_like(dfs[0].y),
        where=(dfs[1].multi != 0),
    ))

    df_div = dfh.div(dfs[3], dfs[4])
    assert np.allclose(df_div.y, [1, 1, 0, 0])
    assert np.allclose(np.sign(df_div.yerr), [1, 1, 0, 0])
    assert np.allclose(
        df_div.yferr,
        [
            np.sqrt((0.6 / 3)**2 + 0.2**2),
            np.sqrt((0.1 / 1)**2 + 0.1**2),
            0,
            0,
        ]
    )
