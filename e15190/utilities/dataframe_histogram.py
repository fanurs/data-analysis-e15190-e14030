import functools

import numpy as np
import pandas as pd

def arg_list_to_tuple(func):
    def wrapper(*args):
        args = [tuple(x) if isinstance(x, (list, pd.Index)) else x for x in args]
        result = func(*args)
        return result
    return wrapper

class Identifier:
    def __init__(self, df, fill_errors=False):
        self.df = df
        if fill_errors:
            self.fill_errors()
    
    @staticmethod
    @arg_list_to_tuple
    @functools.lru_cache(maxsize=100)
    def _check_length(cols):
        if len(cols) > 4:
            raise ValueError('Too many columns to determine xname. Only 1D histogram is supported.')
        if len(cols) < 2:
            raise ValueError('Too few columns to determine xname. At least 2 columns are required.')

    @staticmethod
    def _get_x(cols):
        Identifier._check_length(cols)
        return cols[0]

    @staticmethod
    def _get_y(cols):
        Identifier._check_length(cols)
        return cols[1]
    
    @staticmethod
    def _get_yerr(cols):
        Identifier._check_length(cols)
        y = Identifier._get_y(cols)
        if y + 'err' in cols:
            return y + 'err'
        if y + '_err' in cols:
            return y + '_err'
        
    @staticmethod
    def _get_yferr(cols):
        Identifier._check_length(cols)
        y = Identifier._get_y(cols)
        if y + 'ferr' in cols:
            return y + 'ferr'
        if y + '_ferr' in cols:
            return y + '_ferr'

    @functools.cached_property
    def x_name(self):
        return self._get_x(self.df.columns)

    @functools.cached_property
    def y_name(self):
        return self._get_y(self.df.columns)
    
    @functools.cached_property
    def yerr_name(self):
        return self._get_yerr(self.df.columns)
    
    @functools.cached_property
    def yferr_name(self):
        return self._get_yferr(self.df.columns)
    
    @property
    def x(self):
        return self.df[self.x_name]
    
    @property
    def y(self):
        return self.df[self.y_name]

    @property
    def yerr(self):
        return self.df[self.yerr_name]

    @property
    def yferr(self):
        return self.df[self.yferr_name]

    def fill_errors(self):
        if self.yerr_name is not None and self.yferr_name is None:
            yferr_name = self.yerr_name.replace('err', 'ferr')
            self.df[yferr_name] = np.abs(np.divide(
                self.df[self.yerr_name],
                self.df[self.y_name],
                out=np.zeros_like(self.df[self.yerr_name]),
                where=(self.df[self.y_name] != 0),
            ))
            del self.yferr_name
        elif self.yerr_name is None and self.yferr_name is not None:
            yerr_name = self.yferr_name.replace('ferr', 'err')
            self.df[yerr_name] = np.abs(self.df[self.yferr_name] * self.df[self.y_name])
            del self.yerr_name

def same_x_values(dfa, dfb):
    return np.allclose(dfa[Identifier(dfa).x_name], dfb[Identifier(dfb).x_name])

def _add_scalar(df, scalar):
    d = Identifier(df, fill_errors=True)
    result = pd.DataFrame({
        d.x_name: d.df[d.x_name],
        d.y_name: d.df[d.y_name] + scalar,
        d.yerr_name: d.df[d.yerr_name],
    })
    result[d.yferr_name] = np.abs(np.divide(
        result[d.yerr_name], result[d.y_name],
        out=np.zeros_like(result[d.yerr_name]),
        where=(result[d.y_name] != 0),
    ))
    return result

def add(dfa, dfb):
    if isinstance(dfb, (int, float)):
        return _add_scalar(dfa, dfb)
    if not same_x_values(dfa, dfb):
        raise ValueError('Cannot add dataframes with different x values.')
    a = Identifier(dfa, fill_errors=True)
    b = Identifier(dfb, fill_errors=True)
    result = pd.DataFrame({
        a.x_name: a.df[a.x_name],
        a.y_name: a.df[a.y_name] + dfb[b.y_name],
        a.yerr_name: np.sqrt(a.df[a.yerr_name]**2 + dfb[b.yerr_name]**2),
    })
    result[a.yferr_name] = np.abs(np.divide(
        result[a.yerr_name], result[a.y_name],
        out=np.zeros_like(result[a.yerr_name]),
        where=(result[a.y_name] != 0),
    ))
    return result

def _sub_scalar(df, scalar):
    d = Identifier(df, fill_errors=True)
    result = pd.DataFrame({
        d.x_name: d.df[d.x_name],
        d.y_name: d.df[d.y_name] - scalar,
        d.yerr_name: d.df[d.yerr_name],
    })
    result[d.yferr_name] = np.abs(np.divide(
        result[d.yerr_name], result[d.y_name],
        out=np.zeros_like(result[d.yerr_name]),
        where=(result[d.y_name] != 0),
    ))
    return result

def sub(dfa, dfb):
    if isinstance(dfb, (int, float)):
        return _sub_scalar(dfa, dfb)
    if not same_x_values(dfa, dfb):
        raise ValueError('Cannot subtract dataframes with different x values.')
    a = Identifier(dfa, fill_errors=True)
    b = Identifier(dfb, fill_errors=True)
    result = pd.DataFrame({
        a.x_name: a.df[a.x_name],
        a.y_name: a.df[a.y_name] - dfb[b.y_name],
        a.yerr_name: np.sqrt(a.df[a.yerr_name]**2 + dfb[b.yerr_name]**2),
    })
    result[a.yferr_name] = np.abs(np.divide(
        result[a.yerr_name], result[a.y_name],
        out=np.zeros_like(result[a.yerr_name]),
        where=(result[a.y_name] != 0),
    ))
    return result

def _mul_scalar(df, scalar):
    d = Identifier(df, fill_errors=True)
    return pd.DataFrame({
        d.x_name: d.df[d.x_name],
        d.y_name: d.df[d.y_name] * scalar,
        d.yerr_name: d.df[d.yerr_name] * scalar,
        d.yferr_name: d.df[d.yferr_name],
    })

def mul(dfa, dfb):
    if isinstance(dfb, (int, float)):
        return _mul_scalar(dfa, dfb)
    if not same_x_values(dfa, dfb):
        raise ValueError('Cannot multiply dataframes with different x values.')
    a = Identifier(dfa, fill_errors=True)
    b = Identifier(dfb, fill_errors=True)
    result = pd.DataFrame({
        a.x_name: a.df[a.x_name],
        a.y_name: a.df[a.y_name] * dfb[b.y_name],
        a.yferr_name: np.sqrt(a.df[a.yferr_name]**2 + dfb[b.yferr_name]**2),
    })
    result[a.yerr_name] = np.abs(result[a.y_name] * result[a.yferr_name])
    return result[[a.x_name, a.y_name, a.yerr_name, a.yferr_name]]

def _div_scalar(df, scalar):
    d = Identifier(df, fill_errors=True)
    return pd.DataFrame({
        d.x_name: d.df[d.x_name],
        d.y_name: d.df[d.y_name] / scalar,
        d.yerr_name: d.df[d.yerr_name] / scalar,
        d.yferr_name: d.df[d.yferr_name],
    })

def div(dfa, dfb):
    if isinstance(dfb, (int, float)):
        return _div_scalar(dfa, dfb)
    if not same_x_values(dfa, dfb):
        raise ValueError('Cannot divide dataframes with different x values.')
    a = Identifier(dfa, fill_errors=True)
    b = Identifier(dfb, fill_errors=True)
    result = pd.DataFrame({
        a.x_name: a.df[a.x_name],
        a.y_name: np.divide(
            a.df[a.y_name], dfb[b.y_name],
            out=np.zeros_like(a.df[a.y_name]),
            where=(dfb[b.y_name] != 0),
        ),
        a.yferr_name: np.sqrt(a.df[a.yferr_name]**2 + dfb[b.yferr_name]**2),
    })
    result[a.yerr_name] = np.abs(result[a.y_name] * result[a.yferr_name])
    return result[[a.x_name, a.y_name, a.yerr_name, a.yferr_name]]

def plot(df, ax=None, **kwargs):
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    hist = Identifier(df, fill_errors=True)

    kw = dict(
        fmt='.',
    )
    kw.update(kwargs)

    return ax.errorbar(
        hist.x,
        hist.y,
        yerr=hist.yerr,
        **kw,
    )
