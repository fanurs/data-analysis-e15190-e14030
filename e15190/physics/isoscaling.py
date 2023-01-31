#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, leastsq

from e15190.utilities import (
    dataframe_histogram as dfh
)

class Isoscaling:
    def __init__(self):
        self.yield_1 = dict()
        self.yield_2 = dict()
        self.ratios = dict() # {(N, Z): ratio_dataframe}
    
    def _add_first(self, df1, df2):
        df1 = dfh.Identifier(df1)
        df2 = dfh.Identifier(df2)
        if df1.x_name != df2.x_name:
            raise ValueError(f'df1 and df2 have different x_names: {df1.x_name}, {df2.x_name}')
        self.x_name = df1.x_name

        if df1.y_name != df2.y_name:
            raise ValueError(f'df1 and df2 have different y_names: {df1.y_name}, {df2.y_name}')
        self.y_name = df1.y_name

        if df1.yerr_name != df2.yerr_name:
            raise ValueError(f'df1 and df2 have different yerr_names: {df1.yerr_name}, {df2.yerr_name}')
        self.yerr_name = df1.yerr_name

    def add(self, N, Z, df1, df2):
        df1 = dfh.Identifier(df1)
        df2 = dfh.Identifier(df2)
        if len(self.ratios) == 0:
            self._add_first(df1.df, df2.df)

        if df1.x_name != self.x_name:
            raise ValueError(f'df1 has different x_name: {df1.x_name} (expecting {self.x_name})')
        if df2.x_name != self.x_name:
            raise ValueError(f'df2 has different x_name: {df2.x_name} (expecting {self.x_name})')

        if df1.y_name != self.y_name:
            raise ValueError(f'df1 has different y_name: {df1.y_name} (expecting {self.y_name})')
        if df2.y_name != self.y_name:
            raise ValueError(f'df2 has different y_name: {df2.y_name} (expecting {self.y_name})')

        if df1.yerr_name != self.yerr_name:
            raise ValueError(f'df1 has different yerr_name: {df1.yerr_name} (expecting {self.yerr_name})')
        if df2.yerr_name != self.yerr_name:
            raise ValueError(f'df2 has different yerr_name: {df2.yerr_name} (expecting {self.yerr_name})')

        self.yield_1[(N, Z)] = df1.df
        self.yield_2[(N, Z)] = df2.df
        self.ratios[(N, Z)] = dfh.div(df2.df, df1.df)
    
    def remove(self, N, Z):
        del self.ratios[(N, Z)]
    
    @property
    def all_x_values(self) -> list:
        x = set()
        for df_ratio in self.ratios.values():
            x.update(df_ratio.x.tolist())
        return list(sorted(x))

    def fit_with_independent_normalizations(self) -> pd.DataFrame:
        result = []
        x_vals = self.all_x_values
        for x in x_vals:
            N_vals, Z_vals = [], []
            iso_ratios, iso_ratios_err = [], []
            for (N, Z), df_ratio in self.ratios.items():
                row = df_ratio.query(f'{self.x_name} == {x}')
                if len(row) > 1:
                    raise ValueError(f'x value {x} is not unique in {row}')
                if len(row) == 0:
                    continue
                iso_ratios.append(row[self.y_name].iloc[0])
                iso_ratios_err.append(row[self.yerr_name].iloc[0])
                N_vals.append(N)
                Z_vals.append(Z)
            if len(iso_ratios) < 3:
                continue
            para, perr = self.fit_single(iso_ratios, N_vals, Z_vals, sigma=iso_ratios_err)
            perr = np.sqrt(np.diag(perr))
            result.append([
                x,
                [N_Z_pair for N_Z_pair in zip(N_vals, Z_vals)],
                *para, *perr,
            ])
        return pd.DataFrame(result,
            columns=['x', 'N_Z', 'C', 'alpha', 'beta', 'C_err', 'alpha_err', 'beta_err']
        )
    
    @staticmethod
    def fit_single(ratio_vals, N_vals, Z_vals, **kwargs):
        ratio_vals, N_vals, Z_vals = map(np.array, (ratio_vals, N_vals, Z_vals))
        model = lambda N_Z, C, alpha, beta: C * np.exp(alpha * N_Z[:, 0] + beta * N_Z[:, 1])
        if 'x0' not in kwargs:
            kwargs['p0'] = [1, 0.1, 0.1]
        return curve_fit(model, np.array([N_vals, Z_vals]).T, ratio_vals, **kwargs)

    @staticmethod
    def fit_leastsq(function, x, y, p0, yerr=None):
        if yerr is None:
            residual = lambda para: y - function(x, para)
        else:
            residual = lambda para: (y - function(x, para)) / yerr
        para, perr, *_ = leastsq(residual, p0, full_output=1, epsfcn=1e-8)
        if len(y) <= len(p0):
            raise ValueError(f'Too many parameters to fit. len(y) = {len(y)}, len(p0) = {len(p0)}')
        s_sq = np.sum(residual(para)**2) / (len(y) - len(p0))
        perr = perr * s_sq
        return para, np.sqrt(np.diag(perr))
    
    def fit_with_one_normalization(self) -> pd.DataFrame:
        def decode_data(data):
            n = len(data)
            return {'N': data[:n // 2], 'Z': data[n // 2:]}

        def decode_para(para):
            n = len(para)
            return {'C': para[0], 'alpha': para[1:(n + 1) // 2], 'beta': para[(n + 1) // 2:]}

        n_repeats = []
        def model(data, para):
            data = decode_data(data)
            para = decode_para(para)
            alpha = np.repeat(para['alpha'], n_repeats)
            beta = np.repeat(para['beta'], n_repeats)
            return para['C'] * np.exp(alpha * data['N'] + beta * data['Z'])

        # collect the "x" (data) and "y" (iso_ratios) to fit isoscaling
        data = [[], []] # [N_vals, Z_vals]
        iso_ratios, iso_ratios_err = [], []
        x_vals_to_fit = []
        for x in self.all_x_values:
            _data = [[], []]
            _iso_ratios, _iso_ratios_err = [], []
            for (N, Z), df_ratio in self.ratios.items():
                row = df_ratio.query(f'{self.x_name} == {x}')
                if len(row) > 1:
                    raise ValueError(f'x value {x} is not unique in {row}')
                if len(row) == 0:
                    continue
                _iso_ratios.append(row[self.y_name].iloc[0])
                _iso_ratios_err.append(row[self.yerr_name].iloc[0])
                _data[0].append(N)
                _data[1].append(Z)
            _n_data = len(_iso_ratios)
            if _n_data < 3:
                continue
            x_vals_to_fit.append(x)
            iso_ratios.extend(_iso_ratios)
            iso_ratios_err.extend(_iso_ratios_err)
            data[0].extend(_data[0])
            data[1].extend(_data[1])
            n_repeats.append(_n_data)
        data = np.hstack(data)

        # fit the isoscaling
        para, perr = self.fit_leastsq(model, data, iso_ratios, p0=[
            1, *[0] * len(x_vals_to_fit), *[0] * len(x_vals_to_fit),
        ], yerr=iso_ratios_err)

        # collect and beautify the results
        para, perr = map(decode_para, (para, perr))
        N_Z_pairs = [tuple(ele) for ele in data.reshape(2, -1).T]
        if len(N_Z_pairs) != np.sum(n_repeats):
            raise ValueError(f'Expected {np.sum(n_repeats)} N_Z pairs, but got {len(N_Z_pairs)}')
        N_Z = []
        for n_repeat in n_repeats:
            N_Z.append(N_Z_pairs[:n_repeat])
            N_Z_pairs = N_Z_pairs[n_repeat:]
        return pd.DataFrame({
            'x': x_vals_to_fit,
            'N_Z': N_Z,
            'C': para['C'], 'alpha': para['alpha'], 'beta': para['beta'],
            'C_err': perr['C'], 'alpha_err': perr['alpha'], 'beta_err': perr['beta'],
        })
    
    def fit(self, fix_normalization=True) -> pd.DataFrame:
        if fix_normalization:
            return self.fit_with_one_normalization()
        return self.fit_with_independent_normalizations()

    def get_albergo_temperature(self, reaction_idx) -> pd.DataFrame:
        deuteron = (1, 1)
        triton = (2, 1)
        helium3 = (1, 2)
        helium4 = (2, 2)

        yield_ = self.yield_1 if reaction_idx == 1 else self.yield_2
        x = None
        for (N, Z) in [deuteron, triton, helium3, helium4]:
            if x is None:
                x = set(yield_[(N, Z)][self.x_name])
            else:
                x = x.intersection(yield_[(N, Z)][self.x_name])
        x = sorted(x)

        df_deuteron = yield_[deuteron].query(f'{self.x_name} in {x}')
        df_triton = yield_[triton].query(f'{self.x_name} in {x}')
        df_helium3 = yield_[helium3].query(f'{self.x_name} in {x}')
        df_helium4 = yield_[helium4].query(f'{self.x_name} in {x}')

        temperature = dfh.div(dfh.mul(df_deuteron, df_helium4), dfh.mul(df_triton, df_helium3))
        temperature['y'] = 14.29 / np.log(1.59 * temperature['y'])
        temperature['yerr'] = 14.29 / (temperature['y'] * (np.log(1.59 * temperature['y']))**2)
        temperature['yferr'] = temperature['yerr'] / temperature['y']
        return temperature
