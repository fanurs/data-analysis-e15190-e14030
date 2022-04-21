import json
import os
from pathlib import Path
from typing import Literal

import duckdb as dk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import RANSACRegressor

from e15190.neutron_wall.position_calibration import NWCalibrationReader
from e15190.neutron_wall.cache import RunCache
from e15190.utilities import fast_histogram as fh, misc, slicer

class AmBeRun:
    DATABASE_DIR = '$DATABASE_DIR/neutron_wall/ambe'
    CACHE_RELPATH = r'cache/run-{run:04d}.db'

    def __init__(self, AB: Literal['A', 'B']):
        self.AB = AB.upper()
        self.ab = AB.lower()
        self.df = None
    
    @staticmethod
    def _get_daniele_root_files_dir(json_path=None) -> Path:
        if json_path is None:
            json_path = os.path.expandvars('$DATABASE_DIR/local_paths.json')
        with open(json_path) as file:
            return Path(json.load(file)['daniele_root_files_dir'])

    def read(self, runs, from_cache=True, **kwargs) -> pd.DataFrame:
        """Update ``self.df``"""
        rc = RunCache(
            src_path_fmt=str(self._get_daniele_root_files_dir() / r'CalibratedData_{run:04d}.root'),
            cache_path_fmt=str(Path(os.path.expandvars(self.DATABASE_DIR)) / self.CACHE_RELPATH),
        )

        kw = dict(
            from_cache=from_cache,
            sql_cmd='WHERE ' + ' AND '.join([
                f'NW{self.AB}_time_L - NW{self.AB}_time_R < 20',
                f'NW{self.AB}_time_L - NW{self.AB}_time_R > -20',
                f'NW{self.AB}_total_L * NW{self.AB}_total_R > {50**2}',
            ]),
        )
        kw.update(kwargs)

        self.runs = [runs] if isinstance(runs, int) else runs
        self.df = rc.read(
            self.runs,
            {
                f'NW{self.AB}.fnumbar'                      : f'NW{self.AB}_bar',
                f'NW{self.AB}.fLeft'                        : f'NW{self.AB}_total_L',
                f'NW{self.AB}.fRight'                       : f'NW{self.AB}_total_R',
                f'NW{self.AB}.fTimeLeft'                    : f'NW{self.AB}_time_L',
                f'NW{self.AB}.fTimeRight'                   : f'NW{self.AB}_time_R',
            },
            **kw,
        )

        self.df = misc.convert_64_to_32(self.df)
        self.df[f'NW{self.AB}_bar'] = self.df[f'NW{self.AB}_bar'].astype(np.int16)
        return self.df

    def add_x_positions(self, drop_time=True) -> pd.DataFrame:
        bar = f'NW{self.AB}_bar'
        time_L = f'NW{self.AB}_time_L'
        time_R = f'NW{self.AB}_time_R'

        calib_reader = NWCalibrationReader(self.AB)
        df_result = None
        for run, df_run in self.df.groupby('run'):
            pars = calib_reader(run, extrapolate=True)
            pars = pars.loc[df_run[bar]].to_numpy()
            df_run['pos_x'] = pars[:, 0] + pars[:, 1] * (df_run[time_L] - df_run[time_R])
            df_result = pd.concat([df_result, df_run], axis=0)
        df_result = misc.convert_64_to_32(df_result)

        if drop_time:
            df_result = df_result.drop([time_L, time_R], axis=1)
        self.df = df_result
        return self.df

    def randomize_ADC(self, seed=None):
        columns = [
            f'NW{self.AB}_total_L',
            f'NW{self.AB}_total_R',
        ]
        self.df = misc.randomize_columns(self.df, columns, seed)
        self.df = misc.convert_64_to_32(self.df)
        return self.df

    def fit_compton_edges(self, verbose=False):
        self.edge_points = dict()
        self.edge_fits = dict()
        for bar, df_bar in self.df.groupby(f'NW{self.AB}_bar'):
            if verbose:
                print(f'Fitting Compton edge for NW{self.AB} bar-{bar:02d}')
            fitter = ComptonEdgeFitter(df_bar).fit()
            self.edge_points[bar] = fitter.df_edge
            self.edge_fits[bar] = fitter.edge_fit

    def save_params(self, path):
        df_coefs = []
        for bar, edge_fit in self.edge_fits.items():
            df_coefs.append([bar, *edge_fit.coef])
        df_coefs = pd.DataFrame(df_coefs, columns=['bar', 'p0', 'p1', 'p2'])
        df_coefs.to_csv(path, index=False)
    
    def save_plot(self, path, bar):
        with mpl.rc_context({'backend': 'Agg'}):
            plotter = FitPlotter(self)
            fig, _ = plotter.plot_bar(bar)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.draw()
            plt.close()


class UshapeRansacEstimator:
    def model(self, x, p0, p1, p2):
        return p0 + p1 * x + p2 * x**2

    def fit(self, X, y):
        self.par, _ = optimize.curve_fit(self.model, X.flatten(), y)
        return self
    
    def predict(self, X):
        return self.model(X.flatten(), *self.par)

    def score(self, X, y):
        y_pred = self.predict(X)
        y_true = y
        u = np.sum((y_true - y_pred)**2)
        v = np.sum((y_true - np.mean(y_true))**2)
        v = np.maximum(v, 1e-12) # avoid division by zero
        return 1 - u / v
    
    def get_params(self, deep=True):
        return dict()
    
    def set_params(self, **params):
        return UshapeRansacEstimator()


class ComptonEdgeFitter:
    def __init__(self, df_bar):
        if 'NWB_total_L' in df_bar.columns:
            total_L, total_R = 'NWB_total_L', 'NWB_total_R'
        if 'NWA_total_L' in df_bar.columns:
            total_L, total_R = 'NWA_total_L', 'NWA_total_R'

        self.df = pd.DataFrame({
            'pos_x': df_bar['pos_x'],
            'total_GM': np.sqrt(df_bar[total_L] * df_bar[total_R]),
        })
    
    @staticmethod
    def _gaus(t, amplt, mean, sigma):
        return amplt * np.exp(-0.5 * ((t - mean) / sigma) ** 2)
    
    @classmethod
    def fit_1d(cls, arr, bound=None):
        if bound is None:
            bound = (200, 600)
        
        h_range = [0, 1000]
        h_bins = 10_000
        y = fh.histo1d(arr, range=h_range, bins=h_bins)
        x = np.linspace(*h_range, h_bins)
        y_conv = np.convolve(
            y,
            cls._gaus(np.linspace(-5, 5, 1000), 1, 0, 1),
            mode='same',
        )
        y_conv /= np.sum(y_conv)
        spline = UnivariateSpline(x, y_conv, k=5, s=0)
        deriv = spline.derivative()

        obj_func = lambda t: -1e8 * np.abs(deriv(t))
        brute_x = optimize.brute(obj_func, ranges=[bound], Ns=100)[0]
        min_x = optimize.minimize(obj_func, x0=brute_x, bounds=[bound]).x[0]
        if abs((min_x - brute_x) / brute_x) > 0.1:
            min_x = brute_x
        return min_x
    
    def get_slice_fits(self):
        x_ranges_mid = slicer.create_ranges(-70, 70, width=10, step=5)
        x_ranges_left = slicer.create_ranges(-110, -60, width=20, step=5)
        x_ranges_right = slicer.create_ranges(60, 110, width=20, step=5)
        x_ranges = np.concatenate([
            x_ranges_left,
            x_ranges_mid,
            x_ranges_right,
        ])

        self.df_edge = []
        df_ = self.df
        for x_range in x_ranges:
            arr = dk.query(f'''
                SELECT total_GM from df_
                WHERE pos_x > {x_range[0]} AND pos_x < {x_range[1]};
            ''').df()['total_GM']
            edge = self.fit_1d(arr)
            self.df_edge.append([np.mean(x_range), edge])
        self.df_edge = pd.DataFrame(self.df_edge, columns=['pos_x', 'edge'])
        return self.df_edge
    
    def fit(self):
        self.df_edge = self.get_slice_fits()
        X = np.array(self.df_edge['pos_x'])[:, None]
        y = np.array(self.df_edge['edge'])
        ransac = RANSACRegressor(
            base_estimator=UshapeRansacEstimator(),
            min_samples=0.7,
        ).fit(X, y)
        self.edge_fit = np.polynomial.Polynomial(ransac.estimator_.par)
        return self


class FitPlotter:
    def __init__(self, fitter):
        self.fitter = fitter
        self.AB = self.fitter.AB

    def plot_bar(self, bar):
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 7), constrained_layout=True)
        
        ax = axes[0, 0]
        self.plot_hist2d(ax=ax, bar=bar)
        self.plot_compton_points_on_2d(ax=ax, bar=bar)
        self.plot_compton_fit_on_2d(ax=ax, bar=bar)

        ax = axes[0, 1]
        pos_x_range = [-5, 5]
        self.plot_hist1d(ax=ax, bar=bar, pos_x_range=pos_x_range)
        self.plot_compton_edge_on_1d(ax=ax, bar=bar, pos_x_range=pos_x_range)

        ax = axes[1, 0]
        pos_x_range = [-80, -70]
        self.plot_hist1d(ax=ax, bar=bar, pos_x_range=pos_x_range)
        self.plot_compton_edge_on_1d(ax=ax, bar=bar, pos_x_range=pos_x_range)

        ax = axes[1, 1]
        pos_x_range = [70, 80]
        self.plot_hist1d(ax=ax, bar=bar, pos_x_range=pos_x_range)
        self.plot_compton_edge_on_1d(ax=ax, bar=bar, pos_x_range=pos_x_range)

        return fig, axes
    
    def plot_hist2d(self, ax, bar):
        df_bar = self.fitter.df.query(f'NW{self.AB}_bar == {bar}')
        fh.plot_histo2d(
            ax.hist2d,
            df_bar['pos_x'],
            np.sqrt(df_bar[f'NW{self.AB}_total_L'] * df_bar[f'NW{self.AB}_total_R']),
            range=[[-120, 120], [0, 1500]],
            bins=[500, 500],
            cmap='viridis',
            norm=mpl.colors.LogNorm(vmin=1),
        )
        ax.set_title(f'NWB-bar{bar:02d}')
        ax.set_xlabel(r'Position $x$ [cm]')
        ax.set_ylabel('G.M. of TOTAL [ADC]')
        ax.set_xlim(-120, 120)
        ax.set_ylim(0, )

    def plot_compton_points_on_2d(self, ax, bar):
        df_edge = self.fitter.edge_points[bar]
        ax.scatter(df_edge['pos_x'], df_edge['edge'], s=3, color='pink', zorder=101)
        ax.scatter(df_edge['pos_x'], df_edge['edge'], s=6, color='black', zorder=100)

    def plot_compton_fit_on_2d(self, ax, bar):
        fit = self.fitter.edge_fits[bar]
        x_plt = np.linspace(-120, 120, 500)
        ax.plot(x_plt, fit(x_plt), color='red', linewidth=0.8)

    def plot_hist1d(self, ax, bar, pos_x_range):
        df_bar = self.fitter.df.query(f'''
            NW{self.AB}_bar == {bar} & pos_x > {pos_x_range[0]} & pos_x < {pos_x_range[1]}
        ''')
        fh.plot_histo1d(
            ax.hist,
            np.sqrt(df_bar[f'NW{self.AB}_total_L'] * df_bar[f'NW{self.AB}_total_R']),
            range=[50, 850],
            bins=400,
            histtype='stepfilled',
            color='lightgray',
            edgecolor='black',
        )
        title = r'$x = %.0f \pm %.0f$ cm' % (np.mean(pos_x_range), np.abs(np.diff(pos_x_range)) / 2)
        ax.set_title(f'NW{self.AB}-bar{bar:02d}: ' + title)
        ax.set_xlabel('G.M. of TOTAL [ADC]')
        ax.set_xlim(0, 850)
        ax.set_ylim(0, )
    
    def plot_compton_edge_on_1d(self, ax, bar, pos_x_range):
        fit = self.fitter.edge_fits[bar]
        x = np.mean(pos_x_range)
        ax.axvline(fit(x), color='red', linewidth=0.8)


if __name__ == '__main__':
    VERBOSE = True

    ambe_run = AmBeRun('B')
    # ambe_run.read([4798, 4799, 4800, 4801, 4802])
    ambe_run.read([3072, 3073])
    ambe_run.randomize_ADC()
    ambe_run.add_x_positions()
    ambe_run.fit_compton_edges(verbose=VERBOSE)

    path = Path(os.path.expandvars(AmBeRun.DATABASE_DIR))
    path /= f'compton_edges_{min(ambe_run.runs)}-{max(ambe_run.runs)}.csv'
    ambe_run.save_params(path)

    directory = Path(os.path.expandvars(AmBeRun.DATABASE_DIR))
    directory /= f'gallery/run-{min(ambe_run.runs)}-{max(ambe_run.runs)}'
    directory.mkdir(parents=True, exist_ok=True)
    for bar in range(1, 25):
        path = directory / f'NW{ambe_run.AB}-bar{bar:02d}.png'
        ambe_run.save_plot(path, bar)
        if VERBOSE:
            print(f'Saved plot to "{path}"')
