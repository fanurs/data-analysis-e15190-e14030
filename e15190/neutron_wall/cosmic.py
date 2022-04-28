#%%
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor

from e15190.neutron_wall.geometry import Wall as NWwall
from e15190.neutron_wall.position_calibration import NWCalibrationReader
from e15190.neutron_wall.cache import RunCache
from e15190.runlog.query import Query
from e15190.utilities import fast_histogram as fh, misc

#%%
class CosmicRun:
    DATABASE_DIR = '$DATABASE_DIR/neutron_wall/cosmic'
    CACHE_RELPATH = r'cache/run-{run:04d}.db'

    def __init__(self, AB: Literal['A', 'B'], min_multi=5):
        self.AB = AB.upper()
        self.ab = AB.lower()
        self.min_multi = min_multi
        self.df = None
        self.df_tracks = None
    
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
                f'NW{self.AB}_multi >= {self.min_multi}',
                f'NW{self.AB}_time_L - NW{self.AB}_time_R < 20',
                f'NW{self.AB}_time_L - NW{self.AB}_time_R > -20',
            ]),
        )
        kw.update(kwargs)

        self.runs = [runs] if isinstance(runs, int) else runs
        self.df = rc.read(
            self.runs,
            {
                f'NW{self.AB}.fmulti'                       : f'NW{self.AB}_multi',
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

        # original multi is no longer reliable as some rows in an entry could be filtered out
        self._refilter_multi()
        return self.df
    
    def _refilter_multi(self):
        self.df[f'NW{self.AB}_multi'] = self.df.groupby(level=('run', 'entry')).size()
        return self.df.query(f'NW{self.AB}_multi >= {self.min_multi}')
    
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

    def add_y_positions(self):
        nw_bars = NWwall(self.AB).bars
        heights = {b: bar.pca.mean_[1] for b, bar in nw_bars.items()}
        self.df['pos_y'] = self.df[f'NW{self.AB}_bar'].map(heights)
        return self.df

    def randomize_ADC(self, seed=None):
        columns = [
            f'NW{self.AB}_total_L',
            f'NW{self.AB}_total_R',
        ]
        self.df = misc.randomize_columns(self.df, columns, seed)
        self.df = misc.convert_64_to_32(self.df)
        return self.df
    
    def get_tracks(self, from_cache=True, update_cache=True, append=True):
        fitter = TrackFitter(self.df)
        self.df_tracks = fitter.get_tracks(from_cache=from_cache)
        if update_cache:
            fitter.update_cache(append=append)
        return self.df_tracks
    
    def calculate_track_angles(self):
        self.df_tracks['angle'] = np.degrees(np.arctan2(self.df_tracks['dx'], self.df_tracks['dy']))
        self.df_tracks['angle'] = np.abs(np.mod(self.df_tracks['angle'] + 90, 180) - 90)
        return self.df_tracks


class TrackFitter:
    def __init__(self, df):
        self.df = df
        self.df_tracks = None

    @staticmethod
    def fit_single_track(x, y) -> list:
        x, y = np.array(x), np.array(y)
        linear = RANSACRegressor(
            min_samples=0.8,
            max_trials=10,
            residual_threshold=30.0,
        ).fit(y[:, None], x)
        pca = PCA(n_components=1).fit(np.vstack([y, x]).T[linear.inlier_mask_])
        return [len(x), *pca.mean_[::-1], *pca.components_[0][::-1]]

    @classmethod
    def fit_tracks(cls, df) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with indices ``('run', 'entry')``, columns ``'pos_x'`` and
            ``'pos_y'``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            func = lambda event: cls.fit_single_track(event['pos_x'], event['pos_y'])
            tracks = df.groupby(['run', 'entry']).apply(func)
        return pd.DataFrame(
            tracks.tolist(),
            columns=['multi', 'x', 'y', 'dx', 'dy'],
        )
    
    def get_tracks(self, from_cache=True) -> pd.DataFrame:
        df_tracks_cache = self.read_cache()
        df = self.df[['pos_x', 'pos_y']].reset_index(['subentry'], drop=True)
        target_indices = df.index.unique()

        if df_tracks_cache is None or not from_cache:
            self.df_tracks = self.fit_tracks(df).set_index(target_indices)
            return self.df_tracks

        cached_indices = df_tracks_cache.index
        new_indices = target_indices.difference(cached_indices)

        self.df_tracks = df_tracks_cache.loc[target_indices.intersection(cached_indices)]
        if len(self.df_tracks) == 0:
            self.df_tracks = None

        if len(df.loc[new_indices]) > 0:
            df_tracks_new = self.fit_tracks(df.loc[new_indices])
            df_tracks_new = df_tracks_new.set_index(new_indices)
            self.df_tracks = pd.concat([self.df_tracks, df_tracks_new])

        self.df_tracks = self.df_tracks.sort_index()
        return self.df_tracks
    
    @staticmethod
    def _cache_path(run):
        return Path(os.path.expandvars(CosmicRun.DATABASE_DIR)) / CosmicRun.CACHE_RELPATH.format(run=run)
    
    def update_cache_per_run(self, run, append=True):
        df_merge = self.df_tracks.loc[run]
        if append:
            df_old = self.read_cache_per_run(run)
            df_merge = pd.concat([df_merge, df_old])
            df_merge = df_merge.reset_index('entry').drop_duplicates('entry').set_index('entry')
        df_merge = df_merge.sort_index()

        path = self._cache_path(run)
        with sqlite3.connect(str(path)) as conn:
            df_merge.to_sql('tracks', con=conn, if_exists='replace')
    
    def update_cache(self, append=True):
        for run in self.df_tracks.index.get_level_values('run').unique():
            self.update_cache_per_run(run, append=append)
    
    def read_cache_per_run(self, run):
        path = self._cache_path(run)
        with sqlite3.connect(str(path)) as conn:
            if not conn.execute('''
                SELECT name FROM sqlite_master WHERE type="table" AND name="tracks";
            ''').fetchall():
                return None
            df = pd.read_sql('SELECT * FROM tracks;', con=conn)
        return df.set_index('entry', drop=True)
    
    def read_cache(self):
        df_result = None
        for run in self.df.index.get_level_values('run').unique():
            df_run = self.read_cache_per_run(run)
            if df_run is None:
                continue
            df_run.insert(0, 'run', run)
            df_run = df_run.set_index(['run'], append=True, drop=True)
            df_run = df_run.reorder_levels(['run', 'entry'])
            df_result = pd.concat([df_result, df_run])
        return df_result

cr = CosmicRun('B')
cr.read([4803, 4806, 4809, 4810, 4926, 4927])
cr.add_x_positions()
cr.add_y_positions()
cr.randomize_ADC()
cr.get_tracks()
cr.calculate_track_angles()

#%%
i = 25
df = cr.df.copy()
indices = df.reset_index('subentry').index.unique()
entry = df.loc[indices[i]]
track = cr.df_tracks.loc[indices[i]]
def track_line(x, track):
    return (x - track.x) * (track.dy / track.dx) + track.y

fig, ax = plt.subplots()
ax.scatter(entry.pos_x, entry.pos_y, s=3, color='blue')
x_plt = np.linspace(-120, 120, 10)
ax.plot(x_plt, track_line(x_plt, track), color='red')
ax.set_xlim(-120, 120)
ax.set_ylim(-120, 120)
ax.set_aspect('equal')
plt.show()

#%%
class UshapeRansacEstimator:
    AMBE_PATH = '$DATABASE_DIR/neutron_wall/ambe/compton_edges_3072-3073.csv'

    def __init__(self, bar):
        self.bar = bar
        self.ambe_fit = self._get_ambe_fit(bar)

    @classmethod
    def _get_ambe_fit(cls, bar, with_intercept=False):
        path = Path(os.path.expandvars(cls.AMBE_PATH))
        df = pd.read_csv(path).set_index('bar')
        par = df.loc[bar]
        if with_intercept:
            return lambda x: par[0] + par[1] * x + par[2] * x**2
        else:
            return lambda x: par[1] * x + par[2] * x**2
    
    def model(self, x, y_shift):
        return self.ambe_fit(x) + y_shift

    def fit(self, X, y):
        self.par, _ = curve_fit(self.model, X.flatten(), y)
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
        return dict(bar=self.bar)
    
    def set_params(self, **params):
        return UshapeRansacEstimator(self.bar)

#%%
path = '$DATABASE_DIR/neutron_wall/light_output_calibration/nwb_pulse_height_calibration.dat'
path = os.path.expandvars(path)
df_light = pd.read_csv(path, delim_whitespace=True, comment='#').set_index('bar')
def get_light_from_adc(adc, pos, a, b, c, d, e):
    result = 4.196 * adc / (a + b * pos + c * pos**2)
    result = d + result * e
    return result

#%%
class UshapeFitter:
    def __init__(self, bar, cosmic_run):
        self.bar = bar
        self.AB = cosmic_run.AB
        self.df = cosmic_run.df.query(' & '.join([
            f'NW{self.AB}_bar == {self.bar}',
            f'NW{self.AB}_total_L < 4000',
            f'NW{self.AB}_total_R < 4000',
            f'NW{self.AB}_total_L > 100',
            f'NW{self.AB}_total_R > 100',
        ]))
        self.df = self.df.reset_index('subentry')
        self.df_tracks = cosmic_run.df_tracks
    
    def get_angle_cut(self, angle, half_width=10.0):
        track_idx = self.df_tracks.query(f'abs(angle - {angle}) < {half_width}').index
        idx = self.df.index.intersection(track_idx)
        return self.df.loc[idx].set_index('subentry', append=True)
    
    def fit(self, angle, half_width=10.0):
        subdf = self.get_angle_cut(angle, half_width)
        subdf = subdf.query('abs(pos_x) < 60')
        fitter = RANSACRegressor(
            base_estimator=UshapeRansacEstimator(self.bar),
            min_samples=0.8,
            max_trials=100,
            residual_threshold=150,
        )
        fitter.fit(
            np.array(subdf.pos_x)[:, None],
            np.sqrt(subdf[f'NW{self.AB}_total_L'] * subdf[f'NW{self.AB}_total_R'])
        )
        return fitter.estimator_.par[0]

for bar in range(1, 25):
    u_fitter = UshapeFitter(bar, cr)
    x = [4.2, 11.96, 17.0536, 22.1550]
    y = [UshapeRansacEstimator._get_ambe_fit(bar, True)(0)]
    angles = [0, 44.4, 56.3]
    for angle in angles:
        y.append(u_fitter.fit(angle))

    fig, ax = plt.subplots()
    ax.scatter(x, y, s=3)
    x_plt = np.linspace(0, 4000, 100)
    light = get_light_from_adc(x_plt, 0, *df_light.loc[bar])
    plt.plot(light, x_plt, color='red')
    ax.set_title(bar)
    ax.set_xlim(-1, 80)
    ax.set_ylim(-100, 4100)
    ax.grid()
    plt.show()


#%%
bar = 7
df_bar = cr.df.query(f'NWB_bar == {bar} & NWB_total_L < 4000 & NWB_total_R < 4000 & pos_x < 120 & pos_x > -120')
df_bar = df_bar.reset_index('subentry')
# indices = df_bar.index.intersection(cr.df_tracks.query('abs(angle - 0) < 10').index)
# indices = df_bar.index.intersection(cr.df_tracks.query('abs(angle - 44.4) < 10').index)
indices = df_bar.index.intersection(cr.df_tracks.query('abs(angle - 56.3) < 10').index)
df_bar = df_bar.loc[indices].set_index('subentry', append=True)
df_bar

u_fitter = RANSACRegressor(
    base_estimator=UshapeRansacEstimator(bar),
    min_samples=0.8,
    max_trials=100,
    residual_threshold=150,
)
df_fit = df_bar.query('pos_x < 60 & pos_x > -60')
u_fitter.fit(
    np.array(df_fit.pos_x)[:, None],
    np.sqrt(df_fit.NWB_total_L * df_fit.NWB_total_R),
)
print(u_fitter.estimator_.par)

fig, ax = plt.subplots()
x_plt = np.linspace(-120, 120, 100)
ax.plot(
    x_plt,
    u_fitter.predict(x_plt[:, None]),
    color='red',
)
fh.plot_histo2d(
    ax.hist2d,
    df_bar.pos_x,
    np.sqrt(df_bar.NWB_total_L * df_bar.NWB_total_R),
    range=[[-120, 120], [0, 4000]],
    bins=[100, 100],
    cmap='viridis',
    norm=mpl.colors.LogNorm(vmin=1),
)
ax.set_title(f'NWB bar {bar}')
ax.set_xlabel(r'Position $x$ [cm]')
ax.set_ylabel(r'$\sqrt{A_L\cdot A_R} / 4.196$')
plt.show()

#%%
subdf = df_bar.query('pos_x > -10 & pos_x < 10')
fig, ax = plt.subplots()
fh.plot_histo1d(
    ax.hist,
    np.sqrt(subdf.NWB_total_L * subdf.NWB_total_R),
    range=[0, 2000],
    bins=50,
    histtype='step',
)
plt.show()