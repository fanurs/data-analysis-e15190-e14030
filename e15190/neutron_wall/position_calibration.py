import concurrent.futures
import inspect
import pathlib
import warnings
import sys

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, stats
from sklearn import neighbors
import uproot

from .. import PROJECT_DIR
from ..utilities import local_manager

DATABASE_DIR = pathlib.Path(PROJECT_DIR, 'database', 'neutron_wall', 'position_calibration')
CACHE_DIR = pathlib.Path(DATABASE_DIR, 'cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class NWBPositionCalibrator:
    def __init__(self, max_workers=12):
        self.AB = 'B'
        self.ab = self.AB.lower()
        self.decompression_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # read in the expected positions of Veto Wall shadows
        path = pathlib.Path(DATABASE_DIR, f'VW_shadows_on_NW{self.AB}.csv')
        vw_shadow = pd.read_csv(path)
        vw_shadow.set_index('vw_bar', drop=True, inplace=True)
        self.vw_shadow = {bar: vw_shadow.loc[bar][f'nw{self.ab}_x'] for bar in vw_shadow.index}

        # hyperparameters
        self.threshold_light_GM = 5.0 # MeVee
        self.nw_bars = list(range(1, 25))
        self.vw_bars = list(range(3, 22))
        self.nw_edges = [-93.2285, 99.8115] # cm
        self.nw_length = self.nw_edges[1] - self.nw_edges[0]

        # holders to be filled with data
        self.run = None
        self.df_vw = None
        self.df_nw = None
        self.rough_calib_params = None
        self.calib_params = None

    def read_run(self, run, use_cache=False, save_cache=True, raise_not_found=True):
        # check for existing cache
        cache_path = pathlib.Path(CACHE_DIR, f'run-{run:04d}.h5')
        if use_cache and cache_path.is_file():
            with pd.HDFStore(cache_path, 'r') as file:
                if f'nw{self.ab}' in file and 'vw' in file:
                    self.df_nw = file[f'nw{self.ab}']
                    self.df_vw = file['vw']
                    return True

        # prepare path
        self.run = run
        root_dir = local_manager.get_local_path('daniele_root_files_dir')
        filename = f'CalibratedData_{self.run:04d}.root'
        path = pathlib.Path(root_dir, filename).resolve()

        # specify branches of interest and their aliases
        vw_branches = {
            'VetoWall.fnumbar': 'bar',
        }
        nw_branches = {
            f'NW{self.AB}.fnumbar': 'bar',
            f'NW{self.AB}.fTimeLeft': 'time_L',
            f'NW{self.AB}.fTimeRight': 'time_R',
            f'NW{self.AB}.fGeoMeanSaturationCorrected': 'light_GM',
        }

        # read in branches into pandas.DataFrame
        try:
            kw = dict(
                library='pd',
                decompression_executor=self.decompression_executor,
                interpretation_executor=self.interpretation_executor,
            )
            self.df_vw = uproot.concatenate(f'{str(path)}:E15190', vw_branches.keys(), **kw)
            self.df_nw = uproot.concatenate(f'{str(path)}:E15190', nw_branches.keys(), **kw)
        except FileNotFoundError as error:
            if raise_not_found:
                raise FileNotFoundError(f'Fail to find "{str(path)}"') from None
            return False

        # clean up the dataframes before use
        self.df_vw.rename(columns=vw_branches, inplace=True)
        self.df_nw.rename(columns=nw_branches, inplace=True)
        self.df_nw['time_diff'] = self.df_nw.eval('time_L - time_R')
        self.df_nw = self.df_nw[['bar', 'time_diff', 'light_GM']]

        # save dataframes to HDF file
        if save_cache:
            with pd.HDFStore(cache_path, 'a') as file:
                file.append(f'nw{self.ab}', self.df_nw, append=False)
                file.append('vw', self.df_vw, append=False)

        return True

    def _rough_calibrate(self):
        # some HARD-CODED numbers to help estimating the edges (roughly)
        qvals_L = np.linspace(0.01, 0.05, 5)
        qvals_R = 1 - qvals_L
        qval_L, qval_R = 0.01, 0.99

        # estimate rough edges in time_diff (td) using quantiles and some simple linear fits
        self.rough_calib_params = dict()
        for nw_bar in self.nw_bars:
            df = self.df_nw.query('bar == @nw_bar & light_GM > @self.threshold_light_GM')

            quantiles = np.quantile(df['time_diff'], np.hstack([qvals_L, qvals_R]))
            quantiles_L = quantiles[:len(qvals_L)]
            quantiles_R = quantiles[len(qvals_L):]

            res_L = stats.linregress(qvals_L, quantiles_L)
            res_R = stats.linregress(qvals_R, quantiles_R)
            td_L = res_L.intercept + res_L.slope * qval_L
            td_R = res_R.intercept + res_R.slope * qval_R
            self.rough_calib_params[nw_bar] = np.array([
                np.mean(self.nw_edges) - self.nw_length * (0.5 + td_L / (td_R - td_L)),
                self.nw_length / (td_R - td_L),
            ])

        # apply calibration to finalize the rough positions
        params = np.vstack(self.df_nw['bar'].map(self.rough_calib_params))
        self.df_nw['rough_pos'] = params[:, 0] + params[:, 1] * self.df_nw['time_diff']

    def calibrate(self, verbose=False):
        self._rough_calibrate()

        # cast the VW shadows, i.e. find the intersecting entries between NW and VW
        vw_bar_entries = self.df_vw.groupby('bar').groups
        nw_entries_set = set(self.df_nw.index.get_level_values(0))
        mask_entries = {
            vw_bar: list(set(np.vstack(vw_entries)[:, 0]).intersection(nw_entries_set))
            for vw_bar, vw_entries in vw_bar_entries.items()
        }

        # apply basic cuts on NW dataframe; more cuts will be made, on top of these, later
        df_cut = self.df_nw[
            (np.abs(self.df_nw['rough_pos']) < 150.0) &
            (self.df_nw['light_GM'] > self.threshold_light_GM)
        ]

        # prepare instances that will help peak identification later
        kde = neighbors.KernelDensity(bandwidth=1.0, rtol=1e-4, kernel='tophat')
        kde_fmt = lambda x: np.array([x]).transpose()
        kde_func = lambda x, kde=kde: np.exp(kde.score_samples(kde_fmt(x)))
        gaus = lambda x, amplt, x0, width: amplt * np.exp(-0.5 * ((x - x0) / width)**2)

        # to collect calibration parameters for each NW bar
        self.calib_params = dict()
        for nw_i, nw_bar in enumerate(self.nw_bars):
            if verbose:
                print(
                    f'\rCalibrating NW{self.AB}-bar{nw_bar:02d} ({nw_i+1}/{len(self.nw_bars)})',
                    end='', flush=True,
                )

            df_nw = df_cut.query('bar == @nw_bar')

            # start collecting all calibration points on this NW bar
            calib_x = dict()
            for vw_bar in self.vw_bars:
                # case the VW shadow on this NW bar
                df = df_nw.loc[mask_entries[vw_bar]]

                # estimate the VW shadow peak position
                est_peak_x = df['rough_pos'].median()

                # get the distribution (in KDE) around the estimated VW shadow
                x_fit = np.linspace(est_peak_x - 15, est_peak_x + 15, 100)
                kde.fit(kde_fmt(df['rough_pos']))

                try:
                    # fine tune the shadow peak position and save it for calibration
                    p, _ = optimize.curve_fit(
                        gaus, x_fit, kde_func(x_fit),
                        p0=[kde_func([est_peak_x])[0], est_peak_x, 2.0],
                    )
                    calib_x[vw_bar] = p[1]
                except:
                    print()
                    warnings.warn(
                        inspect.cleandoc(f'''
                        Fail to fit VW-bar{vw_bar:02d} shadow on NW{self.AB}-bar{nw_bar:02d}.
                        Using simple median of the shadow as calibration point.
                        '''),
                        Warning, stacklevel=2,
                    )
                    calib_x[vw_bar] = est_peak_x

                if abs(calib_x[vw_bar] - est_peak_x) > 10:
                    print()
                    warnings.warn(
                        inspect.cleandoc(f'''
                        VW-bar{vw_bar:02d} shadow on NW{self.AB}-bar{nw_bar:02d} given by Gaussian fit lies far away (> 10 cm) from the initial estimate using simple median.
                        Resorting to simple median as calibration point.
                        '''),
                        Warning, stacklevel=2,
                    )
                    calib_x[vw_bar] = est_peak_x
            
            # done collecting all calibration points for this NW bar, compute calibration parameters
            common_vw_bars = sorted(set(calib_x).intersection(set(self.vw_shadow)))
            x = np.array([calib_x[vw_bar] for vw_bar in common_vw_bars])
            y = np.array([self.vw_shadow[vw_bar] for vw_bar in common_vw_bars])
            res = stats.linregress(x, y)
            self.calib_params[nw_bar] = np.array([
                self.rough_calib_params[nw_bar][0] + self.rough_calib_params[nw_bar][1] * res.intercept,
                self.rough_calib_params[nw_bar][1] * res.slope,
            ])

        # apply final position calibration
        params = np.vstack(self.df_nw['bar'].map(self.calib_params))
        self.df_nw['pos'] = params[:, 0] + params[:, 1] * self.df_nw['time_diff']