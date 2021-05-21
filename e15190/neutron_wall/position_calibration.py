import concurrent.futures
import inspect
import pathlib
import warnings

import matplotlib.pyplot as plt
import numexpr
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn import neighbors
import uproot
import time

from .. import PROJECT_DIR
from ..utilities import local_manager, tables, timer

DATABASE_DIR = pathlib.Path(PROJECT_DIR, 'database', 'neutron_wall', 'position_calibration')
CALIB_PARAMS_DIR = pathlib.Path(DATABASE_DIR, 'calib_params')
CALIB_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
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
        self.gaus_params = None

    def read_run(self, run, use_cache=False, save_cache=True, raise_not_found=True):
        self.run = run

        # check for existing cache
        cache_path = pathlib.Path(CACHE_DIR, f'run-{run:04d}.h5')
        if use_cache and cache_path.is_file():
            with pd.HDFStore(cache_path, 'r') as file:
                if f'nw{self.ab}' in file and 'vw' in file:
                    self.df_nw = file[f'nw{self.ab}']
                    self.df_vw = file['vw']
                    return True

        # prepare path
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
        df_cut = self.df_nw.query('light_GM > @self.threshold_light_GM')
        bar_grp = df_cut.groupby('bar').groups
        for nw_bar in self.nw_bars:
            df = df_cut.loc[bar_grp[nw_bar]]

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

    def calibrate(self, verbose=False, save_params=True):
        self._rough_calibrate()

        # cast the VW shadows, i.e. find the intersecting entries between NW and VW
        vw_bar_entries = self.df_vw.groupby('bar').groups
        nw_entries = self.df_nw.index.get_level_values(0)
        mask_entries = {
            vw_bar: np.intersect1d(np.vstack(vw_entries)[:, 0], nw_entries)
            for vw_bar, vw_entries in vw_bar_entries.items()
        }

        # apply basic cuts on NW dataframe; more cuts will be made, on top of these, later
        df_cut = self.df_nw[
            (np.abs(self.df_nw['rough_pos']) < 150.0) &
            (self.df_nw['light_GM'] > self.threshold_light_GM)
        ]

        # prepare instances that will help peak identification later
        kde = neighbors.KernelDensity(bandwidth=1.0, rtol=1e-4)
        kde_fmt = lambda x: np.array([x]).transpose()
        kde_func = lambda x, kde=kde: np.exp(kde.score_samples(kde_fmt(x)))
        self.gaus = lambda x, amplt, x0, width: amplt * np.exp(-0.5 * ((x - x0) / width)**2)

        # to collect calibration parameters for each NW bar
        self.calib_params = dict()
        self.gaus_params = dict()
        for nw_i, nw_bar in enumerate(self.nw_bars):
            if verbose:
                msg = f'\rCalibrating NW{self.AB}-bar{nw_bar:02d} ({nw_i+1}/{len(self.nw_bars)})'
                print(msg, end='', flush=True)

            df_nw = df_cut.query('bar == @nw_bar')

            # start collecting all calibration points on this NW bar
            calib_x = dict()
            for vw_bar in self.vw_bars:
                # cast the VW shadow on this NW bar
                df = df_nw.loc[mask_entries[vw_bar]]

                # estimate the VW shadow peak position
                count, x = np.histogram(df['rough_pos'], range=[-150, 150], bins=100)
                x = 0.5 * (x[1:] + x[:-1])
                est_peak_x = x[np.argmax(count)]

                gaus_fit_ok = True
                try:
                    fit_halfwidth = 20.0

                    # first fit around the estimated shadow peak position
                    kde.fit(kde_fmt(df.query('abs(rough_pos - @est_peak_x) < @fit_halfwidth')['rough_pos']))
                    x_fit = np.linspace(est_peak_x - fit_halfwidth, est_peak_x + fit_halfwidth, 20)
                    gaus_param, _ = optimize.curve_fit(
                        self.gaus, x_fit, kde_func(x_fit),
                        p0=[kde_func([est_peak_x])[0], est_peak_x, 10.0],
                    )

                    # second fit to further converge toward the shadow peak position
                    kde.fit(kde_fmt(df.query('abs(rough_pos - @gaus_param[1]) < @fit_halfwidth')['rough_pos']))
                    x_fit = np.linspace(gaus_param[1] - fit_halfwidth, gaus_param[1] + fit_halfwidth, 40)
                    gaus_param, _ = optimize.curve_fit(self.gaus, x_fit, kde_func(x_fit), p0=gaus_param)
                except:
                    msg = '\n' + inspect.cleandoc(
                        f'''
                        Fail to fit VW-bar{vw_bar:02d} shadow on NW{self.AB}-bar{nw_bar:02d}.
                        Using simple statistical mode of the shadow as calibration point.
                        '''),
                    print()
                    warnings.warn(msg, Warning, stacklevel=2)
                    gaus_fit_ok = False

                max_allowed_offset = 10.0 # cm
                if gaus_fit_ok and abs(gaus_param[1] - est_peak_x) > max_allowed_offset:
                    msg = '\n' + inspect.cleandoc(
                        f'''
                        VW-bar{vw_bar:02d} shadow on NW{self.AB}-bar{nw_bar:02d}:
                        Gaussian fit is far (> {max_allowed_offset:.0f} cm) from initial estimation.
                        Resorting to using simple statistical mode as calibration point.
                        ''')
                    print()
                    warnings.warn(msg, Warning, stacklevel=2)
                    gaus_fit_ok = False

                # save Gaussian fits
                if gaus_fit_ok:
                    self.gaus_params[(nw_bar, vw_bar)] = gaus_param
                else:
                    # create a very sharp Gaussian so that when plotted it is clear that estimated peak was used
                    self.gaus_params[(nw_bar, vw_bar)] = [5.0 * kde_func([est_peak_x])[0], est_peak_x, 0.2]
                calib_x[vw_bar] = self.gaus_params[(nw_bar, vw_bar)][1]

            # done collecting all calibration points for this NW bar, compute calibration parameters
            common_vw_bars = sorted(set(calib_x).intersection(set(self.vw_shadow)))
            x = np.array([calib_x[vw_bar] for vw_bar in common_vw_bars])
            y = np.array([self.vw_shadow[vw_bar] for vw_bar in common_vw_bars])
            res = stats.linregress(x, y)
            self.calib_params[nw_bar] = np.array([
                self.rough_calib_params[nw_bar][0] * res.slope + res.intercept,
                self.rough_calib_params[nw_bar][1] * res.slope,
            ])

        # apply final position calibration
        params = np.vstack(self.df_nw['bar'].map(self.calib_params))
        self.df_nw['pos'] = params[:, 0] + params[:, 1] * self.df_nw['time_diff']

        # save parameters
        if save_params:
            self.save_parameters()

    def save_parameters(self):
        # turn calibration parameters into pandas dataframe
        df = pd.DataFrame(self.calib_params).transpose()
        df.columns = ['p0', 'p1']
        df.index.name = f'nw{self.ab}-bar'

        # write to file
        path = pathlib.Path(CALIB_PARAMS_DIR, f'run-{self.run:04d}-nw{self.ab}.dat')
        tables.to_fwf(df, path, drop_index=False)
