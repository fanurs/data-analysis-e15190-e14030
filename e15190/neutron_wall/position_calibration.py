import concurrent.futures
import copy
import heapq
import inspect
import json
import pathlib
import warnings
import sys

import matplotlib as mpl
mpl_default_backend = mpl.get_backend()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from scipy import optimize, stats
from shapely.geometry import Polygon
from sklearn import neighbors
import uproot

from e15190 import PROJECT_DIR
from e15190.neutron_wall import geometry as nw_geom
from e15190.veto_wall import geometry as vw_geom
from e15190.runlog import query
from e15190.utilities import geometry as geom
from e15190.utilities import fast_histogram as fh
from e15190.utilities import local_manager, tables
from e15190.utilities import ray_triangle_intersection as rti
from e15190.utilities import styles
styles.set_matplotlib_style(mpl)

DATABASE_DIR = PROJECT_DIR / 'database/neutron_wall/position_calibration'
CACHE_DIR = DATABASE_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CALIB_PARAMS_DIR = DATABASE_DIR / 'calib_params'
CALIB_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR = DATABASE_DIR / 'gallery'
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

class NWBPositionCalibrator:
    def __init__(
        self,
        max_workers=8,
        verbose=False,
        stdout_path=None,
        recalculate_vw_shadows=False,
    ):
        mpl.use('Agg')
        self.verbose = verbose
        self.stdout_path = stdout_path
        self.AB = 'B'
        self.ab = self.AB.lower()
        self.decompression_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.cmap = copy.copy(plt.cm.viridis_r)
        self.cmap.set_under('white')

        # read in the expected positions of Veto Wall shadows
        path = DATABASE_DIR / f'VW_shadows_on_NW{self.AB}.csv'
        if recalculate_vw_shadows:
            df = self.get_veto_wall_shadows()
            tables.to_fwf(df, path)
        self.vw_shadow = pd.read_csv(path, delim_whitespace=True)
        self.vw_shadow.set_index(['nw_bar', 'vw_bar'], inplace=True, drop=False)

        # hyperparameters
        self.threshold_light_GM = 5.0 # MeVee
        self.nw_bars = list(range(1, 24 + 1))
        self.vw_bars = list(range(4, 21 + 1)) # 3 and 22 have partial shadows, hence dropped
        self.nw_edges = [-93.2285, 99.8115] # cm; for rough calibration only
        self.nw_length = self.nw_edges[1] - self.nw_edges[0]

        # holders to be filled with data
        self.run = None
        self.df_vw = None
        self.df_nw = None
        self.rough_calib_params = None
        self.calib_params = None
        self._rough_to_pos_calib_params = None
        self.gaus_params = None
        self.outlier_vw_bars = None
    
    def get_veto_wall_shadows(self):
        df = []
        nwall = nw_geom.Wall(self.AB, contain_pyrex=False)
        vwall = vw_geom.Wall()

        vw_polygons = {
            b: Polygon(bar.get_theta_phi_alphashape())
            for b, bar in vwall.bars.items()
        }
        for nb, nw_bar in nwall.bars.items():
            nw_polygon = Polygon(nw_bar.get_theta_phi_alphashape())
            for vb, vw_bar in vwall.bars.items():
                vw_polygon = vw_polygons[vb]
                centroid = nw_polygon.intersection(vw_polygon).centroid
                if centroid.is_empty:
                    continue

                rays = np.tile([1.0, centroid.x, centroid.y], (1, 1))
                rays = geom.spherical_to_cartesian(rays)
                nw_bar.construct_plotly_mesh3d()
                triangles = nw_bar.triangle_mesh.get_triangles()
                origin = np.array([0.0, 0.0, 0.0])
                intersections = rti.moller_trumbore(origin, rays, triangles)
                hit = nw_bar.get_hit_positions(
                    hit_t=0.5,
                    simulation_result=dict(
                        origin=origin,
                        intersections=intersections,
                    )
                )[0]

                df.append([nb, vb, hit[0]])
        df = pd.DataFrame(df, columns=['nw_bar', 'vw_bar', f'nw{self.ab}_x'])

        df = df.query(f'vw_bar != {df.vw_bar.min()} & vw_bar != {df.vw_bar.max()}')
        df.reset_index(drop=True, inplace=True)
        return df
    
    def draw_expected_veto_wall_shadows(self, ax):
        nwall = nw_geom.Wall(self.AB, contain_pyrex=False)
        nw_func = lambda x, a, b: a / x + b # an empirical functional for NW bar in spherical coordinates 
        for b, bar in nwall.bars.items():
            if b == 0: continue # this bar is not used in the experiment

            # get the alphashape of the bar in lab (theta, phi)
            ashape = np.degrees(bar.get_theta_phi_alphashape(delta=20.0))

            # fit a line that empirically represents the longest axis of the bar
            hits = bar.randomize_from_local_x(
                np.linspace(-0.5 * bar.length, 0.5 * bar.length, 100),
                local_ynorm=0, local_znorm=0,
            )
            hits = geom.cartesian_to_spherical(hits) # turns into a curve in spherical coordinates
            par, _ = optimize.curve_fit(nw_func, np.degrees(hits[:, 1]), np.degrees(hits[:, 2]))
            func = lambda x, par=par: nw_func(x, *par)

            # draw and annotate
            kw_patch = dict(fill=True, alpha=0.5, edgecolor='navy', linewidth=0.5)
            kw_line = dict(linestyle='dashed', linewidth=0.5)
            kw_annot = dict(va='center', ha='center', fontsize=8, zorder=100)
            x_plt = np.linspace(20, 60, 100) # degree
            if b % 2 == 0:
                ax.add_patch(mpl.patches.Polygon(ashape, facecolor='cyan', **kw_patch))
                ax.plot(x_plt, func(x_plt), color='cyan', **kw_line)
                ax.annotate(str(b), xy=(24.0, func(24.0)), color='blue', **kw_annot)
            else:
                ax.add_patch(mpl.patches.Polygon(ashape, facecolor='pink', **kw_patch))
                ax.plot(x_plt, func(x_plt), color='pink', **kw_line)
                ax.annotate(str(b), xy=(22.0, func(22.0)), color='red', **kw_annot)

        vwall = vw_geom.Wall()
        for b, bar in vwall.bars.items():
            ashape = np.degrees(bar.get_theta_phi_alphashape(delta=5.0))

            # draw and annotate
            kw_bar = dict(linewidth=0.5, zorder=20)
            kw_annot = dict(ha='left', fontsize=8, zorder=100)
            if b % 2 == 0:
                ax.plot(ashape[:, 0], ashape[:, 1], color='gray', **kw_bar)
                ax.annotate(
                    str(b).rjust(2),
                    xy=(ashape[np.argmax(ashape[:, 1]), 0] + 0.2, ashape[:, 1].max()),
                    va='bottom',
                    color='black',
                    **kw_annot,
                )
            else:
                ax.plot(ashape[:, 0], ashape[:, 1], color='lightgreen', **kw_bar)
                ax.annotate(
                    str(b).rjust(2),
                    xy=(ashape[np.argmax(ashape[:, 1]), 0] + 0.2, ashape[:, 1].min()),
                    va='top',
                    color='green',
                    **kw_annot,
                )
        
        # design
        ax.set_xlim(20, 60)
        ax.set_ylim(-40, 40)
        ax.set_xlabel(r'$\theta$ (deg)')
        ax.set_ylabel(r'$\phi$ (deg)')

    def read_run(self, run, verbose=None, use_cache=True, save_cache=True, raise_not_found=True):
        if verbose is None:
            verbose = self.verbose

        self.run = run
        if self.stdout_path is not None:
            sys.stdout = open(self.stdout_path, 'w')
            sys.stderr = sys.stdout
        if verbose:
            print(f'Reading run-{run:04d}...', flush=True)

        # check for existing cache
        cache_path = pathlib.Path(CACHE_DIR, f'run-{run:04d}.h5')
        if use_cache and cache_path.is_file():
            if verbose:
                print(f'Reading from cache "{str(cache_path)}"...', flush=True)
            with pd.HDFStore(cache_path, 'r') as file:
                if f'nw{self.ab}' in file and 'vw' in file:
                    self.df_nw = file[f'nw{self.ab}']
                    self.df_vw = file['vw']
                    if verbose:
                        print('Done reading.', flush=True)
                    return True

        # prepare path
        root_dir = local_manager.get_local_path('daniele_root_files_dir')
        filename = f'CalibratedData_{self.run:04d}.root'
        path = pathlib.Path(root_dir, filename).resolve()
        if verbose:
            print(f'Reading from "{str(path)}"...', flush=True)

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
        if verbose:
            print('Done reading.', flush=True)

        # save dataframes to HDF file
        if save_cache:
            with pd.HDFStore(cache_path, 'a') as file:
                file.append(f'nw{self.ab}', self.df_nw, append=False)
                file.append('vw', self.df_vw, append=False)
            print(f'Saved data to cache file "{str(cache_path)}"...', flush=True)

        return True

    def _rough_calibrate(self):
        """Perform a rough position calibration using simple quantiles on the hit distribution of each NW bar.

        After this procedure, instead of raw time difference (time left - time right),
        we should have a rough position calibrated good to a precision of ~15 cm.
        The result is saved to `self.df_nw` as a new column named `rough_pos`.
        This makes it easier to further calibrate the positions using VW shadows.
        """
        # some HARD-CODED numbers to help estimating the edges (roughly)
        qvals_L = np.linspace(0.01, 0.05, 5)
        qvals_R = 1 - qvals_L
        qval_L, qval_R = 0.01, 0.99

        # estimate rough edges in time_diff (td) using quantiles and some simple linear fits
        self.rough_calib_params = dict()
        df_cut = self.df_nw.query('light_GM > @self.threshold_light_GM')
        for nw_bar in self.nw_bars:
            df = df_cut.query('bar == @nw_bar')
            if len(df) < 1e3:
                msg = '\n' + inspect.cleandoc(
                    f'''
                    Do not have enough statistics to do position calibration.
                    ''')
                print()
                warnings.warn(msg, Warning, stacklevel=2)
                return False

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
        return True

    def calibrate(self, verbose=None, save_params=True):
        if verbose is None:
            verbose = self.verbose

        success = self._rough_calibrate()
        if not success:
            return False

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
        self.gaus = lambda x, amplt, x0, sigma: amplt * np.exp(-0.5 * ((x - x0) / sigma)**2)

        # to collect calibration parameters for each NW bar
        self.calib_params = dict()
        self._rough_to_pos_calib_params = dict()
        self.gaus_params = dict()
        self.outlier_vw_bars = dict()
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
                        ''')
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

            # done collecting all calibration points for this NW bar
            common_vw_bars = sorted(set(calib_x).intersection(set(self.vw_shadow.vw_bar)))
            x = {vw_bar: calib_x[vw_bar] for vw_bar in common_vw_bars} # rough_pos from Gaussian fits
            y = { # expected position from the simulation
                vw_bar: self.vw_shadow.loc[(nw_i, vw_bar)][f'nw{self.ab}_x']
                for vw_bar in common_vw_bars
            }

            # first simple filter to remove calibration points that have very different rough_pos (> 5.0 cm)
            bad_vw_bars1 = [vw_bar for vw_bar in common_vw_bars if abs(x[vw_bar] - y[vw_bar]) > 5.0]
            x = {vw_bar: x[vw_bar] for vw_bar in x if vw_bar not in bad_vw_bars1}
            y = {vw_bar: y[vw_bar] for vw_bar in x if vw_bar not in bad_vw_bars1}
            res1 = stats.linregress(list(x.values()), list(y.values()))

            # second fit after removing the two biggest outliers
            n_outliers = 2
            abs_residuals = {vw_bar: abs(_x - _y) for (vw_bar, _x), _y in zip(x.items(), y.values())}
            bad_vw_bars2 = heapq.nlargest(n_outliers, abs_residuals, key=abs_residuals.__getitem__)
            x = {vw_bar: x[vw_bar] for vw_bar in x if vw_bar not in bad_vw_bars2}
            y = {vw_bar: y[vw_bar] for vw_bar in x if vw_bar not in bad_vw_bars2}
            res2 = stats.linregress(list(x.values()), list(y.values()))

            # save all parameters to memory
            self.outlier_vw_bars[nw_bar] = sorted(np.union1d(bad_vw_bars1, bad_vw_bars2))
            self._rough_to_pos_calib_params[nw_bar] = [res2.intercept, res2.slope]
            self.calib_params[nw_bar] = np.array([
                self.rough_calib_params[nw_bar][0] * res2.slope + res2.intercept,
                self.rough_calib_params[nw_bar][1] * res2.slope,
            ])
        if verbose:
            print()

        # apply final position calibration
        params = np.vstack(self.df_nw['bar'].map(self.calib_params))
        self.df_nw['pos'] = params[:, 0] + params[:, 1] * self.df_nw['time_diff']

        # save parameters
        if save_params:
            self.save_parameters()

        return True

    def save_parameters(self, verbose=None):
        if verbose is None:
            verbose = self.verbose

        # turn calibration parameters into pandas dataframe
        df = pd.DataFrame(self.calib_params).transpose()
        df.columns = ['p0', 'p1']
        df.index.name = f'nw{self.ab}-bar'

        # write to file
        path = pathlib.Path(CALIB_PARAMS_DIR, f'run-{self.run:04d}-nw{self.ab}.dat')
        tables.to_fwf(df, path, drop_index=False)
        if verbose:
            print(f'Done saving calibration parameters to "{str(path)}"...', flush=True)

    def save_to_gallery(self, verbose=None, show_plot=False, save_plot=True):
        if verbose is None:
            verbose = self.verbose

        # customize plot style
        self.cmap = copy.copy(plt.cm.viridis_r)
        self.cmap.set_under('white')
        mpl_custom = {
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'figure.dpi': 500,
            'figure.facecolor': 'white',
            'xtick.top': True,
            'xtick.direction': 'in',
            'xtick.minor.visible': True,
            'ytick.right': True,
            'ytick.direction': 'in',
            'ytick.minor.visible': True,
        }
        for key, val in mpl_custom.items():
            mpl.rcParams[key] = val

        # start plotting
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9), constrained_layout=True)

        rc = (0, 0)
        if verbose:
            print('Plotting 2D hit pattern (Full)...', flush=True)
        self._draw_hit_pattern2d(ax[rc], self.df_nw)

        rc = (0, 1)
        if verbose:
            print('Plotting 2D hit pattern (Even VW bars)...', flush=True)
        nw_entries = self.df_nw.index.get_level_values(0)
        vw_entries = self.df_vw.query('bar % 2 == 0').index.get_level_values(0)
        df = self.df_nw.loc[np.intersect1d(vw_entries, nw_entries)]
        gaus_params_even = {
            (nw_bar, vw_bar): gparam
            for (nw_bar, vw_bar), gparam in self.gaus_params.items()
            if vw_bar % 2 == 0
        }
        self._draw_hit_pattern2d(ax[rc], df, gaus_params_even)

        rc = (0, 2)
        if verbose:
            print('Plotting 2D hit pattern (Odd VW bars)...', flush=True)
        vw_entries = self.df_vw.query('bar % 2 == 1').index.get_level_values(0)
        df = self.df_nw.loc[np.intersect1d(vw_entries, nw_entries)]
        gaus_params_odd = {
            (nw_bar, vw_bar): gparam
            for (nw_bar, vw_bar), gparam in self.gaus_params.items()
            if vw_bar % 2 == 1
        }
        self._draw_hit_pattern2d(ax[rc], df, gaus_params_odd)

        rc = (1, 0)
        if verbose:
            print('Plotting position (linear) fit residuals...', flush=True)
        self._draw_residuals(ax[rc])

        rc = (1, 1)
        if verbose:
            print('Plotting Gaussian fits (Even bars)...', flush=True)
        self._draw_gaus_fits(ax[rc], gaus_params_even)

        rc = (1, 2)
        if verbose:
            print('Plotting Gaussian fits (Odd bars)...', flush=True)
        self._draw_gaus_fits(ax[rc], gaus_params_odd)

        # set common attributes across subplots
        for axe in ax.flatten():
            axe.set_xlim(-150, 150)
            axe.set_ylim(-0.5, 25.5)
            axe.set_ylabel(f'NW{self.AB} bar')
            axe.set_xlabel(f'run-{self.run:04d}: NW{self.AB} calibrated position (cm)')
        ax[1, 0].set_xlabel(f'run-{self.run:04d}: Expected positions of VW shadows on NW{self.AB} (cm)')
        fig.align_labels()

        # update canvas
        plt.draw()
        if verbose:
            print('Done plotting.', flush=True)

        # save figure as image file
        path = GALLERY_DIR / f'run-{self.run:04d}.png'
        if save_plot:
            fig.savefig(path, dpi=500, bbox_inches='tight')
            if verbose:
                print(f'Saved figure to "{str(path)}".', flush=True)

        # show or close plot
        if show_plot:
            plt.show()
        else:
            plt.close()

    def _draw_hit_pattern2d(self, ax, df, gaus_params=None):
        ax.hist2d(
            df['pos'], df['bar'],
            range=[[-150, 150], [-0.5, 25.5]], bins=[150, 26],
            cmap=self.cmap, vmin=1,
        )
        ax.grid(axis='y', color='darkgray', linestyle='dashed')

        if gaus_params is not None:
            # mark the expected shadow positions (from Inventor or laser measurement)
            color = 'dimgray'
            vw_bars = sorted(set(np.vstack(list(gaus_params.keys()))[:, 1]))
            axt = ax.twiny()
            xticks = []
            Polynomial = np.polynomial.Polynomial
            for vw_bar in vw_bars:
                subdf = self.vw_shadow.query(f'vw_bar == {vw_bar}')
                line = Polynomial.fit(subdf['nw_bar'], subdf[f'nw{self.ab}_x'], 1)
                y_plt = np.linspace(-0.5, 25.5, 3)
                ax.plot(line(y_plt), y_plt, color=color, linewidth=0.8, linestyle='dashed')
                xticks.append(line(25.5))
            axt.set_xlim(-150, 150)
            axt.set_xticks(xticks)
            axt.set_xticklabels(vw_bars)
            axt.set_xticks([], minor=True)
            axt.tick_params(axis='x', which='both', colors=color)
            axt.spines['top'].set_color(color)

            # mark the shadow peaks identified by Gaussian fits
            data = []
            for (nw_bar, vw_bar), gparam in gaus_params.items():
                intercept, slope = self._rough_to_pos_calib_params[nw_bar]
                pos = intercept + slope * gparam[1]
                sigma = slope * gparam[2]
                data.append([nw_bar, pos, sigma])
            data = pd.DataFrame(data, columns=['nw_bar', 'pos', 'sigma'])
            ax.hlines(
                data['nw_bar'], data['pos'] - data['sigma'], data['pos'] + data['sigma'],
                color='orangered', linewidth=1.5, zorder=100,
            )
            ax.vlines(
                data['pos'], data['nw_bar'] - 0.25, data['nw_bar'] + 0.25,
                color='darkred', linewidth=0.5, zorder=101,
            )

    def _draw_residuals(self, ax):
        colors = {
            'bright': {0: 'darkorange', 1: 'limegreen'},
            'dark': {0: 'darkgoldenrod', 1: 'darkgreen'},
        }
        outlier_color = 'dimgray'
        connect_lcolor = 'navy'
        nw_grid_color = 'dimgray'
        vw_grid_color = 'gray'
        residual_scalar = 0.1

        for nw_bar in self.nw_bars:
            ctheme = 'bright' if nw_bar % 2 else 'dark'
            outliers = self.outlier_vw_bars[nw_bar]

            data = []
            for vw_bar in self.vw_bars:
                gparam = self.gaus_params[(nw_bar, vw_bar)] # in rough pos
                intercept, slope = self._rough_to_pos_calib_params[nw_bar]
                vw_pos = self.vw_shadow.loc[(nw_bar, vw_bar)][f'nw{self.ab}_x']
                residual = vw_pos - (intercept + slope * gparam[1])
                scaled_residual = residual_scalar * residual
                data.append([vw_bar, vw_pos, scaled_residual])

                # annotate the residual of each point
                annotation = f'{residual:.1f}' if residual < 10.0 else f'{residual:.0f}'
                ax.annotate(
                    annotation, (vw_pos, scaled_residual + nw_bar - 0.4),
                    fontsize=3, ha='center', va='center',
                    color=outlier_color if vw_bar in outliers else colors[ctheme][vw_bar % 2],
                    zorder=100,
                )
            data = pd.DataFrame(data, columns=['vw_bar', 'vw_pos', 'scaled_residual'])

            # draw a simple line joining all the good and outliers points to guide eyes
            ax.plot(
                data['vw_pos'], data['scaled_residual'] + nw_bar,
                linewidth=0.4, color=connect_lcolor,
                zorder=50,
            )

            # draw the scatter points with alternating theme of colors
            for r in range(2):
                color = colors[ctheme][r]

                # draw the good points (not outliers)
                subdata = data.query('vw_bar % 2 == @r & vw_bar not in @outliers')
                ax.scatter(
                    subdata['vw_pos'], subdata['scaled_residual'] + nw_bar,
                    s=4, marker='o', linewidth=0.0, color=color,
                    zorder=100,
                )

                # draw the outlier points
                subdata = data.query('vw_bar % 2 == @r & vw_bar in @outliers')
                ax.scatter(
                    subdata['vw_pos'], subdata['scaled_residual'] + nw_bar,
                    s=3, marker='X', linewidth=0.1, edgecolor=color, color=outlier_color,
                    zorder=100,
                )
        
        # nw grid lines (horizontal)
        for nw_bar in self.nw_bars:
            ax.axhline(nw_bar, linewidth=0.3, linestyle='dashed', color=nw_grid_color)

        # vw grid lines (vertical)
        Polynomial = np.polynomial.Polynomial
        xticks = dict()
        for vw_bar in self.vw_bars:
            subdf = self.vw_shadow.query(f'vw_bar == {vw_bar}')
            if len(subdf) == 0: continue
            line = Polynomial.fit(subdf['nw_bar'], subdf[f'nw{self.ab}_x'], 1)
            y_plt = np.linspace(-0.5, 25.5, 3)
            ax.plot(line(y_plt), y_plt, color=vw_grid_color, linewidth=0.3, linestyle='dashed')
            xticks[vw_bar] = line(25.5)
        
        # vw ticks on mod 5 bars
        axt = ax.twiny()
        axt.set_xticks([xt for b, xt in xticks.items() if b % 5 == 0])
        axt.set_xticklabels([str(b) for b in xticks if b % 5 == 0])
        axt.set_xticks(list(xticks.values()), minor=True)
        axt.tick_params(axis='x', which='both', colors=vw_grid_color)
        axt.spines['top'].set_color(vw_grid_color)
        axt.set_xlim(-150, 150)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-0.5, 25.5)
    
    def _draw_gaus_fits(self, ax, gaus_params):
        colors = {
            'bright': {
                'hist': {0: 'orangered', 1: 'skyblue'},
                'gaus': {0: 'hotpink', 1: 'cyan'},
            },
            'dark': {
                'hist': {0: 'darkred', 1: 'darkblue'},
                'gaus': {0: 'lightcoral', 1: 'dodgerblue'},
            },
        }
        outlier_color = 'dimgray'
        nw_grid_color = 'dimgray'
        vw_grid_color = 'gray'

        nw_entries = sorted(set(self.df_nw.index.get_level_values(0)))
        vw_bars = sorted(set(np.vstack(list(gaus_params.keys()))[:, 1]))
        for nw_bar in sorted(self.nw_bars, reverse=True):
            df_nw = self.df_nw.query('bar == @nw_bar')
            ctheme = 'bright' if nw_bar % 2 else 'dark'
            outliers = self.outlier_vw_bars[nw_bar]

            for vw_i, vw_bar in enumerate(vw_bars):
                # filter entries
                vw_entries = self.df_vw.query('bar == @vw_bar').index.get_level_values(0)
                entries = np.intersect1d(vw_entries, nw_entries)
                df = df_nw.loc[entries][['rough_pos', 'pos']]

                # retrieve Gaussian fit parameters
                gparam = gaus_params[(nw_bar, vw_bar)]
                intercept, slope = self._rough_to_pos_calib_params[nw_bar]
                gparam[1] = intercept + slope * gparam[1]
                gparam[2] = slope * gparam[2]
                fit_range = [gparam[1] - 10, gparam[1] + 10]
                df = df.query('pos > @fit_range[0] & pos < @fit_range[1]')

                # plot the distribution around VW shadow as histogram
                hrange = [int(np.floor(fit_range[0])), int(np.ceil(fit_range[1]))]
                y = fh.histo1d(df['pos'], range=hrange, bins=hrange[1] - hrange[0])
                x = np.linspace(*hrange, hrange[1] - hrange[0] + 1) # edges
                y = np.repeat(y, 2)
                x = np.hstack([x[0], np.repeat(x[1:-1], 2), x[-1]])
                y_max = y.max()
                ax.plot( # to draw like a histogram of style 'step'
                    x, y / y_max + nw_bar - 0.5,
                    color=outlier_color if vw_bar in outliers else colors[ctheme]['hist'][vw_i % 2],
                    linewidth=0.3 if vw_bar in outliers else 0.4,
                    zorder=100,
                )

                # plot the Gaussian fit
                x = np.linspace(*fit_range, 100)
                ax.plot(
                    x, self.gaus(x, *gparam) / gparam[0] + nw_bar - 0.5,
                    color=colors[ctheme]['gaus'][vw_i % 2],
                    linestyle='dashed' if vw_bar in outliers else 'solid',
                    linewidth=0.3 if vw_bar in outliers else 0.4,
                    zorder=100,
                )

        # nw grid lines (horizontal)
        for nw_bar in self.nw_bars:
            ax.axhline(nw_bar - 0.5, linewidth=0.3, linestyle='dashed', color=nw_grid_color)

        # vw grid lines (vertical)
        Polynomial = np.polynomial.Polynomial
        xticks = []
        for vw_bar in vw_bars:
            subdf = self.vw_shadow.query(f'vw_bar == {vw_bar}')
            if len(subdf) == 0: continue
            line = Polynomial.fit(subdf['nw_bar'], subdf[f'nw{self.ab}_x'], 1)
            y_plt = np.linspace(-0.5, 25.5, 3)
            ax.plot(line(y_plt), y_plt, color=vw_grid_color, linewidth=0.3, linestyle='dashed')
            xticks.append(line(25.5))
        
        # vw ticks
        axt = ax.twiny()
        axt.set_xlim(-150, 150)
        axt.set_xticks(xticks)
        axt.set_xticklabels(vw_bars)
        axt.set_xticks([], minor=True)
        axt.tick_params(axis='x', which='both', colors=vw_grid_color)
        axt.spines['top'].set_color(vw_grid_color)

class NWCalibrationReader:
    def __init__(self, AB, force_find_breakpoints=False):
        self.AB = AB
        self.ab = self.AB.lower()

        self.json_path = DATABASE_DIR / 'calib_params.json'
        self.dat_path = DATABASE_DIR / 'calib_params.dat'

        if not self.json_path.is_file() or force_find_breakpoints:
            self.find_calibration_breakpoints(save_to_database=True)
        self._update_json_content()
    
    def _update_json_content(self):
        with open(self.json_path, 'r') as file:
            self.json_content = json.load(file)
        self.json_content = {int(key): value for key, value in self.json_content.items()}

    def find_calibration_breakpoints(self, save_to_database=True):
        """Use Change Point Detection "CPD" algorithm from `ruptures` to
        identify where calibration parameters have shifted.

        Parameters:
            save_to_database : bool, default True
                If `True`, save all the results into JSON and DAT files; if
                `False`, nothing will be saved to files. Default is `True`.
        
        Returns:
            A dictionary of breakpoints with keys being bar numbers. The values
            are lists of dictionaries, each of which has two keys, namely,
            `run_range` and `parameters`.
        """
        # read in all calibration parameters
        runs = []
        df_par = {'p0': None, 'p1': None}
        elog = query.ElogQuery()
        for run in elog.df['run']:
            _df = self._get_calib_params(run)
            if _df is None:
                continue
            for par in df_par:
                if df_par[par] is None:
                    df_par[par] = _df[par].to_frame()
                else:
                    df_par[par] = pd.concat([df_par[par], _df[par]], axis='columns')
            runs.append(run)
        for par, _df in df_par.items():
            _df.columns = runs
            _df = _df.transpose()
            _df.index.rename('run', inplace=True)
            df_par[par] = _df # rows: run; cols: bar

        # apply change point detection (CPD) on the calibration parameters
        breakpoints = {bar: [] for bar in df_par['p0'].columns}
        for cut in ['run < 3500', 'run > 3500']:
            df = {par: _df.query(cut) for par, _df in df_par.items()}
            for bar in breakpoints:
                params = {par: df[par][bar] for par in df}
                data = np.vstack(list(params.values())).transpose()
                cpd = rpt.KernelCPD('rbf', min_size=1).fit(data)
                bp_indices = cpd.predict(pen=5.0)
                runs = params['p0'].index
                bp_runs = runs[bp_indices[:-1]]
                breakpoints[bar].extend([runs[0]] + list(bp_runs) + [runs[-1] + 1])
        
        # calculate the mid-50 averages
        for bar, bp_runs in breakpoints.items():
            bp_runs = np.array(bp_runs)
            run_ranges = np.vstack([bp_runs[:-1], bp_runs[1:] - 1]).transpose()
            new_value = []
            for run_range in run_ranges:
                cut = f'run >= {run_range[0]} & run <= {run_range[1]}'
                if eval(cut.replace('run', '3500').replace('&', 'and')):
                    continue
                p_means = dict()
                for par in df_par:
                    pars = df_par[par].query(cut)[bar]
                    pars = pars[(pars > pars.quantile(0.25)) & (pars < pars.quantile(0.75))]
                    p_means[par] = pars.mean()
                
                new_value.append({
                    'run_range': [int(run) for run in run_range],
                    'parameters': [round(p_mean, 6) for p_mean in p_means.values()],
                })
            breakpoints[bar] = new_value
        
        if save_to_database:
            # as .json
            with open(self.json_path, 'w') as file:
                json.dump(breakpoints, file, indent=4)

            # as .dat
            df = []
            for bar, infos in breakpoints.items():
                for info in infos:
                    df.append([bar, *info['run_range'], *info['parameters']])
            df = pd.DataFrame(df, columns=['bar', 'run_start', 'run_stop', 'p0', 'p1'])
            tables.to_fwf(df, self.dat_path, floatfmt=['02.0f', '04.0f', '04.0f', '.6f', '.6f'])

        return breakpoints
    
    def _get_calib_params(self, run):
        """Reader in the run-by-run calibration parameters from *.dat files.

        Parameters:
            run : int
                Experimental run number.
        
        Returns:
            If run is found in database, returns a `pandas.DataFrame`.
            Otherwise, returns `None`
        """
        path = pathlib.Path(
            PROJECT_DIR,
            'database/neutron_wall/position_calibration/calib_params',
            f'run-{run:04d}-nw{self.ab}.dat',
        )
        if path.is_file():
            df_par = pd.read_csv(path, delim_whitespace=True, comment='#')
            df_par.set_index(f'nw{self.ab}-bar', drop=True, inplace=True)
            return df_par
        else:
            return None

    def __call__(self, run, extrapolate=False, refresh=False):
        """Returns position calibration parameters for all bars as `pandas.DataFrame`.

        Parameters:
            run : int
                Experimental run number.
            extrapolate : bool, default False
                If `True`, uses the parameters from the closest run for run out
                of bound; if `False`, raises a `ValueError` whenever the run is
                out of bound. Default is `False`.
            refresh : bool, default False
                If `True`, always read in the parameters from JSON file; if
                `False`, simply uses `self.json_content` that had been loaded to
                memory upon initialization.
        
        Returns:
            A `pandas.DataFrame` with rows of bars and columns of p0 and p1.
        """
        if refresh:
            self._update_json_content()

        # start collecting the parameters
        df = []
        for bar, infos in self.json_content.items():
            # extract all parameters for the run of interest
            run_found = False
            for info in infos:
                if run >= info['run_range'][0] and run <= info['run_range'][1]:
                    df.append([bar, *info['parameters']])
                    run_found = True
                    break
            if run_found:
                continue

            # in the case when run was not found
            if not extrapolate:
                raise ValueError(f'Found no position calibration parmeters for run-{run:04d}')

            # run not found, constant extrapolation from the closest run_range
            closest_index = None
            closest_diff = 1e8
            for i_info, info in enumerate(infos):
                for run_edge in info['run_range']:
                    diff = abs(run - run_edge)
                    if diff < closest_diff:
                        closest_index = i_info
                        closest_diff = diff
            info = infos[closest_index]
            df.append([bar, *info['parameters']])

        # save parameters into pandas.DataFrame
        par_cols = [f'p{i}' for i in range(len(info['parameters']))]
        df = pd.DataFrame(df, columns=[f'nw{self.ab}-bar', *par_cols])
        df.set_index(f'nw{self.ab}-bar', inplace=True, drop=True)
        return df
