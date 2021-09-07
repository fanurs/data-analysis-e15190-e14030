import concurrent.futures
import hashlib
import json
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import scipy.stats
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import uproot 

from e15190 import PROJECT_DIR
from e15190.utilities import fast_histogram as fh
from e15190.utilities import styles
styles.set_matplotlib_style(mpl)

class PulseShapeDiscriminator:
    database_dir = PROJECT_DIR / 'database/neutron_wall/pulse_shape_discrimination'
    root_files_dir = PROJECT_DIR / 'database/root_files'
    light_GM_range = [1.0, 200.0] # MeVee
    pos_range = [-120.0, 120.0] # cm
    adc_range = [0, 4000] # the upper limit is 4096, but those would definitely be saturated.

    def __init__(self, AB, max_workers=12):
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.decompression_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.features = [
            'pos'
            'total_L',
            'total_R',
            'fast_L',
            'fast_R',
            'light_GM',
        ]
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.root_files_dir.mkdir(parents=True, exist_ok=True)
        self.particles = {'gamma': 0.0, 'neutron': 1.0}
        self.center_line = {'L': None, 'R': None}
        self.cfast_total = {'L': None, 'R': None}
    
    @classmethod
    def _cut_for_root_file_data(cls, AB):
        cuts = [
            f'NW{AB}_light_GM > {cls.light_GM_range[0]}',
            f'NW{AB}_light_GM < {cls.light_GM_range[1]}',
            f'NW{AB}_pos > {cls.pos_range[0]}',
            f'NW{AB}_pos < {cls.pos_range[1]}',
            f'NW{AB}_total_L >= {cls.adc_range[0]}',
            f'NW{AB}_total_L <= {cls.adc_range[1]}',
            f'NW{AB}_total_R >= {cls.adc_range[0]}',
            f'NW{AB}_total_R <= {cls.adc_range[1]}',
            f'NW{AB}_fast_L >= {cls.adc_range[0]}',
            f'NW{AB}_fast_L <= {cls.adc_range[1]}',
            f'NW{AB}_fast_R >= {cls.adc_range[0]}',
            f'NW{AB}_fast_R <= {cls.adc_range[1]}',
        ]
        return ' & '.join([f'({c.strip()})' for c in cuts])

    def read_run_from_root_file(self, run, tree_name=None, apply_cut=True):
        path = self.root_files_dir / f'run-{run:04d}.root'

        # determine the tree_name
        if tree_name is None:
            with uproot.open(str(path)) as file:
                objects = list(set(key.split(';')[0] for key in file.keys()))
            if len(objects) == 1:
                tree_name = objects[0]
            else:
                raise Exception(f'Multiple objects found in {path}')

        # load in the data
        nw_branches = [
            'bar',
            'total_L',
            'total_R',
            'fast_L',
            'fast_R',
            'pos',
            'light_GM',
        ]
        branch_names = [f'NW{self.AB}_{name}' for name in nw_branches]
        branch_names.append('VW_multi')
        with uproot.open(str(path) + ':' + tree_name) as tree:
            df = tree.arrays(
                branch_names,
                library='pd',
                decompression_executor=self.decompression_executor,
                interpretation_executor=self.interpretation_executor,
            )

        if apply_cut:
            df = df.query(self._cut_for_root_file_data(self.AB))
        return df
    
    def cache_run(self, run, tree_name=None):
        """Read in the data from ROOT file and save relevant branches to an HDF5 file.

        The data will be categorized according to bar number, because future
        retrieval by this class will most likely analyze only one bar at a time.
        """
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        df = self.read_run_from_root_file(run, tree_name=tree_name)

        # convert all float64 columns into float32
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        # write cache to HDF5 files bar by bar
        columns = [col for col in df.columns if col != f'NW{self.AB}_bar']
        for bar, subdf in df.groupby(f'NW{self.AB}_bar'):
            subdf.reset_index(drop=True, inplace=True)
            with pd.HDFStore(path, mode='a') as file:
                file.put(f'nw{self.ab}{bar:02d}', subdf[columns], format='fixed')
    
    def _read_single_run(self, run, bar, from_cache=True):
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        if not from_cache or not path.exists():
            self.cache_run(run)
        
        with pd.HDFStore(path, mode='r') as file:
            df = file.get(f'nw{self.ab}{bar:02d}')
        return df
    
    def read(self, run, bar, from_cache=True, verbose=False):
        if isinstance(run, int):
            runs = [run]
        else:
            runs = run
        
        df = None
        for run in runs:
            if verbose:
                print(f'\rReading run-{run:04d}', end='', flush=True)
            df_run = self._read_single_run(run, bar, from_cache=from_cache)
            df_run.insert(0, 'run', run)
            if df is None:
                df = df_run.copy()
            else:
                df = pd.concat([df, df_run], ignore_index=True)
        if verbose:
            print('\r', flush=True)
        self.bar = bar
        self.df = df
        self.df.columns = [name.replace(f'NW{self.AB}_', '') for name in self.df.columns]
    
    def randomize_integer_features(self, seed=None):
        rng = np.random.default_rng(seed=seed)
        for name, column in self.df.iteritems():
            if name not in self.features:
                continue
            if np.issubdtype(column.dtype, np.integer):
                self.df[name] += rng.uniform(low=-0.5, high=+0.5, size=len(column))
                self.df[name] = self.df[name].astype(np.float32)
    
    def normalize_features(self, df=None):
        self.feature_scaler = StandardScaler().fit(self.df[self.features])
        _df = self.feature_scaler.transform(self.df[self.features])
        if df is None:
            self.df[self.features] = _df
        else:
            return _df
    
    def denormalize_features(self, df=None):
        _df = self.feature_scaler.inverse_transform(self.df[self.features])
        del self.feature_scaler
        if df is None:
            self.df[self.features] = _df
        else:
            return _df
    
    def remove_vetowall_coincidences(self):
        self.df = self.df.query('VW_multi == 0')
        self.df.drop('VW_multi', axis=1, inplace=True)
    
    def remove_distorted_fast_total(self):
        self.total_threshold = dict()
        x_thres_min = 3000
        for side in ['L', 'R']:
            # produce 2D histogram of fast-total
            total, fast = f'total_{side}', f'fast_{side}'
            lin_reg = scipy.stats.linregress(self.df[total], self.df[fast])
            predict = lambda x: lin_reg.slope * x + lin_reg.intercept
            hrange = [[0, 4000], [-500, 500]]
            hbins = [100, 4000]
            hcounts = fh.histo2d(
                self.df[total],
                self.df[fast] - predict(self.df[total]),
                range=hrange,
                bins=hbins,
            )

            # calculate the x and y centers of the 2D histogram
            x_centers = np.linspace(*hrange[0], hbins[0] + 1)
            y_centers = np.linspace(*hrange[1], hbins[1] + 1)
            x_centers = 0.5 * (x_centers[1:] + x_centers[:-1])
            y_centers = 0.5 * (y_centers[1:] + y_centers[:-1])

            # determine threshold for nonlinearity
            y_means = np.sum(hcounts, axis=1)
            y_means = np.divide(
                np.dot(hcounts, y_centers), y_means,
                where=(y_means != 0),
                out=np.zeros_like(y_means),
            )
            y_thres = np.max(np.abs(y_means[x_centers < x_thres_min]))
            x_thres = None
            for ix in range(np.argmax(y_means), 0, -1):
                if y_means[ix] < y_thres:
                    x_thres = x_centers[ix]
                    break
            if x_thres is None:
                warnings.warn(f'Fail to find x_thres (TOTAL_{side}) above {x_thres_min}. Use {x_thres_min = }')
                x_thres = x_thres_min
            else:
                x_thres = int(np.round(x_thres, -2))
            self.total_threshold[side] = x_thres

        self.df = self.df.query(' & '.join([
            f'total_L < {self.total_threshold["L"]}',
            f'total_R < {self.total_threshold["R"]}',
        ]))

    def preprocessing(self):
        self.remove_vetowall_coincidences()
        self.remove_distorted_fast_total()
        self.randomize_integer_features()

    @staticmethod
    def find_topN_peaks(
        arr,
        n_peaks=2,
        bins=150,
        hrange=None,
        hist_log=False,
        kernel_width=0.05,
        kernel_half_range=2.0,
        fit_range_width=None,
        return_gaus_fit=False,
        use_kde=False,
        kde_bandwidth=None,
    ):
        """Find the first-N highest peaks from the distribution of arr.

        The algorithm is of recursive nature. It will always find only one peak -
        the highest peak - at a time, then it subtracts the peak out from the
        current distribution and proceeds to find the next highest peak. In each
        recursion, the algorithm simply looks for the highest count of the
        distribution (in histogram), meaning that it does not check whether the peak
        is of any statistical significance. While users can specify as many
        `n_peaks` as they want, the result only makes sense if `arr` does actually
        contain that many peaks.
        
        The algorithm uses a simple local Gaussian fit to each peak in order to
        improve the precision of the peak locations.

        Parameters:
            arr : 1D array
                The array to find the peaks in its distribution. Normalization of
                `arr` is done to set the mean to 0 and the standard deviation to 1,
                so that the algorithm and hyperparameters are more robust to various
                ranges of `arr`. Nonetheless, any results will be de-normalized
                before being returned, i.e. they will be in the original scale or
                unit.

                All other arguments that use the same units as `arr` will always be
                specified in the original unit. The normalization is always done
                internally, so that users do not have to worry about it.
            n_peaks : int, default 2
                The number of peaks to find.
            bins : int, default 150
                The number of bins to use in the histogram that approximates the
                distribution of `arr`.
            hrange : 2-tuple, default None
                The range of the histogram in the unit of `arr`. If None, the range
                is set to `[-3, 3]` in the normalized unit.
            hist_log : bool, default False
                If True, the counts of the histogram will be log-transformed,
                i.e. `h = log(h + 1)`, where `log` is the natural logarithm
            kernel_width : float, default 0.05
                The width of the kernel used to smooth the histogram.
            kernel_half_range : float, default 2.0
                The half-range of the kernel.
            fit_range_width : float, default None
                The width, in the unit of `arr`, specifies regions around the
                estimated peak positions that will be used to fit the Gaussian
                curves. If None, the width is set to `1.0` in the normalized unit.
            return_gaus_fit : bool, default False
                If True, the Gaussian fit to each peak is also returned.
            use_kde : bool, default False
                If True, KDE is used to find the peak location after the
                Gaussian fit.
            kde_bandwidth : float, default None
                The bandwidth of the KDE in the unit of `arr`. If None, the
                bandwidth is set to `0.05` in the normalized unit.
            
        Returns:
            If `return_gaus_fit` is False, returns a 1D array of the peak positions
            `xpeaks`, sorted from left to right; if `return_gaus_fit` is True,
            returns a tuple `(xpeaks, gpars)`, where `gpars` contain the Gaussian
            parameters for each peak.
        """
        gaus = lambda t, A, t0, sigma: A * np.exp(-0.5 * ((t - t0) / sigma)**2)

        # normalize arr and other x-related arguments
        arr_mean, arr_std = np.mean(arr), np.std(arr)
        transform = lambda x, x0=arr_mean, w=arr_std: (x - x0) / w
        arr = transform(arr)
        hrange = [-3, 3] if hrange is None else transform(np.array(hrange))
        fit_range_width = 1.0 if fit_range_width is None else transform(fit_range_width, x0=0.0)
        kde_bandwidth = 0.05 if kde_bandwidth is None else transform(kde_bandwidth, x0=0.0)

        # construct a histogram for peak-finding
        y = fh.histo1d(arr, range=hrange, bins=bins)
        y = y / np.max(y)
        x = np.linspace(*hrange, bins + 1)
        x = 0.5 * (x[1:] + x[:-1])
        if hist_log:
            y = np.log(y + 1)

        # Gaussian convolute to smooth out the histogram so peak finding is more reliable
        kernel = gaus(
            np.linspace(-kernel_half_range, kernel_half_range, bins),
            1.0,
            0.0,
            kernel_width * float(np.diff(hrange)) / (2 * kernel_half_range),
        )
        y = np.convolve(y, kernel, mode='same')

        # recursively find the highest peak, one at a time
        xpeaks = []
        gpars = []
        for ip in range(n_peaks):
            ipeak = np.argmax(y)
            xpeak, ypeak = x[ipeak], y[ipeak]
            fit_range = [xpeak - 0.5 * fit_range_width, xpeak + 0.5 * fit_range_width]
            mask = (x > fit_range[0]) & (x < fit_range[1])

            try:
                # fit a Gaussian curve to the peak
                gpar, _ = scipy.optimize.curve_fit(
                    gaus,
                    x[mask], y[mask],
                    p0=[ypeak, xpeak, 0.1],
                )
            except RuntimeError as err:
                if 'Optimal parameters not found' not in str(err):
                    raise RuntimeError(err)
                
                # usually the minimization fails because the peak is too narrow
                # so we try a finer sampling of the (x, y) points using interpolation
                # as well as fixing both the amplitude and the width of the Gaussian
                # varying only sigma 
                x_fine = np.linspace(fit_range[0], fit_range[1], 200)
                y_fine = scipy.interpolate.Akima1DInterpolator(x, y)(x_fine)
                gpar, _ = scipy.optimize.curve_fit(
                    lambda t, sigma: gaus(t, ypeak, xpeak, sigma),
                    x_fine, y_fine,
                    p0=[0.1],
                )
                gpar = [ypeak, xpeak, gpar[0]]

            if fit_range[0] < gpar[1] < fit_range[1]:
                xpeak = gpar[1]
            xpeaks.append(xpeak)
            gpars.append(gpar)

            # subtract the peak out from the current distribution
            y = np.clip(y - gaus(x, *gpar), 0.0, None)
        
        # use KDE to further pinpoint the peak positions
        if use_kde:
            kde = scipy.stats.gaussian_kde(arr, bw_method=kde_bandwidth)
            for ix, xpeak in enumerate(xpeaks):
                xpeaks[ix] = float(scipy.optimize.minimize_scalar(
                    lambda x: -kde(x),
                    method='bounded',
                    bounds=[xpeak - 5 * kde_bandwidth, xpeak + 5 * kde_bandwidth],
                ).x)

        # de-normalize the results
        xpeaks = np.array(xpeaks) * arr_std + arr_mean
        gpars = np.array(gpars)
        gpars[:, 1] = gpars[:, 1] * arr_std + arr_mean
        gpars[:, 2] = gpars[:, 2] * arr_std

        # sort peaks from left to right in x
        isorted = np.argsort(xpeaks)
        xpeaks = xpeaks[isorted]
        gpars = gpars[isorted]
        
        return xpeaks if not return_gaus_fit else (xpeaks, gpars)

    @staticmethod
    def extrapolate_linearly(func, boundary):
        func_deriv = nd.Derivative(func, n=1)
        if boundary is None or boundary == [None, None]:
            return func
        elif boundary[0] is None:
            x_ranges = [lambda x: x <= boundary[1], lambda x: x > boundary[1]]
            functions = [
                func,
                lambda x: func_deriv(boundary[1]) * (x - boundary[1]) + func(boundary[1]),
            ]
        elif boundary[1] is None:
            x_ranges = [lambda x: x < boundary[0], lambda x: x >= boundary[0]]
            functions = [
                lambda x: func_deriv(boundary[0]) * (x - boundary[0]) + func(boundary[0]),
                func,
            ]
        else:
            x_ranges = [
                lambda x: x < boundary[0],
                lambda x: (x >= boundary[0]) & (x <= boundary[1]),
                lambda x: x > boundary[1],
            ]
            functions = [
                lambda x: func_deriv(boundary[0]) * (x - boundary[0]) + func(boundary[0]),
                func,
                lambda x: func_deriv(boundary[1]) * (x - boundary[1]) + func(boundary[1]),
            ]

        def extrapolated_func(x):
            nonlocal x_ranges, functions
            x = np.array(x)
            return np.piecewise(
                x,
                [x_range(x) for x_range in x_ranges],
                [function for function in functions],
            )
        return extrapolated_func
    
    def normalize_psd_values(self, psd_column, **find_peak_kwargs):
        xpeaks = self.find_topN_peaks(self.df[psd_column], **find_peak_kwargs)

        # identify the particles, neutron or gamma, that correspond to the two peaks
        particle_identified = dict()
        for side in ['L', 'R']:
            total, fast = f'total_{side}', f'fast_{side}'
            if self.center_line[side] is None:
                self.center_line[side] = np.polynomial.Polynomial.fit(self.df[total], self.df[fast], 1)

            residual_means = [None] * 2
            for ip, comparator in enumerate(['<', '>']):
                subdf = self.df.query(f'{psd_column} {comparator} {np.mean(xpeaks)}')
                residual_means[ip] = np.mean(subdf[fast] - self.center_line[side](subdf[total]))
            
            if residual_means[0] < residual_means[1]:
                particle_identified[side] = ['neutron', 'gamma']
            else:
                particle_identified[side] = ['gamma', 'neutron']
        if particle_identified['L'] == particle_identified['R']:
            particle_identified = particle_identified['L']
        else:
            raise Exception(f'Contradicting particles from both sides:{particle_identified}')
        
        # normalize the psd values in psd_column according to self.particles
        pid = {particle: particle_identified.index(particle) for particle in particle_identified}
        slope = (self.particles['neutron'] - self.particles['gamma']) / (xpeaks[pid['neutron']] - xpeaks[pid['gamma']])
        self.df[psd_column] = slope * (self.df[psd_column] - xpeaks[pid['gamma']]) + self.particles['gamma']

    def discrimination_using_pca(self, normalize_psd_values=True):
        self.normalize_features()
        self.pca = decomposition.PCA(n_components=len(self.features))
        self.pca.fit(self.df[self.features])
        self.psd_params_from_pca = None
        for component in self.pca.components_:
            total_L = component[self.features.index('total_L')]
            total_R = component[self.features.index('total_R')]
            if np.sign(total_L) != np.sign(total_R):
                continue
            fast_L = component[self.features.index('fast_L')]
            fast_R = component[self.features.index('fast_R')]
            slope_L = fast_L / total_L
            slope_R = fast_R / total_R
            if np.isclose(slope_L, -1.0, atol=0.2) and np.isclose(slope_R, -1.0, atol=0.2):
                self.psd_params_from_pca = component.copy()
                self.psd_params_from_pca *= np.sign(total_L)
                break
        if self.psd_params_from_pca is None:
            raise Exception('No PSD parameters found from PCA')

        # finalize result
        self.df['psd_pca'] = np.dot(self.df[self.features], self.psd_params_from_pca)
        if normalize_psd_values:
            self.normalize_psd_values('psd_pca')
        self.denormalize_features()

    @staticmethod
    def create_ranges(low, upp, width, step=None):
        if step is None:
            step = width
        halfstep = 0.5 * step
        centers = np.arange(low + halfstep, upp - halfstep + 1e-3 * step, step)
        return np.vstack([centers - halfstep, centers + halfstep]).T

    def fit_fast_total(
        self,
        side,
        position_range=None,
        ax=None,
        find_peaks_kwargs=None,
    ):
        # initialize function arguments
        if find_peaks_kwargs is None:
            find_peaks_kwargs = dict()
        total = f'total_{side}'
        fast = f'fast_{side}'
        if position_range is None:
            position_range = [-100, -30] if side == 'L' else [30, 100]

        # prepare the data for fitting and center the fast-total pairs
        df = self.df.query(f'{position_range[0]} < pos < {position_range[1]}')
        df = df[[total, fast]]
        self.center_line[side] = np.polynomial.Polynomial.fit(df[total], df[fast], 1)
        df['cfast'] = df[fast] - self.center_line[side](df[total])

        # collect the peak information at various x-slices and save as control points
        ctrl_pts = {particle: [] for particle in self.particles}
        x_min = np.round(np.quantile(df[total], [0, 1e-4]).mean(), -2)
        if x_min > 100.0:
            warnings.warn(f'np.round({total}, -2) = {x_min} > 100.0')
        slices = [
            dict(
                x_range=[100, 1500], width=100, step=100, particles=['neutron', 'gamma'],
            ),
            dict(
                x_range=[1500, 2500], width=250, step=125, particles=['neutron', 'gamma'],
                find_peaks_kw=dict(
                    kernel_width=0.01,
                )
            ),
            dict(
                x_range=[2500, 4000], width=500, step=250, particles=['neutron'],
            ),
        ]
        for sl in slices:
            x_ranges = self.create_ranges(*sl['x_range'], width=sl['width'], step=sl['step'])
            for x_range in x_ranges:
                subdf = df.query(f'{x_range[0]} < {total} < {x_range[1]}')
                if len(subdf) < 100:
                    continue
                n_peaks = len(sl['particles'])
                kw = find_peaks_kwargs.copy()
                if 'find_peaks_kw' in sl:
                    kw.update(sl['find_peaks_kw'])
                cfasts = self.find_topN_peaks(subdf['cfast'], n_peaks=n_peaks, **kw)
                x_mean = np.mean(x_range)
                for ip, particle in enumerate(sl['particles']):
                    ctrl_pts[particle].append([x_mean, cfasts[ip]])
        ctrl_pts = {particle: np.array(pts) for particle, pts in ctrl_pts.items()}

        # throw away control points that are vertically too far from the previous point
        # all following points (to the right) are abandoned too
        removed_ctrl_pts = dict()
        for particle in ctrl_pts:
            x = ctrl_pts[particle][:, 0]
            y = ctrl_pts[particle][:, 1]
            dx = np.diff(x)
            dy = np.diff(y)
            slopes = np.divide(dy, dx, out=np.zeros_like(dy), where=dx != 0)
            n_heads = np.sum(x < 1500)
            slopes = slopes[n_heads:]
            dy = dy[n_heads:]
            indices = np.where((np.abs(slopes) > 0.12) | (np.abs(dy) > 25))[0] # HARD-CODED
            i_outlier = None if len(indices) == 0 else indices[0]
            if i_outlier is not None:
                i_outlier += n_heads + 1
                removed_ctrl_pts[particle] = ctrl_pts[particle][i_outlier:]
                ctrl_pts[particle] = ctrl_pts[particle][:i_outlier]

        # fit the control points
        pars = {particle: None for particle in self.particles}
        for particle in pars:
            if particle == 'neutron':
                kwargs = dict(
                    p0=np.zeros(3 + 1),
                )
            else:
                kwargs = dict(
                    p0=np.zeros(2 + 1),
                    method='trf',
                    loss='soft_l1',
                    f_scale=0.1,
                )
            pars[particle], _ = scipy.optimize.curve_fit(
                lambda x, *p: np.polynomial.Polynomial(p)(x),
                ctrl_pts[particle][:, 0], ctrl_pts[particle][:, 1],
                **kwargs,
            )
        
        # finalize fast-total relations
        fast_total = {particle: None for particle in self.particles}
        for particle in fast_total:
            fast_total[particle] = self.extrapolate_linearly(
                np.polynomial.Polynomial(pars[particle]),
                [ctrl_pts[particle][0, 0], ctrl_pts[particle][-1, 0]],
            )

        # plot for debug or check
        if ax:
            # some styling
            ax.grid(linestyle='dashed', color='cyan')
            ax.set_axisbelow(True)
            ax.set_xlabel(f'TOTAL-{side}')
            ax.set_ylabel(f'Centered FAST-{side}')

            # plot the centered fast-total 2D histogram
            h = fh.plot_histo2d(
                ax.hist2d,
                df[total], df['cfast'],
                range=[[0, 4000], [-150, 200]],
                bins=[1000, 175],
                cmap=mpl.cm.viridis,
                norm=mpl.colors.LogNorm(vmin=1),
            )
            plt.colorbar(h[3], ax=ax, pad=-0.05, fraction=0.1, aspect=30.0)

            # plot boundaries of slices
            for sl in slices:
                ax.axvline(sl['x_range'][1], color='green', linewidth=2, linestyle='dotted')

            # plot the control points
            kw = dict(s=8, color='white', linewidth=1.0, zorder=10)
            pts = ctrl_pts # shorthand
            ax.scatter(pts['neutron'][:, 0], pts['neutron'][:, 1], edgecolor='darkorange', marker='s', **kw)
            ax.scatter(pts['gamma'][:, 0], pts['gamma'][:, 1], edgecolor='magenta', marker='o', **kw)

            # plot the removed control points
            kw = dict(s=25, marker='X', edgecolor='black', linewidth=1.0, zorder=20)
            pts = removed_ctrl_pts # shorthand
            if 'neutron' in removed_ctrl_pts:
                ax.scatter(pts['neutron'][:, 0], pts['neutron'][:, 1], color='darkorange', **kw)
            if 'gamma' in removed_ctrl_pts:
                ax.scatter(pts['gamma'][:, 0], pts['gamma'][:, 1], color='magenta', **kw)
            
            # plot the fitted polynomials and its linearly extrapolated version
            colors = dict(neutron='red', gamma='red')
            for particle in fast_total:
                x_plt = np.linspace(0, 4000, 1000)
                ax.plot(
                    x_plt, np.polynomial.Polynomial(pars[particle])(x_plt),
                    color='gray', linewidth=1.0, linestyle='dashed',
                )
                ax.plot(
                    x_plt, fast_total[particle](x_plt),
                    color=colors[particle], linewidth=1.2, linestyle='solid',
                )
        
        self.cfast_total[side] = fast_total
        return self.cfast_total[side]

    def value_assign(self, side):
        total = f'total_{side}'
        fast = f'fast_{side}'
        cfast = self.df[fast] - self.center_line[side](self.df[total])
        fgamma = self.cfast_total[side]['gamma'](self.df[total])
        fneutron = self.cfast_total[side]['neutron'](self.df[total])
        self.df[f'vpsd_{side}'] = (cfast - fgamma) / (fneutron - fgamma)

    @staticmethod
    def find_two_2dpeaks(x, y, **find_peaks_kwargs):
        pca = decomposition.PCA(n_components=2)
        pca.fit(np.array([x, y]).T)
        first_pca = pca.transform(np.array([x, y]).T)[:, 0]
        xpeaks = PulseShapeDiscriminator.find_topN_peaks(first_pca, n_peaks=2, **find_peaks_kwargs)
        peak0 = pca.mean_ + pca.components_[0, :] * xpeaks[0]
        peak1 = pca.mean_ + pca.components_[0, :] * xpeaks[1]
        slope_01 = (peak1[1] - peak0[1]) / (peak1[0] - peak0[0])
        if slope_01 > 0:
            return np.array([peak0, peak1])
        else:
            return np.array([peak1, peak0])

    def fit_position_correction(self, light_GM_cut='light_GM > 3'):
        # identify the neutron-gamma centroids for different position slices
        pos_ranges = self.create_ranges(-100, 100, width=30, step=20)
        centroids = {'neutron': [], 'gamma': []}
        for i, pos_range in enumerate(pos_ranges):
            subdf = self.df.query(f'{pos_range[0]} < pos < {pos_range[1]} & {light_GM_cut}')[['vpsd_L', 'vpsd_R']]
            g_centroid, n_centroid = self.find_two_2dpeaks(subdf['vpsd_L'], subdf['vpsd_R'], kernel_width=0.01)
            centroids['neutron'].append(n_centroid)
            centroids['gamma'].append(g_centroid)

        # interpolate the centroids as functions of positions
        centroids = {particle: np.array(centroids[particle]) for particle in centroids}
        positions = np.mean(pos_ranges, axis=1)
        centroid_curves = {p: scipy.interpolate.Akima1DInterpolator(positions, centroids[p]) for p in centroids}
        for particle in centroid_curves:
            centroid_curves[particle].extrapolate = True
        
        # apply the centroid correction to the data
        subdf = self.df.query(light_GM_cut)
        g_pos = centroid_curves['gamma'](subdf['pos'])
        n_pos = centroid_curves['neutron'](subdf['pos'])
        xy = subdf[['vpsd_L', 'vpsd_R']].to_numpy() - g_pos
        gn_vec = n_pos - g_pos
        gn_rot90 = np.vstack([-gn_vec[:, 1], gn_vec[:, 0]]).T
        x = np.sum(xy * gn_vec, axis=1) / np.sum(np.square(gn_vec), axis=1)
        y = np.sum(xy * gn_rot90, axis=1) / np.sum(np.square(gn_rot90), axis=1)

        # using PCA to fine tune the centroid positions onto gamma: (0, 0) and neutron: (1, 0)
        mask = (-1 < x) & (x < 2) & (-1 < y) & (y < 1)
        pca_xy = decomposition.PCA(n_components=2).fit(np.vstack([x[mask], y[mask]]).T)
        if pca_xy.components_[0, 0] < 0:
            pca_xy.components_[0] *= -1
        if pca_xy.components_[1, 1] < 0:
            pca_xy.components_[1] *= -1
        x, y = pca_xy.transform(np.vstack([x, y]).T).T
        xpeaks = self.find_topN_peaks(x, n_peaks=2, use_kde=True)
        x = (x - xpeaks[0]) / (xpeaks[1] - xpeaks[0])

        # save all position correction parameters
        self.position_correction_params = {
            'centroid_curves': centroid_curves,
            'pca_xy': pca_xy,
            'pca_xpeaks': xpeaks,
        }

    def position_correction(self):
        """A function that maps (vpsd_L, vpsd_R, pos) to (ppsd, ppsd_perp)
        """
        pars = self.position_correction_params # shorthand

        centroids = {
            'gamma': pars['centroid_curves']['gamma'](self.df['pos']),
            'neutron': pars['centroid_curves']['neutron'](self.df['pos']),
        }
        xy = self.df[['vpsd_L', 'vpsd_R']].to_numpy() - centroids['gamma']
        gn_vec = centroids['neutron'] - centroids['gamma']
        gn_rot90 = np.vstack([-gn_vec[:, 1], gn_vec[:, 0]]).T
        x = np.sum(xy * gn_vec, axis=1) / np.sum(np.square(gn_vec), axis=1)
        y = np.sum(xy * gn_rot90, axis=1) / np.sum(np.square(gn_rot90), axis=1)
        x, y = pars['pca_xy'].transform(np.vstack([x, y]).T).T
        x = (x - pars['pca_xpeaks'][0]) / (pars['pca_xpeaks'][1] - pars['pca_xpeaks'][0])

        self.df['ppsd'] = x
        self.df['ppsd_perp'] = y

    @property
    def _run_hash_str(self):
        runs = sorted(set(self.df['run']))
        runs_hash = hashlib.sha256(','.join([str(run) for run in runs]).encode()).hexdigest()
        return f'run-{min(runs):04d}-{max(runs):04d}-h{runs_hash[-5:]}'

    def save_parameters(self, path=None):
        # prepare JSON path
        filename = f'NW{self.AB}-bar{self.bar:02d}.json'
        if path is None:
            path = self.database_dir / 'calib_params' / self._run_hash_str / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # the container for all JSON serialized parameters to be written to file
        pars = {
            'runs': sorted(set(self.df['run'])),
            'run_hash_str': self._run_hash_str,
            'NW': self.AB,
            'bar': self.bar,
        }

        # prepare fast-total relations into JSON serializable
        totals = np.arange(0, 4100 + 1e-9, 20.0)
        pars['fast_total'] = {
            'totals': totals,
            'center_line_L': self.center_line['L'](totals),
            'center_line_R': self.center_line['R'](totals),
            'gamma_cfasts_L': self.cfast_total['L']['gamma'](totals),
            'neutron_cfasts_L': self.cfast_total['L']['neutron'](totals),
            'gamma_cfasts_R': self.cfast_total['R']['gamma'](totals),
            'neutron_cfasts_R': self.cfast_total['R']['neutron'](totals),
        }

        # prepare the position correction parameters into JSON serializable
        pos_pars = self.position_correction_params
        positions = np.arange(-100, 100 + 1e-9, 5.0)
        pars['position_correction'] = {
            'centroid_curves': {
                'positions': positions,
                'gamma_centroids': pos_pars['centroid_curves']['gamma'](positions),
                'neutron_centroids': pos_pars['centroid_curves']['neutron'](positions),
            },
            'pca': {
                'mean': pos_pars['pca_xy'].mean_.tolist(),
                'components': pos_pars['pca_xy'].components_.tolist(),
                'xpeaks': pos_pars['pca_xpeaks'].tolist(),
            },
        }
        
        # write to file
        def default(obj):
            if isinstance(obj, np.ndarray):
                return np.round(obj, 5).tolist()
            raise TypeError(f'{obj.__class__.__name__} is not JSON serializable')
        with open(path, 'w') as file:
            json.dump(pars, file, indent=4, default=default)
    
    def _draw_fast_total(self, side, ax, cut):
        total, fast = f'total_{side}', f'fast_{side}'
        subdf = self.df.query(cut)[[total, fast]]
        subdf['cfast'] = subdf[fast] - self.center_line[side](subdf[total])

        # two-dimensional histogram
        h = fh.plot_histo2d(
            ax.hist2d,
            subdf[total], subdf['cfast'],
            range=[[0, 4000], [-150, 200]],
            bins=[500, 175],
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=ax, pad=-0.02, fraction=0.08, aspect=50.0)

        # fast-total relations
        total_plt = np.linspace(0, 4000, 200)
        for particle, ft in self.cfast_total[side].items():
            ax.plot(total_plt, ft(total_plt), color='black', linewidth=1.5, zorder=10)
            ax.plot(total_plt, ft(total_plt), color='gold', linewidth=1.2, zorder=20)
        
        # designs
        ax.set_xlim(0, 4000)
        ax.set_ylim(-150, 200)
        ax.set_xlabel(f'TOTAL-{side}')
        ax.set_ylabel(f'Centered FAST-{side}')

    def _draw_vpsd2d(self, ax, cut):
        subdf = self.df.query(cut)

        # two-dimensional histogram
        h = fh.plot_histo2d(
            ax.hist2d,
            subdf['vpsd_L'], subdf['vpsd_R'],
            range=[[-2, 3], [-2, 3]],
            bins=[250, 250],
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=ax, pad=-0.02, fraction=0.08, aspect=50.0)

        # design
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 3)
        ax.set_xlabel('VPSD-L')
        ax.set_ylabel('VPSD-R')

    def _draw_ppsd2d(self, ax, cut):
        subdf = self.df.query(cut)

        # two-dimensional histogram
        h = fh.plot_histo2d(
            ax.hist2d,
            subdf['ppsd'], subdf['ppsd_perp'],
            range=[[-2, 3], [-2, 2]],
            bins=[250, 200],
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=ax, pad=-0.02, fraction=0.08, aspect=50.0)

        # design
        ax.set_xlim(-2, 3)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('PPSD')
        ax.set_ylabel('PPSD-perpendicular')
    
    def _draw_ppsd2d_vs_lightGM(self, ax, cut):
        subdf = self.df.query(cut)

        # two-dimensional histogram
        fh.plot_histo2d(
            ax.hist2d,
            subdf['ppsd'], subdf['light_GM'],
            range=[[-3, 4], [0, 60]],
            bins=[350, 300],
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        ax.fill_betweenx([0, 3], -4, 5, color='silver', alpha=0.6, edgecolor='black', linewidth=1.5)
        ax.axvline(0.5, color='black', linewidth=1.5, zorder=10)
        ax.axvline(0.5, color='gold', linewidth=1.2, zorder=20)

        # design
        ax.set_xlim(-3, 4)
        ax.set_ylim(0, 60)
        ax.set_xlabel('PPSD')
        ax.set_ylabel('G.M. light (MeVee)')

    def _draw_ppsd1d(self, ax, cut):
        subdf = self.df.query(cut)
        fh.plot_histo1d(
            ax.hist,
            subdf['ppsd'],
            range=[-3, 4],
            bins=700,
            histtype='step',
            color='navy',
            density=True,
        )
        ax.axvline(0.5, color='gold', linewidth=1.2)

        # design
        ax.set_xlim(-3, 4)
        ax.set_ylim(0, )
        ax.set_xlabel('PPSD')
        ax.set_ylabel('Probability density')
        
    def save_to_gallery(self, path=None, cut='light_GM > 3', show_plot=False):
        filename = f'{self._run_hash_str}-NW{self.AB}-bar{self.bar:02d}.png'
        if path is None:
            path = self.database_dir / 'gallery' / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(13, 8), constrained_layout=True)
        fig.suptitle(f'{self._run_hash_str}: NW{self.AB}-bar{self.bar:02d}')

        rc = (0, 0)
        _cut = ' & '.join([f'({cut})', '(-100 < pos < -40)'])
        self._draw_fast_total('L', ax[rc], _cut)

        rc = (1, 0)
        _cut = ' & '.join([f'({cut})', '(40 < pos < 100)'])
        self._draw_fast_total('R', ax[rc], _cut)

        rc = (0, 1)
        self._draw_vpsd2d(ax[rc], cut)

        rc = (1, 1)
        self._draw_ppsd2d(ax[rc], cut)

        rc = (0, 2)
        self._draw_ppsd2d_vs_lightGM(ax[rc], 'light_GM > 1')

        rc = (1, 2)
        self._draw_ppsd1d(ax[rc], cut)

        plt.draw()
        fig.savefig(path, dpi=500, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
