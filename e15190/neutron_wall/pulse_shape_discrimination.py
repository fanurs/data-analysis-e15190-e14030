#!/usr/bin/env python
import concurrent.futures
import hashlib
import json
import pathlib
import warnings

import matplotlib as mpl
mpl_default_backend = mpl.get_backend()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import scipy.stats
from sklearn import decomposition
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
import uproot 

from e15190 import PROJECT_DIR
from e15190.neutron_wall.position_calibration import NWCalibrationReader
from e15190.utilities import fast_histogram as fh
from e15190.utilities import styles
styles.set_matplotlib_style(mpl)


class FastTotalRansacEstimator:
    """Custom base estimator object for RANSAC.

    By default, RANSAC regressor in scikit-learn only uses linear regression.
    To use custom regressor, we need to construct a base estimator object.
    See
    `sklearn.linear_model.RANSACRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html>`__
    for more details.
    """
    def __init__(self):
        pass
    
    def model(self, x, *args, **kwargs):
        """The fast-total relation function.

        This function is currently just an empty function. When inheriting from
        this class, define this model to enable all other methods that depend on
        the model.

        Parameters
        ----------
        x : array-like
            The TOTAL values.
        
        Returns
        -------
        y : array-like
            The predicted FAST values.
        """
        pass
    
    def fit(self, X, y, **kwargs):
        self.par, _ = scipy.optimize.curve_fit(self.model, X.flatten(), y, **kwargs)
        return self
    
    def predict(self, X):
        return self.model(X.flatten(), *self.par)
    
    def score(self, X, y):
        """Coefficient of determination.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target values, or ground truth.
        
        Returns
        -------
        score : float
            The coefficient of determination. For model that perfectly predicts
            all true values, the score is 1. For model that always predicts the
            mean of all true values, the score is 0. Models with worse
            performance will have negative scores.
        """
        y_pred = self.predict(X)
        y_true = y
        u = np.sum((y_true - y_pred)**2)
        v = np.sum((y_true - np.mean(y_true))**2)
        v = np.maximum(v, 1e-12) # avoid division by zero
        return 1 - u / v
    
    def get_params(self, deep=True):
        return dict()
    
    def set_params(self, **params):
        return FastTotalRansacEstimator()


class FastTotalRansacEstimatorGamma(FastTotalRansacEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def model(self, x, p0, p1):
        """The fast-total relation for gammas.

        Parameters
        ----------
        x : array_like
            The TOTAL values.
        p0 : float
            The intercept.
        p1 : float
            The slope.
        
        Returns
        -------
        y : array_like
            The predicted FAST values.
        """
        return p0 + p1 * x


class FastTotalRansacEstimatorNeutron(FastTotalRansacEstimator):
    # default hyperparameters
    x_switch = 1500.0

    def __init__(self, *args, **kwargs):
        super().__init__()

    def model(self, x, a0, a1, a2):
        """The fast-total relation for neutrons.

        Unlike gammas, whose fast-total relation is mostly linear, the observed
        fast-total relation of neutrons has non-linearity at low TOTAL light
        region. Hence, we fit the fast-total relation using piecewise function.
        When ``TOTAL >= x_switch``, we use linear function; when
        ``TOTAL < x_switch``, we quadratic function. At ``TOTAL = x_switch``,
        the two functions are continuous and smooth. This removes the freedom of
        choosing the intercept and slope of the linear function, hence there are
        only three parameters, ``a0``, ``a1`` and ``a2``.

        Parameters
        ----------
        x : array_like
            The TOTAL values.
        a0 : float
            The intercept.
        a1 : float
            The slope.
        a2 : float
            The coefficient of the quadratic term.

        Returns
        -------
        y : array_like
            The predicted FAST values.
        """
        xs = self.x_switch
        b1 = a1 + 2 * a2 * xs
        b0 = a0 + a1 * xs + a2 * xs**2 - b1 * xs
        return np.where(
            x < xs,
            a0 + a1 * x + a2 * x**2,
            b0 + b1 * x,
        )


class FastTotalFitter:
    def __init__(self, total, cfast, particle, **kwargs):
        self.data = pd.DataFrame({'total': total, 'cfast': cfast})
        self.fitted_model = None
        self.particle = particle
        if self.particle == 'gamma':
            self.estimator = FastTotalRansacEstimatorGamma(**kwargs)
        elif self.particle == 'neutron':
            self.estimator = FastTotalRansacEstimatorNeutron(**kwargs)
    
    def _fitted_model(self, x):
        if np.issubdtype(type(x), np.float) or np.issubdtype(type(x), np.integer):
            X = np.array([[x]])
        if np.array(x).ndim == 1:
            X = np.array(x)[:, None]
        return self.ransac.predict(X)
    
    def fit(self, **kwargs):
        X = np.array(self.data.total)[:, None]
        y = np.array(self.data.cfast)
        kw = dict(
            base_estimator=self.estimator,
            min_samples=0.1,
            residual_threshold=10,
        )
        kw.update(kwargs)
        self.ransac = RANSACRegressor(**kw).fit(X, y)
        self.fitted_model = self._fitted_model
        return self
    
    @property
    def score_of_data(self):
        """A custom score to evaluate the quality of data.

        A simple metric would be to compute the ratio between the number of
        inliers and the total number of data points. But in our case, outliers
        are commonly expected at large TOTAL values due to lack of statistics.
        Hence, we want to weight the numbers. For outliers at low TOTAL values,
        more weights would be assigned to act as heavy penalties; for outliers
        at high TOTAL values, less weights would be assigned.

        Returns
        -------
        score : float
            The metric that measures the quality of data. A score of 1 implies
            zero outliers. A score of 0 implies all data points are outliers.
            For our application, we found empirically that scores between 0.6 to
            0.8 can be set as a separator between good and bad data.
        """
        inliers = self.ransac.inlier_mask_
        weights = 1 / np.arange(1, len(inliers) + 1)**0.5
        return np.sum(inliers * weights) / np.sum(weights)

    def plot_data(self, ax=None, mask=None, **kwargs):
        if ax is not None:
            plt.sca(ax)
        if mask is None:
            x = self.data.total
            y = self.data.cfast
        else:
            x = self.data.total[mask]
            y = self.data.cfast[mask]
        kw = dict(fmt='o')
        kw.update(kwargs)
        plt.errorbar(x, y, **kw)
    
    def plot_fit(self, ax=None, **kwargs):
        if ax is not None:
            plt.sca(ax)
        kw = dict(linestyle='solid')
        kw.update(kwargs)
        x_plt = np.linspace(self.data.total.min(), self.data.total.max(), 100)
        plt.plot(x_plt, self.fitted_model(x_plt), **kw)


class PulseShapeDiscriminator:
    database_dir = PROJECT_DIR / 'database/neutron_wall/pulse_shape_discrimination'
    root_files_dir = None # the input root files directory (Daniele's ROOT files)
    light_GM_range = [1.0, 200.0] # MeVee
    pos_range = [-120.0, 120.0] # cm
    adc_range = [0, 4000] # the upper limit is 4096, but those would definitely be saturated.

    # some hyper-parameters for optimizing the fast-total fit
    ft_breakpoint1 = 1500.0 # TOTAL in raw ADC
    ft_breakpoint2 = 2500.0 # TOTAL in raw ADC
    min_samples_gamma = 0.25 # the minimum percentage of inliers
    min_samples_neutron = 0.7 # the minimum percentage of inliers
    x_switch_neutron = 1300.0 # TOTAL in raw ADC

    def __init__(self, AB, max_workers=12):
        """Construct a class to perform pulse shape discrimination.

        Parameters
        ----------
        AB : 'A' or 'B'
            The neutron wall to use.
        max_workers : int, default 12
            The maximum number of workers to use for parallelization. This value
            will be passed to the `concurrent.futures.ThreadPoolExecutor` for
            constructing ``self.decompression_executor`` and
            ``self.interpretation_executor``. Each executor is assigned with a
            thread pool of size ``max_workers``.
        """
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.max_workers = max_workers
        self.decompression_executor = None
        self.interpretation_executor = None
        self.features = [
            'pos',
            'total_L',
            'total_R',
            'fast_L',
            'fast_R',
            'light_GM',
        ]
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.particles = {'gamma': 0.0, 'neutron': 1.0}
        self.center_line = {'L': None, 'R': None}
        self.cfast_total = {'L': None, 'R': None}
        self.ctrl_pts = {'L': None, 'R': None}
        self.fitter = {'L': None, 'R': None}

        # initialize input root files directory
        path = PROJECT_DIR / 'database/local_paths.json'
        with open(path, 'r') as file:
            self.root_files_dir = pathlib.Path(json.load(file)['daniele_root_files_dir'])
    
    @classmethod
    def update_hyperparameters(cls, args):
        if args.ft_breakpoint1 is not None:
            cls.ft_breakpoint1 = args.ft_breakpoint1
        if args.ft_breakpoint2 is not None:
            cls.ft_breakpoint2 = args.ft_breakpoint2
        if args.min_samples_gamma is not None:
            cls.min_samples_gamma = args.min_samples_gamma
        if args.min_samples_neutron is not None:
            cls.min_samples_neutron = args.min_samples_neutron
        if args.x_switch_neutron is not None:
            cls.x_switch_neutron = args.x_switch_neutron
    
    @classmethod
    def _cut_for_root_file_data(cls, AB):
        """Returns a string that specifies the first common cut.

        This cut is the first common cut that is applied to all the data before
        analysis. Their purpose is only to remove the events that are
        unphysical. Further analysis cuts will have to be applied separately for
        each bar.

        Parameters
        ----------
        AB : 'A' or 'B'
            The neutron wall to use.
        """
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
        """Read in the data from Daniele's ROOT file.

        Parameters
        ----------
        run : int
            The run number.
        tree_name : str, default None
            The name of the tree to read. If None, the default tree name is
            automatically determined. An exception is raised if multiple objects
            are found in the ROOT file.
        apply_cut : bool, default True
            Whether to apply the first common cut to the data. If ``True``,
            ``self._cut_for_root_file_data(self.AB)`` is used as the cut.
        """
        path = self.root_files_dir / f'CalibratedData_{run:04d}.root'

        # determine the tree_name
        if tree_name is None:
            with uproot.open(str(path)) as file:
                objects = list(set(key.split(';')[0] for key in file.keys()))
            if len(objects) == 1:
                tree_name = objects[0]
            else:
                raise Exception(f'Multiple objects found in {path}')

        # load in the data
        branches = { # new name -> old name
            f'NW{self.AB}_bar'     : f'NW{self.AB}.fnumbar',
            f'NW{self.AB}_total_L' : f'NW{self.AB}.fLeft',
            f'NW{self.AB}_total_R' : f'NW{self.AB}.fRight',
            f'NW{self.AB}_fast_L'  : f'NW{self.AB}.ffastLeft',
            f'NW{self.AB}_fast_R'  : f'NW{self.AB}.ffastRight',
            f'NW{self.AB}_time_L'  : f'NW{self.AB}.fTimeLeft',
            f'NW{self.AB}_time_R'  : f'NW{self.AB}.fTimeRight',
            f'NW{self.AB}_light_GM': f'NW{self.AB}.fGeoMeanSaturationCorrected',
             'VW_multi'            :  'VetoWall.fmulti',
        }
        if self.decompression_executor is None:
            self.decompression_executor = concurrent.futures.ThreadPoolExecutor(self.max_workers)
        if self.interpretation_executor is None:
            self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(self.max_workers)
        with uproot.open(str(path) + ':' + tree_name) as tree:
            df = tree.arrays(
                list(branches.values()),
                library='pd',
                decompression_executor=self.decompression_executor,
                interpretation_executor=self.interpretation_executor,
            )
        df.columns = list(branches.keys())

        # apply position calibration
        calib_reader = NWCalibrationReader(self.AB)
        df[f'NW{self.AB}_pos'] = self.get_position(
            df[[f'NW{self.AB}_bar', f'NW{self.AB}_time_L', f'NW{self.AB}_time_R']],
            calib_reader(run),
        )
        df.drop([f'NW{self.AB}_time_L', f'NW{self.AB}_time_R'], axis=1, inplace=True)

        if apply_cut:
            df = df.query(self._cut_for_root_file_data(self.AB))
        return df
    
    @staticmethod
    def get_position(df_time, pos_calib_params):
        """Calculate the positions from time and calibration parameters.

        Parameters
        ----------
        df_time : pandas.DataFrame
            The dataframe with columns ``bar``, ``time_L``, and ``time_R``.
        pos_calib_params : panda.DataFrame
            The calibration parameters. The index is bar number; the first
            column is the calibration offset; the second column is the
            calibration slope.
        
        Returns
        -------
        positions : numpy.1darray
            The calibrated positions in cm.
        """
        pars = pos_calib_params # shorthand
        df_time.columns = ['bar', 'time_L', 'time_R']
        pars = pars.loc[df_time['bar']].to_numpy()
        return pars[:, 0] + pars[:, 1] * (df_time['time_L'] - df_time['time_R'])
    
    def cache_run(self, run, tree_name=None):
        """Read in the data from ROOT file and save relevant branches to an HDF5 file.

        The data will be categorized according to bar number, because future
        retrieval by this class will most likely analyze only one bar at a time.

        Parameters
        ----------
        run : int
            The run number.
        tree_name : str, default None
            The name of the tree to read. If None, the default tree name is
            automatically determined. An exception is raised if multiple objects
            are found in the ROOT file.
        """
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        df = self.read_run_from_root_file(run, tree_name=tree_name)

        # convert all float64 columns into float32
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        # write cache to HDF5 files bar by bar
        columns = [col for col in df.columns if col != f'NW{self.AB}_bar']
        path.unlink(missing_ok=True) # remove pre-existing file, if any
        path.parent.mkdir(exist_ok=True, parents=True)
        for bar, subdf in df.groupby(f'NW{self.AB}_bar'):
            subdf.reset_index(drop=True, inplace=True)
            with pd.HDFStore(path, mode='a') as file:
                file.put(f'nw{self.ab}{bar:02d}', subdf[columns], format='fixed')
    
    def _read_single_run(self, run, bar, from_cache=True):
        """Read in the data from a single bar in a single run.

        Parameters
        ----------
        run : int
            The run number.
        bar : int
            The bar number.
        from_cache : bool, default True
            Whether to read the data from the cache. If ``False``, the data is
            read from the ROOT file.
        
        Returns
        -------
        df : pandas.DataFrame
            The dataframe with the data.
        """
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        if not from_cache or not path.exists():
            self.cache_run(run)
        
        with pd.HDFStore(path, mode='r') as file:
            df = file.get(f'nw{self.ab}{bar:02d}')
        return df
    
    def read(self, run, bar, from_cache=True, verbose=False):
        """Read in the data needed to do pulse shape discrimintion.

        Attributes ``self.bar`` and ``self.df`` will be updated.  The data will
        be read from the HDF5 file if it exists, otherwise they will be read in
        from the ROOT file (generated by Daniele's framework).

        Parameters
            run : int or list of ints
                The run number(s) to read in. Most of the time, this will be a
                list of at least five runs. Otherwise, the algorithm will have a
                hard time find the pulse shape discrimintion parameters due to
                low statistics.
            bar : int
                The bar number to read in.
            from_cache : bool, default True
                Whether to read in the data from the HDF5 cache. If False, the
                data will be read in from the ROOT file. A new cache will be
                created, overwriting any existing cache.
            verbose : bool, default False
                Whether to print out the progress of the read in.
        """
        if isinstance(run, int):
            runs = [run]
        else:
            runs = run
        
        df = None
        for i_run, run in enumerate(runs):
            if verbose:
                print(f'\rReading run-{run:04d}  ({i_run + 1}/{len(runs)})', end='', flush=True)
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
        """Randomize the integer columns in ``self.df``.

        For each integer column, we add a uniform random number between -0.5 and
        +0.5 to the values. This is done in-place.

        Parameters
        ----------
        seed : int, default None
            The seed for the random number generator. If None, the randomization
            non-reproducible.
        """
        rng = np.random.default_rng(seed=seed)
        for name, column in self.df.iteritems():
            if name not in self.features:
                continue
            if np.issubdtype(column.dtype, np.integer):
                self.df[name] += rng.uniform(low=-0.5, high=+0.5, size=len(column))
                self.df[name] = self.df[name].astype(np.float32)
    
    def normalize_features(self, df=None):
        """Normalize features.

        Normalization is done such that the resulting variables have mean 0 and
        standard deviation 1. Under the hood, this is done by
        ``sklearn.preprocessing.StandardScaler``. The normalization parameters
        are stored in ``self.feature_scaler``. This allows for denormalization
        later on.

        Parameters
        ----------
        df : pandas.DataFrame, default None
            The dataframe to normalize. If None, the dataframe in ``self.df``
            is used, and the normalization is done in-place.

        Returns
        -------
        normed_df : pandas.DataFrame
            The normalized dataframe.
        """
        self.feature_scaler = StandardScaler().fit(self.df[self.features])
        _df = self.feature_scaler.transform(self.df[self.features])
        if df is None:
            self.df[self.features] = _df
        return _df
    
    def denormalize_features(self, df=None):
        """Denormalize features.

        Recovers the original values of the features from the normalized ones.
        This is done by applying the inverse transformation by
        ``self.feature_scaler``. The attribute ``self.feature_scaler`` will also
        be deleted after this operation to prevent further denormalization.

        Parameters
        ----------
        df : pandas.DataFrame, default None
            The dataframe to denormalize. If None, the dataframe in ``self.df``
            is used, and the denormalization is done in-place.
        
        Returns
        -------
        denormed_df : pandas.DataFrame
            The denormalized dataframe.
        """
        _df = self.feature_scaler.inverse_transform(self.df[self.features])
        del self.feature_scaler
        if df is None:
            self.df[self.features] = _df
        return _df
    
    def remove_vetowall_coincidences(self, df=None):
        """Remove Veto Wall coincidences.

        This is done to effectively remove charged particles from the data.
        Currently, this is done by removing the entire event as long as there is
        any non-zero Veto Wall multiplicity. In the future, this could be
        further improved by removing only hits that fall within a certain solid
        angle from the point of contact of the veto wall.

        Parameters
        ----------
        df : pandas.DataFrame, default None
            The dataframe to remove the Veto Wall coincidences from. If None,
            the dataframe in ``self.df`` is used in-place.
        
        Returns
        -------
        df_result : pandas.DataFrame
            The dataframe with the Veto Wall coincidences removed.
        """
        _df = self.df.query('VW_multi == 0')
        _df.drop('VW_multi', axis=1, inplace=True)
        if df is None:
            self.df = _df
        return _df
    
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
        
        return xpeaks if not return_gaus_fit else gpars

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
        find_peaks_kwargs=None,
    ):
        """Fit and update ``self.cfast_total``.

        Parameters
        ----------
        side : 'L' or 'R'
            The side to fit.
        position_range : list of 2 floats, default None
            The range of positions to fit . If None, use the default ranges.
            For left side, the range is [-100, -30]; for right side, the range
            is [30, 100].
        find_peaks_kwargs : dict, default None
            The keyword arguments for :py:func:`find_topN_peaks`.
        
        Returns
        -------
        cfast_total : dict
            The fitted parameters.
        """
        # initialize function arguments
        if find_peaks_kwargs is None:
            find_peaks_kwargs = dict()
        total = f'total_{side}'
        fast = f'fast_{side}'
        cfast = 'cfast'
        if position_range is None:
            if side == 'L':
                position_range = [-100.0, -30.0]
            else: # 'R'
                position_range = [30.0, 100.0]
        
        # center the fast-total pairs
        df = self.df.query(f'{position_range[0]} < pos < {position_range[1]}')
        df = df[[total, fast]]
        self.center_line[side] = np.polynomial.Polynomial.fit(df[total], df[fast], 1)
        df[cfast] = df[fast] - self.center_line[side](df[total])

        # collect peak information at various x-slices and save as control points
        ctrl_pts = {particle: [] for particle in self.particles}
        x_min = np.round(np.quantile(df[total], [0, 1e-4]).mean(), -2)
        if x_min > 100.0:
            warnings.warn(f'np.round({total}, -2) = {x_min} > 100.0')
        slices = [
            dict(
                x_range=[100.0, self.ft_breakpoint1], width=100, step=75, particles=['neutron', 'gamma'],
            ),
            dict(
                x_range=[self.ft_breakpoint1, self.ft_breakpoint2], width=250, step=100, particles=['neutron', 'gamma'],
                find_peaks_kw=dict(
                    kernel_width=0.01,
                ),
            ),
            dict(
                x_range=[self.ft_breakpoint2, 4000.0], width=400, step=150, particles=['neutron'],
            ),
        ]
        for slice_ in slices:
            x_ranges = self.create_ranges(*slice_['x_range'], width=slice_['width'], step=slice_['step'])
            for x_range in x_ranges:
                subdf = df.query(f'{x_range[0]} < {total} < {x_range[1]}')
                if len(subdf) < 100:
                    continue
                n_peaks = len(slice_['particles'])
                kw = find_peaks_kwargs.copy()
                if 'find_peaks_kw' in slice_:
                    kw.update(slice_['find_peaks_kw'])
                cfasts = self.find_topN_peaks(subdf['cfast'], n_peaks=n_peaks, **kw)
                x_mean = np.mean(x_range)
                for ip, particle in enumerate(slice_['particles']):
                    ctrl_pts[particle].append([x_mean, cfasts[ip]])
        ctrl_pts = {particle: pd.DataFrame(pts, columns=['total', 'fast']) for particle, pts in ctrl_pts.items()}

        # fit the control points
        FastTotalRansacEstimatorNeutron.x_switch = self.x_switch_neutron
        fitter = {
            'gamma': FastTotalFitter(ctrl_pts['gamma'].total, ctrl_pts['gamma'].fast, 'gamma'),
            'neutron': FastTotalFitter(ctrl_pts['neutron'].total, ctrl_pts['neutron'].fast, 'neutron')
        }
        fitter['gamma'].fit(min_samples=self.min_samples_gamma)
        fitter['neutron'].fit(min_samples=self.min_samples_neutron)
        self.fitter[side] = fitter
        ctrl_pts['gamma']['valid'] = fitter['gamma'].ransac.inlier_mask_
        ctrl_pts['neutron']['valid'] = fitter['neutron'].ransac.inlier_mask_

        self.ctrl_pts[side] = ctrl_pts
        self.cfast_total[side] = {
            'gamma': fitter['gamma'].fitted_model,
            'neutron': fitter['neutron'].fitted_model,
        }
        return self.cfast_total[side]

    def value_assign(self, side):
        """Value-assign a PSD value to each entry.

        Assigns gamma to 0 and neutron to 1. Values in between, e.g. 0.5, would
        mean that the algorithm has a hard time confidently distinguishing
        between gamma and neutron. However, values that are far from the
        assigned values, e.g. 10.0 or -10.0, are likely to be noise too. So when
        trying to select a particle, it is better to specify a range of assigned
        values, VPSDs, e.g. [-1, 0.5] for gamma and [0.5, 2] for neutron.

        Of course, this function is only assigning the PSD value based on
        information from one side of the NW bar, which does not give very good
        neutron-gamma separation. Also, hit position information is not used at
        all. Eventually, the method `PulseShapeDiscriminator.fit()` would
        provide a better PSD parameter that is built on top of this one-sided
        VPSD parameter.

        Parameters
        ----------
        side : 'L' or 'R'
            The side of the NW bar to do the value-assignment.
        """
        total = f'total_{side}'
        fast = f'fast_{side}'
        cfast = self.df[fast] - self.center_line[side](self.df[total])
        fgamma = self.cfast_total[side]['gamma'](self.df[total])
        fneutron = self.cfast_total[side]['neutron'](self.df[total])
        self.df[f'vpsd_{side}'] = np.where(
            fneutron < fgamma,
            (cfast - fgamma) / (fneutron - fgamma),
            -9999.0,
        )

    @staticmethod
    def find_two_2dpeaks(x, y, **find_peaks_kwargs):
        pca = decomposition.PCA(n_components=2)
        pca.fit(np.array([x, y]).T)
        first_pca = pca.transform(np.array([x, y]).T)[:, 0]
        xpeaks = PulseShapeDiscriminator.find_topN_peaks(first_pca, n_peaks=2, **find_peaks_kwargs)
        peak0 = pca.mean_ + pca.components_[0, :] * xpeaks[0]
        peak1 = pca.mean_ + pca.components_[0, :] * xpeaks[1]
        return np.array(sorted([peak0, peak1], key=lambda v: np.linalg.norm(v)))

    def fit_position_correction(self, light_GM_cut='light_GM > 2'):
        vpsd_cut = '(-4 < vpsd_L < 5) & (-4 < vpsd_R < 5)'

        # identify the neutron-gamma centroids for different position slices
        pos_ranges = self.create_ranges(-100, 100, width=30, step=20)
        centroids = {'neutron': [], 'gamma': []}
        for i, pos_range in enumerate(pos_ranges):
            subdf = self.df.query(' & '.join([
                f'{pos_range[0]} < pos < {pos_range[1]}',
                vpsd_cut,
                light_GM_cut,
            ]))[['vpsd_L', 'vpsd_R']]
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
        subdf = self.df.query(light_GM_cut + ' & ' + vpsd_cut)
        g_pos = centroid_curves['gamma'](subdf['pos'])
        n_pos = centroid_curves['neutron'](subdf['pos'])
        xy = subdf[['vpsd_L', 'vpsd_R']].to_numpy() - g_pos
        gn_vec = n_pos - g_pos
        gn_rot90 = np.vstack([-gn_vec[:, 1], gn_vec[:, 0]]).T
        x = np.sum(xy * gn_vec, axis=1) / np.sum(np.square(gn_vec), axis=1)
        y = np.sum(xy * gn_rot90, axis=1) / np.sum(np.square(gn_rot90), axis=1)

        # using PCA to fine tune the centroid positions onto gamma: (0, 0) and neutron: (1, 0)
        mask = (-2 < x) & (x < 3) & (-1 < y) & (y < 1)
        pca_xy = decomposition.PCA(n_components=2).fit(np.vstack([x[mask], y[mask]]).T)
        if pca_xy.components_[0, 0] < 0:
            pca_xy.components_[0] *= -1
        if pca_xy.components_[1, 1] < 0:
            pca_xy.components_[1] *= -1
        x, y = pca_xy.transform(np.vstack([x, y]).T).T
        xpeaks = self.find_topN_peaks(x[(x > -2) & (x < 3)], n_peaks=2, use_kde=True)
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

    def fit(self):
        """To calculate the pulse shape discrimination parameters.
        
        This is a wrapper for multiple functions. If you run into any bugs and
        need to debug the program, it is recommended to run the individual
        functions one by one.
        """
        self.preprocessing()
        for side in 'LR':
            self.fit_fast_total(side)
            self.value_assign(side)
        self.fit_position_correction()
        self.position_correction()

    @staticmethod
    def figure_of_merit(arr, **find_peaks_kwargs):
        """Calculate the figure-of-merit that quantifies the two-peak separation

        The formula is F.O.M. = |x0 - x1| / (FWHM0 + FWHM1), which is standard
        adopted in many literature on the subject of pulse shape discrimination.
        One slight modification here is that, instead of using the actual FWHM,
        we estimate it by first fitting a Gaussian around the peak, then take
        FWHM to be ~2.3548 times sigma, the standard deviation.
        """
        find_peaks_kwargs.update(dict(return_gaus_fit=True))
        gpars = PulseShapeDiscriminator.find_topN_peaks(arr, **find_peaks_kwargs)
        fwhm = lambda sigma: 2 * np.sqrt(2 * np.log(2)) * sigma
        return np.abs(gpars[1, 1] - gpars[0, 1]) / (fwhm(gpars[0, 2]) + fwhm(gpars[1, 2]))

    @property
    def _run_hash_str(self):
        """Returns a string that uniquely identifies the runs.
        """
        runs = sorted(set(self.df['run']))
        runs_hash = hashlib.sha256(','.join([str(run) for run in runs]).encode()).hexdigest()
        return f'run-{min(runs):04d}-{max(runs):04d}-h{runs_hash[-5:]}'

    def save_parameters(self, path=None):
        """Save pulse shape discrimination parameters as a JSON file.

        By default, this function saves the parameters to the database as a JSON
        file. An example path would be:
        `$PROJECT_DIR/database/calib_params/run-4082-4123-ha88b0/NWB-bar01.json`

        Parameters
        ----------
        path : str or pathlib.Path, default None
            The path to save the parameters. If None, the parameters will be
            saved to the default database.
        """
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

        pars['hyper_params'] = {
            'ft_breakpoint1': self.ft_breakpoint1,
            'ft_breakpoint2': self.ft_breakpoint2,
            'min_samples_gamma': self.min_samples_gamma,
            'min_samples_neutron': self.min_samples_neutron,
            'x_switch_neutron': self.x_switch_neutron,
        }

        # prepare control points into JSON serializable
        pars['ctrl_pts'] = {
            'L': {
                'gamma': {
                    'total': list(self.ctrl_pts['L']['gamma']['total']),
                    'fast': list(self.ctrl_pts['L']['gamma']['fast']),
                    'valid': list(self.ctrl_pts['L']['gamma']['valid']),
                },
                'neutron': {
                    'total': list(self.ctrl_pts['L']['neutron']['total']),
                    'fast': list(self.ctrl_pts['L']['neutron']['fast']),
                    'valid': list(self.ctrl_pts['L']['neutron']['valid']),
                },
            },
            'R': {
                'gamma': {
                    'total': list(self.ctrl_pts['R']['gamma']['total']),
                    'fast': list(self.ctrl_pts['R']['gamma']['fast']),
                    'valid': list(self.ctrl_pts['R']['gamma']['valid']),
                },
                'neutron': {
                    'total': list(self.ctrl_pts['R']['neutron']['total']),
                    'fast': list(self.ctrl_pts['R']['neutron']['fast']),
                    'valid': list(self.ctrl_pts['R']['neutron']['valid']),
                },
            },
        }

        # quality metric for data and fits
        pars['score_of_data'] = {
            'L': {
                'neutron': self.fitter['L']['neutron'].score_of_data,
                'gamma': self.fitter['L']['gamma'].score_of_data,
            },
            'R': {
                'neutron': self.fitter['R']['neutron'].score_of_data,
                'gamma': self.fitter['R']['gamma'].score_of_data,
            },
        }

        # fast-total relations as polynomials
        pars['fast_total_polynomial'] = {
            'L': {
                'neutron': list(self.fitter['L']['neutron'].ransac.estimator_.par),
                'gamma': list(self.fitter['L']['gamma'].ransac.estimator_.par),
            },
            'R': {
                'neutron': list(self.fitter['R']['neutron'].ransac.estimator_.par),
                'gamma': list(self.fitter['R']['gamma'].ransac.estimator_.par),
            },
        }


        # prepare fast-total relations into JSON serializable
        # Here, we use fine grid of the relations so that any
        # smooth interpolation would be able to reconstruct
        # the fast-total relations without the analytic model.
        totals = np.arange(0, 4100 + 1e-9, 20.0)
        pars['fast_total'] = {
            'nonlinear_total_L': self.total_threshold['L'],
            'nonlinear_total_R': self.total_threshold['R'],
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


class Gallery:
    @staticmethod
    def save_as_png(psd_obj, path=None, cut='light_GM > 3', show_plot=False, save=True):
        """Save a diagnostic plot to the gallery as a PNG file.

        Parameters
        ----------
        path : str or pathlib.Path, default None
            The path to save the plot. If None, the plot is saved to the default
            database.
        cut : str, default 'light_GM > 3'
            The cut to apply to the data when plotting. All panels in the figure
            will apply this cut, and potentially other cuts, joined by logical
            AND, except for the panel that draws the PPSD as a function of light
            GM.
        show_plot : bool, default False
            If True, the plot will be shown in run time, i.e. the command
            `plt.show()` will be called.
        save : bool, default True
            If `True`, the plot will be saved to the database. `False` option is
            useful when users are using this function to inspect the plot
            without saving.
        """
        # prepare image path and directory
        if path is None:
            path = psd_obj.database_dir / 'gallery' / psd_obj._run_hash_str # sub-directory
            path = path / f'NW{psd_obj.AB}-bar{psd_obj.bar:02d}.png'
        elif isinstance(path, str):
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # determine matplotlib backend
        if show_plot:
            mpl.use(mpl_default_backend)
        else:
            mpl.use('Agg')

        # construct figure
        fig, ax = plt.subplots(
            ncols=3, nrows=3,
            figsize=(13, 11),
            constrained_layout=True,
        )
        fig.suptitle(f'{psd_obj._run_hash_str}: NW{psd_obj.AB}-bar{psd_obj.bar:02d}')

        plt.sca(ax[0, 0])
        Gallery.draw_cfast_total(psd_obj, 'L', ' & '.join([f'({cut})', '(-100 < pos < -40)']))
        
        plt.sca(ax[0, 1])
        Gallery.draw_cfast_total(psd_obj, 'R', ' & '.join([f'({cut})', '(40 < pos < 100)']))

        plt.sca(ax[0, 2])
        Gallery.draw_vpsd2d(psd_obj, cut)

        plt.sca(ax[1, 0])
        Gallery.draw_centroid_as_func_of_position(psd_obj)

        plt.sca(ax[1, 1])
        Gallery.draw_ppsd2d(psd_obj, cut)

        plt.sca(ax[1, 2])
        Gallery.draw_ppsd_as_func_of_lightGM(psd_obj, 'light_GM > 1')

        plt.sca(ax[2, 0])
        Gallery.draw_figure_of_merits(psd_obj, cut)

        plt.sca(ax[2, 1])
        Gallery.draw_ppsd_as_func_of_position(psd_obj, cut)

        plt.sca(ax[2, 2])
        Gallery.draw_ppsd1d(psd_obj, cut)

        plt.draw()
        if save:
            fig.savefig(path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def draw_cfast_total(psd_obj, side, cut=None, ax=None):
        """Draw centered-fast v.s. total.

        Centered-fast, or cfast, is defined as the residual of fast w.r.t. to
        the center line.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        side : str
            The side of the bar, either 'L' or 'R'.
        cut : str, default None
            The cut to apply to the data to be plotted. This has nothing to do
            with the actual data used for analysis.
        ax : matplotlib.axes.Axes
            Default None, and `plt.gca()` is used.
        """
        # prepare data for plotting
        total, fast = f'total_{side}', f'fast_{side}'
        if cut is None:
            subdf = psd_obj.copy()[[total, fast]]
        else:
            subdf = psd_obj.df.query(cut)[[total, fast]]
        subdf['cfast'] = subdf[fast] - psd_obj.center_line[side](subdf[total])

        if ax is not None:
            plt.sca(ax)

        # plot the 2D histogram of cfast-total
        x_range = [0, 4000]
        y_range = [-150, 200]
        h = fh.plot_histo2d(
            plt.hist2d,
            subdf[total], subdf['cfast'],
            range=[x_range, y_range],
            bins=[500, 175],
            cmap=plt.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=plt.gca(), pad=-0.02, fraction=0.08, aspect=50.0)

        # plot the cfast-total relation
        # to emulate a colored line with black edges, we draw two overlapping lines
        # this is to improve the visibility
        total_plt = np.linspace(*x_range, 200)
        for particle, ft in psd_obj.cfast_total[side].items():
            plt.plot(total_plt, ft(total_plt), color='black', linewidth=1.5, zorder=10)
            plt.plot(total_plt, ft(total_plt), color='gold', linewidth=1.2, zorder=20)
        # use a different color for the linear segment of neutron line
        total_plt = np.linspace(psd_obj.x_switch_neutron, x_range[1], 100)
        ft = psd_obj.cfast_total[side]['neutron']
        plt.plot(total_plt, ft(total_plt), color='pink', linewidth=1.2, zorder=20)
        cfast_total = psd_obj.cfast_total[side]
        plt.plot(
            [psd_obj.x_switch_neutron] * 2,
            [y_range[0], 0.5 * (cfast_total['neutron'](psd_obj.x_switch_neutron) + cfast_total['gamma'](psd_obj.x_switch_neutron))],
            color='deeppink', linewidth=1.2, linestyle='dashed', zorder=10,
        )

        # control points, including
        # 1. those that were used to establish the cfast-total relation (filled white)
        # 2. those that were disgarded because of bad Gaussian fit (filled red)
        # The edge color is determined by particle: navy for neutron, darkgreen for gamma.
        for particle, ctrl_pts in psd_obj.ctrl_pts[side].items():
            # style by particles
            if particle == 'neutron':
                kw = dict(fmt='o', color='navy')
            if particle == 'gamma':
                kw = dict(fmt='s', color='darkgreen')

            common_kw = dict(linewidth=0.6, zorder=100)
            
            # control points used in fitting cfast-total relation
            cpoints = ctrl_pts.query('valid == True')
            plt.errorbar(
                cpoints['total'], np.clip(cpoints['fast'], *y_range),
                markerfacecolor='white', markersize=3,
                **kw, **common_kw
            )
        
            # control points that were disgarded
            cpoints = ctrl_pts.query('valid == False')
            plt.errorbar(
                cpoints['total'], np.clip(cpoints['fast'], *y_range),
                markerfacecolor='red', markersize=4,
                **kw, **common_kw
            )
        
        # draw vertical lines to indicate slice segments
        sty = dict(color='black', linewidth=1.2, linestyle='dashed', zorder=10)
        plt.axvline(100.0, **sty)
        plt.axvline(psd_obj.ft_breakpoint2, **sty)
        plt.axvline(psd_obj.ft_breakpoint1, **sty)
        plt.axvline(4000.0, **sty)

        # annotate the score for data quality
        kw = dict(
            xycoords='axes fraction',
            ha='left',
            va='top',
        )
        fitter = psd_obj.fitter
        plt.annotate(
            '\n'.join([
                r'$\gamma$: ' + f'{fitter[side]["gamma"].score_of_data:.3f}',
                r'$n$: ' + f'{fitter[side]["neutron"].score_of_data:.3f}',
            ]),
            (0, 0),
            xytext=(0.05, 0.95),
            xycoords='axes fraction',
            ha='left',
            va='top',
            linespacing=0.8,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
        )

        # final styling
        plt.xlim(x_range)
        plt.ylim(y_range)
        plt.xlabel(f'TOTAL-{side}')
        plt.ylabel(f'Centered FAST-{side}')

    @staticmethod
    def draw_vpsd2d(psd_obj, cut=None, ax=None):
        """Draw two-dimensional VPSD plot.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        cut : str, default None
            The cut to apply to the data to be plotted. This has nothing to do
            with the actual data used for analysis.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        # prepare data for plotting
        if cut is None:
            subdf = psd_obj.copy()[['vpsd_L', 'vpsd_R']]
        else:
            subdf = psd_obj.df.query(cut)[['vpsd_L', 'vpsd_R']]

        if ax is not None:
            plt.sca(ax)

        # plot the 2D VPSD histogram
        h = fh.plot_histo2d(
            plt.hist2d,
            subdf['vpsd_L'], subdf['vpsd_R'],
            range=[[-2, 3], [-2, 3]],
            bins=[250, 250],
            cmap=plt.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=plt.gca(), pad=-0.02, fraction=0.08, aspect=50.0)

        # plot position correction curves
        pos_pars = psd_obj.position_correction_params # shorthand
        positions = np.linspace(-100, 100, 500)
        for particle, curve in pos_pars['centroid_curves'].items():
            centroids = curve(positions)
            color = 'navy' if particle == 'neutron' else 'green' # gamma

            # emulate a golden line with black edges, to improve the visibility
            x, y = centroids[:, 0], centroids[:, 1]
            plt.plot(x, y, color='black', linewidth=1.5, zorder=10)
            plt.plot(x, y, color=color, linewidth=1.2, zorder=20)
        
        # final styling
        plt.xlim(-2, 3)
        plt.ylim(-2, 3)
        plt.xlabel(r'VPSD-L $v^{(\mathrm{L})}$')
        plt.ylabel(r'VPSD-R $v^{(\mathrm{R})}$')

    @staticmethod
    def draw_centroid_as_func_of_position(psd_obj, ax=None):
        """Draw centroid as a function of position.

        "Centroid curve" refers to the trajectory of centroids in the 2D VPSD
        plot when we vary the hit position. In this function, we will be
        plotting four curves in total - two for neutron and two for gamma. For
        each particle, one curve is plotting VPSD-L and the another is plotting
        VPSD-R.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        # ax_L will be used to plot VPSD-L (y ticks on left)
        # ax_R will be used to plot VPSD-R (y ticks on right)
        ax_L = plt.gca() if ax is None else ax
        ax_R = ax_L.twinx()

        curves = psd_obj.position_correction_params['centroid_curves']
        for particle, curve in curves.items():
            # particle-dependent attributes
            if particle == 'neutron':
                color = 'navy'
                symb = r'n'
            if particle == 'gamma':
                color = 'green'
                symb = r'\gamma'

            # plot the centroids (dots)
            centroids = curve(curve.x)
            kw = dict(color=color, markersize=5, linewidth=0.6)
            ax_L.errorbar(
                curve.x, centroids[:, 0],
                fmt='*',
                label=r'$v^{(\mathrm{L})}_%s$' % symb,
                **kw,
            )
            ax_R.errorbar(
                curve.x, centroids[:, 1],
                fmt='P',
                markerfacecolor='white',
                label=r'$v^{(\mathrm{R})}_%s$' % symb,
                **kw,
            )

            # plot the centroid curves (interpolation)
            pos_plt = np.linspace(-110, 110, 500)
            c_plt = curve(pos_plt)
            ax_L.plot(pos_plt, c_plt[:, 0], **kw)
            ax_R.plot(pos_plt, c_plt[:, 1], linestyle='dashed', **kw)
        
        # final styling
        ax_L.set_xlabel(f'NW{psd_obj.AB} hit position ' + r'$x$' + ' (cm)')
        ax_L.set_ylabel(r'VPSD-L $v^{(\mathrm{L})}$')
        ax_R.set_ylabel(r'VPSD-R $v^{(\mathrm{R})}$')
        ax_L.set_xlim(-120, 120)
        ax_L.set_ylim(ax_L.get_ylim()[0], 1.1 * np.diff(ax_L.get_ylim())[0] + ax_L.get_ylim()[0])
        ax_R.set_ylim(ax_R.get_ylim()[0], 1.1 * np.diff(ax_R.get_ylim())[0] + ax_R.get_ylim()[0])
        kw = dict(
            labelspacing=0.0,
            handlelength=0.3,
            handletextpad=0.2,
            borderaxespad=0.3,
            ncol=2,
            columnspacing=0.8,
        )
        ax_L.legend(loc='upper left', **kw)
        ax_R.legend(loc='upper right', **kw)

    @staticmethod
    def draw_ppsd2d(psd_obj, cut=None, ax=None):
        """Draw two-dimensional PPSD histogram.

        The x-axis is the PPSD value, and the y-axis is the perpendicular-PPSD
        value.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        cut : str, default None
            The cut to apply to the PSD object.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        # prepare data for plotting
        if cut is None:
            subdf = psd_obj.df.copy()[['ppsd', 'ppsd_perp']]
        else:
            subdf = psd_obj.df.query(cut)[['ppsd', 'ppsd_perp']]

        if ax is not None:
            plt.sca(ax)

        # plot two-dimensional histogram
        h = fh.plot_histo2d(
            plt.hist2d,
            subdf['ppsd'], subdf['ppsd_perp'],
            range=[[-2, 3], [-2, 2]],
            bins=[250, 200],
            cmap=plt.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=plt.gca(), pad=-0.02, fraction=0.08, aspect=50.0)

        # final styling
        plt.xlim(-2, 3)
        plt.ylim(-2, 2)
        plt.xlabel('PPSD')
        plt.ylabel('PPSD-perpendicular')

    @staticmethod
    def draw_ppsd_as_func_of_lightGM(psd_obj, cut=None, ax=None):
        """Draw PPSD as a function of light_GM (geometric mean)

        This plot can give us a sense of how well can we trust the PPSD at low
        light_GM. The lower the light_GM value, the harder to separate neutrons
        and gammas.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        cut : str, default None
            The cut to apply to the PSD object. It is suggested to use some low
            threshold, e.g. ``light_GM > 1``, so that we can inspect the
            relation as low as 1 MeVee; the default threshold for data analysis,
            however, is empirically established to be at least 3 MeVee or above.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        if cut is None:
            subdf = psd_obj.df.copy()[['ppsd', 'light_GM']]
        else:
            subdf = psd_obj.df.query(cut)[['ppsd', 'light_GM']]

        if ax is not None:
            plt.sca(ax)

        # plot two-dimensional histogram
        h = fh.plot_histo2d(
            plt.hist2d,
            subdf['ppsd'], subdf['light_GM'],
            range=[[-3, 4], [0, 60]],
            bins=[350, 300],
            cmap=plt.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=plt.gca(), pad=-0.02, fraction=0.08, aspect=50.0)
        plt.fill_betweenx([0, 3], -4, 5, color='silver', alpha=0.6, edgecolor='black', linewidth=1.5)

        # plot a vertical separation line for neutron and gamma (PPSD = 0.5)
        # we emulate a golden line with black edge by drawing two overlapping lines
        plt.axvline(0.5, color='black', linewidth=1.2, zorder=10)
        plt.axvline(0.5, color='gold', linewidth=0.9, zorder=20)

        # final styling
        plt.xlim(-3, 4)
        plt.ylim(0, 60)
        plt.xlabel('PPSD')
        plt.ylabel('G.M. light (MeVee)')
    
    @staticmethod
    def draw_ppsd_as_func_of_position(psd_obj, cut=None, ax=None):
        """Draw PPSD as a function of position.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        cut : str, default None
            The cut to apply to the PSD object.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        if cut is None:
            subdf = psd_obj.df.copy()[['pos', 'ppsd']]
        else:
            subdf = psd_obj.df.query(cut)[['pos', 'ppsd']]

        if ax is not None:
            plt.sca(ax)

        # plot the two-dimensional histogram
        h = fh.plot_histo2d(
            plt.hist2d,
            subdf['pos'], subdf['ppsd'],
            range=[[-120, 120], [-3, 4]],
            bins=[240, 350],
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        plt.colorbar(h[3], ax=plt.gca(), pad=-0.02, fraction=0.08, aspect=50.0)
        plt.axhline(0.5, color='black', linewidth=1.2, zorder=10)
        plt.axhline(0.5, color='gold', linewidth=0.9, zorder=20)
        
        # design
        plt.xlim(-120, 120)
        plt.ylim(-3, 4)
        plt.xlabel(f'NW{psd_obj.AB} hit position ' + r'$x$' + ' (cm)')
        plt.ylabel('PPSD')
    
    @staticmethod
    def draw_ppsd1d(psd_obj, cut=None, ax=None):
        """Draw one-dimensional histogram of PPSD.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        cut : str, default None
            The cut to apply to the PSD object.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        if cut is None:
            subdf = psd_obj.df.copy()[['ppsd']]
        else:
            subdf = psd_obj.df.query(cut)[['ppsd']]

        if ax is not None:
            plt.sca(ax)
        
        # plot the histogram
        fh.plot_histo1d(
            plt.hist,
            subdf['ppsd'],
            range=[-3, 4],
            bins=700,
            histtype='step',
            color='navy',
            density=True,
        )
        plt.axvline(0.5, color='gold', linewidth=1.0)

        # design
        plt.xlim(-3, 4)
        plt.ylim(0, )
        plt.xlabel('PPSD')
        plt.ylabel('Probability density')
    
    @staticmethod
    def draw_figure_of_merits(psd_obj, cut=None, ax=None):
        """Draw figure of merit as functions of light_GM and position.

        Parameters
        ----------
        psd_obj : PulseShapeDiscriminator instance
            The PSD object to draw.
        cut : str, default None
            The common cut to apply to the PSD object. Additional cuts are added
            when constructing each F.O.M. point: ``-3 < ppsd < 4`` and
            ``-1 < ppsd_perp < 1``.
        ax : matplotlib.axes.Axes, default None
            Default None, and `plt.gca()` is used.
        """
        if cut is None:
            subdf = psd_obj.df.copy()[['ppsd', 'ppsd_perp', 'pos', 'light_GM']]
        else:
            subdf = psd_obj.df.query(cut)[['ppsd', 'ppsd_perp', 'pos', 'light_GM']]

        if ax is None:
            ax = plt.gca()
        ax_low = ax # for position
        ax_upp = ax.twiny() # for light_GM

        # calculate FOM as function of light GM
        light_GM_ranges = psd_obj.create_ranges(3, 60, width=10, step=5)
        df_light_fom = []
        for light_GM_range in light_GM_ranges:
            subdf_light = subdf.query(' & '.join([
                f'{light_GM_range[0]} <= light_GM < {light_GM_range[1]}',
                '-3 < ppsd < 4',
                '-1 < ppsd_perp < 1',
            ]))
            
            # if too few events, skip
            if len(subdf_light) < 500:
                continue

            light_mean = np.mean(light_GM_range)
            fom = psd_obj.figure_of_merit(subdf_light['ppsd'])
            df_light_fom.append([light_mean, fom])
        df_light_fom = pd.DataFrame(df_light_fom, columns=['light_GM', 'fom'])

        # plot FOM as a function of light GM with the upper x-axis
        ax_upp.errorbar(
            df_light_fom['light_GM'], df_light_fom['fom'],
            fmt='o-', color='crimson', linestyle='dashed', linewidth=0.8,
        )
        ax_upp.set_xlim(0, 60)
        ax_upp.set_ylim(0.5, 1.5)
        ax_upp.set_xlabel('G.M. light (MeVee)', color='crimson')
        ax_upp.set_ylabel('Figure of merit')

        # calculate FOM as a function of position
        pos_ranges = psd_obj.create_ranges(-100, 100, width=30, step=15)
        df_pos_fom = []
        for pos_range in pos_ranges:
            subdf_pos = subdf.query(' & '.join([
                f'{pos_range[0]} <= pos < {pos_range[1]}',
                '-3 < ppsd < 4',
                '-1 < ppsd_perp < 1',
            ]))
            pos_mean = np.mean(pos_range)
            fom = psd_obj.figure_of_merit(subdf_pos['ppsd'])
            df_pos_fom.append([pos_mean, fom])
        df_pos_fom = pd.DataFrame(df_pos_fom, columns=['pos', 'fom'])

        # plot FOM as a funciton of position with the lower x-axis
        ax_low.errorbar(
            df_pos_fom['pos'], df_pos_fom['fom'],
            fmt='^-', markerfacecolor='white', color='navy', linewidth=0.8,
        )
        ax_low.set_xlim(-120, 120)
        ax_low.set_xlabel(f'NW{psd_obj.AB} hit position ' + r'$x$' + ' (cm)', color='navy')
        ax_low.set_ylabel('Figure-of-merit')



class _MainUtilities:
    """Functions, classes and attributes for using this module as a script."""
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            description='Pulse shape discrimination for neutron wall',
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            'AB',
            type=str,
            help='"A" or "B", this selects NWA or NWB'
        )
        parser.add_argument(
            'runs',
            nargs='+',
            help=inspect.cleandoc('''
                Runs to calibrate.

                Usually the script needs at least five runs to give reliable
                calibrations, otherwise there is not enough statistics.
                Consecutive runs can be specified in ranges separated by the
                character "-". Here is an example:
                    > ./pulse_shape_discrimination.py B 8-10 11 20 2-5
                This will calibrate the runs 8, 9, 10, 11, 20, 2, 3, 4, 5, on
                NWB.
            '''),
        )
        parser.add_argument(
            '-b', '--bars',
            nargs='+',
            help=inspect.cleandoc('''
                The bar number(s) to calibrate. If not specfiied, all bars,
                including bar 1 to 24, will be analyzed. Dash "-" can be used to
                specify ranges, e.g. "1-3 10-12" would make the program
                analyzes bars 1, 2, 3, 10, 11, 12.
            '''),
            default=['1-24'],
        )
        parser.add_argument(
            '-c', '--no-cache',
            help=inspect.cleandoc('''
                When this option is given, the script will ignore the HDF5 cache
                files. All data will be read from the ROOT files. New cache
                files will then be created. By default, the script will use the
                cache.
            '''),
            action='store_true',
        )
        parser.add_argument(
            '-d', '--debug',
            help=inspect.cleandoc('''
                When this option is given, no output would be saved. This
                includes both calibration parameters and gallery. The cached
                data is not controlled by this option; for that, refer to "-c"
                or "--no-cache".
            '''),
            action='store_true',
        )
        parser.add_argument(
            '-o', '--output',
            help=inspect.cleandoc('''
                The output directory. If not given, the default is
                "$PROJECT_DIR/database/neutron_wall/pulse_shape_discrimination/".
            '''),
            default=str(PROJECT_DIR / 'database/neutron_wall/pulse_shape_discrimination'),
        )
        parser.add_argument(
            '-s', '--silence',
            help='To silent all status messages.',
            action='store_true',
        )
        parser.add_argument(
            '--ft-breakpoint1',
            type=float,
            help=inspect.cleandoc('''
                The breakpoint 1 for the fast-total fitting ranges. Default is
                1500.0. From this breakpoint onward till the second breakpoint,
                a smoother convolution is applied (due to lower statistics) to
                make the peak finding more stable.
            '''),
        )
        parser.add_argument(
            '--ft-breakpoint2',
            type=float,
            help=inspect.cleandoc('''
                The breakpoint 2 for the fast-total fitting ranges. Default is
                2500.0. If gamma statistics is insufficient at high ADC,
                reducing this value may improve the overall fit. No gamma peak
                finding is performed after this breakpoint.
            '''),
        )
        parser.add_argument(
            '--min-samples-gamma',
            type=float,
            help=inspect.cleandoc('''
                The minimum number of samples for gamma to pass a RANSAC fit.
                Default is 0.25.
            '''),
        )
        parser.add_argument(
            '--min-samples-neutron',
            type=float,
            help=inspect.cleandoc('''
                The minimum number of samples for neutron to pass a RANSAC fit.
                Default is 0.70.
            '''),
        )
        parser.add_argument(
            '--x-switch-neutron',
            type=float,
            help=inspect.cleandoc('''
                The switching point for neutron fast-total relation from
                quadratic to linear. Default is 1300.0.
            '''),
        )
        args = parser.parse_args()

        # process the wall type
        args.AB = args.AB.upper()
        if args.AB not in ['A', 'B']:
            raise ValueError(f'Invalid wall type: "{args.AB}"')
        
        # process the runs
        runs = []
        for run_str in args.runs:
            run_range = [int(run) for run in run_str.split('-')]
            if len(run_range) == 1:
                runs.append(run_range[0])
            elif len(run_range) == 2:
                runs.extend(range(run_range[0], run_range[1] + 1))
            else:
                raise ValueError(f'Unrecognized input: {run_str}')
        args.runs = runs

        # process the bars
        bars = []
        for bar_str in args.bars:
            bar_range = [int(bar) for bar in bar_str.split('-')]
            if len(bar_range) == 1:
                bars.append(bar_range[0])
            elif len(bar_range) == 2:
                bars.extend(range(bar_range[0], bar_range[1] + 1))
            else:
                raise ValueError(f'Unrecognized input: {bar_str}')
        args.bars = bars

        return args

if __name__ == '__main__':
    """
    To use this as a script for analyzing PSD on NWB from, say, run 1000 to
    1100, just type on the terminal:

    .. code-block:: console
        $ python pulse_shape_discrimination.py B 1000-1100

    Use the flag ``-h`` or ``--help`` to see the list of available options.
    """
    import argparse
    import inspect
    
    args = _MainUtilities.get_args()

    PulseShapeDiscriminator.database_dir = pathlib.Path(args.output)
    PulseShapeDiscriminator.update_hyperparameters(args)
    psd = PulseShapeDiscriminator(args.AB)

    read_verbose = (not args.silence)
    for bar in args.bars:
        psd.read(
            run=args.runs,
            bar=bar,
            from_cache=(not args.no_cache),
            verbose=read_verbose,
        )
        read_verbose = False # only show read status once

        if not args.silence:
            print(f'Calibrating NW{args.AB}-bar{bar:02d}...', end='', flush=True)
        
        psd.fit()

        if not args.debug:
            psd.save_parameters()
            Gallery.save_as_png(psd)

        print(' Done', flush=True)