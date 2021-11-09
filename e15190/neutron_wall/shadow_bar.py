import concurrent.futures
import pathlib
from lmfit.model import propagate_err

import numpy as np
import pandas as pd
import uproot

from lmfit import Model, Parameters
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import erf
from e15190 import PROJECT_DIR
from e15190.utilities import fast_histogram as fh
from e15190.runlog.query import Query


class ShadowBar:
    database_dir = PROJECT_DIR / 'database/neutron_wall/shadow_bar'
    root_files_dir = PROJECT_DIR / 'database/root_files'
    light_GM_range = [5.0, 500.0]  # MeVee
    pos_range = [-120.0, 120.0]  # cm
    psd = [0.5, 2]

    def __init__(self, AB, max_workers=8):
        """Initialize the :py:class:`ShadowBar` class.

        Parameters
        ----------
        AB : str
            Either 'A' or 'B'. To specify neutron wall A or B.
        max_workers : int, default 12
            Maximum number of workers to use for parallelization. The parallelization
            is used to read in data from ROOT files as executors for
            `decompression <https://uproot.readthedocs.io/en/latest/uproot.reading.ReadOnlyFile.html#decompression-executor>`__
            and
            `interpretation <https://uproot.readthedocs.io/en/latest/uproot.reading.ReadOnlyFile.html#interpretation-executor>`__
            in
            `uproot <https://uproot.readthedocs.io/>`__.
        """
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.decompression_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
        self.interpretation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
        self.database_dir.mkdir(parents=True, exist_ok=True)

    def _cut_for_root_file_data(self, AB):
        """A simple cut to select valid entries only.

        This is not the final cut. It is only the biggest cut that defines the
        largest subset of data that will be used, i.e. we only throw away
        entries that are not useable or recoverable. Data that are not of our
        *interest* should *not* be thrown away here.
        """
        AB = AB.upper()
        cuts = [
            f'NW{AB}_light_GM > {self.light_GM_range[0]}',
            f'NW{AB}_light_GM < {self.light_GM_range[1]}',
            f'NW{AB}_pos > {self.pos_range[0]}',
            f'NW{AB}_pos < {self.pos_range[1]}',
            f'NW{AB}_psd >{self.psd[0]}',
            f'NW{AB}_psd <{self.psd[1]}',
        ]
        return ' & '.join([f'({c.strip()})' for c in cuts])

    def read_run_from_root_file(self, run, tree_name=None, apply_cut=True):
        """Read in single run from ROOT file.

        The ROOT file here refers to the one generated by ``calibrate.cpp``,
        *not* the "Daniele's ROOT file".

        Parameters
        ----------
        run : int
            Run number.
        tree_name : str, default None
            Name of the tree in the ROOT file. If not specified, the function
            will try to automatically determine the tree name. If multiple trees
            are found within the ROOT file, an exception will be raised as the
            function has no way to know which tree to use.
        apply_cut : bool, default True
            If set to `True`, the function will apply the cut specified by
            :py:func:`_cut_for_root_file_data` to the data. Otherwise, the cut
            is not applied.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing the data for the specified run.
        """
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
        branches = [
            'MB_multi',
            'VW_multi',
            f'NW{self.AB}_bar',
            f'NW{self.AB}_light_GM',
            f'NW{self.AB}_pos',
            f'NW{self.AB}_theta',
            f'NW{self.AB}_phi',
            f'NW{self.AB}_psd',
            f'NW{self.AB}_distance',
            f'NW{self.AB}_time',
            'FA_time_min',
        ]
        with uproot.open(str(path) + ':' + tree_name) as tree:
            df = tree.arrays(
                branches,
                library='pd',
                decompression_executor=self.decompression_executor,
                interpretation_executor=self.interpretation_executor,
            )

        if apply_cut:
            df = df.query(self._cut_for_root_file_data(self.AB))

        # calculate TOF from NW{self.AB}_time - FA_time_min
        df[f'NW{self.AB}_tof'] = df.eval(f'NW{self.AB}_time - FA_time_min')
        # calculate kinetic energy from TOF and NW{self.AB}_distance
        df[f'NW{self.AB}_energy'] = np.power(
            df[f'NW{self.AB}_distance']/df[f'NW{self.AB}_tof'], 2)*0.5*939.565420*(1.0/9.0)*0.01
        # drop branches that won't be used,
        # e.g. NW{self.AB}_distance, NW{self.AB}_time, FA_time_min
        df.drop([f'NW{self.AB}_distance', f'NW{self.AB}_time',
                 'FA_time_min'], axis=1, inplace=True)

        return df

    def cache_run(self, run, tree_name=None):
        """Read in data from ROOT file and save relevant branches to an HDF5 file.

        The data will be grouped according to bar number, because future
        retrieval by this class will most likely analyze only one bar at a time.

        Parameters
        ----------
        run : int
            Run number.
        tree_name : str, default None
            Name of the tree in the ROOT file. If not specified, the function
            will try to automatically determine the tree name. If multiple trees
            are found within the ROOT file, an exception will be raised as the
            function has no way to know which tree to use.
        """
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.read_run_from_root_file(run, tree_name=tree_name)

        # convert all float64 columns into float32
        for col in df.columns:
            if df[col].dtype == np.float64:
                df[col] = df[col].astype(np.float32)

        # write cache to HDF5 files bar by bar
        columns = [col for col in df.columns if col !=
                   f'NW{self.AB}_bar']  # drop bar number
        # remove pre-existing file, if any
        pathlib.Path(path).unlink(missing_ok=True)
        for bar, subdf in df.groupby(f'NW{self.AB}_bar'):
            subdf.reset_index(drop=True, inplace=True)
            with pd.HDFStore(path, mode='a') as file:
                file.put(f'nw{self.ab}{bar:02d}',
                         subdf[columns], format='fixed')

    def _read_single_run(self, run, bar, from_cache=True):
        """Read in single run from either cache or ROOT file.

        If no cache is available or ``from_cache`` is set to ``False``, the
        function first prepare a cache for the run by invoking
        :py:func:`cache_run`, then read in the data from the newly created cache
        into a dataframe to be returned. Otherwise, it reads directly from the
        cache and returns the dataframe.

        Notice that the cache file is always generated for one run at a time,
        with multiple dataframes contained in it, each dataframe corresponding
        to one bar. In other words, if you have invoked
        ``_read_single_run(6666, 1)``,
        then a cache file ``cache/run-6666.h5`` will be generated, and in it
        there are dataframes for not just bar-01, but also *all* the other bars
        (of course, excluding those that you had chosen to filter out).

        Parameters
        ----------
        run : int
            Run number.
        bar : int
            Bar number.
        from_cache : bool, default True
            If set to `True`, the data will be read from the cache if exists.
            Otherwise, it will refresh the cache by reading again from the ROOT
            file, then read from the newly created cache.

        Returns
        -------
        df : pd.DataFrame
            Dataframe containing the data for the specified run and bar.
        """
        path = self.database_dir / f'cache/run-{run:04d}.h5'
        if not from_cache or not path.exists():
            self.cache_run(run)

        with pd.HDFStore(path, mode='r') as file:
            df = file.get(f'nw{self.ab}{bar:02d}')
        return df

    def read(self, run, bar, from_cache=True, verbose=False):
        """Read in the data needed to do shadow bar analysis.

        This function reads in the data as a dataframe, and saves it to
        ``self.df``. It also updates ``self.bar`` for future reference.

        The data will be read from the HDF5 files if they exists, otherwise the
        function will read in from the ROOT files (generated by
        ``calibrate.cpp``).

        Branch names that start with NW{self.AB}_ will be dropped because of
        redunancy. For example, ``NWB_psd`` will be reduced into ``psd``.
        However, other prefixes that indicate detectors other than the neutron
        wall will be kept, e.g. ``MB_multi``, ``FA_time_min``, etc.

        Parameters
        ----------
        run : int or list of ints
            The run number(s).
        bar : int
            Bar number.
        from_cache : bool, default True
            If set to `True`, the data will be read from the cache if exists.
            Otherwise, it will refresh the cache by reading again from the ROOT
            file, then read from the newly created cache.
        verbose : bool, default False
            Whether to print out the progress of the read in.

        Example
        -------
        >>> from e15190.neutron_wall.shadow_bar import ShadowBar
        >>> shade = ShadowBar('B') # ShadowBar object for neutron wall B
        >>> shade.read([4083, 4084], 1, verbose=True) # read in run 4083 and 4084, bar-01
        Reading run-4083  (1/2)
        Reading run-4084  (2/2)
        >>> shade.df[['run', 'theta']] # display the columns run and theta for inspection
                run      theta
        0      4083  32.787231
        1      4083  33.294674
        2      4083  35.215828
        3      4083  40.720539
        4      4083  31.086285
        ...     ...        ...
        25253  4084  38.263584
        25254  4084  35.900543
        25255  4084  40.574760
        25256  4084  31.906084
        25257  4084  49.476383
        <BLANKLINE>
        [25258 rows x 2 columns]

        """
        if isinstance(run, int):
            runs = [run]
        else:
            runs = run

        df = None
        for i_run, run in enumerate(runs):
            if verbose:
                print(
                    f'\rReading run-{run:04d}  ({i_run + 1}/{len(runs)})', end='', flush=True)
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
        self.df.columns = [name.replace(
            f'NW{self.AB}_', '') for name in self.df.columns]

    def remove_vetowall_coincidences(self):
        self.df = self.df.query('VW_multi == 0')
        self.df.drop('VW_multi', axis=1, inplace=True)

    def gate_on_tof(self):
        self.df = self.df.query('tof>0 & tof<250')
        self.df.drop('tof', axis=1, inplace=True)

    def preprocessing(self):
        self.remove_vetowall_coincidences()
        self.gate_on_tof()

    def make_space_above(ax, topmargin=1):
        """ increase figure size to make topmargin (in inches) space for 
        titles, without changing the axes sizes"""
        fig = ax.flatten()[0].figure
        s = fig.subplotpars
        w, h = fig.get_size_inches()

        figh = h - (1-s.top)*h + topmargin
        fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
        fig.set_figheight(figh)

    def fit_energy_range(self, ene, background_position):
        subdf = self.df[(self.df['energy'] > ene-10) &
                        (self.df['energy'] < ene+10)]

        # Backward Angle (range theta >40 and <48 backward angle) (range>32 and <40 forward angle)
        if background_position == 'F':
            subdf = subdf[(subdf['theta'] > 32) & (subdf['theta'] < 40)]
        elif background_position == 'B':
            subdf = subdf[(subdf['theta'] > 40) & (subdf['theta'] < 48)]
        else:
            print('shadow_bar position is not correct')

        fig, ax = plt.subplots(dpi=120, figsize=(4, 3))
        ax.set_xlabel(r'Lab $\theta$')
        ax.set_ylabel(r'Density')
        if background_position == 'F':
            y, x, _ = ax.hist(subdf['theta'], range=[
                              32, 40], bins=50, histtype='step', density='true')
        elif background_position == 'B':
            y, x, _ = ax.hist(subdf['theta'], range=[
                              40, 48], bins=50, histtype='step', density='true')
        else:
            print('shadow_bar position is not correct')
        x = 0.5 * (x[1:] + x[:-1])
        y_err = np.sqrt(y)

        return x, y, y_err

    def fit_parameters(self, background_position, x, y, y_err):
        def f_kin1(x, k0, k1):
            return k0 + k1 * x

        def f_dip1(x, A, x_L, x_R, s_L, s_R):
            return A*(-erf((x-x_L)/(s_L)) + erf((x-x_R)/(s_R)))

        def f_fit_total(x, k0, k1, A, x_L, x_R, s_L, s_R):
            return k0 + k1 * x + A*(1.0+0.50*-erf((x-x_L)/(np.sqrt(2)*s_L)) + 0.5*erf((x-x_R)/(np.sqrt(2)*s_R)))

        param = Parameters()
        # backward angles
        if background_position == 'B':
            param.add('k0', value=1, min=-2.0, max=2.0)
            param.add('k1', value=1, min=-2.0, max=2.0)
            param.add('A', value=10, min=-10.0, max=40.0)
            param.add('x_L', value=42, min=40.0, max=44.0)
            param.add('x_R', value=45, min=44.0, max=48.0)
            param.add('s_L', value=1, min=0.0, max=2.0)
            param.add('s_R', value=1, min=0.0, max=2.0)

        # forward angles
        elif background_position == 'F':
            param.add('k0', value=1, min=-2.0, max=2.0)
            param.add('k1', value=1, min=-2.0, max=2.0)
            param.add('A', value=10, min=-10.0, max=40.0)
            param.add('x_L', value=34, min=30.0, max=35.0)
            param.add('x_R', value=37, min=35.0, max=40.0)
            param.add('s_L', value=1, min=0.0, max=2.0)
            param.add('s_R', value=1, min=0.0, max=2.0)
        else:
            print('shadow_bar position is not correct')

        gmodel = Model(f_kin1)+Model(f_dip1)
        w = np.divide(
            1,
            y_err,
            where=(y_err != 0),
            out=np.zeros(len(y_err)),
        )
        result = gmodel.fit(y, param, x=x, weights=w, calc_covar=True,
                            method='least_squares', nan_policy='propagate')
        #result = gmodel.fit(y,param,x=x,weights=w,calc_covar=True)
        print(result.fit_report())

        # calcualte the 1-sigma error bar for kinematic part of model at each x
        # def y_error(x,k0,k1,delko,delk1,covk0k1):
        #     return np.sqrt(np.exp(-x/k1)**2*delko**2+(k0*x*np.exp(-x/k1)/k1**2)**2*delk1**2+2*k0*x*(np.exp(-x/k1))**2/k1**2*covk0k1)
        # err_y=y_error(x,result.params['k0'].value,result.params['k1'].value,result.params['k0'].stderr,result.params['k1'].stderr,covar1[0,1])

        return result

    def calculate_background(self, result, x, y):
        # evaluate components
        comps = result.eval_components(x=x)
        covar1 = result.covar

        def y_error(x, delko, delk1, covk0k1):
            return np.sqrt(delko**2+(x*delk1)**2+2*x*covk0k1)
        err_y = y_error(
            x, result.params['k0'].stderr, result.params['k1'].stderr, covar1[0, 1])

        # upper and lower components of kinematic function
        upper_kin = comps['f_kin1']+err_y
        lower_kin = comps['f_kin1']-err_y

        # eval 1-sigma uncertainty of the best fit
        dely = result.eval_uncertainty(x=x)
        background = np.amin(result.best_fit/comps['f_kin1'])
        background_upper = np.amin((result.best_fit+dely)/upper_kin)
        background_lower = np.amin((result.best_fit-dely)/lower_kin)
        background_err_ul = np.amin(
            (result.best_fit+dely)/upper_kin)-np.amin(result.best_fit/comps['f_kin1'])
        background_err_ll = np.amin(
            result.best_fit/comps['f_kin1'])-np.amin((result.best_fit-dely)/lower_kin)
        return err_y, background*1e2, 1e2*0.50*(background_err_ul+background_err_ll)

    def _draw_Energy_vs_lightGM(self, ax):
        # two-dimensional histogram
        h = fh.plot_histo2d(
            ax.hist2d,
            self.df['light_GM'], self.df['energy'],
            range=[[0, 100], [0, 120]],
            bins=[1000, 1200],
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm(vmin=1),
        )
        # design

        ax.set_ylabel('Energy (MeV)')
        ax.set_xlabel('G.M. light (MeVee)')

    def _draw_counts_vs_theta(self, ene, ax):
        subdf = self.df[(self.df['energy'] > ene-10) &
                        (self.df['energy'] < ene+10)]
        # one-dimensional histogram
        h = fh.plot_histo1d(
            ax.hist,
            subdf['theta'],
            range=[25, 55],
            bins=150,
            histtype='step',
            density='true',
        )
        # design

        ax.set_xlabel('theta (degree)')
        ax.set_ylabel('Counts')

    def _draw_counts_vs_position(self, ene, ax):
        subdf = self.df[(self.df['energy'] > ene-10) &
                        (self.df['energy'] < ene+10)]
        # one-dimensional histogram
        h = fh.plot_histo1d(
            ax.hist,
            subdf['pos'],
            range=[-110, 110],
            bins=220,
            histtype='step',
            density='true',
        )
        # design

        ax.set_xlabel('position (cm)')
        ax.set_ylabel('Counts')

    def _draw_fitfunction_vs_theta(self, x, y, result, err_y, ax):
        comps = result.eval_components(x=x)
        # upper and lower components of kinematic function
        upper_kin = comps['f_kin1']+err_y
        lower_kin = comps['f_kin1']-err_y
        dely = result.eval_uncertainty(x=x)
        ax.plot(x, y)
        ax.plot(x, result.best_fit)
        ax.plot(x, result.best_fit-dely, 'r--')
        ax.plot(x, result.best_fit+dely, 'm--')
        ax.plot(x, comps['f_kin1'], 'g--')
        ax.fill_between(x, result.best_fit-dely,
                        result.best_fit+dely, color='#888888')
        ax.plot(x, upper_kin, 'y--', label='upper limit')
        ax.plot(x, lower_kin, 'c--', label='lower limit')
        ax.set_ylabel(r'$Y_\mathrm{data}$')
        ax.set_xlabel(r'Lab $\theta$ (deg)')

    def _draw_background(self, x, y, result, err_y, ax):
        comps = result.eval_components(x=x)
        upper_kin = comps['f_kin1']+err_y
        lower_kin = comps['f_kin1']-err_y
        dely = result.eval_uncertainty(x=x)

        background = np.amin(result.best_fit/comps['f_kin1'])
        background_upper = np.amin((result.best_fit+dely)/upper_kin)
        background_lower = np.amin((result.best_fit-dely)/lower_kin)
        background_err_ul = np.amin(
            (result.best_fit+dely)/upper_kin)-np.amin(result.best_fit/comps['f_kin1'])
        background_err_ll = np.amin(
            result.best_fit/comps['f_kin1'])-np.amin((result.best_fit-dely)/lower_kin)
        ax.plot(x, result.best_fit/comps['f_kin1'], 'r--', label='normalized')
        ax.plot(x, (result.best_fit-dely)/lower_kin,
                'c--', label='lower limit')
        ax.plot(x, (result.best_fit+dely)/upper_kin,
                'y--', label='upper limit')
        ax.set_ylabel(r'$Y_\mathrm{data}(\theta)/Y_\mathrm{kin}(\theta)$')
        ax.set_xlabel(r'Lab $\theta$ (deg)')
        ax.axhline(background, linestyle='dashed', color='red',
                   label='background = %.1f%%' % (1e2*background))
        ax.axhline(background_upper, linestyle='dashed', color='yellow',
                   label='%.1f%%' % (1e2*background_upper))
        ax.axhline(background_lower, linestyle='dashed', color='cyan',
                   label='%.1f%%' % (1e2*background_lower))

    def save_to_gallery(self, sf, background_position, ene, x, y, result, err_y, path=None, show_plot=False, save=True):
        """Save a diagnostic plot to the gallery as a PNG file.

        Parameters
            path : str or pathlib.Path, default None
                The path to save the plot. If None, the plot is saved to the
                default database.
            cut : str, default 'light_GM > 3'
                The cut to apply to the data when plotting. All panels in the
                figure will apply this cut, and potentially other cuts, joined
                by logical AND, except for the panel that draws the PPSD as a
                function of light GM.
            show_plot : bool, default False
                If True, the plot will be shown in run time, i.e. the command
                `plt.show()` will be called.
            save : bool, default True
                If `True`, the plot will be saved to the database. `False`
                option is useful when users are using this function to inspect
                the plot without saving.
        """
        #filename = f'{self._run_hash_str}-NW{self.AB}-bar{self.bar:02d}.png'
        filename = '%s+%s_%.02d_MeV_bar_%02d_%s_ene_%.02d.png' % (
            sf['beam'], sf['target'], sf['beam_energy'], self.bar, background_position, ene)
        # filename=f'Ca48Sn124_bar_{self.bar}_{background_position}_{ene}.png'
        if path is None:
            path = self.database_dir / 'gallery' / filename
        elif isinstance(path, str):
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(
            10, 10), constrained_layout=True)
        fig.suptitle(
            f'Ca48Sn124_@140 MeV_bar_{self.bar}_{background_position}_Ene_{ene}+-10')

        rc = (0, 0)
        # self._draw_Energy_vs_lightGM(ax[rc])
        self._draw_counts_vs_position(ene, ax[rc])
        rc = (0, 1)
        self._draw_counts_vs_theta(ene, ax[rc])

        rc = (1, 0)
        self._draw_fitfunction_vs_theta(x, y, result, err_y, ax[rc])

        rc = (1, 1)
        self._draw_background(x, y, result, err_y, ax[rc])
        plt.draw()
        if save:
            fig.savefig(path, dpi=500, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

    def fit(self, background_position, path=None):
        self.preprocessing()
        fig, ax = plt.subplots(figsize=(10, 7))
        self._draw_Energy_vs_lightGM(ax)
        sf = Query.get_run_info(self.df['run'].iloc[0])
        filename = '%s+%s_%.02d_MeV_bar_%02d_%s.txt' % (
            sf['beam'], sf['target'], sf['beam_energy'], self.bar, background_position)
        if path is None:
            path = self.database_dir / 'gallery' / filename
        elif isinstance(path, str):
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as file:
            file.write('{} {} {} {} \n'.format(
                "bar", "energy", "background", "error"))

            for ene in np.arange(20, 110, 20):
                x, y, y_err = self.fit_energy_range(ene, background_position)
                result = self.fit_parameters(background_position, x, y, y_err)
                err_y, bg, bg_err = self.calculate_background(
                    self.fit_parameters(background_position, x, y, y_err), x, y)
                self.save_to_gallery(sf, background_position, ene, x, y,
                                     result, err_y, path=None, show_plot=False, save=True)
                file.write('{} {} {} {} \n'.format(self.bar, ene, bg, bg_err))


'Done'
