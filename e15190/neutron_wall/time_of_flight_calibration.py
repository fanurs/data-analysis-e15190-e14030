#!/usr/bin/env python
import inspect
import os
from pathlib import Path
import traceback
from typing import Dict, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT
from scipy.optimize import curve_fit

from e15190.neutron_wall import geometry as nwgeo
from e15190.neutron_wall.position_calibration import NWCalibrationReader
from e15190.utilities import dataframe_histogram as dfh, root6 as rt, slicer, styles, tables
from e15190.utilities.peak_finder import PeakFinderGaus1D

MPL_DEFAULT_BACKEND = mpl.get_backend()
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT_DECLARED = False # global variable to prevent multiple ROOT declarations

class TimeOfFlightCalibrator:
    DATABASE_DIR = '$DATABASE_DIR/neutron_wall/time_of_flight_calibration/'
    SPEED_OF_LIGHT_IN_AIR = 29.9702547 # cm / ns
    POS_RANGE = [-100, 100]
    DIST_RANGE_WIDTH = 8 # cm
    DIST_RANGE_N_STEPS = 9

    def __init__(self, AB, run, init_rdf=True):
        self.AB = AB.upper()
        self.ab = AB.lower()
        self.run = run

        self.lazy_items = dict()
        self.pg_fit_params = dict()
        self.tof_offset = dict()
        self.tof_offset_err = dict()

        if init_rdf:
            self.declare_all_to_root_interpreter()
            self.create_base_rdf()
    
    @staticmethod
    def get_input_path(run: int, directory=None, filename_fmt='CalibratedData_%04d.root') -> Path:
        if directory is None:
            directory = os.path.expandvars('$DATABASE_DIR/root_files_daniele')
        return Path(directory, filename_fmt % run)
    
    def _get_position_calibration_parameters(self) -> pd.DataFrame:
        reader = NWCalibrationReader(self.AB)
        self.df_pos_calib = reader(self.run)
        self.df_pos_calib.index.name = 'bar'
        return self.df_pos_calib
    
    def _get_empirical_distance_parameters(self) -> pd.DataFrame:
        wall = nwgeo.Wall(self.AB)
        df = []
        for b, bar in wall.bars.items():
            if b <= 0:
                continue
            fit_low, fit_upp = bar.get_empirical_distance_bounds()
            df.append([b, *fit_low, *fit_upp])
        df = pd.DataFrame(df, columns=['bar', 'low0', 'low1', 'low2', 'upp0', 'upp1', 'upp2'])
        self.df_emp_distance = df.set_index('bar', drop=True)
        return self.df_emp_distance

    def declare_all_to_root_interpreter(self):
        """Declare all relevant functions and variables to the ROOT interpreter.
    
        Functions that are declared are:
            - GetPositionParam0(bar)
            - GetPositionParam1(bar)
            - GetDistParam0(bar)
            - GetDistParam1(bar)
            - GetDistParam2(bar)
            - GetPosition(time_L, time_R, p0, p1)
            - GetDistance(pos, p0, p1, p2)
        
        This function is called only when ``ROOT_DECLARED`` is False. The
        variable is set to True after this function is called. To call this
        function again, user have to set ``ROOT_DECLARED`` to False manually,
        though such action would most likely cause an error in the ROOT
        interpreter due to multiple declarations.

        Parameters
        ----------
        p0 : np.array
            Array of intercepts. The indices have to match the bar numbers. Pad
            with zeros if necessary.
        p1 : np.array
            Array of slopes. The indices have to match the bar numbers. Pad with
            zeros if necessary.
        """
        self._get_position_calibration_parameters()
        self._get_empirical_distance_parameters()

        global ROOT_DECLARED
        if ROOT_DECLARED:
            return
        formatter = lambda x: ', '.join(map(str, x))
        ROOT.gInterpreter.Declare('''
            #pragma cling optimize(3)
            using RInt = ROOT::RVec<int>;
            using RDouble = ROOT::RVec<double>;

            const RDouble pos_param0 = {%s};
            const RDouble pos_param1 = {%s};
            const RDouble dist_param0 = {%s};
            const RDouble dist_param1 = {%s};
            const RDouble dist_param2 = {%s};

            RDouble GetPosParam0(RInt bar)  { return ROOT::VecOps::Take(pos_param0, bar - 1); }
            RDouble GetPosParam1(RInt bar)  { return ROOT::VecOps::Take(pos_param1, bar - 1); }
            RDouble GetDistParam0(RInt bar) { return ROOT::VecOps::Take(dist_param0, bar - 1); }
            RDouble GetDistParam1(RInt bar) { return ROOT::VecOps::Take(dist_param1, bar - 1); }
            RDouble GetDistParam2(RInt bar) { return ROOT::VecOps::Take(dist_param2, bar - 1); }

            RDouble GetPosition(RDouble time_L, RDouble time_R, RDouble p0, RDouble p1) {
                return p0 + p1 * (time_L - time_R);
            }

            RDouble GetDistance(RDouble pos, RDouble p0, RDouble p1, RDouble p2) {
                return p0 + p1 * pos + p2 * pos * pos;
            }
            ''' % tuple(map(formatter, [
                self.df_pos_calib['p0'],
                self.df_pos_calib['p1'],
                0.5 * (self.df_emp_distance['low0'] + self.df_emp_distance['upp0']),
                0.5 * (self.df_emp_distance['low1'] + self.df_emp_distance['upp1']),
                0.5 * (self.df_emp_distance['low2'] + self.df_emp_distance['upp2']),
            ]))
        )
        ROOT_DECLARED = True

    def create_base_rdf(self, multithread=True):
        path = self.get_input_path(self.run)
        tree_name = rt.infer_tree_name(path)
        if multithread:
            ROOT.EnableImplicitMT()
        self.rdf = ROOT.RDataFrame(tree_name, str(path))
        self.rdf = (self.rdf
            .Define('posParam0', f'GetPosParam0(NW{self.AB}.fnumbar)')
            .Define('posParam1', f'GetPosParam1(NW{self.AB}.fnumbar)')
            .Define('distParam0', f'GetDistParam0(NW{self.AB}.fnumbar)')
            .Define('distParam1', f'GetDistParam1(NW{self.AB}.fnumbar)')
            .Define('distParam2', f'GetDistParam2(NW{self.AB}.fnumbar)')

            .Define('pos', f'GetPosition(NW{self.AB}.fTimeLeft, NW{self.AB}.fTimeRight, posParam0, posParam1)')
            .Define('dist', f'GetDistance(pos, distParam0, distParam1, distParam2)')
        )
        return self.rdf

    def get_distance_of_flight_histogram(self, bar, lazy=True):
        nw_cuts = [
            f'NW{self.AB}.fnumbar == {bar}',
            f'pos > {self.POS_RANGE[0]}',
            f'pos < {self.POS_RANGE[1]}',
        ]
        h = (self.rdf
            .Define('nw_cut', ' && '.join(nw_cuts))
            .Define('dist_cut', 'dist[nw_cut]')
            .Histo1D(('', '', 50 * 10, 430, 480), 'dist_cut')
        )
        return h if lazy else h.GetValue()

    def get_time_of_flight_histogram(
        self,
        bar: int,
        dist_cut: Optional[Tuple[str, str]] = None,
        lazy: bool = True,
    ):
        """Returns the time of flight histogram for a given bar.

        Parameters
        ----------
        bar : int
            Bar number.
        dist_cut : 2-tuple of strings, default None
            Optional distance cut. When provided, it is user's responsibility to
            ensure there is non-zero count after the cut. Default is None, where
            no cut is applied.
        lazy : bool, default True
            If True, return a lazy result pointer to the histogram. If False,
            return the histogram instantly.
        
        Returns
        -------
        histogram : RResultPtr or TH1D
            The time-of-flight spectrum.
        """
        nw_cuts = [
            f'NW{self.AB}.fnumbar == {bar}',
            f'pos > {self.POS_RANGE[0]}',
            f'pos < {self.POS_RANGE[1]}',
        ]
        if dist_cut is not None:
            nw_cuts.append(f'dist > {dist_cut[0]}')
            nw_cuts.append(f'dist < {dist_cut[1]}')
        h = (self.rdf
            .Define('nw_cut', ' && '.join(nw_cuts))
            .Define('tof_raw', f'0.5 * (NW{self.AB}.fTimeLeft + NW{self.AB}.fTimeRight) - ForwardArray.fTimeMean')
            .Define('tof_raw_cut', 'tof_raw[nw_cut]')
            .Histo1D(('', '', 50 * 5, 0, 50), 'tof_raw_cut')
        )
        return h if lazy else h.GetValue()

    @staticmethod
    def get_first_peak(x, y):
        """Find the first peak from the left.

        Useful when locating the prompt gamma peak in a time-of-flight spectrum.
        """
        x, y = map(np.array, (x, y))
        peaks = []
        for i in range(1, len(x) - 1):
            if y[i - 1] < y[i] and y[i] > y[i + 1]: # peak
                peaks.append([x[i], y[i]])
        
        highest_y_so_far = None
        best_score, best_x = -1, None
        for x, y in peaks:
            score = 1 if highest_y_so_far is None else y / highest_y_so_far
            if score >= 1:
                highest_y_so_far = y
            if score > best_score:
                best_score, best_x = score, x
        return best_x
        
    def get_distance_range(self, bar):
        pars = self.df_emp_distance.loc[bar]
        pos = np.linspace(*self.POS_RANGE, 200)
        fit_low = np.polyval(pars[['low2', 'low1', 'low0']], pos)
        fit_upp = np.polyval(pars[['upp2', 'upp1', 'upp0']], pos)
        return fit_low.min(), fit_upp.max()

    def get_all_distance_of_flight_stats(self, bar, lazy=True) -> Dict[str, Union[float, ROOT.RDF.RResultPtr[ROOT.Double_t]]]:
        nw_cuts = [
            f'NW{self.AB}.fnumbar == {bar}',
            f'pos > {self.POS_RANGE[0]}',
            f'pos < {self.POS_RANGE[1]}',
        ]
        rdf = (self.rdf
            .Define('nw_cut', ' && '.join(nw_cuts))
            .Define('dist_cut', 'dist[nw_cut]')
        )
        result = {
            'mean': rdf.Mean('dist_cut'),
            'stdev': rdf.StdDev('dist_cut'),
        }
        return result if lazy else {k: v.GetValue() for k, v in result.items()}
    
    def get_stdev_distance_of_flight(self, bar, lazy=True) -> Union[float, ROOT.RDF.RResultPtr[ROOT.Double_t]]:
        nw_cuts = [
            f'NW{self.AB}.fnumbar == {bar}',
            f'pos > {self.POS_RANGE[0]}',
            f'pos < {self.POS_RANGE[1]}',
        ]
        result = (self.rdf
            .Define('nw_cut', ' && '.join(nw_cuts))
            .Define('dist_cut', 'dist[nw_cut]')
            .StdDev('dist_cut')
        )
        return result if lazy else result.GetValue()

    def get_all_time_of_flight_histograms(self, bar, lazy=True):
        dist_range = self.get_distance_range(bar)
        dist_ranges = slicer.create_ranges(*dist_range, width=self.DIST_RANGE_WIDTH, n_steps=self.DIST_RANGE_N_STEPS)
        histos = dict()
        histos['all'] = self.get_time_of_flight_histogram(bar, lazy=lazy)
        for d_range in dist_ranges:
            histos[tuple(d_range)] = self.get_time_of_flight_histogram(bar, dist_cut=d_range, lazy=lazy)
        return histos

    def get_prompt_gamma_fits(self, bar_tof_histos):
        rand = np.random.RandomState()
        h_all = rt.histo_conversion(bar_tof_histos['all'])
        x_all = self.get_first_peak(h_all.x, h_all.y)
        pars = dict()
        for i, (d_range, h) in enumerate(bar_tof_histos.items()):
            h = rt.histo_conversion(h)
            h = h.query(f'x > {x_all - 3} & x < {x_all + 2}')
            ppp = []
            x, y = h.x.to_numpy(), h.y.to_numpy()
            idx = np.arange(len(x))
            for _ in range(5):
                rand.shuffle(idx)
                par, err = PeakFinderGaus1D._find_highest_peak(x[idx], y[idx], error=True)
                ppp.append(par[1])

            pars[d_range] = {
                'amplt': par[0], 'mean': par[1], 'sigma': par[2],
                'amplt_err': err[0], 'mean_err': err[1], 'sigma_err': err[2],
            }
            if d_range == 'all':
                x_all = pars[d_range]['mean']
        return pars

    @staticmethod
    def _speed_of_light_model(distance, tof_offset):
        return distance / TimeOfFlightCalibrator.SPEED_OF_LIGHT_IN_AIR + tof_offset

    def get_speed_of_flight_fit(self, prompt_gamma_fit_params):
        pars = prompt_gamma_fit_params.copy() # shorthand
        pars.pop('all')
        df = []
        for d_range, par in pars.items():
            df.append([np.mean(d_range), par['mean'], par['mean_err']])
        df = pd.DataFrame(df, columns=['dist', 'tof', 'tof_eff'])
        para, perr = curve_fit(self._speed_of_light_model, df.dist, df.tof, sigma=df.tof_eff, p0=[0])
        return para[0], np.sqrt(np.diag(perr))[0]
    
    def collect_all_lazy(self, bars):
        self.lazy_items = {
            bar: {
                'histos': self.get_all_time_of_flight_histograms(bar, lazy=True),
                'all_dof_stats': self.get_all_distance_of_flight_stats(bar, lazy=True),
            }
            for bar in bars
        }
        return self.lazy_items

    def fit_bar(self, bar):
        if bar not in self.lazy_items:
            self.lazy_items[bar] = {
                'histos': self.get_all_time_of_flight_histograms(bar, lazy=True),
                'all_dof_stats': self.get_all_distance_of_flight_stats(bar, lazy=True),
            }

        histos = {key: h.GetValue() for key, h in self.lazy_items[bar]['histos'].items()}

        self.pg_fit_params[bar] = self.get_prompt_gamma_fits(histos)
        self.tof_offset[bar], self.tof_offset_err[bar] = self.get_speed_of_flight_fit(self.pg_fit_params[bar])
    
    def save_parameters(self, path=None):
        if path is None:
            path = Path(os.path.expandvars(self.DATABASE_DIR)) / 'calib_params' / f'run-{self.run:04d}-nw{self.ab}.dat'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = []
        for (bar, offset), offset_err in zip(self.tof_offset.items(), self.tof_offset_err.values()):
            df.append([bar, offset, offset_err])
        df = pd.DataFrame(df, columns=['bar', 'tof_offset', 'tof_offset_err'])
        tables.to_fwf(
            df, path,
            comment=inspect.cleandoc(f'''
            # run-{self.run:04d} NW{self.AB}
            # tof offset in unit of ns
            '''),
        )

class Plotter:
    """Plotter utility class.

    This is an utility class to plot the results from :py:class:`TimeOfFlightCalibrator`.

    >>> calib = TimeOfFlightCalibrator('B', 4100)
    >>> calib.fit_bar(bar=24) # will take a while
    >>> fig, axes = Plotter.plot(calib, bar=24)
    >>> plt.show()

    """
    base_layout = '''
        ABCXXX
        DEFXXX
        GHIYYY
    '''

    @classmethod
    def generate_mosaic_layout(cls, n_points):
        max_n_points = TimeOfFlightCalibrator.DIST_RANGE_N_STEPS
        if n_points > max_n_points:
            raise ValueError('Too many points to plot.')
        
        all_letters = [chr(ord('A') + i) for i in range(max_n_points)]
        letters = [chr(ord('A') + i) for i in range(n_points)]
        unused_letters = [l for l in all_letters if l not in letters]
        layout = cls.base_layout
        for l in unused_letters:
            layout = layout.replace(l, '.')
        return layout
    
    @staticmethod
    def plot_time_of_flight_histogram(ax, dist_range, histo, param):
        df = rt.histo_conversion(histo)
        dfh.hist(df, ax, histtype='stepfilled', color='gray')
        peak_x_range = [param['mean'] - 5, param['mean'] + 5]
        ax.set_xlim(*peak_x_range)
        x_plt = np.linspace(*peak_x_range, 100)
        gaus_par = [param['amplt'], param['mean'], param['sigma']]
        ax.plot(
            x_plt, PeakFinderGaus1D.gaus(x_plt, *gaus_par),
            color='red',
            lw=1.5,
            linestyle='dashed',
        )

        title = dist_range
        if dist_range != 'all':
            title = r'DoF$=%.1f\pm%.1f$ cm' % (np.mean(dist_range), np.diff(dist_range)[0] / 2)
        ax.set_title(title)
    
    @staticmethod
    def plot_speed_of_light_fit(ax, obj, bar):
        params = obj.pg_fit_params[bar].copy()
        all_dof_stats = obj.lazy_items[bar]['all_dof_stats'].copy()
        tof_offset = obj.tof_offset[bar]
        tof_offset_err = obj.tof_offset_err[bar]

        param_all = params.pop('all')
        df = []
        for d_range, param in params.items():
            df.append([np.mean(d_range), param['mean'], param['mean_err']])
        df = pd.DataFrame(df, columns=['dist', 'tof', 'tof_err'])
        ax.errorbar(
            df.dist, df.tof,
            yerr=df.tof_err,
            fmt='o',
            color='black',
        )
        x_plt = np.linspace(df.dist.min(), df.dist.max(), 100)
        ax.plot(x_plt, obj._speed_of_light_model(x_plt, tof_offset), color='green', lw=1.5)
        ax.set_title(r'$\mathrm{DoF} = c_\mathrm{air}\cdot\mathrm{ToF} - t_0$')
        ax.annotate(
            r'$t_0 = %.3f \pm %.3f$ ns' % (tof_offset, tof_offset_err),
            (0.5, 0.95),
            xycoords='axes fraction',
            ha='center', va='top',
            fontsize=15,
        )
        ax.errorbar(
            [all_dof_stats['mean'].GetValue()], [param_all['mean']],
            xerr=[all_dof_stats['stdev'].GetValue()],
            yerr=[param_all['mean_err']],
            fmt='o',
            linewidth=2,
            color='blue',
            markerfacecolor='white',
            label=r'all (not used to fit)',
        )
        ax.legend(loc='lower right', fontsize=15)
    
    @classmethod
    def plot(cls, obj, bar):
        styles.set_matplotlib_style(mpl)
        mpl.rcParams.update({
            'figure.dpi': 200,
            'axes.grid': True,
        })

        histos = {key: val.GetValue() for key, val in obj.lazy_items[bar]['histos'].items()}
        params = obj.pg_fit_params[bar].copy()

        histo_all = histos.pop('all')
        n_histos = len(histos)
        layout = cls.generate_mosaic_layout(n_histos)

        fig, axes = plt.subplot_mosaic(layout, figsize=(15, 10), constrained_layout=True)

        ax = axes['X']
        cls.plot_speed_of_light_fit(ax, obj, bar)
        ax.set_xlabel('DoF [cm]')
        ax.set_ylabel('Uncalibrated ToF [ns]')

        ax = axes['Y']
        cls.plot_time_of_flight_histogram(ax, 'all', histo_all, params['all'])
        ax.set_xlabel('Uncalibrated ToF [ns]')

        for i in range(n_histos):
            ax_name = chr(ord('A') + i)
            ax = axes[ax_name]
            dist_range = list(histos.keys())[i]
            histo = histos[dist_range]
            param = params[dist_range]
            cls.plot_time_of_flight_histogram(ax, dist_range, histo, param)
            ax.get_shared_y_axes().join(ax, axes['A'])
            if ax_name not in ['A', 'D', 'G']:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('Counts')
            if ax_name in ['G', 'H', 'I']:
                ax.set_xlabel('Uncalibrated ToF [ns]')
        return fig, axes
    
    @classmethod
    def save_plot(cls, obj, bar, path=None):
        if path is None:
            path = Path(os.path.expandvars(TimeOfFlightCalibrator.DATABASE_DIR)) / f'gallery/run-{obj.run:04d}/NW{obj.AB}-bar{bar:02d}.png'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        mpl.use('Agg')
        fig, _ = cls.plot(obj, bar)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        mpl.use(MPL_DEFAULT_BACKEND)

class _MainUtilities:
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(
            description='Calibrate the time-of-flight for neutron wall.',
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            'AB',
            type=str,
            choices=['A', 'B'],
            help='"A" or "B". This selects "NWA" or "NWB".',
        )
        parser.add_argument(
            'run',
            type=int,
            help='Run number.',
        )
        parser.add_argument(
            '-b', '--bars',
            nargs='+',
            type=str,
            default=['1-24'],
            help=misc.MainUtilities.wrap('''
                The bar number(s). Default "1-24". Dash can be used to specify
                ranges.
            ''')
        )
        parser.add_argument(
            '-d', '--debug',
            action='store_true',
            help=misc.MainUtilities.wrap('''
                When this flag is set, no output would be saved, i.e.  the
                calibration results, both the parameters and diagnostic plots,
                will not be saved.
            ''')
        )
        parser.add_argument(
            '-o', '--output',
            default="$DATABASE_DIR/neutron_wall/time_of_flight_calibration/",
            help=misc.MainUtilities.wrap('''
                The output directory for the calibration results. Default is
                "$DATABASE_DIR/neutron_wall/time_of_flight_calibration/".
            ''')
        )
        parser.add_argument(
            '-s', '--silence',
            action='store_true',
            help='To silence the status messages.',
        )

        args = parser.parse_args()
        args.bars = sorted(misc.MainUtilities.parse_bars(args.bars))
        return args

if __name__ == '__main__':
    import argparse
    from e15190.utilities import misc

    args = _MainUtilities.get_args()
    TimeOfFlightCalibrator.DATABASE_DIR = Path(os.path.expandvars(args.output))
    try:
        calib = TimeOfFlightCalibrator(args.AB, args.run, init_rdf=True)
        calib.collect_all_lazy(args.bars)
        for bar in args.bars:
            if not args.silence:
                print(f'Calibrating bar-{bar:02d}...', flush=True)
            calib.fit_bar(bar)
            Plotter.save_plot(calib, bar=bar)
            calib.save_parameters() # save result each time a new bar is calibrated so that we don't everything if the program crashes
        if not args.silence:
            print('Done.', flush=True)
    except Exception as err:
        print(traceback.format_exc())
