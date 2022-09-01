import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT
from scipy.optimize import curve_fit

from e15190.neutron_wall import geometry as nwgeo
from e15190.neutron_wall.position_calibration import NWCalibrationReader
from e15190.utilities import dataframe_histogram as dfh, root6 as rt, slicer, styles
from e15190.utilities.peak_finder import PeakFinderGaus1D

MPL_DEFAULT_BACKEND = mpl.get_backend()
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT_DECLARED = False # global variable to prevent multiple ROOT declarations

class TimeOfFlightCalibrator:
    SPEED_OF_LIGHT_IN_AIR = 29.9702547 # cm / ns
    POS_RANGE = [-100, 100]
    DIST_RANGE_WIDTH = 8 # cm
    DIST_RANGE_N_STEPS = 9

    def __init__(self, AB, run, init_rdf=True):
        self.AB = AB.upper()
        self.ab = AB.lower()
        self.run = run
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
            - GetPosition(time_L, time_R, p0, p1)
            - GetParam0(bar)
            - GetParam1(bar)
        
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
        ROOT.gInterpreter.Declare('''
            #pragma cling optimize(3)
            using RInt = ROOT::RVec<int>;
            using RDouble = ROOT::RVec<double>;

            const RDouble pos_param0 = {%s};
            const RDouble pos_param1 = {%s};
            const RDouble dist_low_param0 = {%s};
            const RDouble dist_low_param1 = {%s};
            const RDouble dist_low_param2 = {%s};
            const RDouble dist_upp_param0 = {%s};
            const RDouble dist_upp_param1 = {%s};
            const RDouble dist_upp_param2 = {%s};

            RDouble GetPositionParam0(RInt bar) { return ROOT::VecOps::Take(pos_param0, bar - 1); }
            RDouble GetPositionParam1(RInt bar) { return ROOT::VecOps::Take(pos_param1, bar - 1); }
            RDouble GetDistLowParam0(RInt bar)  { return ROOT::VecOps::Take(dist_low_param0, bar - 1); }
            RDouble GetDistLowParam1(RInt bar)  { return ROOT::VecOps::Take(dist_low_param1, bar - 1); }
            RDouble GetDistLowParam2(RInt bar)  { return ROOT::VecOps::Take(dist_low_param2, bar - 1); }
            RDouble GetDistUppParam0(RInt bar)  { return ROOT::VecOps::Take(dist_upp_param0, bar - 1); }
            RDouble GetDistUppParam1(RInt bar)  { return ROOT::VecOps::Take(dist_upp_param1, bar - 1); }
            RDouble GetDistUppParam2(RInt bar)  { return ROOT::VecOps::Take(dist_upp_param2, bar - 1); }

            RDouble GetPosition(RDouble time_L, RDouble time_R, RDouble pos_param0, RDouble pos_param1) {
                return pos_param0 + pos_param1 * (time_L - time_R);
            }

            RDouble GetDistance(
                RDouble pos,
                RDouble low0, RDouble low1, RDouble low2,
                RDouble upp0, RDouble upp1, RDouble upp2
            ) {
                RDouble low = low0 + low1 * pos + low2 * pos * pos;
                RDouble upp = upp0 + upp1 * pos + upp2 * pos * pos;
                RDouble random_nums;
                for (int i = 0; i < pos.size(); i++) random_nums.push_back(gRandom->Rndm());
                return low + (upp - low) * random_nums;
            }
            ''' % (
                ', '.join(map(str, self.df_pos_calib['p0'].to_numpy())),
                ', '.join(map(str, self.df_pos_calib['p1'].to_numpy())),
                ', '.join(map(str, self.df_emp_distance['low0'].to_numpy())),
                ', '.join(map(str, self.df_emp_distance['low1'].to_numpy())),
                ', '.join(map(str, self.df_emp_distance['low2'].to_numpy())),
                ', '.join(map(str, self.df_emp_distance['upp0'].to_numpy())),
                ', '.join(map(str, self.df_emp_distance['upp1'].to_numpy())),
                ', '.join(map(str, self.df_emp_distance['upp2'].to_numpy()))

            )
        )
        ROOT_DECLARED = True

    def create_base_rdf(self, multithread=True):
        path = self.get_input_path(self.run)
        tree_name = rt.infer_tree_name(path)
        if multithread:
            ROOT.EnableImplicitMT()
        self.rdf = ROOT.RDataFrame(tree_name, str(path))
        self.rdf = (self.rdf
            .Define('p0', f'GetPositionParam0(NW{self.AB}.fnumbar)')
            .Define('p1', f'GetPositionParam1(NW{self.AB}.fnumbar)')
            .Define('low0', f'GetDistLowParam0(NW{self.AB}.fnumbar)')
            .Define('low1', f'GetDistLowParam1(NW{self.AB}.fnumbar)')
            .Define('low2', f'GetDistLowParam2(NW{self.AB}.fnumbar)')
            .Define('upp0', f'GetDistUppParam0(NW{self.AB}.fnumbar)')
            .Define('upp1', f'GetDistUppParam1(NW{self.AB}.fnumbar)')
            .Define('upp2', f'GetDistUppParam2(NW{self.AB}.fnumbar)')

            .Define('pos', f'GetPosition(NW{self.AB}.fTimeLeft, NW{self.AB}.fTimeRight, p0, p1)')
            .Define('dist', f'GetDistance(pos, low0, low1, low2, upp0, upp1, upp2)')
        )
        return self.rdf

    def estimate_distance_resolution(self, bar: int, lazy=True) -> ROOT.TH1:
        nw_cuts = [
            f'NW{self.AB}.fnumbar == {bar}',
            f'pos > {self.POS_RANGE[0]}',
            f'pos < {self.POS_RANGE[1]}',
        ]
        h = (self.rdf
            .Define('dist2', f'GetDistance(pos, low0, low1, low2, upp0, upp1, upp2)')
            .Define('nw_cut', ' && '.join(nw_cuts))
            .Define('dist_diff', f'dist - dist2')
            .Histo1D(('', '', 200, -10, 10), 'dist_diff')
        )
        return h if lazy else h.GetValue()

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
            .Histo1D(('', '', 50 * 10, 0, 50), 'tof_raw_cut')
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

    def get_mean_distance_of_flight(self, bar, lazy=True):
        nw_cuts = [
            f'NW{self.AB}.fnumbar == {bar}',
            f'pos > {self.POS_RANGE[0]}',
            f'pos < {self.POS_RANGE[1]}',
        ]
        h = (self.rdf
            .Define('nw_cut', ' && '.join(nw_cuts))
            .Define('dist_cut', 'dist[nw_cut]')
            .Mean('dist_cut')
        )
        return h if lazy else h.GetValue()

    def get_all_time_of_flight_histograms(self, bar):
        dist_range = self.get_distance_range(bar)
        dist_ranges = slicer.create_ranges(*dist_range, width=self.DIST_RANGE_WIDTH, n_steps=self.DIST_RANGE_N_STEPS)
        histos = dict()
        histos['all'] = self.get_time_of_flight_histogram(bar)
        for d_range in dist_ranges:
            histos[tuple(d_range)] = self.get_time_of_flight_histogram(bar, dist_cut=d_range)
        return histos

    def get_prompt_gamma_fits(self, bar_tof_histos):
        rand = np.random.RandomState()
        h_all = rt.histo_conversion(bar_tof_histos['all'])
        x_all = self.get_first_peak(h_all.x, h_all.y)
        print(f'{x_all=}')
        pars = dict()
        for i, (d_range, h) in enumerate(bar_tof_histos.items()):
            h = rt.histo_conversion(h)
            h = h.query(f'x > {x_all - 3} & x < {x_all + 2}')
            ppp = []
            x, y = h.x.to_numpy(), h.y.to_numpy()
            if i == 8:
                print(x)
                print(y)
            idx = np.arange(len(x))
            for _ in range(5):
                rand.shuffle(idx)
                par, err = PeakFinderGaus1D._find_highest_peak(x[idx], y[idx], error=True)
                ppp.append(par[1])
            if i == 8:
                print(ppp)

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
        self.tof_offset = para[0]
        self.tof_offset_err = np.sqrt(np.diag(perr))[0]
        return self.tof_offset, self.tof_offset_err
    
    def fit(self, bar):
        self.bar = bar
        self.histos = self.get_all_time_of_flight_histograms(self.bar)
        self.mean_dist_all = self.get_mean_distance_of_flight(self.bar).GetValue()
        self.histos = {key: h.GetValue() for key, h in self.histos.items()}
        self.pg_fit_params = self.get_prompt_gamma_fits(self.histos)
        self.tof_offset, self.tof_offset_err = self.get_speed_of_flight_fit(self.pg_fit_params)


class Plotter:
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
    def plot_speed_of_light_fit(ax, obj):
        params = obj.pg_fit_params.copy()
        param_all = params.pop('all')
        df = []
        for d_range, param in params.items():
            df.append([np.mean(d_range), param['mean'], param['mean_err']])
        df = pd.DataFrame(df, columns=['dist', 'tof', 'tof_err'])
        ax.errorbar(
            df.dist, df.tof,
            yerr=df.tof_err,
            fmt='.',
            color='black',
        )
        x_plt = np.linspace(df.dist.min(), df.dist.max(), 100)
        ax.plot(x_plt, obj._speed_of_light_model(x_plt, obj.tof_offset), color='green', lw=1.5)
        ax.set_title(r'$\mathrm{DoF} = c_\mathrm{air}\cdot\mathrm{ToF} - t_0$')
        ax.annotate(
            r'$t_0 = %.3f \pm %.3f$ ns' % (obj.tof_offset, obj.tof_offset_err),
            (0.5, 0.95),
            xycoords='axes fraction',
            ha='center', va='top',
            fontsize=15,
        )

        ax.axhline(param_all['mean'], color='blue', linestyle='dashed', lw=1, label='all')
        ax.axvline(obj.mean_dist_all, color='blue', linestyle='dashed', lw=1)

        ax.legend()
    
    @classmethod
    def plot(cls, obj):
        styles.set_matplotlib_style(mpl)
        mpl.rcParams.update({
            'figure.dpi': 200,
            'axes.grid': True,
        })

        histos = obj.histos.copy()
        params = obj.pg_fit_params.copy()

        histo_all = histos.pop('all')
        n_histos = len(histos)
        layout = cls.generate_mosaic_layout(n_histos)

        fig, axes = plt.subplot_mosaic(layout, figsize=(15, 10), constrained_layout=True)

        ax = axes['X']
        cls.plot_speed_of_light_fit(ax, obj)
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
