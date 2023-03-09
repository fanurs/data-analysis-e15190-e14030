#!/usr/bin/env python3
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import argparse
from glob import glob
import itertools
import json
import os
from pathlib import Path
import re
from typing import Literal, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
import pandas as pd
import ruptures as rpt
from scipy.optimize import minimize
import ROOT

from e15190.utilities import (
    fast_histogram as fh,
    root6 as rt,
    styles,
)

def main_function(AB: Literal['A', 'B'], run: int) -> None:
    from time import perf_counter
    path = str(ADCPreprocessor._get_input_root_path(run))
    try:
        tree_name = rt.infer_tree_name(path)
    except ValueError:
        print(f'Run {run:04d} has no tree. Exited.', flush=True)
        exit(1)
    n_entries = rt.get_n_entries(str(ADCPreprocessor._get_input_root_path(run)), tree_name)
    if n_entries < 5e5:
        print(f'Run {run:04d} has only {n_entries} entries. Exited.', flush=True)
        exit(1)
    
    gallery = Gallery()
    for bar in range(1, 24 + 1):
        t0 = perf_counter()
        plot_path = Path(os.path.expandvars(ADCPreprocessor.database_dir)) / f'gallery/run-{run:04d}/NW{AB}-{bar:02d}.png'
        if plot_path.exists():
            print(f'Run {run:04d} NW{AB}-{bar:02d} already done.', flush=True)
            continue
        try:
            preprocessor = ADCPreprocessor(AB, run, bar)
            preprocessor.alias()
            preprocessor.filter_neutron_wall_multiplicity()
            preprocessor.define_randomized_adc()
            preprocessor.fit()
            preprocessor.save_parameters()
            gallery.plot(preprocessor, save=True)
            print(f'Run {run:04d} NW{AB}-{bar:02d} done ({perf_counter() - t0:.1f} s)', flush=True)
        except Exception as err:
            print(f'Run {run:04d} NW{AB}-{bar:02d} FAILED ({perf_counter() - t0:.1f} s): {err}', flush=True)

class ADCPreprocessor:
    # input_root_path_fmt = '$DATABASE_DIR/root_files_daniele/CalibratedData_{run:04d}.root'
    input_root_path_fmt = '$DATABASE_DIR/root_files/run-{run:04d}.root'
    database_dir = '$DATABASE_DIR/neutron_wall/adc_preprocessing/'

    def __init__(self, AB: Literal['A', 'B'], run: int, bar: int, enable_implicit_mt=True):
        self.AB = AB.upper()
        self.ab = AB.lower()
        self.run = run
        self.bar = bar
        self.rdf = self._get_input_root_rdataframe(run, enable_implicit_mt=enable_implicit_mt)
    
    @classmethod
    def _get_input_root_path(cls, run: int) -> Path:
        return Path(os.path.expandvars(cls.input_root_path_fmt.format(run=run)))
    
    @classmethod
    def _get_input_root_rdataframe(cls, run: int, enable_implicit_mt=True) -> ROOT.RDataFrame:
        if enable_implicit_mt:
            ROOT.EnableImplicitMT()
        tree_name = rt.infer_tree_name(cls._get_input_root_path(run))
        return ROOT.RDataFrame(tree_name, str(cls._get_input_root_path(run)))
    
    def alias(self) -> None:
        self.rdf = (self.rdf
            # .Alias('VW_multi', 'VetoWall.fmulti')
            # .Alias('MB_multi', 'uBall.fmulti')
            # .Alias('total_L', f'NW{self.AB}.fLeft')
            # .Alias('total_R', f'NW{self.AB}.fRight')
            # .Alias('fast_L', f'NW{self.AB}.ffastLeft')
            # .Alias('fast_R', f'NW{self.AB}.ffastRight')
            .Alias('bar', f'NW{self.AB}_bar')
            .Alias('total_L', f'NW{self.AB}_total_L')
            .Alias('total_R', f'NW{self.AB}_total_R')
            .Alias('fast_L', f'NW{self.AB}_fast_L')
            .Alias('fast_R', f'NW{self.AB}_fast_R')
            .Alias('pos_x', f'NW{self.AB}_pos_x')
        )
    
    def filter_neutron_wall_multiplicity(self, lower_bound: int = 1) -> None:
        self.rdf = self.rdf.Filter(f'NW{self.AB}_multi > {lower_bound - 1}')

    def _define_randomized_adc(self, gate: Literal['total', 'fast'], side: Literal['L', 'R']) -> str:
        name = f'{gate}_{side}'
        new_name = f'{gate}r_{side}'
        self.rdf = self.rdf.Define(new_name, ' + '.join([
            name,
            f'({name} > 0 && {name} < 4096) * gRandom->Uniform(-0.5, 0.5)',
            f'({name} == 0) * gRandom->Uniform(0, 0.5)',
        ]))
        return new_name
    
    def define_randomized_adc(self) -> list[str]:
        return [
            self._define_randomized_adc(gate, side)
            for gate in ['total', 'fast']
            for side in ['L', 'R']
        ]

    def get_fit_histograms(self, get_value=False) -> dict[
        str, ROOT.RDF.RResultPtr[ROOT.TH1D] | ROOT.RDF.RResultPtr[ROOT.TH2D] | ROOT.TH1D | ROOT.TH2D
    ]:
        rdf_bar = self.rdf.Define('base_cut', f'bar == {self.bar}')
        histograms = {
            'fast_total_L': rdf_bar.Histo2D(('', '', 1025, 0, 4100, 1025, 0, 4100), 'fastr_L', 'totalr_L', 'base_cut'),
            'fast_total_R': rdf_bar.Histo2D(('', '', 1025, 0, 4100, 1025, 0, 4100), 'fastr_R', 'totalr_R', 'base_cut'),
            'log_ratio_total': (rdf_bar
                .Filter('VW_multi == 0')
                .Define('cut', ' && '.join([
                    'base_cut',
                    'sqrt(totalr_R * totalr_L) > 25',
                    'totalr_R < 3000', 'totalr_L < 3000',
                    'fastr_L > 0', 'fastr_R > 0',
                ]))
                .Define('y', 'log(totalr_R / totalr_L)')
                .Histo2D(('', '', 250, -125, 125, 500, -5, 5), 'pos_x', 'y', 'cut')
            ),
        }
        if get_value:
            histograms = {k: v.GetValue() for k, v in histograms.items()}
        return histograms
    
    def fit(self) -> dict[str, NonLinearCorrector | SaturationCorrector]:
        histograms = self.get_fit_histograms(get_value=True)
        self.correctors = {
            'fast_total_L': NonLinearCorrector(histograms['fast_total_L']).fit(),
            'fast_total_R': NonLinearCorrector(histograms['fast_total_R']).fit(),
            'log_ratio_total': SaturationCorrector(histograms['log_ratio_total']).fit(),
        }
        return self.correctors
    
    def define_corrected_adc(self) -> None:
        ft_L = self.correctors['fast_total_L']
        ft_R = self.correctors['fast_total_R']
        lrt = self.correctors['log_ratio_total'].model.coef
        self.rdf = (self.rdf
            .Define('is_saturated_total_L', 'total_L > 4095.5')
            .Define('is_saturated_total_R', 'total_R > 4095.5')
            .Define('total_ratio', f'exp({lrt[0]} + {lrt[1]} * pos_x)')

            # the corrected ADC values
            .Define('totalf_L', ' + '.join([
                'totalr_L',
                '(is_saturated_total_L && !is_saturated_total_R) * (totalr_R / total_ratio - totalr_L)',
                f'(!is_saturated_total_L && fastr_L > {ft_L.x_switch} && fastr_L < {ft_L.stationary_point_x}) * ({ft_L.linear_fit.coef[0] - ft_L.quad_p0} + {ft_L.linear_fit.coef[1] - ft_L.quad_p1} * fastr_L + {-ft_L.quad_p2} * fastr_L * fastr_L)',
                f'(!is_saturated_total_L && fastr_L > {ft_L.stationary_point_x}) * ({ft_L.stationary_point_y} - totalr_L)',
            ]))
            .Define('totalf_R', ' + '.join([
                'totalr_R',
                '(is_saturated_total_R && !is_saturated_total_L) * (totalr_L * total_ratio - totalr_R)',
                f'(!is_saturated_total_R && fastr_R > {ft_R.x_switch} && fastr_R < {ft_R.stationary_point_x}) * ({ft_R.linear_fit.coef[0] - ft_R.quad_p0} + {ft_R.linear_fit.coef[1] - ft_R.quad_p1} * fastr_R + {-ft_R.quad_p2} * fastr_R * fastr_R)',
                f'(!is_saturated_total_R && fastr_R > {ft_R.stationary_point_x}) * ({ft_R.stationary_point_y} - totalr_R)',
            ]))
            .Alias('fastf_L', 'fastr_L')
            .Alias('fastf_R', 'fastr_R')
        )

    def get_corrected_histograms(self, get_value=False):
        rdf_bar = self.rdf.Define('base_cut', f'bar == {self.bar} && fastf_L > 0 && fastf_R > 0')
        histograms = {
            'fast_total_L': (rdf_bar
                .Define('cut', 'base_cut && totalr_R < 4095.5')
                .Histo2D(('', '', 1125, 0, 4500, 1125, 0, 4500), 'fastf_L', 'totalf_L', 'cut')
            ),
            'fast_total_R': (rdf_bar
                .Define('cut', 'base_cut && totalr_L < 4095.5')
                .Histo2D(('', '', 1125, 0, 4500, 1125, 0, 4500), 'fastf_R', 'totalf_R', 'cut')
            ),
            'log_ratio_total': (rdf_bar
                .Filter('VW_multi == 0')
                .Define('cut', 'base_cut && sqrt(totalf_R * totalf_L) > 25')
                .Define('y', 'log(totalf_R / totalf_L)')
                .Histo2D(('', '', 250, -125, 125, 500, -5, 5), 'pos_x', 'y', 'cut')
            ),
            'total_L_R': (rdf_bar
                .Filter('VW_multi == 0')
                .Histo2D(('', '', 600, 0, 6000, 600, 0, 6000), 'totalf_L', 'totalf_R', 'base_cut')
            ),
        }
        if get_value:
            histograms = {k: v.GetValue() for k, v in histograms.items()}
        return histograms

    def save_parameters(self, path: Optional[str | Path] = None) -> Path:
        if path is None:
            path = Path(os.path.expandvars(self.database_dir)) / f'calib_params/run-{self.run:04d}/nw{self.ab}-bar{self.bar:02d}.json'
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        quad_L = [self.correctors['fast_total_L'].__dict__[k] for k in ['quad_p0', 'quad_p1', 'quad_p2']]
        quad_R = [self.correctors['fast_total_R'].__dict__[k] for k in ['quad_p0', 'quad_p1', 'quad_p2']]

        params = {
            'fast_total_L': {
                'nonlinear_fast_threshold': self.correctors['fast_total_L'].x_switch,
                'linear_fit_params': str(list(self.correctors['fast_total_L'].linear_fit.coef)),
                'quadratic_fit_params': str(quad_L),
                'stationary_point_x': self.correctors['fast_total_L'].stationary_point_x,
                'stationary_point_y': self.correctors['fast_total_L'].stationary_point_y,
            },
            'fast_total_R': {
                'nonlinear_fast_threshold': self.correctors['fast_total_R'].x_switch,
                'linear_fit_params': str(list(self.correctors['fast_total_R'].linear_fit.coef)),
                'quadratic_fit_params': str(quad_R),
                'stationary_point_x': self.correctors['fast_total_R'].stationary_point_x,
                'stationary_point_y': self.correctors['fast_total_R'].stationary_point_y,
            },
            'log_ratio_total': {
                'attenuation_length': self.correctors['log_ratio_total'].attenuation_length,
                'attenuation_length_error': self.correctors['log_ratio_total'].attenuation_length_err,
                'gain_ratio': self.correctors['log_ratio_total'].gain_ratio,
                'gain_ratio_error': self.correctors['log_ratio_total'].gain_ratio_err,
            },
        }

        json_str = json.dumps(params, indent=4)
        json_str = re.sub(r'\]\"', ']', re.sub(r'\"\[', '[', json_str))
        with open(path, 'w') as file:
            file.write(json_str)
        return path


class Corrector:
    def __init__(self, histogram: ROOT.TH2D):
        self.histogram = histogram
        self.df_histogram = rt.histo_conversion(self.histogram, keep_zeros=False, ignore_errors=True)


class NonLinearCorrector(Corrector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_linear_fit(self, fit_range: tuple[float, float]) -> Polynomial:
        df_fit = self.df_histogram.query(f'x > {fit_range[0]} & x < {fit_range[1]}')
        return Polynomial.fit(df_fit.x, df_fit.y, 1, w=df_fit.z).convert()

    @staticmethod
    def get_quadratic_parameters(lin_p0: float, lin_p1: float, quad_p2: float, x_switch: float) -> tuple[float, float, float]:
        quad_p0 = lin_p0 + quad_p2 * x_switch**2
        quad_p1 = lin_p1 - 2 * quad_p2 * x_switch
        return quad_p0, quad_p1, quad_p2

    @staticmethod
    def get_stationary_point(quad_p0: float, quad_p1: float, quad_p2: float) -> tuple[float, float]:
        stationary_point_x = -quad_p1 / (2 * quad_p2)
        stationary_point_y = quad_p0 - quad_p1**2 / (4 * quad_p2)
        return stationary_point_x, stationary_point_y

    @classmethod
    def fast_total_model(cls, x: np.ndarray, lin_p0: float, lin_p1: float, quad_p2: float, x_switch: float) -> np.ndarray:
        quad_p0, quad_p1, quad_p2 = cls.get_quadratic_parameters(lin_p0, lin_p1, quad_p2, x_switch)
        stationary_point_x, stationary_point_y = cls.get_stationary_point(quad_p0, quad_p1, quad_p2)
        if stationary_point_x < x_switch:
            return np.where(
                x < x_switch,
                lin_p0 + lin_p1 * x,
                quad_p0 + quad_p1 * x + quad_p2 * x**2,
            )
        return np.piecewise(
            x,
            [
                x < x_switch,
                (x >= x_switch) & (x < stationary_point_x),
            ],
            [
                lambda x: lin_p0 + lin_p1 * x,
                lambda x: quad_p0 + quad_p1 * x + quad_p2 * x**2,
                lambda _: stationary_point_y,
            ]
        )
    
    @classmethod
    def cost_function(
        cls,
        params: tuple[float, float],
        df_fit: pd.DataFrame,
        linear_fit_params: ArrayLike[float, float],
        min_stationary_point: float,
    ) -> float:
        quad_p2, x_switch = params
        x, y = df_fit.x.to_numpy(), df_fit.y.to_numpy()
        weights = (x / 4096)**2

        residuals = np.abs(y - cls.fast_total_model(x, *linear_fit_params, quad_p2, x_switch))
        mask = (residuals < 150) # drop outliers
        mse = np.sum((weights[mask] * residuals[mask])**2) / np.sum(weights[mask])
        stationary_point = x_switch - 0.5 * linear_fit_params[1] / quad_p2 # -b / 2a, from ax^2 + bx + c = 0
        penalty = max(0, min_stationary_point - stationary_point) # penalty if stationary point lower than min_stationary_point
        return mse + (0.1 * penalty)**2
    
    def fit(self, linear_fit_range: tuple[float, float] = (1200.0, 2500.0), fit_min_threshold=2500.0) -> NonLinearCorrector:
        self.linear_fit = self.get_linear_fit(linear_fit_range)
        df_fit = self.df_histogram.query(f'x > {fit_min_threshold}')
        best_residuals = np.inf
        for x0 in itertools.product(
            [-2e-4, -3e-4, -5e-4, -8e-4, -1e-3, -2e-3, -3e-3, -5e-3, -8e-3],
            [2600, 2800, 3000, 3200, 3400]
        ):
            min_result = minimize(
                self.cost_function,
                x0=x0,
                method='Nelder-Mead',
                bounds=[(-1e-2, -1e-4), (2500, 4000)],
                args=(df_fit, self.linear_fit.coef, (4096 - self.linear_fit.coef[0]) / self.linear_fit.coef[1]),
            )
            if min_result.fun < best_residuals:
                best_residuals = min_result.fun
                self.min_result = min_result
        self.quad_p2, self.x_switch = self.min_result.x
        self.model = lambda x: self.fast_total_model(x, *self.linear_fit.coef, self.quad_p2, self.x_switch)
        self.quad_p0, self.quad_p1, _ = self.get_quadratic_parameters(*self.linear_fit.coef, self.quad_p2, self.x_switch)
        self.stationary_point_x = -self.quad_p1 / (2 * self.quad_p2)
        self.stationary_point_y = self.quad_p0 - self.quad_p1 ** 2 / (4 * self.quad_p2)
        return self


class SaturationCorrector(Corrector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fit(self, fit_range: tuple[float, float] = (-50.0, 50.0)) -> SaturationCorrector:
        df_fit = self.df_histogram.query(f'x > {fit_range[0]} & x < {fit_range[1]}')

        # I couldn't find a way to get the covariance matrix
        # Hence the older np.polyfit instead of the newer np.polynomial.polynomial.Polynomial.fit
        pars, perr = np.polyfit(df_fit.x, df_fit.y, 1, w=df_fit.z, full=False, cov='unscaled')
        perr = np.sqrt(np.diag(perr))
        pars, perr = pars[::-1], perr[::-1] # reverse order to get p0, p1

        self.model = Polynomial(pars)
        self.gain_ratio = np.exp(pars[0])
        self.gain_ratio_err = np.exp(pars[0]) * perr[0]
        self.attenuation_length = 2 / pars[1]
        self.attenuation_length_err = self.attenuation_length * (perr[1] / pars[1])
        return self


class Gallery:
    def __init__(self):
        self.cmap = copy.copy(plt.cm.jet)
        self.cmap.set_under('white')
        styles.set_matplotlib_style(mpl)

    def plot_histo2d(self, ax: plt.Axes, histogram: ROOT.TH2D):
        df_hist = rt.histo_conversion(histogram, keep_zeros=False, ignore_errors=True)
        return fh.plot_histo2d(
            ax.hist2d,
            df_hist.x, df_hist.y, weights=df_hist.z,
            range=[
                (histogram.GetXaxis().GetXmin(), histogram.GetXaxis().GetXmax()),
                (histogram.GetYaxis().GetXmin(), histogram.GetYaxis().GetXmax())
            ],
            bins=[histogram.GetNbinsX(), histogram.GetNbinsY()],
            cmap=self.cmap,
            norm=mpl.colors.LogNorm(vmin=1),
        )
    
    def plot(self, preprocessor: ADCPreprocessor, path: Optional[str | Path] = None, save: bool = False):
        pp = preprocessor
        pp.define_corrected_adc()
        histograms = pp.get_corrected_histograms()

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 15), constrained_layout=True)
        fig.suptitle(f'run-{pp.run:04d}: NW{pp.AB}-{pp.bar:02d}')

        ax = axes[0, 0]
        self.plot_histo2d(ax, pp.correctors['fast_total_L'].histogram)
        x_plt = np.linspace(0, 4100, 200)
        ax.plot(x_plt, pp.correctors['fast_total_L'].model(x_plt), color='black', linewidth=0.8)
        ax.axvline(pp.correctors['fast_total_L'].x_switch, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, 4100)
        ax.set_ylim(0, 4100)
        ax.set_xlabel(r'Left FAST [ADC]')
        ax.set_ylabel(r'Uncorrected left TOTAL [ADC]')

        ax = axes[0, 1]
        self.plot_histo2d(ax, pp.correctors['fast_total_R'].histogram)
        x_plt = np.linspace(0, 4100, 200)
        ax.plot(x_plt, pp.correctors['fast_total_R'].model(x_plt), color='black', linewidth=0.8)
        ax.axvline(pp.correctors['fast_total_R'].x_switch, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, 4100)
        ax.set_ylim(0, 4100)
        ax.set_xlabel(r'Right FAST [ADC]')
        ax.set_ylabel(r'Uncorrected right TOTAL [ADC]')

        ax = axes[1, 0]
        self.plot_histo2d(ax, histograms['fast_total_L'])
        x_plt = np.linspace(0, 4100, 200)
        ax.plot(x_plt, pp.correctors['fast_total_L'].linear_fit(x_plt), color='black', linewidth=0.8)
        ax.axvline(pp.correctors['fast_total_L'].x_switch, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, 4100)
        ax.set_ylim(0, 4100)
        ax.set_xlabel(r'Left FAST [ADC]')
        ax.set_ylabel(r'Corrected left TOTAL [ADC]')

        ax = axes[1, 1]
        self.plot_histo2d(ax, histograms['fast_total_R'])
        x_plt = np.linspace(0, 4100, 200)
        ax.plot(x_plt, pp.correctors['fast_total_R'].linear_fit(x_plt), color='black', linewidth=0.8)
        ax.axvline(pp.correctors['fast_total_R'].x_switch, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(0, 4100)
        ax.set_ylim(0, 4100)
        ax.set_xlabel(r'Right FAST [ADC]')
        ax.set_ylabel(r'Corrected right TOTAL [ADC]')

        ax = axes[2, 0]
        self.plot_histo2d(ax, histograms['log_ratio_total'])
        x_plt = np.linspace(-125, 125, 200)
        ax.plot(x_plt, pp.correctors['log_ratio_total'].model(x_plt), color='black', linewidth=0.8, linestyle='--')
        ax.set_xlim(-125, 125)
        ax.set_ylim(-5, 5)
        ax.set_xlabel(r'Hit position $x$ [cm]')
        ax.set_ylabel(r'Corrected $\ln(Q_2^\mathrm{R}/Q_2^\mathrm{L})$')

        ax = axes[2, 1]
        self.plot_histo2d(ax, histograms['total_L_R'])
        ax.set_xlim(0, 6000)
        ax.set_ylim(0, 6000)
        ax.set_xlabel(r'Corrected left TOTAL [ADC]')
        ax.set_ylabel(r'Corrected right TOTAL [ADC]')

        plt.draw()
        if save:
            path = path or Path(os.path.expandvars(pp.database_dir)) / f'gallery/run-{pp.run:04d}/NW{pp.AB}-{pp.bar:02d}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()


class CalibParameter:
    bars = list(range(1, 24 + 1))

    @classmethod
    def collect_parameters(cls, filepath: str | Path) -> dict:
        with open(filepath, 'r') as file:
            return json.load(file)

    @classmethod
    def collect_all_parameters_of_bar(cls, AB: Literal['A', 'B'], bar: int, max_workers=None) -> dict[str, pd.DataFrame]:
        directory = Path(os.path.expandvars(ADCPreprocessor.database_dir))
        paths = glob(str(directory / f'calib_params/run-*/nw{AB.lower()}-bar{bar:02d}.json'))
        paths = [Path(path) for path in paths]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            parameters = {
                int(path.parent.name[-4:]): executor.submit(cls.collect_parameters, path)
                for path in paths
            }
        parameters = {run: parameter.result() for run, parameter in parameters.items()}
        parameters = {run: parameters[run] for run in sorted(parameters)}
        results = {'fast_total_L': [], 'fast_total_R': [], 'log_ratio_total': []}
        for run, params in parameters.items():
            for side in ['L', 'R']:
                par = params[f'fast_total_{side}']
                results[f'fast_total_{side}'].append([
                    run, par['nonlinear_fast_threshold'],
                    *par['linear_fit_params'], *par['quadratic_fit_params'],
                    par['stationary_point_x'], par['stationary_point_y'],
                ])
            par = params['log_ratio_total']
            results['log_ratio_total'].append([
                run, par['attenuation_length'], par['attenuation_length_error'], par['gain_ratio'], par['gain_ratio_error'],
            ])
        for side in ['L', 'R']:
            results[f'fast_total_{side}'] = pd.DataFrame(results[f'fast_total_{side}'], columns=[
                'run', 'nonlinear_fast_threshold',
                'linear_fit_params[0]', 'linear_fit_params[1]',
                'quadratic_fit_params[0]', 'quadratic_fit_params[1]', 'quadratic_fit_params[2]',
                'stationary_point_x', 'stationary_point_y',
            ]).set_index('run', drop=False, inplace=False)
        results['log_ratio_total'] = pd.DataFrame(results['log_ratio_total'], columns=[
            'run', 'attenuation_length', 'attenuation_length_error', 'gain_ratio', 'gain_ratio_error',
        ]).set_index('run', drop=False, inplace=False)
        return results
    
    @classmethod
    def collect_all_parameters(cls, AB: Literal['A', 'B'], max_workers=None) -> dict[str, dict[int, pd.DataFrame]]:
        results = {bar: cls.collect_all_parameters_of_bar(AB, bar, max_workers=max_workers) for bar in cls.bars}
        names = list(results.values().__iter__().__next__().keys())
        return {name: {bar: results[bar][name] for bar in cls.bars} for name in names}
 
    @classmethod
    def _determine_breakpoint_ranges_of_bar(
        cls,
        parameters: pd.DataFrame,
        full_run_range: tuple[int, int],
        extra_breakpoints: tuple[int, ...],
        ruptures_kwargs: Optional[dict],
    ) -> np.ndarray:
        runs = list(parameters.index)

        # prepare data for change point detection
        data = []
        for run in range(full_run_range[0], full_run_range[1] + 1):
            if run not in runs: continue
            data.append([run, *parameters.loc[run]])
        data = pd.DataFrame(data, columns=['run', *parameters.columns]).set_index('run', drop=True, inplace=False)
        data = (data - data.mean(axis=0)) / data.std(axis=0)

        # change point detection
        kw = dict(kernel='rbf', min_size=20)
        if ruptures_kwargs is not None:
            kw.update(ruptures_kwargs)
        bp_runs = rpt.KernelCPD(**kw).fit_predict(data.values, pen=1)
        bp_runs = data.index[bp_runs[1:-1]]

        # add extra breakpoints
        if extra_breakpoints is not None:
            bp_runs = sorted([*extra_breakpoints, *bp_runs])
            for extra_bp in extra_breakpoints:
                idx = bp_runs.index(extra_bp)
                if idx <= 0: continue
                del bp_runs[idx - 1]
                if idx >= len(bp_runs) - 1: continue
                del bp_runs[idx]

        # add full run range
        if full_run_range[0] < bp_runs[0]:
            bp_runs = [full_run_range[0], *bp_runs]
        if bp_runs[-1] < full_run_range[1]:
            bp_runs = [*bp_runs, full_run_range[1]]

        # convert to ranges
        bp_ranges = np.array(list(zip(bp_runs[:-1], bp_runs[1:])))
        bp_ranges[:, 0] += 1
        assert (bp_ranges[:, 0] <= bp_ranges[:, 1]).all()
        return bp_ranges
    
    @classmethod
    def determine_breakpoints_ranges_of_bar(
        cls,
        parameters: pd.DataFrame,
        full_run_range: tuple[int, int],
        extra_breakpoints: tuple[int, ...],
        ruptures_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> np.ndarray:
        if 'run' in parameters.columns:
            parameters = parameters.set_index('run', inplace=False, drop=True)
        if 'nonlinear_fast_threshold' in parameters.columns:
            stationary_point_y_range = kwargs.get('stationary_point_y_range', (3750.0, 4000.0))
            parameters = parameters[parameters['stationary_point_y'].between(*stationary_point_y_range)]
            parameters = parameters[
                ['linear_fit_params[0]', 'linear_fit_params[1]', 'quadratic_fit_params[2]', 'nonlinear_fast_threshold']
            ] # the other columns are not independent
            return cls._determine_breakpoint_ranges_of_bar(parameters, full_run_range, extra_breakpoints, ruptures_kwargs)
        parameters = parameters[['attenuation_length', 'gain_ratio']]
        return cls._determine_breakpoint_ranges_of_bar(parameters, full_run_range, extra_breakpoints, ruptures_kwargs)
    
    @classmethod
    def determine_breakpoints_ranges(
        cls,
        parameters: dict[int, pd.DataFrame],
        full_run_range: tuple[int, int] = (2000, 5000),
        extra_breakpoints: Optional[tuple[int, ...]] = (3500, ),
        ruptures_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict[int, np.ndarray]:
        return {
            bar: cls.determine_breakpoints_ranges_of_bar(
                parameters[bar], full_run_range, extra_breakpoints, ruptures_kwargs, **kwargs,
            ) for bar in parameters
        }

    @classmethod
    def get_savable_of_bar(cls, breakpoint_ranges: np.ndarray, parameters: pd.DataFrame) -> pd.DataFrame:
        if 'run' in parameters.columns:
            parameters = parameters.set_index('run', inplace=False, drop=True)
        result = []
        for bp_range in breakpoint_ranges:
            subparams = np.array(parameters.loc[bp_range[0]:bp_range[1] + 1])
            quantiles = np.quantile(subparams, [0.25, 0.75], axis=0)
            mask = (subparams >= quantiles[0]) & (subparams <= quantiles[1])
            subparams[~mask] = np.nan
            means = np.nanmean(subparams, axis=0)
            result.append([*bp_range, *means])
        return pd.DataFrame(result, columns=['run_start', 'run_stop', *parameters.columns])
    
    @classmethod
    def get_savable(cls, breakpoint_ranges: dict[int, np.ndarray], parameters: dict[int, pd.DataFrame]) -> pd.DataFrame:
        df = None
        for (bar, bp_range), (_, params) in zip(breakpoint_ranges.items(), parameters.items()):
            df_bar = cls.get_savable_of_bar(bp_range, params)
            # insert bar as first column
            df_bar.insert(0, 'bar', bar)
            df = pd.concat([df, df_bar]) if df is not None else df_bar
        return df
    
    @staticmethod
    def round_by_significant_digits(df: pd.DataFrame, significant_digits: int) -> pd.DataFrame:
        df_rounded = df.copy()
        for col, dtype in zip(df_rounded.columns, df_rounded.dtypes):
            if dtype == np.float64 or dtype == np.float32:
                col_values = df[col].values
                magnitude = 10**(significant_digits - 1 - np.floor(np.log10(np.abs(col_values))))
                df_rounded[col] = np.round(col_values * magnitude) / magnitude
        return df_rounded
    
    @classmethod
    def save(
        cls,
        name: str,
        breakpoint_ranges: dict[int, np.ndarray],
        parameters: dict[int, pd.DataFrame],
        filepath: Optional[str | Path] = None,
    ) -> Path:
        df = cls.get_savable(breakpoint_ranges, parameters)

        if name in ['fast_total_L', 'slow_total_R']:
            independent_columns = [
                'linear_fit_params[0]', 'linear_fit_params[1]', 'quadratic_fit_params[2]', 'nonlinear_fast_threshold',
            ]
            df = df[['bar', 'run_start', 'run_stop', *independent_columns]]
            df['quadratic_fit_params[0]'], df['quadratic_fit_params[1]'], _ = NonLinearCorrector.get_quadratic_parameters(*[
                df[col].values for col in  independent_columns
            ])
            df['stationary_point_x'], df['stationary_point_y'] = NonLinearCorrector.get_stationary_point(*[
                df[f'quadratic_fit_params[{i}]'].values for i in range(3)
            ])
            df = df[[
                'bar', 'run_start', 'run_stop',
                'nonlinear_fast_threshold',
                'linear_fit_params[0]', 'linear_fit_params[1]',
                'quadratic_fit_params[0]', 'quadratic_fit_params[1]', 'quadratic_fit_params[2]',
                'stationary_point_x', 'stationary_point_y',
            ]]
        df = cls.round_by_significant_digits(df, 8)

        # determine file path
        if filepath is None:
            filepath = Path(os.path.expandvars(ADCPreprocessor.database_dir)) / f'calib_params_{name}.csv'
        filepath = Path(filepath)
        if filepath.suffix != '.csv':
            filepath = filepath.with_suffix('.csv')
        
        # save CSV
        df.to_csv(filepath, index=False)

        # save JSON
        content = dict()
        for _, row in df.iterrows():
            bar_str = str(int(row['bar']))
            if bar_str not in content:
                content[bar_str] = []
            row_content = {
                'run_range': str([int(row['run_start']), int(row['run_stop'])]),
            }
            if name in ['fast_total_L', 'fast_total_R']:
                row_content.update({
                    'nonlinear_fast_threshold': float(row['nonlinear_fast_threshold']),
                    'linear_fit_params': str([float(row[f'linear_fit_params[{i}]']) for i in range(2)]),
                    'quadratic_fit_params': str([float(row[f'quadratic_fit_params[{i}]']) for i in range(3)]),
                    'stationary_point_x': float(row['stationary_point_x']),
                    'stationary_point_y': float(row['stationary_point_y']),
                })
            else:
                row_content.update({
                    'attenuation_length': float(row['attenuation_length']),
                    'attenuation_length_error': float(row['attenuation_length_error']),
                    'gain_ratio': float(row['gain_ratio']),
                    'gain_ratio_error': float(row['gain_ratio_error']),
                })
            content[bar_str].append(row_content)
        json_str = json.dumps(content, indent=4)
        json_str = re.sub(r'\]\"', ']', re.sub(r'\"\[', '[', json_str))
        with open(filepath.with_suffix('.json'), 'w') as file:
            file.write(json_str)
        return filepath


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('AB', type=str, choices=['A', 'B'])
    argparser.add_argument('run', type=int)
    argparser.add_argument('--summary', action='store_true')
    args = argparser.parse_args()
    if args.summary:
        for name in ['fast_total_L', 'fast_total_R', 'log_ratio_total']:
            paras = CalibParameter.collect_all_parameters(args.AB, max_workers=8)
            bp_ranges = CalibParameter.determine_breakpoints_ranges(paras[name])
            CalibParameter.save(name, bp_ranges, paras[name])
    else:
        main_function(args.AB, args.run)
