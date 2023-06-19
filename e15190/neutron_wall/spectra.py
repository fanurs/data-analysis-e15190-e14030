#!/usr/bin/env python3
from __future__ import annotations
import os
import pathlib
from typing import Callable, Literal, Optional

from astropy import constants
import numpy as np
import pandas as pd
import ROOT
from scipy.integrate import quad

from e15190.runlog.query import Query
from e15190.microball import impact_parameter
from e15190.neutron_wall import (
    efficiency as nw_eff,
    geometry as nw_geom,
    shadow_bar as nw_shade,
)
from e15190.utilities import (
    atomic_mass_evaluation as ame,
    physics,
    root6 as rt,
)

SPEED_OF_LIGHT = constants.c.to('cm/ns').value
MASS_NEUTRON = (constants.m_n * constants.c**2).to('MeV').value

class Spectrum:
    ROOT_FILE_DIR = '$DATABASE_DIR/root_files/'

    def __init__(self, runs: list[int]):
        self.root_file_dir = pathlib.Path(os.path.expandvars(self.ROOT_FILE_DIR))
        self.runs = runs
        self.root_file_paths = [self.root_file_dir / f'run-{run:04d}.root' for run in runs]

        reaction = self._infer_reaction(self.runs)
        self.target: str = reaction['target'] # type: ignore
        self.beam: str = reaction['beam'] # type: ignore
        self.beam_energy: float = reaction['beam_energy'] # MeV/u # type: ignore
        self.reaction_str = f'{self.beam}{self.target}e{self.beam_energy:.0f}' # e.g. ca48ni64e140

        self.shadow_bar_present = self._infer_shadow_bar_present(self.runs)
    
    @staticmethod
    def _infer_reaction(runs: list[int]) -> dict[Literal['target', 'beam', 'beam_energy'], str | float]:
        """Infer the reaction from the runs.

        Parameters
        ----------
        runs : list[int]
            A list of runs.

        Returns
        -------
        reaction : dict[Literal['target', 'beam', 'beam_energy'], str | float]
            A dictionary containing the reaction information.

        Raises
        ------
        ValueError
            If the runs are not from the same reaction.
        """
        first_run_info = Query.get_run_info(runs[0])
        for run in runs[1:]:
            run_info = Query.get_run_info(run)
            for key in ['target', 'beam', 'beam_energy']:
                if run_info[key] != first_run_info[key]:
                    raise ValueError(f'run-{run:04d} is not from the same reaction as run-{runs[0]:04d}.')
        return {
            'target': first_run_info['target'].lower(),
            'beam': first_run_info['beam'].lower(),
            'beam_energy': float(first_run_info['beam_energy']),
        }

    @staticmethod
    def _infer_shadow_bar_present(runs: list[int]) -> bool:
        """Infer whether the shadow bar is present.

        Parameters
        ----------
        runs : list[int]
            A list of runs.
        
        Returns
        -------
        presence : bool
            True if the shadow bar is present for all runs, False otherwise.
        
        Raises
        ------
        ValueError
            If the runs have different shadow bar presence.
        """
        presence = Query.get_run_info(runs[0])['shadow_bar']
        for run in runs[1:]:
            if Query.get_run_info(run)['shadow_bar'] != presence:
                raise ValueError(f'run-{run:04d} has different shadow bar presence than run-{runs[0]:04d}.')
        return presence

    def build_rdataframe(self, tree_name: Optional[str] = None, implicit_mt=True, inplace=False) -> ROOT.RDataFrame:
        """Build the RDataFrame and store it as :py:attr:`self.rdf`.

        Parameters
        ----------
        tree_name : str, optional
            Name of the tree, by default None. If None, the tree name will be
            inferred from the root file paths.
        implicit_mt : bool, default True
            Whether to enable implicit multi-threading.
        inplace : bool, default False
            Whether to build the RDataFrame in place.
        """
        if implicit_mt:
            ROOT.EnableImplicitMT()
        else:
            ROOT.DisableImplicitMT()
        
        if tree_name is None:
            tree_name = rt.infer_tree_name(self.root_file_paths)
        result = ROOT.RDataFrame(tree_name, list(map(str, self.root_file_paths)))
        if not inplace:
            return result
        self.rdf = result
    
    def filter_microball_multiplicity(
        self,
        bhat_range: Optional[tuple[float, float]] = None,
        multi_range: Optional[tuple[int, int]] = None,
        inplace=False,
    ) -> Optional[ROOT.RDataFrame]:
        """Filter the RDataFrame by microball multiplicity.

        Parameters
        ----------
        bhat_range : tuple[float, float], optional
            The range of normalized impact parameter, e.g. (0.0, 0.4). If this
            is not provided, ``multi_range`` must be provided.
        multi_range : tuple[int, int], optional
            The inclusive range of microball multiplicity, e.g. (14, 30). If
            ``bhat_range`` is provided, this will be ignored.
        inplace : bool, default False
            Whether to apply the filter in place.
        
        Returns
        -------
        rdf : ROOT.RDataFrame or None
            The filtered RDataFrame. If ``inplace=True``, returns None.
        """
        if bhat_range is not None:
            multi_float_range = impact_parameter.get_multiplicity_range(
                self.beam, self.target, int(self.beam_energy),
                bhat_range=bhat_range,
            )
            multi_range = (
                int(np.ceil(min(multi_float_range))),
                int(np.floor(max(multi_float_range))),
            )

        if multi_range is None:
            raise ValueError('Either bhat_range or multi_range must be provided.')

        result = self.rdf.Filter(f'MB_multi >= {multi_range[0]} && MB_multi <= {multi_range[1]}')
        if not inplace:
            return result
        self.rdf = result
    
    def get_coincidence_peak(
        self,
        branch_name='TDC_mb_nw',
        rdf: Optional[ROOT.RDataFrame] = None,
        return_histogram=False,
    ) -> dict[Literal['mean', 'stdev'], float] | ROOT.TH1:
        """Get the coincidence peak of the TDC.

        You may want to apply the microball multiplicity filter before calling
        this function.
        """
        if rdf is None:
            rdf = self.rdf

        hist = rdf.Histo1D(('', '', 20_000, -10_000, 10_000), branch_name).GetValue()
        if return_histogram:
            return hist

        df = (rt.histo_conversion(hist)[['x', 'y']]
            .astype(np.float32)
            .query('x > -9000') # values below were assigned for bad data
        )

        # identify the peak
        idx = np.argmax(df.y)
        hist = rdf.Histo1D(('', '', 2000, df.x.iloc[idx] - 10, df.x.iloc[idx] + 10), branch_name).GetValue()
        df = rt.histo_conversion(hist)[['x', 'y']].astype(np.float32)
        tol = 1e-3
        old_mean, old_stdev = 0.0, 0.0
        for _ in range(10):
            mean = np.average(df.x, weights=df.y)
            stdev = np.sqrt(np.average((df.x - mean)**2, weights=df.y))
            if np.abs(mean - old_mean) < tol and np.abs(stdev - old_stdev) < tol:
                break
            old_mean, old_stdev = mean, stdev
            df = df.query(f'x > {mean - 3 * stdev} and x < {mean + 3 * stdev}')
        else:
            raise RuntimeError('Failed to converge in 10 iterations.')

        return {'mean': mean, 'stdev': stdev}

    def filter_tdc(
        self,
        tdc_range: tuple[float, float],
        tdc_branch_name='TDC_mb_nw',
        rdf: Optional[ROOT.RDataFrame] = None,
        inplace=False,
    ) -> Optional[ROOT.RDataFrame]:
        """Filter the RDataFrame by TDC.

        Parameters
        ----------
        tdc_range : tuple[float, float]
            The range of TDC.
        rdf : ROOT.RDataFrame, optional
            The RDataFrame to be filtered. If None, the RDataFrame stored in
            :py:attr:`self.rdf` will be used.
        inplace : bool, default False
            Whether to apply the filter in place.
        
        Returns
        -------
        rdf : ROOT.RDataFrame or None
            The filtered RDataFrame. If ``inplace=True``, returns None.
        """
        if rdf is None:
            rdf = self.rdf
        result = rdf.Filter(f'{tdc_branch_name} > {tdc_range[0]} && {tdc_branch_name} < {tdc_range[1]}')
        if not inplace:
            return result
        self.rdf = result

    def filter_charged_particles(self, rdf: Optional[ROOT.RDataFrame] = None, inplace=False) -> Optional[ROOT.RDataFrame]:
        """Veto charged particles.

        This function will filter out charged particles by requiring the
        multiplicity of veto wall to be 0.

        Parameters
        ----------
        rdf : ROOT.RDataFrame, optional
            The RDataFrame to be filtered. If None, the RDataFrame stored in
            :py:attr:`self.rdf` will be used.
        inplace : bool, default False
            Whether to apply the filter in place.
        
        Returns
        -------
        rdf : ROOT.RDataFrame or None
            The filtered RDataFrame. If ``inplace=True``, returns None.
        """
        if rdf is None:
            rdf = self.rdf
        result = rdf.Filter('VW_multi == 0') # select when multiplicity of veto wall is 0
        if not inplace:
            return result
        self.rdf = result
    
    def define_energy(self, name='energy', rdf: Optional[ROOT.RDataFrame] = None, inplace=False) -> Optional[ROOT.RDataFrame]:
        """Define the total relativistic energy of the neutron.

        This function will define a new column ``energy`` in the RDataFrame. It
        first computes the relativistic beta using time-of-flight and distance
        between the target and the hit position on NWB. Then it calculates the
        total energy assuming a mass of neutron. If charged particles and gamma
        rays have not been excluded, the energies for those hits will be off
        because they are not neutrons.
    
        Parameters
        ----------
        name : str, default 'energy'
            Name of the new column. The values will be in MeV.
        rdf : ROOT.RDataFrame, optional
            The RDataFrame to be filtered. If None, the RDataFrame stored in
            :py:attr:`self.rdf` will be used.
        inplace : bool, default False
            Whether to apply the definition in place.
        
        Returns
        -------
        rdf : ROOT.RDataFrame or None
            The RDataFrame with the new column. If ``inplace=True``, returns
            None.
        """
        if rdf is None:
            rdf = self.rdf
        result = (rdf
            .Define('_beta', f'NWB_distance / NWB_tof / {SPEED_OF_LIGHT}') # relativistic beta
            .Define(name, f'{MASS_NEUTRON} / sqrt(1 - _beta * _beta)') # total relativistic energy
        )
        if not inplace:
            return result
        self.rdf = result
    
    def define_neutron_cut(
        self,
        name='cut',
        light_bias=3.0,
        position_cut=True,
        positive_adc_cut=True,
        extra_cuts: Optional[list[str]] = None,
        rdf: Optional[ROOT.RDataFrame] = None,
        inplace=False,
    ) -> Optional[ROOT.RDataFrame]:
        """Define the neutron cut, ``cut``.

        Parameters
        ----------
        name : str, default 'cut'
            Name of the new column.
        light_bias : float, default 3.0
            The light output bias in MeVee.
        position_cut : bool, default True
            Whether to apply the position cut. Shadow bar cuts are applied
            automatically if shadow bars are present in those runs. See :py:attr:`self.shadow_bar_present`.
        positive_adc_cut : bool, default True
            Whether to apply the positive ADC cut. If ``True``, all the fast and
            total ADCs must be positive.
        extra_cuts : list[str], optional
            Extra cuts to be applied. Each cut should be a string that can be
            parsed by PyROOT. Any columns that are defined in the RDF can be
            used in the cut. The cuts will be joined by ``&&``.
        rdf : ROOT.RDataFrame, optional
            The RDataFrame to be filtered. If None, the RDataFrame stored in
        inplace : bool, default False
            Whether to apply the cut in place.
        """
        if rdf is None:
            rdf = self.rdf

        conditions = [f'NWB_light_GM > {light_bias}']

        Bar = nw_geom.Bar
        if position_cut:
            conditions.append(f'NWB_pos_x > {Bar.edges_x[0]}')
            conditions.append(f'NWB_pos_x < {Bar.edges_x[-1]}')
        if position_cut and self.shadow_bar_present:
            for bar in Bar.shadowed_bars:
                conditions.append(f'!(NWB_bar == {bar} && {Bar.left_shadow_x[0]} < NWB_pos_x && NWB_pos_x < {Bar.left_shadow_x[-1]})')
                conditions.append(f'!(NWB_bar == {bar} && {Bar.right_shadow_x[0]} < NWB_pos_x && NWB_pos_x < {Bar.right_shadow_x[-1]})')
        
        if positive_adc_cut:
            conditions.append('NWB_fast_L > 0')
            conditions.append('NWB_fast_R > 0')
            conditions.append('NWB_total_L > 0')
            conditions.append('NWB_total_R > 0')
        
        if extra_cuts:
            conditions.extend(extra_cuts)

        result = rdf.Define(name, ' && '.join(conditions))
        if not inplace:
            return result
        self.rdf = result
    
    @staticmethod
    def _drop_spectrum_outliers(df: pd.DataFrame, drop_outliers: float, yname='y') -> pd.DataFrame:
        """Drop outliers in the dataframe.

        This function assumes outliers only occur at the ends of the spectrum.
        In other words, if a point in the middle is an outlier, it will not be
        dropped. This decision is made to keep the resulting spectrum
        continuous. Empirically, for this purpose, if an outlier happens in the
        middle, it is likely that there is some problem with the data anyway.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe to drop outliers from. Must have a column named defined by
            `yname`.
        drop_outliers : float
            Outliers are defined as points whose ratio to the neighboring point is
            larger than `drop_outliers`.
        yname : str, default 'y'
            The name of the column in `df` that contains the values to check for
            outliers.
        
        Returns
        -------
        df : pandas.DataFrame
            The dataframe with outliers dropped.
        """
        df = df.copy()
        df = df[df[yname] != 0]

        y = np.array(df[yname])
        inter_ratio = y[1:] / y[:-1]
        inlier_links = (inter_ratio < drop_outliers) & (inter_ratio > 1 / drop_outliers)

        args = np.where(inlier_links)
        inlier_links[:args[0][0]] = False
        inlier_links[args[0][-1] + 1:] = False
        inlier_links[args[0][0]:args[0][-1] + 1] = True

        inlier_mask = np.insert(inlier_links, args[0][0], True)
        return df[inlier_mask]

class LabKinergyTheta(Spectrum):
    def __init__(self, runs: list[int]):
        super().__init__(runs)
    
    def get_spectrum(
        self,
        theta_deg_range: tuple[float, float],
        range=(0.0, 300.0),
        bins=300,
    ) -> pd.DataFrame:
        """Get the lab kinetic energy spectrum at a given theta range.

        Returns a histogram represented by a DataFrame with columns: x, y, yerr, yferr.

        Parameters
        ----------
        theta_deg_range : tuple[float, float]
            The theta range in degrees.
        range : tuple[float, float], default (0.0, 300.0)
            The (kinetic) energy range in MeV.
        bins : int, default 300
            The number of bins.
        
        Returns
        -------
        spectrum : pandas.DataFrame
            DataFrame with columns 'x', 'y', 'yerr' and 'yferr'.
        """
        pass

class LabPtransverseRapidity(Spectrum):
    # centrality cut
    bhat_range = (0.0, 0.4)

    # a sharp theta cut to minimize edge effects
    theta_range = tuple(np.radians([28.0, 51.5]))

    # a sharp kinetic energy cut to remove data with low reliability
    kinergy_range = (10.0, 300.0) # MeV

    def __init__(self, runs: list[int]):
        super().__init__(runs)

        self.beam_lab_rapidity = physics.BeamTargetReaction(self.beam, self.target, int(self.beam_energy)).beam_lab_rapidity

        # container for lazy evaluation
        self.lazy = dict()
    
    def build_rdataframe(self, tree_name: str | None = None, implicit_mt=True) -> ROOT.RDataFrame:
        """Build the RDataFrame and store it as :py:attr:`self.rdf`.

        Different from :py:func:`Spectrum.build_rdataframe`, this function will
        always build the RDataFrame in place.

        This function will apply filters and define new quantities ready for the
        construction of transverse momentum spectra. It will also populate the
        lazy container with ``total_count`` and ``tdc_count``.

        This function is written for quick construction of spectra. Many cuts
        and definitions are written with the most common decisions. If you want
        to customize the cuts and definitions, you shall use functions in
        :py:class:`Spectrum` or directly interact with :py:attr:`self.rdf`.

        """
        super().build_rdataframe(inplace=True)

        self.filter_microball_multiplicity(bhat_range=self.bhat_range, inplace=True)
        self.lazy['total_count'] = self.rdf.Count()

        tdc_peak = self.get_coincidence_peak()
        tdc_peak['stdev'] = np.clip(tdc_peak['stdev'], 0.4, 0.8)
        self.filter_tdc((tdc_peak['mean'] - 5 * tdc_peak['stdev'], tdc_peak['mean'] + 5 * tdc_peak['stdev']), inplace=True)
        self.lazy['tdc_count'] = self.rdf.Count()

        self.filter_charged_particles(inplace=True)

        self.define_energy(name='energy', inplace=True)

        self.rdf = (self.rdf
            .Define('sin_theta', f'sin(NWB_theta * {np.pi} / 180)')
            .Define('cos_theta', f'cos(NWB_theta * {np.pi} / 180)')

            .Define('momentum', f'sqrt(energy * energy - {MASS_NEUTRON} * {MASS_NEUTRON})')
            .Define('transverse_momentum', 'momentum * sin_theta')

            .Define('longitudinal_momentum', 'momentum * cos_theta')
            .Define('rapidity', f'0.5 * log((energy + longitudinal_momentum) / (energy - longitudinal_momentum))')
            .Define('norm_rapidity', f'rapidity / {self.beam_lab_rapidity}')
        )

        self.define_neutron_cut(inplace=True,
            name='base_cut',
            light_bias=3.0, # MeVee
            position_cut=True,
            positive_adc_cut=True,
            extra_cuts=[
                f'energy - {MASS_NEUTRON} > 10', # energy threshold
                '(NWB_psd > 0.5 || NWB_total_L > 3500 || NWB_total_R > 3500)', # PSD cut
                f'NWB_theta > {np.degrees(self.theta_range[0]):.3f}',
                f'NWB_theta < {np.degrees(self.theta_range[1]):.3f}',
            ],
        )

    @staticmethod
    def theta_curve(theta: float, mass=MASS_NEUTRON) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Return a function that calculates the transverse momentum for a given rapidity.

        The theta and mass (particle) are fixed.

        Parameters
        ----------
        theta : float
            Polar angle in lab frame in radian.
        mass : float, default MASS_NEUTRON
            Mass of the particle in MeV/c^2.

        Returns
        -------
        transverse_momentum : Callable[[float | np.ndarray], float | np.ndarray
            Function that calculates the transverse momentum for a given (unnormalized) rapidity in MeV/c.
        """
        def transverse_momentum(rapidity: float | np.ndarray) -> float | np.ndarray:
            st = np.sinh(rapidity) * np.tan(theta)
            quantity = 1 - st**2
            return mass * st / np.sqrt(np.where(quantity > 0, quantity, np.nan))
        return transverse_momentum
    
    @staticmethod
    def kinergy_curve(kinergy: float, mass=MASS_NEUTRON) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Return a function that calculates the transverse momentum for a given rapidity.

        The kinetic energy and mass (particle) are fixed.

        Parameters
        ----------
        kinergy : float
            Kinetic energy in MeV. This should be the total kinetic energy of
            the whole isotope (particle). So if you are given the kinetic energy
            per nucleon, you should multiply it by the mass number.
        mass : float, default MASS_NEUTRON
            Mass of the particle in MeV/c^2.
        
        Returns
        -------
        transverse_momentum : Callable[[float | np.ndarray], float | np.ndarray
            Function that calculates the transverse momentum for a given rapidity in MeV/c.
        """
        def transverse_momentum(rapidity: float | np.ndarray) -> float | np.ndarray:
            quantity = (kinergy + mass)**2 / np.cosh(rapidity)**2 - mass**2
            return np.sqrt(np.where(quantity > 0, quantity, np.nan))
        return transverse_momentum

    @staticmethod
    def theta_curves() -> tuple[Callable, Callable]:
        """A 2-tuple of theta curves that sets the boundaries of the phase space.

        Returns
        -------
        theta_curves : tuple[Callable, Callable]
            First element is the curve for the lower boundary, second element is
            the curve for the upper boundary.
        """
        return tuple(LabPtransverseRapidity.theta_curve(theta) for theta in LabPtransverseRapidity.theta_range)
    
    @staticmethod
    def kinergy_curves() -> tuple[Callable, Callable]:
        """A 2-tuple of kinergy curves that sets the boundaries of the phase space.

        Returns
        -------
        kinergy_curves : tuple[Callable, Callable]
            First element is the curve for the lower boundary, second element is
            the curve for the upper boundary.
        """
        return tuple(LabPtransverseRapidity.kinergy_curve(kinergy) for kinergy in LabPtransverseRapidity.kinergy_range)

    @staticmethod
    def is_inside(rapidity: float | np.ndarray, transverse_momentum: float | np.ndarray) -> bool | np.ndarray:
        """Check if the given points are inside the phase space.

        Under the hood, this method converts back both rapidity and transverse
        momentum into lab (kinetic) energy and lab theta angle. Then it checks
        if the given points are inside the energy cuts and theta cuts.

        Parameters
        ----------
        rapidity : float or np.ndarray
            The (unnormalized) rapidity in lab frame.
        transverse_momentum : float or np.ndarray
            The transverse momentum in MeV/c.

        Returns
        -------
        is_inside : bool or np.ndarray
            True if the given points are inside the phase space, False
            otherwise. Points on the boundary are considered inside.
        """
        rapidity, transverse_momentum = map(np.array, (rapidity, transverse_momentum))
        kinergy = np.sqrt(transverse_momentum**2 + MASS_NEUTRON**2) * np.cosh(rapidity) - MASS_NEUTRON
        theta = np.arctan2(transverse_momentum, (kinergy + MASS_NEUTRON) * np.tanh(rapidity))
        return np.all([
            theta >= LabPtransverseRapidity.theta_range[0],
            theta <= LabPtransverseRapidity.theta_range[1],
            kinergy >= LabPtransverseRapidity.kinergy_range[0],
            kinergy <= LabPtransverseRapidity.kinergy_range[1],
        ], axis=0)

    @staticmethod
    def coverage_curve(rapidity_range: tuple[float, float]) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Returns a function that calculates the phase space coverage for a given transverse momentum.

        Parameters
        ----------
        rapidity_range : tuple[float, float]
            The range of unnormalized rapidity in lab frame.
        
        Returns
        -------
        coverage : Callable[[float | np.ndarray], float | np.ndarray]
            A function that calculates the coverage of the phase space for a
            given transverse momentum in MeV/c.
        """
        is_inside = LabPtransverseRapidity.is_inside

        # rough grid to estimate the range
        rapidity = np.linspace(*rapidity_range, 15)
        ptransverse = np.linspace(0, 800, 800 * 5)
        y, pt = np.meshgrid(rapidity, ptransverse)
        mask = np.any(is_inside(y, pt), axis=1)

        # fine grid
        rapidity = np.linspace(*rapidity_range, 500)
        delta_ptranverse = ptransverse[1] - ptransverse[0]
        fine_ptransverse_range = (ptransverse[mask].min() - delta_ptranverse, ptransverse[mask].max() + delta_ptranverse)
        ptransverse = np.linspace(*fine_ptransverse_range, 500)
        y, pt = np.meshgrid(rapidity, ptransverse)

        coverage = np.sum(is_inside(y, pt), axis=1) / pt.shape[1]
        return lambda _pt: np.interp(_pt, ptransverse, coverage, left=0.0, right=0.0)
    
    @staticmethod
    def get_average_coverages(rapidity_range: tuple[float, float], transverse_momentum_ranges: np.ndarray) -> np.ndarray:
        """Get average phase space coverages at various transverse momentum ranges.

        Parameters
        ----------
        rapidity_range : tuple[float, float]
            The range of unnormalized rapidity in lab frame.
        transverse_momentum_ranges : arary of shape (n, 2)
            The transverse momentum ranges in MeV/c.
        """
        curve = LabPtransverseRapidity.coverage_curve(rapidity_range)
        avg_coverages = []
        for pt_range in transverse_momentum_ranges:
            avg_coverages.append(quad(curve, *pt_range, epsabs=1e-4, epsrel=1e-4)[0] / (pt_range[1] - pt_range[0]))
        return np.array(avg_coverages)
    
    def get_spectrum_2d(
        self,
        norm_rapidity_range=(0.0, 1.0),
        norm_rapidity_bins=200,
        ptransverse_range=(0.0, 800.0), # MeV/c
        ptransverse_bins=800 * 5,
        inplace=True,
    ) -> Optional[ROOT.TH2]:
        """Get the 2D histogram of transverse momentum vs. beam-normalized rapidity.

        Parameters
        ----------
        norm_rapidity_range : 2-tuple of float, default (0.0, 1.0)
            Range of beam-normalized rapidity in lab frame.
        norm_rapidity_bins : int, default 200
            Number of bins for the beam-normalized rapidity.
        ptransverse_range : 2-tuple of float, default (0.0, 800.0)
            Range of transverse momentum in MeV/c.
        ptransverse_bins : int, default 800 * 5
            Number of bins for the transverse momentum.
        inplace : bool, default True
            When True, the histogram will be stored in :py:attr:`self.lazy` and
            evaluated lazily. When False, the histogram will be evaluated
            immediately and returned.
        
        Returns
        -------
        hist_2d : ROOT.TH2 or None
            The 2D histogram. If ``inplace=True``, returns None.
        """
        if self.rdf is None:
            raise RuntimeError('RDataFrame has not been built. Call build_rdataframe() first.')

        histo = (self.rdf
            .Define('hx', 'norm_rapidity[base_cut]')
            .Define('hy', 'transverse_momentum[base_cut]')
            .Histo2D(
                ('', '', norm_rapidity_bins, *norm_rapidity_range, ptransverse_bins, *ptransverse_range),
                'hx', 'hy',
            )
        )

        if inplace: # lazy
            self.lazy['hist_2d'] = histo
        else:
            return histo.GetValue()

    def get_spectrum_1d(
        self,
        norm_rapidity_range: tuple[float, float],
        ptransverse_range: tuple[float, float],
        ptransverse_bins: int,
        geometric_efficiency=True,
        intrinsic_efficiency=True,
        background_subtraction=True,
        coverage_correction=True,
    ) -> Optional[pd.DataFrame]:
        """Get the lab transverse momentum spectrum at a given rapidity range.

        Parameters
        ----------
        norm_rapidity_range : 2-tuple of float
            Range of beam-normalized rapidity in lab frame.
        ptransverse_range : 2-tuple
            Histogram range of :math:`p_T/A` in MeV/c.
        ptransverse_bins : int
            Number of bins for the histogram.

        Returns
        -------
        spectrum : pandas.DataFrame or None
            DataFrame with columns 'x', 'y', 'yerr' and 'yferr'. If
            ``append_to_lazy=True``, returns None.

        Examples
        --------
        >>> from e15190.neutron_wall import spectra
        >>> spec = spectra.LabPtransverseRapidity(list(range(4224, 4245 + 1)))
        >>> spec.build_rdataframe()
        >>> spec.get_spectrum_2d(inplace=True) # this is needed before you can get 1D spectrum
        >>> df = spec.get_spectrum_1d(
        ...     norm_rapidity_range=(0.4, 0.6),
        ...     ptransverse_range=(100, 600),
        ...     ptransverse_bins=500 // 20,
        ...     geometric_efficiency=True,
        ...     intrinsic_efficiency=True,
        ...     background_subtraction=True,
        ...     coverage_correction=True,
        ... )
        """
        if 'hist_2d' not in self.lazy:
            self.get_spectrum_2d(inplace=True)

        hist2d = self.lazy['hist_2d'].GetValue()
        df_hist2d = (rt.histo_conversion(hist2d, keep_zeros=False, ignore_errors=True)
            .rename(columns={
                'x': 'norm_rapidity',
                'y': 'ptransverse',
                'z': 'nparticles',
            })
            .query(' and '.join([
                f'norm_rapidity > {norm_rapidity_range[0]}',
                f'norm_rapidity < {norm_rapidity_range[1]}',
                f'ptransverse > {ptransverse_range[0]}',
                f'ptransverse < {ptransverse_range[1]}',
            ]))
            .reset_index(drop=True)
        )

        # add columns of theta and kinergy
        df_hist2d['theta'] = np.arctan2(df_hist2d.ptransverse, np.sqrt(df_hist2d.ptransverse**2 + MASS_NEUTRON**2) * np.sinh(df_hist2d.norm_rapidity * self.beam_lab_rapidity)) # radian
        df_hist2d['kinergy'] = np.sqrt(df_hist2d.ptransverse**2 + MASS_NEUTRON**2) * np.cosh(df_hist2d.norm_rapidity * self.beam_lab_rapidity) - MASS_NEUTRON

        df_hist2d['weight'] = 1.0

        if geometric_efficiency:
            func = nw_geom.Wall('B', contain_pyrex=False).get_geometry_efficiency(
                shadowed_bars=self.shadow_bar_present,
                skip_bars=[0, 25],
            )
            _eff = func(df_hist2d.theta) # type: ignore
            df_hist2d['weight'] = np.divide(df_hist2d.weight, _eff, where=(_eff != 0), out=np.zeros_like(_eff))
        
        if intrinsic_efficiency:
            func = nw_eff.geNt4.get_intrinsic_efficiency()
            _eff = func(df_hist2d.kinergy) # type: ignore
            df_hist2d['weight'] = np.divide(df_hist2d.weight, _eff, where=(_eff != 0), out=np.zeros_like(_eff))
        
        if background_subtraction:
            func = nw_shade.Shadow.get_background_function(beam=self.beam, target=self.target, beam_energy=self.beam_energy)
            _eta = func(df_hist2d.kinergy) # type: ignore
            df_hist2d['weight'] *= 1 - _eta
        
        # normalization
        df_hist2d['weight'] /= self.lazy['tdc_count'].GetValue()
        df_hist2d['weight'] /= norm_rapidity_range[1] - norm_rapidity_range[0]
        df_hist2d['weight'] /= (ptransverse_range[1] - ptransverse_range[0]) / ptransverse_bins
        
        # bin into 1D histogram
        df_hist2d['z'] = df_hist2d.weight * df_hist2d.nparticles
        ptransverse_bin_width = (ptransverse_range[1] - ptransverse_range[0]) / ptransverse_bins
        df_hist2d['bin_idx'] = list(map(int, (df_hist2d.ptransverse - ptransverse_range[0]) / ptransverse_bin_width))
        df_hist2d = df_hist2d.drop(columns=['norm_rapidity', 'ptransverse', 'theta', 'kinergy', 'weight'])
        df_hist2d = df_hist2d.groupby('bin_idx').sum().reset_index()
        df_hist2d['ptransverse'] = (df_hist2d.bin_idx + 0.5) * ptransverse_bin_width + ptransverse_range[0]
        df_hist2d['zerr'] = df_hist2d.z / np.sqrt(df_hist2d.nparticles)
        df_hist2d['zferr'] = df_hist2d.zerr / df_hist2d.z

        if coverage_correction:
            ptransverse_edges = np.linspace(*ptransverse_range, ptransverse_bins + 1)
            ptransverse_ranges = np.vstack([ptransverse_edges[:-1], ptransverse_edges[1:]]).T
            coverages = self.get_average_coverages(np.array(norm_rapidity_range) * self.beam_lab_rapidity, ptransverse_ranges)

            _eff = np.array(coverages[df_hist2d.bin_idx])
            df_hist2d['z'] /= _eff
            df_hist2d['zerr'] /= _eff
        return df_hist2d
