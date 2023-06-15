from copy import copy
import functools
import json
import os
from pathlib import Path
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import ROOT

import e15190
from e15190.runlog import query
from e15190.utilities import (
    atomic_mass_evaluation as ame,
    dataframe_histogram as dfh,
    fast_histogram as fh,
    root6 as rt6,
    physics,
)

class HiraFile:
    DIRECTORY = '$PROJECT_DIR/database/results/hira_spectra'

    def __init__(self, reaction: str):
        """Create a file object to interact with ROOT files produced by Rensheng
        that contains HiRA spectra.

        Parameters
        ----------
        reaction : str
            Reaction notation, e.g. "ca40ni58e140". Some intellisense has been
            implemented in :py:class:`e15190.runlog.query.ReactionParser` to
            automatically convert the reaction notation into the correct format.
        """
        reac_parser = query.ReactionParser()
        self.reaction = reac_parser.convert(reaction, 'aa10bb20e100')
        self.directory = Path(os.path.expandvars(self.DIRECTORY))
        self.filename = self.reaction + '.root'
        self.path = self.directory / self.filename
    
    @functools.cache
    def get_all_particles(self) -> list[str]:
        """Return a list of all particles symbols in the file.
        """
        with rt6.TFile(self.path) as file:
            get_particle = lambda name: name.split('_')[-1]
            return list(dict.fromkeys([
                get_particle(obj.GetName()) for obj in file.GetListOfKeys()
            ]).keys())

    def convert_particle(self, notation: str) -> str:
        """Convert particle notation into Rensheng's convention.

        The ROOT file contains spectra for multiple particles. This function
        converts user's particle notation into the one that is used to named to
        spectra in the ROOT file.

        Parameters
        ----------
        notation : str
            Particle notation, e.g. "H1", "He4".
        
        Returns
        -------
        converted : str
            Converted particle notation.
        
        Raises
        ------
        KeyError : If the particle is not found in the file.

        Example
        -------
        >>> file = HiraFile('ca48ni64e140')
        >>> file.convert_particle('H1')
        'p'
        """
        all_particles = {
            ame.get_A_Z(particle, simple_tuple=True): particle
            for particle in self.get_all_particles()
        }
        isotope = ame.get_A_Z(notation, simple_tuple=True)
        return all_particles[isotope]

    @functools.cache
    def get_all_histogram_names(self, particle=None) -> list[str]:
        """Return a list of all histogram names in the file.

        Parameters
        ----------
        particle : str, default None
            If None, return all histogram names. Otherwise, return only
            histogram names for the specified particle. The particle notation
            has some intellisense implemented.
        
        Returns
        -------
        histogram_names : list[str]
            List of histogram names.
        """
        with rt6.TFile(self.path) as file:
            if particle is None:
                return [obj.GetName() for obj in file.GetListOfKeys()]
            particle = self.convert_particle(particle)
            return [
                obj.GetName() for obj in file.GetListOfKeys()
                if obj.GetName().endswith('_' + particle)
            ]

    def get_root_histogram(
        self,
        h_name: Union[str, None] = None,
        particle: Union[str, None] = None,
        keyword: Union[str, list[str], None] = None,
        ignore_multiple_matches: bool = False,
    ) -> ROOT.TH2D:
        """Return a ROOT histogram object.

        Provide either the exact histogram name, or a particle with one or more
        keywords, to specify the histogram.

        Parameters
        ----------
        h_name : str, default None
            Histogram name. If None, the histogram name will be inferred from
            the particle and keyword(s). If specified, the particle and
            keyword(s) will be ignored.
        particle : str, default None
            Particle notation, e.g. "H1", "He4". Intellisense implemented.
            Keyword(s) must be provided too if this is specified.
        keyword : str or list[str], default None
            Keyword(s) to search for in the histogram name.
        ignore_multiple_matches : bool, default False
            If True, ignore the case where multiple histograms are found with
            the same keyword(s); only the one with the longest matching name
            will be returned. If False, raise an error if multiple histograms
            are found with the same keyword(s).
        
        Returns
        -------
        hist : ROOT.TH2D
            ROOT's 2D histogram object.
        
        Raises
        ------
        KeyError : No histogram found for ...
        KeyError : Multiple histograms found for ...
        """
        if h_name is None:
            particle = self.convert_particle(particle)
            h_names = self.get_all_histogram_names(particle)
            h_names = [
                h_name for h_name in h_names if all([
                    kw.lower() in h_name.lower()
                    for kw in (keyword if isinstance(keyword, list) else [keyword])
                ])
            ]
            if len(h_names) == 0:
                raise KeyError(f'No histogram found for {particle} with the given keyword(s)')
            if len(h_names) > 1 and not ignore_multiple_matches:
                raise KeyError(f'Multiple histograms found for {particle} with the given keyword(s)')
            h_name = max(h_names, key=len)
        with rt6.TFile(self.path) as file:
            hist = file.Get(h_name)
            hist.SetDirectory(0)
            return hist

_hira_file = HiraFile('ca40ni58e140') # read any file to learn the particle notation convention
convert_particle = _hira_file.convert_particle

class HiraDict(dict):
    """Dictionary of particle in the convention of Hira's file."""
    def __init__(self, arg=None):
        if arg is not None:
            arg = {
                convert_particle(particle): value
                for particle, value in arg.items()
            }
            super().__init__(arg)

    def __getitem__(self, key):
        return super().__getitem__(convert_particle(key))

    def __setitem__(self, key, value):
        super().__setitem__(convert_particle(key), value)

class Spectrum:
    lab_theta_range = np.radians([30.0, 75.0])
    lab_kinergy_per_A_ranges = { # MeV/A
        'p': (20.0, 198.0),
        'd': (15.0, 263 / 2),
        't': (12.0, 312 / 3),
        '3He': (20.0, 200.0),
        '4He': (18.0, 200.0),
        '6He': (13.0, 200.0),
        '6Li': (22.0, 200.0),
        '7Li': (22.0, 200.0),
        '8Li': (22.0, 200.0),
        '7Be': (22.0, 200.0),
        '9Be': (22.0, 200.0),
        '10Be': (22.0, 200.0),
    }

class LabKinergyTheta(Spectrum):
    def __init__(self, reaction, particle, df_hist=None):
        """
        Parameters
        ----------
        reaction : str
            Reaction notation, e.g. "ca40ni58e140".
        particle : str
            Particle name
        df_hist : pandas.DataFrame, default None
            DataFrame with columns 'x', 'y', 'z', 'zerr' and 'zferr'. 'x'
            represents the lab kinetic energy, 'y' represents the lab theta
            angle in degree, 'z' represents the cross section in ?? unit. If
            None, the program attempts to load histogram automatically with the
            keyword 'Ekin'.
        """
        self.reaction = reaction
        self.beam = query.ReactionParser.read_beam(self.reaction)
        self.target = query.ReactionParser.read_target(self.reaction)
        self.beam_energy = query.ReactionParser.read_energy(self.reaction)
        self.hira_file = HiraFile(reaction)
        self.particle = self.hira_file.convert_particle(particle)
        if df_hist is None:
            self.df_full = self.hira_file.get_root_histogram(particle=particle, keyword=['Ekin', 'Normalized', 'GeoEff'])
            self.df_full = rt6.histo_conversion(self.df_full)
        else:
            self.df_full = df_hist
    
    def get_lab_kinergy_spectrum(
        self,
        theta_range,
        range=(0, 200),
        bins=200,
    ):
        """
        Parameters
        ----------
        theta_range : 2-tuple
            The range of theta in degree.
        range : 2-tuple, default (0, 200)
            The histogram range of the lab kinetic energy in MeV/c.
        bins : int, default 200
            The number of bins in the histogram.
        
        Returns
        -------
        spectrum : pandas.DataFrame
            DataFrame with columns 'x', 'y', 'yerr', 'yferr'.
        """
        df_slice = self.df_full.query(f'y >= {theta_range[0]} and y <= {theta_range[1]}')
        h = fh.histo1d(df_slice.x, weights=df_slice.z, range=range, bins=bins)
        herr = np.sqrt(fh.histo1d(df_slice.x, weights=df_slice.zerr**2, range=range, bins=bins))

        # normalization
        df_kinergy = np.abs(np.diff(range)) / bins
        df_solid_angle = np.sin(np.mean(np.radians(theta_range))) * np.abs(np.diff(np.radians(theta_range))) * (2 * np.pi)
        d_denom = df_kinergy * df_solid_angle

        x_edges = np.linspace(*range, bins + 1)
        return pd.DataFrame({
            'x': 0.5 * (x_edges[1:] + x_edges[:-1]),
            'y': h / d_denom,
            'yerr': herr / d_denom,
            'yferr': np.divide(
                herr, h,
                out=np.zeros_like(herr),
                where=(h != 0),
            )
        })

    def plot2d(self, ax=None, hist=None, cmap='jet', cut=True, **kwargs):
        if ax is None:
            ax = plt.gca()
        if hist is None:
            hist = self.df_full
        
        if cut:
            hist = hist.query(' & '.join([
                f'x >= {self.lab_kinergy_per_A_ranges[self.particle][0]}',
                f'x <= {self.lab_kinergy_per_A_ranges[self.particle][1]}',
                f'y >= {np.degrees(self.lab_theta_range[0])}',
                f'y <= {np.degrees(self.lab_theta_range[1])}',
            ]))
        
        cmap = copy(plt.cm.get_cmap(cmap))
        cmap.set_under('white')
        kw = dict(
            cmap=cmap,
            range=[(0, 210), (25, 80)],
            bins=[210, 55 * 5],
        )
        kw.update(kwargs)

        return fh.plot_histo2d(
            ax.hist2d,
            hist.x, hist.y,
            weights=hist.z,
            **kw,
        )

class LabPtransverseRapidity(Spectrum):
    def __init__(
        self,
        reaction: str,
        particle: str,
        df_hist: Union[pd.DataFrame, None] = None,
        keyword: Union[str, list[str], None] = None,
    ):
        """
        Parameters
        ----------
        reaction : str
            Reaction notation, e.g. "Ca40Ni58E140".
        particle : str
            Particle name. It will be converted to the notation used in `HiraFile`.
        df_hist : pandas.DataFrame, default None
            DataFrame with columns 'x', 'y', 'z', 'zerr' and 'zferr'. 'x'
            represents the rapidity in lab frame (0 ~ 1), 'y' represents the
            transverse momentum in MeV/c, 'z' represents the cross section in ??
            unit. If None, the program attempts to load histogram from file.
        keyword : str or list of str or None, default None
            When `df_hist` is not specified, these keywords will be used to
            search for the histogram. When None, the program will use the
            default keywords, which are 'rapidity' and 'geoeff'.
        """
        self.reaction = reaction
        self.beam = query.ReactionParser.read_beam(self.reaction)
        self.target = query.ReactionParser.read_target(self.reaction)
        self.beam_energy = query.ReactionParser.read_energy(self.reaction)
        self.hira_file = HiraFile(reaction)
        self.particle = self.hira_file.convert_particle(particle)
        if keyword is None:
            keyword = ['rapidity', 'geoeff']
        if df_hist is None:
            self.df_full = self.hira_file.get_root_histogram(particle=particle, keyword=keyword)
            self.df_full = rt6.histo_conversion(self.df_full)
        else:
            self.df_full = df_hist
    
    @classmethod
    def get_lab_kinergy_per_A_range(cls, particle: str) -> tuple[float, float]:
        A, Z = ame.get_A_Z(particle)
        particle = '?pdt'[A] if Z == 1 else f'{A}{ame.get_symb(Z)}'
        return cls.lab_kinergy_per_A_ranges[particle]

    @functools.cached_property
    def beam_rapidity(self) -> float:
        """The beam rapidity in lab frame.

        Returns the experimentally modified beam rapidity. The beam direction is
        defined as the +z-axis.
        """
        return physics.BeamTargetReaction(self.beam, self.target, self.beam_energy).beam_lab_rapidity
    
    @staticmethod
    def theta_curve(theta: float, mass: float) -> Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]:
        """Return a function that calculates the transverse momentum for a given rapidity.

        The theta and mass (particle) are fixed.

        Parameters
        ----------
        theta : float
            Polar angle in lab frame in radian.
        mass : float
            Mass of the particle in MeV/c^2.

        Returns
        -------
        transverse_momentum : Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]
            Function that calculates the transverse momentum for a given rapidity in MeV/c.
        """
        def transverse_momentum(rapidity: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
            st = np.sinh(rapidity) * np.tan(theta)
            quantity = 1 - st**2
            return mass * st / np.sqrt(np.where(quantity > 0, quantity, np.nan))
        return transverse_momentum
    
    @staticmethod
    def kinergy_curve(kinergy: float, mass: float) -> Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]:
        """Return a function that calculates the transverse momentum for a given rapidity.

        The kinetic energy and mass (particle) are fixed.

        Parameters
        ----------
        kinergy : float
            Kinetic energy in MeV. This should be the total kinetic energy of
            the whole isotope (particle). So if you are given the kinetic energy
            per nucleon, you should multiply it by the mass number.
        mass : float
            Mass of the particle in MeV/c^2.
        
        Returns
        -------
        transverse_momentum : Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]
            Function that calculates the transverse momentum for a given rapidity in MeV/c.
        """
        def transverse_momentum(rapidity: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
            quantity = (kinergy + mass)**2 / np.cosh(rapidity)**2 - mass**2
            return np.sqrt(np.where(quantity > 0, quantity, np.nan))
        return transverse_momentum

    @property
    def theta_curves(self) -> tuple[Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]]:
        """A 2-tuple of theta curves that sets the boundaries of the phase space.

        Returns
        -------
        theta_curves : tuple[Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]]
            First element is the curve for the lower boundary, second element is
            the curve for the upper boundary.
        """
        mass = ame.mass(self.particle)
        return tuple([
            self.theta_curve(theta, mass)
            for theta in self.lab_theta_range
        ])
    
    @property
    def kinergy_curves(self) -> tuple[Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]]:
        """A 2-tuple of kinergy curves that sets the boundaries of the phase space.

        Returns
        -------
        kinergy_curves : tuple[Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]]]
            First element is the curve for the lower boundary, second element is
            the curve for the upper boundary.
        """
        mass = ame.mass(self.particle)
        A = ame.get_A_Z(self.particle).A
        return tuple([
            self.kinergy_curve(kinergy * A, mass)
            for kinergy in self.lab_kinergy_per_A_ranges[self.particle]
        ])

    def is_inside(self, normed_rapidity: Union[float, ArrayLike], pt_over_A: Union[float, ArrayLike]) -> Union[bool, ArrayLike]:
        """Check if the given points are inside the phase space.

        Under the hood, this method converts back both normalized rapidity and
        transverse momentum over A into lab kinetic energy and lab theta angle.
        Then it checks if the given points are inside the energy cuts and theta
        cuts.

        Parameters
        ----------
        normed_rapidity : Union[float, ArrayLike]
            The normalized rapidity in lab frame. Normalization is done by
            dividing the beam rapidity.
        pt_over_A : Union[float, ArrayLike]
            The transverse momentum per nucleon in MeV/c.

        Returns
        -------
        is_inside : Union[bool, ArrayLike[bool]]
            True if the given points are inside the phase space, False
            otherwise. Points on the boundary are considered inside.
        """
        y_hat, pt_A = map(np.array, (normed_rapidity, pt_over_A))
        A = ame.get_A_Z(self.particle).A
        mass = ame.mass(self.particle)
        kinergy = np.sqrt((pt_A * A)**2 + mass**2) * np.cosh(y_hat * self.beam_rapidity) - mass
        theta = np.arctan2(pt_A * A, (kinergy + mass) * np.tanh(y_hat * self.beam_rapidity))
        return np.all([
            theta >= self.lab_theta_range[0],
            theta <= self.lab_theta_range[1],
            kinergy >= self.lab_kinergy_per_A_ranges[self.particle][0] * A,
            kinergy <= self.lab_kinergy_per_A_ranges[self.particle][1] * A,
        ], axis=0)

    def correct_coverage(self, df_slice: pd.DataFrame) -> pd.DataFrame:
        """Correct the coverage of the given slice along the transverse momentum axis.

        Parameters
        ----------
        df_slice : pandas.DataFrame
            The slice to correct, with columns 'x', 'y', 'z', 'zerr'. 'x' is the
            normed rapidity, 'y' is the transverse momentum per nucleon, and 'z'
            is the yield.
        
        Returns
        -------
        df_corrected : pandas.DataFrame
            The corrected slice, with the same columns as the input.
        """
        df_corrected = df_slice.copy().reset_index(drop=True)
        df_corrected['inside'] = self.is_inside(df_slice.x, df_slice.y)
        scalars = np.ones(len(df_corrected))
        groupby = df_corrected[['y', 'inside']].groupby('y').sum().inside
        for y_val, subdf in df_corrected.groupby('y'):
            n_total = len(subdf)
            n_inside = groupby.loc[y_val]
            if n_inside == 0:
                continue
            scalars[np.array(subdf.index)] = n_total / n_inside
        df_corrected['z'] *= scalars
        df_corrected['zerr'] *= scalars
        return df_corrected.drop('inside', axis='columns')
    
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

    def get_ptA_spectrum(
        self,
        rapidity_range: Tuple[float, float],
        correct_coverage: bool,
        range: Tuple[float, float],
        bins: int,
        drop_outliers: float = -1.0,
    ):
        """
        Parameters
        ----------
        rapidity_range : 2-tuple of float
            Range of beam-normalized rapidity in lab frame.
        correct_coverage : bool
            If ``True``, missing data due to geometric coverage will be
            corrected. If ``False``, correction will not be applied.
        range : 2-tuple
            Histogram range of :math:`p_T/A` in MeV/c.
        bins : int
            Number of bins for the histogram.
        drop_outliers : float, default -1.0
            Outliers are defined as points whose ratio to the neighboring point
            is larger than `drop_outliers`. If negative, no outliers will be
            dropped.

        Returns
        -------
        spectrum : pandas.DataFrame
            DataFrame with columns 'x', 'y', 'yerr' and 'yferr'.
        """
        df_slice = self.df_full.query(f'x >= {rapidity_range[0]} & x <= {rapidity_range[1]}')
        if correct_coverage:
            df_slice = self.correct_coverage(df_slice)

        h = fh.histo1d(df_slice.y, weights=df_slice.z, range=range, bins=bins)
        herr = np.sqrt(fh.histo1d(df_slice.y, weights=df_slice.zerr**2, range=range, bins=bins))

        # normalization
        d_rapidity = np.abs(np.diff(rapidity_range))
        d_transverse_momentum = np.abs(np.diff(range)) / bins
        d_denom = d_rapidity * d_transverse_momentum

        x_edges = np.linspace(*range, bins + 1)
        df = pd.DataFrame({
            'x': 0.5 * (x_edges[:-1] + x_edges[1:]),
            'y': h / d_denom,
            'yerr': herr / d_denom,
            'yferr': np.divide(
                herr, h,
                out=np.zeros_like(herr),
                where=(h != 0),
            )
        })
        if drop_outliers <= 0.0:
            return df
        return self._drop_spectrum_outliers(df, drop_outliers, 'y')
    
    def plot2d(self, ax=None, hist=None, cmap='jet', **kwargs):
        if ax is None:
            ax = plt.gca()
        if hist is None:
            hist = self.df_full

        cmap = copy(plt.cm.get_cmap(cmap))
        cmap.set_under('white')
        kw = dict(
            cmap=cmap,
            range=[(0, 1), (0, 600)],
            bins=[100, 600],
        )
        kw.update(kwargs)

        return fh.plot_histo2d(
            ax.hist2d,
            hist.x, hist.y,
            weights=hist.z,
            **kw,
        )

    def plot1d_ptA(
        self,
        ax=None,
        hist=None,
        rapidity_range=(0.4, 0.6),
        correct_coverage=True,
        range=(0, 600),
        bins=30,
        drop_outliers: float = 10.0,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()
        if hist is None:
            hist = self.get_ptA_spectrum(
                rapidity_range=rapidity_range,
                correct_coverage=correct_coverage,
                range=range,
                bins=bins,
                drop_outliers=drop_outliers,
            )

        kw = dict(
            fmt='.',
        )
        kw.update(kwargs)
        return ax.errorbar(hist.x, hist.y, yerr=hist.yerr, **kw)


def _trim_x_range(*dfs: pd.DataFrame) -> list[pd.DataFrame]:
    x_min = max(df.x.min() for df in dfs)
    x_max = min(df.x.max() for df in dfs)
    return [df.query(f'{x_min} <= x <= {x_max}').reset_index(drop=True) for df in dfs]

class PseudoNeutron:
    """A class to create pseudo-neutron spectrum from light charged particles.
    
    Examples
    --------
    >>> from e15190.hira.spectra import LabPtransverseRapidity, PseudoNeutron
    >>> pt_spec = LabPtransverseRapidity('ca40ni58e140', 'p')
    >>> spectra = {
    ...     particle: pt_spec.get_ptA_spectrum(
    ...         rapidity_range=(0.4, 0.6),
    ...         correct_coverage=True,
    ...         range=(200, 400),
    ...         bins=10,
    ...         drop_outliers=10.0,
    ...     )
    ...     for particle in ['p', 'd', 't', 'He3']
    ... }
    >>> print(PseudoNeutron(spectra).get_spectrum(
    ...     fit_range=(200, 350),
    ...     switch_point=350,
    ... ))
        x         y      yerr     yferr
    0  210.0  0.031069  0.000265  0.008542
    1  230.0  0.027968  0.000289  0.010341
    2  250.0  0.025876  0.000311  0.012022
    3  270.0  0.021639  0.000297  0.013733
    4  290.0  0.018715  0.000295  0.015768
    5  310.0  0.015789  0.000286  0.018140
    6  330.0  0.013411  0.000283  0.021093
    7  350.0  0.010500  0.000311  0.029595
    8  370.0  0.007961  0.000063  0.007875
    9  390.0  0.005974  0.000062  0.010353
    """

    def __init__(self, spectra: dict[str, pd.DataFrame]):
        self.spectra = HiraDict({
            particle: spectrum
            for particle, spectrum in spectra.items()
        })
    
    def get_spectrum_triton_helium3(self) -> pd.DataFrame:
        particles_needed = ['p', 't', 'He3']
        if set(map(convert_particle, particles_needed)) - set(map(convert_particle, self.spectra.keys())) != set():
            raise ValueError(f'Not all of {particles_needed} are given in `self.spectra`.')
        p, t, He3 = _trim_x_range(self.spectra['p'], self.spectra['t'], self.spectra['He3'])
        return dfh.mul(p, dfh.div(t, He3))
    
    def get_spectrum_deuteron_proton(self, scalar=1.0) -> pd.DataFrame:
        particles_needed = ['p', 'd']
        if set(map(convert_particle, particles_needed)) - set(map(convert_particle, self.spectra.keys())) != set():
            raise ValueError(f'Not all of {particles_needed} are given in `self.spectra`.')
        p, d = _trim_x_range(self.spectra['p'], self.spectra['d'])
        return dfh.mul(dfh.div(d, p), scalar)
    
    def get_spectrum(
        self,
        fit_range: tuple[float, float],
        switch_point: float,
    ) -> pd.DataFrame:
        """Get the pseudo-neutron spectrum.

        This method "stitiches" the deuteron-proton spectrum to the
        triton-helium3 spectrum by a scaling factor. The scaling factor is
        determined by a simple scaling fit.

        Parameters
        ----------
        fit_range : tuple[float, float]
            The range to fit the deuteron-proton spectrum to the triton-helium3
            spectrum. The limits are inclusive.
        switch_point : float
            The point where the two spectra are combined. If both triton-helium3
            and deuteron-proton spectra contain this point, the point from the
            in the triton-helium3 is used.
        """
        t_he3 = self.get_spectrum_triton_helium3()
        d_p = self.get_spectrum_deuteron_proton(scalar=1.0)
        
        # fit deutron-proton to triton-helium3 by a scaling factor
        t_he3_fit = t_he3.query(f'{fit_range[0]} <= x <= {fit_range[1]}')
        d_p_fit = d_p.query(f'{fit_range[0]} <= x <= {fit_range[1]}')
        scalar = (d_p_fit.y / t_he3_fit.y).mean()

        # combine the two spectra
        t_he3 = t_he3.query(f'x <= {switch_point}')
        d_p = dfh.mul(d_p.query(f'x > {switch_point}'), 1 / scalar)
        return pd.concat([t_he3, d_p], ignore_index=True)

class CoalescenseInvariant:
    def __init__(self, spectra: dict[str, pd.DataFrame]):
        self.spectra = HiraDict({})
        for key, val in spectra.items():
            try:
                self.spectra[key] = val
            except:
                pass
    
    def get_proton_spectrum(self) -> pd.DataFrame:
        particles_needed = ['p', 'd', 't', 'He3', 'He4']
        if set(map(convert_particle, particles_needed)) - set(map(convert_particle, self.spectra.keys())) != set():
            raise ValueError(f'Not all of {particles_needed} are given in `self.spectra`.')
        p, d, t, He3, He4 = _trim_x_range(self.spectra['p'], self.spectra['d'], self.spectra['t'], self.spectra['He3'], self.spectra['He4'])
        return dfh.sum(p, d, t, dfh.mul(He3, 2), dfh.mul(He4, 2))

    def get_neutron_spectrum(self, **kwargs) -> pd.DataFrame:
        particles_needed = ['p', 'd', 't', 'He3', 'He4']
        if set(map(convert_particle, particles_needed)) - set(map(convert_particle, self.spectra.keys())) != set():
            raise ValueError(f'Not all of {particles_needed} are given in `self.spectra`.')
        pseudo_n = PseudoNeutron({k: self.spectra[k] for k in ['p', 'd', 't', 'He3']}).get_spectrum(**kwargs)
        pseudo_n, d, t, He3, He4 = _trim_x_range(pseudo_n, self.spectra['d'], self.spectra['t'], self.spectra['He3'], self.spectra['He4'])
        return dfh.sum(pseudo_n, d, dfh.mul(t, 2), He3, dfh.mul(He4, 2))
