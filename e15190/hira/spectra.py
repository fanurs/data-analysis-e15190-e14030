from copy import copy
import functools
import json
from pathlib import Path
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import ROOT

import e15190
from e15190.runlog import query
from e15190.utilities import (
    atomic_mass_evaluation as ame,
    fast_histogram as fh,
    root6 as rt6,
    physics,
)

class HiraFile:
    PATH_KEY = 'rensheng_hira_root_files_dir'

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
        self.reaction = reac_parser.convert(reaction, 'Aa10Bb20E100')

        with open(e15190.DATABASE_DIR / 'local_paths.json', 'r') as file:
            content = json.load(file)
        self.directory = Path(content[self.PATH_KEY])
        self.filename = 'f1_MergedData_bHat_0.00_0.40.root'
        self.path = self.directory / self.reaction / self.filename
    
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
        
class LabKinergyTheta:
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
            angle in degree, 'z' represents the cross section in ??  unit. If
            None, the program attempts to load histogram automatically with the
            keyword 'Ekin'.
        """
        self.reaction = reaction
        self.hira_file = HiraFile(reaction)
        self.particle = self.hira_file.convert_particle(particle)
        if df_hist is None:
            self.df_full = self.hira_file.get_root_histogram(particle=particle, keyword='Ekin')
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
                f'y >= {self.lab_theta_range[0]}',
                f'y <= {self.lab_theta_range[1]}',
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

class LabPtransverseRapidity:
    lab_theta_range = np.radians([30.0, 75.0])
    lab_kinergy_per_A_ranges = { # MeV/A
        'p': (20.0, 198.0),
        'd': (15.0, 263.0 / 2),
        't': (12.0, 312.0 / 3),
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
        beam = query.ReactionParser.read_beam(self.reaction)
        target = query.ReactionParser.read_target(self.reaction)
        beam_energy = query.ReactionParser.read_energy(self.reaction)
        return physics.BeamTargetReaction(beam, target, beam_energy).beam_lab_rapidity
    
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

        Parameters
        ----------
        normed_rapidity : Union[float, ArrayLike]
            The normed rapidity in lab frame. Normalization is done by dividing
            the beam rapidity.
        pt_over_A : Union[float, ArrayLike]
            The transverse momentum per nucleon in MeV/c.

        Returns
        -------
        is_inside : Union[bool, ArrayLike[bool]]
            True if the given points are inside the phase space, False
            otherwise. Points on the boundary are considered inside.
        """
        x, y = map(np.array, (normed_rapidity, pt_over_A))
        A = ame.get_A_Z(self.particle).A
        mass = ame.mass(self.particle)
        kinergy = np.sqrt((y * A)**2 + mass**2) * np.cosh(x * self.beam_rapidity) - mass
        theta = np.arctan2(y * A, (kinergy + mass) * np.tanh(x * self.beam_rapidity))
        return np.all([
            np.degrees(theta) >= self.lab_theta_range[0],
            np.degrees(theta) <= self.lab_theta_range[1],
            kinergy >= self.lab_kinergy_per_A_ranges[self.particle][0] * A,
            kinergy <= self.lab_kinergy_per_A_ranges[self.particle][1] * A,
        ], axis=0)

    def correct_coverage(self, df_slice):
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
    
    def get_ptA_spectrum(
        self,
        rapidity_range=(0.4, 0.6),
        correct_coverage=True,
        range=(0, 600),
        bins=600,
    ):
        """
        Parameters
        ----------
        rapidity_range : 2-tuple, default is (0.4, 0.6)
            Range of beam-normalized rapidity in lab frame.
        correct_coverage : bool, default is True
            If ``True``, missing data due to geometric coverage will be
            corrected. If ``False``, correction will not be applied.
        range : 2-tuple, default is (0, 600)
            Histogram range of :math:`p_T/A` in MeV/c.
        bins : int, default is 600
            Number of bins for the histogram.

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
        return pd.DataFrame({
            'x': 0.5 * (x_edges[:-1] + x_edges[1:]),
            'y': h / d_denom,
            'yerr': herr / d_denom,
            'yferr': np.divide(
                herr, h,
                out=np.zeros_like(herr),
                where=(h != 0),
            )
        })
    
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
            )

        kw = dict(
            fmt='.',
        )
        kw.update(kwargs)
        return ax.errorbar(hist.x, hist.y, yerr=hist.yerr, **kw)
