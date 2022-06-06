from copy import copy
import functools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    def __init__(self, reaction):
        reac_parser = query.ReactionParser()
        self.reaction = reac_parser.convert(reaction, 'Aa10Bb20E100')

        with open(e15190.DATABASE_DIR / 'local_paths.json', 'r') as file:
            content = json.load(file)
        self.directory = Path(content[self.PATH_KEY])
        self.filename = 'f1_MergedData_bHat_0.00_0.40.root'
        self.path = self.directory / self.reaction / self.filename
    
    @functools.lru_cache()
    def get_all_particles(self):
        with rt6.TFile(self.path) as file:
            get_particle = lambda name: name.split('_')[-1]
            return list(dict.fromkeys([
                get_particle(obj.GetName()) for obj in file.GetListOfKeys()
            ]).keys())

    def convert_particle(self, notation):
        """Convert notation into Rensheng's convention.
        """
        all_particles = {
            ame.get_A_Z(particle, simple_tuple=True): particle
            for particle in self.get_all_particles()
        }
        isotope = ame.get_A_Z(notation, simple_tuple=True)
        return all_particles[isotope]

    @functools.lru_cache()
    def get_all_histograms(self, particle=None):
        with rt6.TFile(self.path) as file:
            if particle is None:
                return [obj.GetName() for obj in file.GetListOfKeys()]
            particle = self.convert_particle(particle)
            return [
                obj.GetName() for obj in file.GetListOfKeys()
                if obj.GetName().endswith('_' + particle)
            ]

    @functools.lru_cache()
    def get_root_histogram(self, h_name=None, particle=None, keyword=None):
        if h_name is None:
            particle = self.convert_particle(particle)
            h_names = self.get_all_histograms(particle)
            h_names = [h_name for h_name in h_names if keyword.lower() in h_name.lower()]
            h_name = max(h_names, key=len)
        with rt6.TFile(self.path) as file:
            hist = file.Get(h_name)
            hist.SetDirectory(0)
            return hist
        
class LabKinergyTheta:
    lab_theta_range = (30.0, 75.0) # degree
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
    lab_theta_range = (30.0, 75.0) # degree
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
            Reaction notation, e.g. "Ca40Ni58E140".
        particle : str
            Particle name
        df_hist : pandas.DataFrame, default None
            DataFrame with columns 'x', 'y', 'z', 'zerr' and 'zferr'. 'x'
            represents the rapidity in lab frame (0 ~ 1), 'y' represents the
            transverse momentum in MeV/c, 'z' represents the cross section in ??
            unit. If None, the program attempts to load histogram automatically
            with the keyword 'rapidity'.
        """
        self.reaction = reaction
        self.particle = particle
        self.hira_file = HiraFile(reaction)
        if df_hist is None:
            self.df_full = self.hira_file.get_root_histogram(particle=particle, keyword='rapidity')
            self.df_full = rt6.histo_conversion(self.df_full)
        else:
            self.df_full = df_hist
    
    @functools.cached_property
    def beam_rapidity(self):
        beam = query.ReactionParser.read_beam(self.reaction)
        target = query.ReactionParser.read_target(self.reaction)
        beam_energy = query.ReactionParser.read_energy(self.reaction)
        return physics.BeamTargetReaction(beam, target, beam_energy).beam_lab_rapidity
    
    @staticmethod
    def theta_curve(theta, mass):
        def total_transverse_momentum(rapidity):
            return np.sin(theta) * mass / np.sqrt((np.cos(theta) / np.tanh(rapidity))**2 - 1)
        return total_transverse_momentum
    
    @staticmethod
    def kinergy_curve(kinergy, mass):
        def total_transverse_momentum(rapidity):
            return np.sqrt((kinergy + mass)**2 / np.cosh(rapidity)**2 - mass**2)
        return total_transverse_momentum

    @property
    def theta_curves(self):
        mass = ame.mass(self.particle)
        return [
            self.theta_curve(np.radians(theta), mass)
            for theta in self.lab_theta_range
        ]
    
    @property
    def kinergy_curves(self):
        mass = ame.mass(self.particle)
        return [
            self.kinergy_curve(kinergy * ame.get_A_Z(self.particle).A, mass)
            for kinergy in self.lab_kinergy_per_A_ranges[self.particle]
        ]

    def is_inside(self, normed_rapidity, pt_over_A):
        x, y = normed_rapidity, pt_over_A
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
