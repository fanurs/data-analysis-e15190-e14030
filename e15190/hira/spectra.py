from copy import copy
import functools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import e15190
from e15190.runlog import query
from e15190.utilities import atomic_mass_evaluation as ame, fast_histogram as fh, root6 as rt6, physics

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
    
    def get_all_histograms(self, particle=None):
        with rt6.TFile(self.path) as file:
            if particle is None:
                return [obj.GetName() for obj in file.GetListOfKeys()]
            return [
                obj.GetName() for obj in file.GetListOfKeys()
                if obj.GetName().endswith('_' + particle)
            ]
    
    def get_all_particles(self):
        with rt6.TFile(self.path) as file:
            get_particle = lambda name: name.split('_')[-1]
            return list(dict.fromkeys([
                get_particle(obj.GetName()) for obj in file.GetListOfKeys()
            ]).keys())
    
    def get_root_histogram(self, h_name=None, particle=None, keyword=None):
        if h_name is None:
            h_names = self.get_all_histograms(particle)
            h_names = [h_name for h_name in h_names if keyword.lower() in h_name.lower()]
            h_name = max(h_names, key=len)
        with rt6.TFile(self.path) as file:
            hist = file.Get(h_name)
            hist.SetDirectory(0)
            return hist
        
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

    def __init__(self, particle, reaction, df_hist):
        """
        Parameters
        ----------
        particle : str
            Particle name.
        reaction : str
            Reaction notation, e.g. "Ca40Ni58E140".
        df_hist : pandas.DataFrame
            DataFrame with columns 'x', 'y', 'z', 'zerr' and 'zferr'. 'x'
            represents the rapidity in lab frame (0 ~ 1), 'y' represents the
            transverse momentum in MeV/c, 'z' represents the cross section in ??
            unit.
        """
        self.particle = particle
        self.reaction = reaction
        self.df_full = df_hist
    
    @functools.cached_property
    def beam_rapidity(self):
        beam = query.ReactionParser.read_beam(self.reaction)
        beam_energy = query.ReactionParser.read_energy(self.reaction)

        beam_mass = ame.mass(ame.get_A_Z(beam))
        beam_ene = beam_energy * (beam_mass / ame.amu.to('MeV').value) + beam_mass
        beam_momentum = physics.energy_to_momentum(beam_mass, beam_ene)
        return physics.rapidity(beam_ene, beam_momentum)
    
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

    def correct_coverage(self, df_slice, z_threshold=0):
        df_corrected = df_slice.copy()
        for _, subdf in df_slice.groupby('y'):
            n_total = len(subdf)
            n_inside = np.sum(self.is_inside(subdf.x, subdf.y) & (subdf.z > z_threshold))
            if n_inside == 0:
                continue
            with pd.option_context('mode.chained_assignment', None):
                df_corrected.loc[subdf.index, 'z'] *= n_total / n_inside
                df_corrected.loc[subdf.index, 'zerr'] *= n_total / n_inside
        return df_corrected
    
    def get_ptA_spectrum(
        self,
        rapidity_range=(0.4, 0.6),
        z_threshold=0,
        correct_coverage=True,
        hrange=(0, 600),
        bins=600,
    ):
        df_slice = self.df_full.query(f'x >= {rapidity_range[0]} & x <= {rapidity_range[1]}')
        if correct_coverage:
            df_slice = self.correct_coverage(df_slice, z_threshold)

        h = fh.histo1d(df_slice.y, weights=df_slice.z, range=hrange, bins=bins)
        herr = np.sqrt(fh.histo1d(df_slice.y, weights=df_slice.zerr**2, range=hrange, bins=bins))

        # normalization
        d_rapidity = np.abs(np.diff(rapidity_range))
        d_transverse_momentum = np.abs(np.diff(hrange)) / bins

        return pd.DataFrame({
            'x': np.linspace(*hrange, bins),
            'y': h / (d_rapidity * d_transverse_momentum),
            'yerr': herr / (d_rapidity * d_transverse_momentum),
            'fyerr': np.divide(
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

    def plot1d_ptA(self, ax=None, hist=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if hist is None:
            hist = self.get_ptA_spectrum(
                hrange=(0, 600),
                bins=30,
            )

        kw = dict(
            fmt='.',
        )
        kw.update(kwargs)
        return ax.errorbar(hist.x, hist.y, yerr=hist.yerr, **kw)
