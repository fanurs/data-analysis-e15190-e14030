from copy import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import e15190
from e15190.runlog import query
from e15190.utilities import root6 as rt6, fast_histogram as fh

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
    
    def get_root_histogram(self, h_name):
        with rt6.TFile(self.path) as file:
            hist = file.Get(h_name)
            hist.SetDirectory(0)
            return hist
        
class TransverseMomentumSpectrum:
    def __init__(self, df_hist, bounds=(0.4, 0.6)):
        """
        Parameters
        ----------
        df_hist : pandas.DataFrame
            DataFrame with columns 'x', 'y', 'z', 'zerr' and 'zferr'. 'x'
            represents the rapidity in lab frame (0 ~ 1), 'y' represents the
            transverse momentum in MeV/c, 'z' represents the cross section in ??
            unit.
        """
        self.df_full = df_hist
        self.bounds = bounds
        self.df_slice = self.df_full.query(f'x >= {self.bounds[0]} & x <= {self.bounds[1]}')
    
    def correct_coverage(self, z_threshold=0, inplace=False):
        df_corrected = self.df_slice.copy()
        for y_val, subdf in self.df_slice.groupby('y'):
            n_total = len(subdf)
            n_nonzeros = np.sum(subdf.z > z_threshold)
            if n_nonzeros == 0:
                continue
            with pd.option_context('mode.chained_assignment', None):
                df_corrected.loc[subdf.index, 'z'] *= n_total / n_nonzeros
        if inplace:
            self.df_slice = df_corrected
        return df_corrected
    
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

    def plot1d(self, ax=None, hist=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if hist is None:
            hist = self.df_slice

        kw = dict(
            range=(0, 600),
            bins=600,
        )
        kw.update(kwargs)

        return fh.plot_histo1d(
            ax.hist,
            hist.y,
            weights=hist.z,
            **kw,
        )
