import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from e15190.utilities import styles
styles.set_matplotlib_style(mpl)

class Geometry:
    DATABASE_DIR = '$DATABASE_DIR/microball'
    def __init__(self, build_setup=True):
        if build_setup:
            self.table = self.get_sarantites_table()
            self.build_setup()
    
    @staticmethod
    def get_sarantites_table(path=None):
        if path is None:
            path = Path(os.path.expandvars(Geometry.DATABASE_DIR)) / 'table_sarantites.dat'
        path = Path(path)
        df = pd.read_csv(path, delim_whitespace=True, comment='#')
        df.set_index('ring', inplace=True, drop=True)
        return df
    
    def build_setup(self, rings_and_dets=None):
        if rings_and_dets is None:
            self.rings_and_dets = {
                2: [1, 2, 3, 4, 5, 6, 7, 8, 9,10],
                3: [1, 2,             7, 8, 9,10,11,12],
                4: [1, 2, 3,          7, 8, 9,10,11,12],
                5: [1, 2, 3,          7, 8, 9,10,11,12,13,14],
                7: [1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12],
                8: [1, 2, 3,    5, 6, 7, 8, 9,10],
            }
    
    def _for_yingxun(self):
        df = []
        for ring in self.rings_and_dets:
            for det in self.rings_and_dets[ring]:
                theta_range = self.get_theta_range(ring, det)
                phi_range = self.get_phi_range(ring, det)
                df.append([
                    ring, det,
                    *theta_range, *phi_range,
                    *self.table.loc[ring],
                ])
        df = pd.DataFrame(
            df,
            columns=[
                'ring', 'det',
                'theta_min', 'theta_max',
                'phi_min', 'phi_max',
                *self.table.columns,
            ],
        )
        df.set_index(['ring', 'det'], inplace=True)
        df = df[[
            'theta_min', 'theta_max',
            'phi_min', 'phi_max',
            'p_range_1', 'alpha_range_1',
            'p_range_2', 'alpha_range_2',
        ]]
        return df
    
    def get_theta_range(self, ring, *args):
        theta = self.table.loc[ring]['theta']
        half_theta = self.table.loc[ring]['half_theta']
        return theta - half_theta, theta + half_theta
    
    def get_phi_range(self, ring, det):
        n_dets = self.table.loc[ring]['n_dets']
        if not 1<= det <= n_dets:
            raise ValueError('det must be in [1, n_dets]')
        phi_width = 360.0 / n_dets
        phi = (phi_width) * (det - 1)
        return phi - 0.5 * phi_width, phi + 0.5 * phi_width
    
    def draw_coverage(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Draw the coverage of the Microball.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to `matplotlib.pyplot.subplots`.
        
        Returns
        -------
        fig : matplotlib.pyplot.Figure
            The figure.
        ax : matplotlib.pyplot.Axes
            The axes.
        """
        fig, ax = plt.subplots(**kwargs)
        ring_colors = plt.cm.viridis(np.linspace(0.4, 1, len(self.rings_and_dets)))
        for ir, ring in enumerate(self.rings_and_dets):
            for det in self.rings_and_dets[ring]:
                theta_min, theta_max = self.get_theta_range(ring, det)
                phi_min, phi_max = self.get_phi_range(ring, det)
                ax.add_patch(plt.Rectangle(
                    (phi_min, theta_min),
                    phi_max - phi_min, theta_max - theta_min,
                    fill=True,
                    facecolor=ring_colors[ir],
                    edgecolor='k',
                    linewidth=1,
                ))
                ax.text(
                    phi_min + 0.5 * (phi_max - phi_min),
                    theta_min + 0.5 * (theta_max - theta_min),
                    f'R{ring}-{det:02d}',
                    ha='center', va='center',
                )
        ax.set_title('Microball coverage in E15190-E14030')
        ax.set_xlim(-30, 360)
        ax.set_ylim(0, 180)
        ax.set_xlabel(r'$\phi$ (deg)')
        ax.set_ylabel(r'$\theta$ (deg)')
        return fig, ax

