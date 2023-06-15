#!/usr/bin/env python3
from __future__ import annotations
import inspect
import io
import json
from os.path import expandvars
from pathlib import Path
import subprocess
from typing import Callable, Literal, Optional

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.interpolate import UnivariateSpline

from e15190.utilities import (
    cache,
    geometry as geom,
    tables,
)

class Bar(geom.RectangularBar):
    edges_x = (-90.0, 90.0)
    left_shadow_x = (-50.0, -15.0)
    right_shadow_x = (15.0, 50.0)
    shadowed_bars = [7, 8, 9, 10, 15, 16, 17]

    def __init__(self, vertices, contain_pyrex=True):
        """Construct a Neutron Wall bar, either from Wall A or Wall B.

        This class only deals with individual bar. The Neutron Wall object is
        implemented in :py:class`Wall`.

        Parameters
        ----------
        vertices : ndarray of shape (8, 3)
            Exactly eight (x, y, z) lab coordinates measured in centimeter (cm).
            As long as all eight vertices are distinct, i.e. can properly define
            a cuboid, the order of vertices does not matter because PCA will be
            applied to automatically identify the 1st, 2nd and 3rd principal
            axes of the bar, which, for Neutron Wall bar, corresponds to the x,
            y and z directions in the local (bar) coordinate system.
        """
        super().__init__(vertices, make_vertices_dict=False)

        # 1st principal defines local-x, should be opposite to lab-x
        if np.dot(self.pca.components_[0], [-1, 0, 0]) < 0:
            self.pca.components_[0] *= -1

        # 2nd principal defines local-y, should nearly parallel to lab-y
        if np.dot(self.pca.components_[1], [0, 1, 0]) < 0:
            self.pca.components_[1] *= -1

        # 3rd principal defines local-z, direction is given by right-hand rule
        x_cross_y = np.cross(self.pca.components_[0], self.pca.components_[1])
        if np.dot(self.pca.components_[2], x_cross_y) < 0:
            self.pca.components_[2] *= -1

        self._make_vertices_dict()

        self.contain_pyrex = True # always starts with True
        """bool : Status of Pyrex thickness"""

        self.pyrex_thickness = 2.54 / 8 # cm
        """float : The thickness of the Pyrex (unit cm)
        
        Should be 0.125 inches or 0.3175 cm.
        """

        # remove Pyrex if requested
        if not contain_pyrex:
            self.remove_pyrex()
        else:
            # database always have included Pyrex, so no action is required
            pass
    
    @property
    def length(self) -> float:
        """Length of the bar.

        Should be around 76 inches without Pyrex.
        """
        result = self.dimension(index=0)
        if not isinstance(result, float):
            raise Exception('Length is not a float')
        return result
    
    @property
    def height(self) -> float:
        """Height of the bar.

        Should be around 3.0 inches without Pyrex.
        """
        result = self.dimension(index=1)
        if not isinstance(result, float):
            raise Exception('Height is not a float')
        return result

    @property
    def thickness(self) -> float:
        """Longitudinal thickness of the bar.

        Should be around 2.5 inches without Pyrex.
        """
        result = self.dimension(index=2)
        if not isinstance(result, float):
            raise Exception('Thickness is not a float')
        return result
    
    def _modify_pyrex(self, mode):
        """To add or remove Pyrex thickness.

        Parameters
        ----------
        mode : 'add' or 'remove'
            If 'add', the Pyrex thickness will be added to the bar; if 'remove',
            the Pyrex thickness will be removed from the bar.

            An Exception will be raised if bar is already in the desired state.
        """
        if mode == 'remove':
            scalar = -1
        elif mode == 'add':
            scalar = +1
        else:
            raise ValueError(f'Unknown mode "{mode}"')
        
        # apply transformation
        self.loc_vertices = {
            iv: v + scalar * self.pyrex_thickness * np.array(iv)
            for iv, v in self.loc_vertices.items()
        }
        self.vertices = {
            iv: np.squeeze(self.to_lab_coordinates(v))
            for iv, v in self.loc_vertices.items()
        }

        # update status
        self.contain_pyrex = (not self.contain_pyrex)

    def remove_pyrex(self):
        """Remove Pyrex thickness from the bar."""
        if not self.contain_pyrex:
            raise Exception('Pyrex has already been removed')
        self._modify_pyrex('remove')
    
    def add_pyrex(self):
        """Add Pyrex thickness to the bar."""
        if self.contain_pyrex:
            raise Exception('Pyrex has already been added')
        self._modify_pyrex('add')

    def randomize_from_local_x(
        self,
        local_x,
        return_frame='lab',
        local_ynorm: float | tuple[float, float] = (-0.5, 0.5),
        local_znorm: float | tuple[float, float] = (-0.5, 0.5),
        random_seed: Optional[int] = None,
    ):
        """Returns randomized point(s) from the given local x coordinate(s).

        This is useful for getting a uniform hit distribution within the bulk of
        the bar. Experimentally, we can only determine the local x coordinate
        for each hit.

        Parameters
        ----------
        local_x : float or list of floats
            The local x coordinate(s) of the point(s) in centimeters.
        return_frame : 'lab' or 'local', default 'lab'
            The frame of the returned point(s) in Cartesian coordinates.
        local_ynorm : float or 2-tuple of floats, default (-0.5, 0.5)
            The range of randomization in the local y coordinate. If float, no
            randomization is performed. Center is at `0.0`; the top surface is
            `+0.5`; the bottom surface is `-0.5`. Values outside this range will
            still be calculated, but those points will be outside the bar.
        local_znorm : float or 2-tuple of floats, default (-0.5, 0.5)
            The range of randomization in the local z coordinate. If float, no
            randomization is performed. Center is at `0.0`; the front surface is
            `+0.5`; the back surface is `-0.5`. Values outside this range will
            still be calculated, but those points will be outside the bar.
        random_seed : int, default None
            The random seed to be used. If None, randomization is
            non-reproducible.
        
        Returns
        -------
        randomized_coordinates : 3-tuple or list of 3-tuples
            The randomized point(s) in Cartesian coordinates.
        """
        rng = np.random.default_rng(random_seed)
        n_pts = len(local_x)

        if isinstance(local_ynorm, (int, float)):
            local_y = [local_ynorm * self.height] * n_pts
        else:
            local_y = rng.uniform(*local_ynorm, size=n_pts) * self.height

        if isinstance(local_znorm, (int, float)):
            local_z = [local_znorm * self.thickness] * n_pts
        else:
            local_z = rng.uniform(*local_znorm, size=n_pts) * self.thickness

        local_result = np.array([local_x, local_y, local_z]).T
        if return_frame == 'local':
            return local_result
        else:
            return self.to_lab_coordinates(local_result)

    def get_empirical_distance_bounds(self, n_pts=50_000, n_iters=6):
        """Returns the empirical bounds of the relation between local x and distance.

        Parameters
        ----------
        n_pts : int, default 50_000
            The number of points to be used for the empirical calculation.
        n_iters : int, default 6
            The number of iterations to be used to refine the bounds.

        Returns
        -------
        fit_low : np.array of size 3
            The coefficients of the lower bound quadratic fit.
        fit_high : np.array of size 3
            The coefficients of the upper bound quadratic fit.
        """
        pos = np.linspace(-120, 120, n_pts)
        coords = self.randomize_from_local_x(pos)
        distance = np.sqrt(np.sum(np.square(coords), axis=1))

        fit_mid = np.polyfit(pos, distance, 2)
        fit_low, fit_upp = fit_mid, fit_mid
        for _ in range(n_iters):
            mask = (distance < np.polyval(fit_low, pos))
            x, y = pos[mask], distance[mask]
            fit_low = np.polyfit(x, y, 2)

            mask = (distance > np.polyval(fit_upp, pos))
            x, y = pos[mask], distance[mask]
            fit_upp = np.polyfit(x, y, 2)
        return fit_low[::-1], fit_upp[::-1]

class Wall:
    def __init__(self, AB, contain_pyrex=False, refresh_from_inventor_readings=False):
        """Construct a neutron wall, A or B.

        A neutron wall is a collection of bars, ``self.bars``.

        Parameters
        ----------
        AB : 'A' or 'B'
            The wall to construct. Currently only 'B' is actively maintained and
            tested.
        contain_pyrex : bool, default False
            Whether the bars contain pyrex.
        refresh_from_inventor_readings : bool, default False
            If `True`, the geometry will be loaded from the inventor readings;
            if `False`, the geometry will be loaded from the existing database,
            stored in ``*.dat`` files.
        """
        self.AB = AB.upper()
        """'A' or 'B' : Neutron wall name in uppercase."""

        self.ab = self.AB.lower()
        """'a' or 'b' : Neutron wall name in lowercase."""

        self.database_dir = '$PROJECT_DIR/database/neutron_wall/geometry'
        self.database_dir = Path(expandvars(self.database_dir))
        """``pathlib.Path`` : ``'$PROJECT_DIR/database/neutron_wall/geometry'``"""

        self.path_inventor_readings = self.database_dir / f'inventor_readings_NW{self.AB}.txt'
        """``pathlib.Path`` : ``self.database_dir / f'inventor_readings_NW{self.AB}.txt'``"""

        self.path_vertices = self.database_dir / f'NW{self.AB}_vertices.dat'
        """``pathlib.Path`` : ``self.database_dir / f'NW{self.AB}_vertices.dat'``"""

        self.path_pca = self.database_dir / f'NW{self.AB}_pca.dat'
        """``pathlib.Path`` : ``self.database_dir / f'NW{self.AB}_pca.dat'``"""

        self.database = None
        """``pandas.DataFrame`` : Dataframe of vertices from all bars"""

        self.bars = None
        """``dict[int, Bar]`` : A collection of :py:class:`Bar` objects.
        
        The bottommost is bar 0, the topmost is bar 24. In the experiment, bar 0
        was not used because it was shadowed.
        """

        # if True, read in again from raw inventor readings
        self._refresh_from_inventor_readings = refresh_from_inventor_readings
        if self._refresh_from_inventor_readings:
            bars = self.read_from_inventor_readings(self.path_inventor_readings)
            self.save_vertices_to_database(self.AB, self.path_vertices, bars)
            self.save_pca_to_database(self.AB, self.path_pca, bars)

        # read in from database
        self.database = pd.read_csv(self.path_vertices, comment='#', delim_whitespace=True)
        index_names = [f'nw{self.ab}-bar', 'dir_x', 'dir_y', 'dir_z']
        self.database.set_index(index_names, drop=True, inplace=True)

        # construct a dictionary of bar objects
        bar_nums = sorted(set(self.database.index.get_level_values(f'nw{self.ab}-bar')))
        if self.AB == 'B':
            bar_nums.remove(0)  # bar 0 is not used
        self.contain_pyrex = contain_pyrex
        self.bars = {
            b: Bar(self.database.loc[b][['x', 'y', 'z']], contain_pyrex=self.contain_pyrex)
            for b in bar_nums
        }
    
    @staticmethod
    def read_from_inventor_readings(filepath):
        """Reads in Inventor measurements and returns a list of :py:class:`Bar` objects.

        Parameters
        ----------
        filepath : str or pathlib.Path
            The path to the file containing the Inventor measurements, e.g.
            :py:attr:`self.path_inventor_readings <path_inventor_readings>`.
        
        Returns
        -------
        bars : list
            A list of :py:class:`Bar` objects, sorted from bottom to top.
        """
        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # containers for process the lines
        bars_vertices = [] # to collect vertices of all bars
        vertex: list[Optional[float]] = [None] * 3
        vertices = [] # to collect all vertices of one particular bar

        for line in lines:
            sline = line.strip().split()

            # continue if this line does not contain measurement
            # else it will be in the format of, e.g.
            # X Position 200.0000 cm
            if 'cm' not in sline:
                continue

            coord_index = 'XYZ'.find(sline[0])
            if vertex[coord_index] is None:
                vertex[coord_index] = float(sline[2])
            else:
                raise Exception('ERROR: Trying to overwrite existing vertex component.')
            
            # append and reset if all 3 components of vertex have been read in
            if sum([ele is not None for ele in vertex]) == 3:
                vertices.append(vertex)
                vertex = [None] * 3
            
            # append and reset if all 8 vertices of a bar have been read in
            if len(vertices) == 8:
                bars_vertices.append(vertices)
                vertices = []
        
        # create Bar objects
        bars = [Bar(vertices) for vertices in bars_vertices]
        bars = sorted(bars, key=lambda bar: bar.pca.mean_[1]) # sort from bottom to top
        return bars
    
    @staticmethod
    def save_vertices_to_database(AB, filepath, bars):
        """Saves the vertices of the bars to database.

        Parameters
        ----------
        AB : 'A' or 'B'
            Neutron wall A or B.
        filepath : str or pathlib.Path
            Path to the database file, e.g. :py:attr:`self.path_vertices
            <path_vertices>`.
        bars : list of :py:class:`Bar` objects
            Sorted from bottom to top. The bottommost bar will be numbered 0;
            the topmost bar will be numbered 24.
        """
        # collect all vertices from all bars and save into a dataframe
        df = []
        for bar_num, bar_obj in enumerate(bars):
            for sign, vertex in bar_obj.vertices.items():
                df.append([bar_num, *sign, *vertex])
                
        df = pd.DataFrame(
            df,
            columns=[
                f'nw{AB.lower()}-bar',
                'dir_x', 'dir_y', 'dir_z',
                'x', 'y', 'z',
            ],
        )

        # save to database
        tables.to_fwf(
            df, filepath,
            comment='# measurement unit: cm',
            floatfmt=[
                '.0f',
                '.0f', '.0f', '.0f',
                '.4f', '.4f', '.4f',
            ],
        )

    @staticmethod
    def save_pca_to_database(AB, filepath, bars):
        """Save the principal components of the bars to database.

        Parameters
        ----------
        AB : str
            Neutron wall A or B.
        filepath : str or pathlib.Path
            Path to the database file, e.g. :py:attr:`self.path_pca <path_pca>`.
        bars : list of Bar objects
            Sorted from bottom to top. The bottommost bar will be numbered 0;
            the topmost bar will be numbered 24.
        """
        # collect all PCA components and means from all bars and save into a dataframe
        df = []
        for bar_num, bar_obj in enumerate(bars):
            if bar_obj.contain_pyrex:
                bar_obj.remove_pyrex()
            df.append([bar_num, 'L', *bar_obj.pca.mean_])
            for ic, component in enumerate(bar_obj.pca.components_):
                scaled_component = bar_obj.dimension(ic) * component
                df.append([bar_num, 'XYZ'[ic], *scaled_component])

        df = pd.DataFrame(
            df,
            columns=[
                f'nw{AB.lower()}-bar',
                'vector',
                'lab-x', 'lab-y', 'lab-z',
            ],
        )

        # save to database
        tables.to_fwf(
            df, filepath,
            comment=inspect.cleandoc('''
                # measurement unit: cm
                # vector:
                #   - L is NW bar center in lab frame
                #   - X, Y, Z are NW bar's principal components in lab frame w.r.t. to L,
                #     with the lengths equal to the bar's respective dimensions (without Pyrex)
            '''),
            floatfmt=[
                '.0f',
                's',
                '.5f', '.5f', '.5f',
            ],
        )
    
    @staticmethod
    @cache.persistent_cache('$PROJECT_DIR/database/neutron_wall/geometry/efficiency_cache.pkl')
    def _get_geometry_efficiency_from_cpp_executable(
        method: Literal['monte_carlo', 'delta_phi'],
        mode: Literal['single', 'range'],
        theta_deg: float | tuple[float, float], # (low, upp)
        AB: Literal['A', 'B'],
        include_pyrex: bool,
        wall_filters_str: str,
        n_rays=-1,
        n_steps=-1,
    ) -> float | np.ndarray:
        arguments = [
            str(Path(expandvars('$PROJECT_DIR')) / 'scripts' / 'geo_efficiency.exe'),
            method, AB, str(int(include_pyrex)), wall_filters_str,
        ]
        if method == 'delta_phi' and mode == 'range':
            if not isinstance(theta_deg, tuple):
                raise TypeError('theta_deg must be a tuple when mode is range.')
            if n_steps <= 0:
                raise ValueError('n_steps must be a positive integer when mode is range.')
            arguments.append(f'{theta_deg[0]}')
            arguments.append(f'{theta_deg[1]}')
            arguments.append(f'{n_steps}')
            output = subprocess.run(arguments, check=True, capture_output=True).stdout
            return np.fromstring(output, sep='\n')

        if method == 'monte_carlo' and mode == 'single':
            if n_rays <= 0:
                raise ValueError('n_rays must be a positive integer when using Monte Carlo method.')
            arguments.append(f'{theta_deg}')
            arguments.append(f'{n_rays}')
        elif method == 'monte_carlo' and mode == 'range':
            raise NotImplementedError('range mode is only implemented for delta_phi method')
        elif method == 'delta_phi' and mode == 'single':
            arguments.append(f'{theta_deg}')
        return float(subprocess.run(arguments, check=True, capture_output=True).stdout.strip())
    
    def _parse_cuts(
        self,
        shadowed_bars: bool,
        skip_bars: list[int],
        cut_edges=True,
        custom_cuts: Optional[dict[int, list[str]]] = None,
    ) -> dict[int, str]:
        if self.bars is None:
            raise ValueError('Bars have not been initialized.')
        cuts = {b: [] for b in self.bars}
        if custom_cuts is not None:
            cuts = custom_cuts
        else:
            if cut_edges:
                cuts = {b: [[Bar.edges_x[0], Bar.edges_x[1]]] for b in cuts}
            else:
                cuts = {b: [[-9999.9, +9999.9]] for b in cuts} # dummy cut
                # the range will not actually exceed the physical edges of the bars
                # when fed into the C++ executable

            shadowed_bar_num = Bar.shadowed_bars if shadowed_bars else []
            for b in shadowed_bar_num:
                init_range = cuts[b][0]
                cuts[b] = [
                    [init_range[0], Bar.left_shadow_x[0]],
                    [Bar.left_shadow_x[1], Bar.right_shadow_x[0]],
                    [Bar.right_shadow_x[1], init_range[1]],
                ]

        if skip_bars is None:
            skip_bars = []
        for b in skip_bars:
            if b in cuts:
                del cuts[b]
        
        # sort cuts to make sure cache results do not repeat
        return {k: sorted(v) for k, v in sorted(cuts.items())}

    def get_geometry_efficiency(
        self,
        shadowed_bars: bool,
        skip_bars: list[int],
        cut_edges=True,
        custom_cuts: Optional[dict[int, list[str]]] = None,
        method: Literal['monte-carlo', 'delta-phi'] = 'delta-phi',
        n_rays=1_000_000,
        n_steps=30,
    ) -> Callable[[ArrayLike], ArrayLike]:
        """
        A simple wrapper around
        :py:func:`_get_geometry_efficiency_from_cpp_executable`. The function
        has a persistent cache saved to
        `$PROJECT_DIR/database/neutron_wall/geometry/efficiency_cache.pkl`.
        Simply remove the file if you do not want to use the cached results.

        First time calling this function will take a while. Subsequent calls
        with the exact same arguments will be much faster due to caching.

        Parameters
        ----------
        shadowed_bars : bool
            If True, shadow bar cut will be applied to NWB bars 7, 8, 9, 10, 15,
            16, 17. If False, no shadow bar cut is applied.
        skip_bars : list of int
            Bars to be skipped. Remember, in the experiment, NWB-bar00 is the
            bottommost bar that was blocked by the ground.
        cut_edges : bool, default True
            If ``'all'``, edge cut will be applied to all bars. If a list of
            int, edge cut will only be applied to the bars in the list.
        custom_cuts : dict of int to list of str, default None
            When custom cuts are applied, all the other cuts are ignored, so
            users should make sure the cuts are complete, e.g. edge cut has to be
            included. The only cut variable being supported is ``'x'``.
        method : 'monte-carlo' or 'delta-phi', default 'delta-phi'
            Method to calculate the geometry efficiency.
        n_rays : int, default 1_000_000
            Number of rays to be used in the Monte Carlo method. This argument
            is ignored if `method` is not `'monte-carlo'`.
        n_steps : int, default 30
            Number of steps to be used in the delta-phi method. This argument
            is ignored if `method` is not `'delta-phi'`.

        Returns
        -------
        geometry_efficiency : Callable[[float], float]
            A function that takes a theta (radian) and returns the geometry efficiency.
        """
        cuts_str = json.dumps(self._parse_cuts(shadowed_bars, skip_bars, cut_edges, custom_cuts))
        kw = dict(
            AB=self.AB,
            include_pyrex=self.contain_pyrex,
            wall_filters_str=cuts_str,
        )

        if method == 'monte-carlo':
            def inner(theta_input: ArrayLike) -> ArrayLike:
                vectorized_func = np.vectorize(self._get_geometry_efficiency_from_cpp_executable)
                return vectorized_func(
                    method='monte_carlo',
                    mode='single',
                    theta_deg=np.degrees(theta_input),
                    n_rays=n_rays,
                    **kw,
                )
            return inner
        
        # method == 'delta-phi'
        def inner(theta_input: ArrayLike) -> ArrayLike:
            if isinstance(theta_input, (int, float)):
                # round to 2 decimal place to avoid cache miss
                theta = round(theta_input, 2)
                return self._get_geometry_efficiency_from_cpp_executable(
                    method='delta_phi', mode='single',
                    theta_deg=np.degrees(theta),
                    **kw,
                )
            else:
                grid_x = np.linspace(25, 55, 3000 + 1)
                grid_y = self._get_geometry_efficiency_from_cpp_executable(
                    method='delta_phi', mode='range',
                    theta_deg=(grid_x[0], grid_x[-1]),
                    n_steps=len(grid_x),
                    **kw,
                )
                return np.interp(np.degrees(theta_input), grid_x, grid_y)
        return inner
