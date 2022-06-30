import inspect
from os.path import expandvars
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from e15190.utilities import geometry as geom
from e15190.utilities import tables

class Bar(geom.RectangularBar):
    def __init__(self, vertices, number=None):
        """Construct a Veto Wall bar.

        This class only deals with individual bar. The Veto Wall object is
        implemented in :py:class`Wall`.
        
        Parameters
        ----------
        vertices : ndarray of shape (8, 3)
            Exactly eight (x, y, z) lab coordinates measured in centimeter (cm).
            As long as all eight vertices are distinct, i.e. can properly define
            a cuboid, the order of vertices does not matter because PCA will be
            applied to automatically identify the 1st, 2nd and 3rd principal
            axes of the bar, which, for Veto Wall bar, corresponds to the y, x
            and z directions in the local (bar) coordinate system.
        number : int, default None
            The bar number. Use ``None`` if the bar number is not known.
        """
        super().__init__(
            vertices,
            calculate_local_vertices=False,
            make_vertices_dict=False,
        )

        # swap the PCA components such that (0, 1, 2) corresponds to (x, y, z)
        # the 1st principal is y (longest), 2nd principal is x, 3rd principal is z
        # so we swap indices 0 and 1
        swap = lambda arr: np.concatenate((arr[[1, 0]], arr[2:]), axis=0)
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        attrs = [
            'components_',
            'explained_variance_',
            'explained_variance_ratio_',
            'singular_values_',
        ]
        for attr in attrs:
            self.pca.__dict__[attr] = swap(self.pca.__dict__[attr])

        # 1st component defines local-x, should be opposite to lab-x
        if np.dot(self.pca.components_[0], [-1, 0, 0]) < 0:
            self.pca.components_[0] *= -1

        # 2nd component defines local-y, should nearly parallel to lab-y
        if np.dot(self.pca.components_[1], [0, 1, 0]) < 0:
            self.pca.components_[1] *= -1

        # 3rd component defines local-z, direction is given by right-hand rule
        x_cross_y = np.cross(self.pca.components_[0], self.pca.components_[1])
        if np.dot(self.pca.components_[2], x_cross_y) < 0:
            self.pca.components_[2] *= -1

        self.loc_vertices = self.to_local_coordinates(self.vertices)
        self._make_vertices_dict()

        self.number = number
    
    @property
    def width(self):
        """Width of the bar.

        Should be around 9.4 cm.
        """
        return self.dimension(index=0)
    
    @property
    def height(self):
        """Height of the bar.

        Should be around 227.2 cm.
        """
        return self.dimension(index=1)

    @property
    def thickness(self):
        """Longitudinal thickness of the bar.

        Should be around 1.0 cm.
        """
        return self.dimension(index=2)

class Wall:
    path_inventor_readings = '$DATABASE_DIR/veto_wall/geometry/inventor_readings_VW.dat'
    path_vertices = '$DATABASE_DIR/veto_wall/geometry/VW_vertices.dat'
    path_pca = '$DATABASE_DIR/veto_wall/geometry/VW_pca.dat'

    def __init__(self, refresh_from_inventor_readings=False):
        # if True, read in again from raw inventor readings
        self._refresh_from_inventor_readings = refresh_from_inventor_readings
        if self._refresh_from_inventor_readings:
            bars_vertices = self.read_from_inventor_readings(self.path_inventor_readings)
            self.process_and_save_to_database(bars_vertices)

        # read in from database
        self.database = pd.read_csv(self.path_vertices, comment='#', delim_whitespace=True)
        index_names = ['vw-bar', 'dir_x', 'dir_y', 'dir_z']
        self.database.set_index(index_names, drop=True, inplace=True)

        # construct a dictionary of bar objects
        bar_nums = sorted(set(self.database.index.get_level_values('vw-bar')))
        self.bars = {b: Bar(self.database.loc[b][['x', 'y', 'z']]) for b in bar_nums}
    
    @classmethod
    def read_from_inventor_readings(cls, filepath=None) -> List[Bar]:
        """Reads in Inventor measurements and returns a list of :py:class:`Bar` objects.

        Parameters
        ----------
        filepath : str or pathlib.Path, default None
            File path of the Inventor measurements. If ``None``, set to
            :py:attr:`self.path_inventor_readings <path_inventor_readings>`.
        
        Returns
        -------
        bars : list of :py:class:`Bar` objects
            Sorted from left to right w.r.t. neutron wall.
        """
        if filepath is None:
            filepath = Path(expandvars(cls.path_inventor_readings))

        with open(filepath, 'r') as file:
            lines = file.readlines()
        
        # containers for process the lines
        bars = []
        vertex = [None] * 3
        vertices = [] # to collect all vertices of one particular bar

        for line in lines:
            sline = line.strip().split()

            if len(sline) > 0 and sline[0].strip()[0] == 'L':
                bar_number = int(sline[0][1:])

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

                bottom_vertex = [*vertex]
                bottom_vertex[1] *= -1 # flip y to bottom, i.e. positive to negative
                vertices.append(bottom_vertex)

                vertex = [None] * 3
            
            # append and reset if all 8 vertices of a bar have been read in
            if len(vertices) == 8:
                bars.append(Bar(vertices, bar_number))
                vertices = []
                bar_number = None
            
        # sort Bar objects
        bars = sorted(bars, key=lambda bar: bar.pca.mean_[1])
        if not all(bars[i].number < bars[i + 1].number for i in range(len(bars) - 1)):
            raise Exception('Failed to sort bars from left to right')
        return bars
    
    @classmethod
    def save_vertices_to_database(cls, filepath=None, bars=None) -> pd.DataFrame:
        """Saves the vertices of the bars to database.

        Parameters
        ----------
        filepath : str or pathlib.Path, default None
            Path to the database file. If ``None``, set to
            :py:attr:`self.path_vertices <path_vertices>`.
        bars : list of :py:class:`Bar` objects, default None
            Sorted from left to right w.r.t. neutron wall. If ``None``, read
            :py:func:`self.read_from_inventor_readings <read_from_inventor_readings>`
            is invoked.
        
        Returns
        -------
        df_vertices : pandas.DataFrame
            Dataframe containing the vertices of the bars.
        """
        if filepath is None:
            filepath = Path(expandvars(cls.path_vertices))
        if bars is None:
            bars = cls.read_from_inventor_readings()

        # collect all vertices from all bars and save into a dataframe
        df= []
        for bar in bars:
            for sign, vertex in bar.vertices.items():
                df.append([bar.number, *sign, *vertex])
                
        df = pd.DataFrame(
            df,
            columns = [
                'vw-bar',
                'dir_x', 'dir_y', 'dir_z',
                'x', 'y', 'z',
            ],
        )

        # save to database
        tables.to_fwf(
            df,
            filepath,
            comment='# measurement unit: cm',
            floatfmt=[
                '.0f',
                '.0f', '.0f', '.0f',
                '.4f', '.4f', '.4f',
            ],
        )

        return df
    
    @classmethod
    def save_pca_to_database(cls, filepath=None, bars=None) -> pd.DataFrame:
        """Saves the principal components of the bars to database.

        Parameters
        ----------
        filepath : str or pathlib.Path, default None
            Path to the database file. If ``None``, set to
            :py:attr:`self.path_pca <path_pca>`.
        bars : list of :py:class:`Bar` objects, default None
            Sorted from left to right w.r.t. neutron wall. If ``None``, read
            :py:func:`self.read_from_inventor_readings <read_from_inventor_readings>`
            is invoked.
        
        Returns
        -------
        df_pca : pandas.DataFrame
            Dataframe containing the principal components of the bars.
        """
        if filepath is None:
            filepath = Path(expandvars(cls.path_pca))
        if bars is None:
            bars = cls.read_from_inventor_readings()
        
        # collect all PCA components and means from all bars and save into a dataframe
        df = []
        for bar in bars:
            df.append([bar.number, 'L', *bar.pca.mean_])
            for ic, component in enumerate(bar.pca.components_):
                scaled_component = bar.dimension(ic) * component
                df.append([bar.number, 'XYZ'[ic], *scaled_component])
        
        df = pd.DataFrame(
            df,
            columns=[
                'vw-bar',
                'vector',
                'lab-x', 'lab-y', 'lab-z',
            ],
        )

        tables.to_fwf(
            df, filepath,
            comment=inspect.cleandoc('''
                # measurement unit: cm
                # vector:
                #   - L is VW bar center in lab frame
                #   - X, Y, Z are VW bar's principal components in lab frame w.r.t. to L,
                #     with the lengths (magnitudes) equal to the bar's respective dimensions.
            '''),
            floatfmt=[
                '.0f',
                's',
                '.5f', '.5f', '.5f',
            ],
        )

        return df