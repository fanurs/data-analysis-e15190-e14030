import numpy as np
import pandas as pd

from e15190 import PROJECT_DIR
from e15190.utilities import geometry as geom
from e15190.utilities import tables

class Bar(geom.RectangularBar):
    def __init__(self, vertices):
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
    def __init__(self, refresh_from_inventor_readings=False):
        # initialize class parameters
        self.database_dir = PROJECT_DIR / 'database/veto_wall/geometry'
        self.path_inventor_readings = self.database_dir / 'inventor_readings_VW.dat'
        self.path_raw = self.database_dir / 'VW.dat'

        # if True, read in again from raw inventor readings
        self._refresh_from_inventor_readings = refresh_from_inventor_readings
        if self._refresh_from_inventor_readings:
            bars_vertices = self.read_from_inventor_readings()
            self.process_and_save_to_database(bars_vertices)

        # read in from database
        self.database = pd.read_csv(self.path_raw, comment='#', delim_whitespace=True)
        index_names = ['vw-bar', 'dir_x', 'dir_y', 'dir_z']
        self.database.set_index(index_names, drop=True, inplace=True)

        # construct a dictionary of bar objects
        bar_nums = sorted(set(self.database.index.get_level_values('vw-bar')))
        self.bars = {b: Bar(self.database.loc[b][['x', 'y', 'z']]) for b in bar_nums}
    
    def read_from_inventor_readings(self):
        with open(self.path_inventor_readings, 'r') as file:
            lines = file.readlines()
        
        # containers for process the lines
        bars_vertices = [] # to collect vertices of all bars
        vertex = [None] * 3
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

                bottom_vertex = [*vertex]
                bottom_vertex[1] *= -1 # flip y to bottom, i.e. positive to negative
                vertices.append(bottom_vertex)

                vertex = [None] * 3
            
            # append and reset if all 8 vertices of a bar have been read in
            if len(vertices) == 8:
                bars_vertices.append(vertices)
                vertices = []
        
        return bars_vertices
    
    def process_and_save_to_database(self, bars_vertices):
        # construct bar objects from vertices and sort
        bar_objects = []
        bar_objects = [Bar(vertices) for vertices in bars_vertices]
        bar_objects = sorted(bar_objects, key=lambda bar: bar.pca.mean_[0], reverse=True)

        # collect all vertices from all bars and save into a dataframe
        df= []
        for bar_num, bar_obj in enumerate(bar_objects):
            bar_num = bar_num + 1
            for sign, vertex in bar_obj.vertices.items():
                df.append([bar_num, *sign, *vertex])
                
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
            self.path_raw,
            comment='# measurement unit: cm',
            floatfmt=[
                '.0f',
                '.0f', '.0f', '.0f',
                '.4f', '.4f', '.4f',
            ],
        )