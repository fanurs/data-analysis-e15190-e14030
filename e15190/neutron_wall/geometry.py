import itertools as itr

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import sympy as sp

from e15190 import PROJECT_DIR
from e15190.utilities import tables

_database_dir = PROJECT_DIR / 'database/neutron_wall/geometry'

class Bar:
    def __init__(self, vertices, contain_pyrex=True):
        self.contain_pyrex = contain_pyrex
        self.pyrex_thickness = 2.54 / 8 # cm
        self.vertices = np.array(vertices)
        self.pca = PCA(n_components=3, svd_solver='full')
        self.pca.fit(self.vertices)

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

        self.loc_vertices = np.array([
            [np.dot(vertex - self.pca.mean_, component) for component in self.pca.components_]
            for vertex in self.vertices
        ])

        # identify relative directions (egocentric)
        rel_directions = [
            tuple([int(np.sign(coord)) for coord in loc_vertex])
            for loc_vertex in self.loc_vertices
        ]

        # map the relative directions back to vertices and local vertices using dictionaries
        self.vertices = {
            rel_dir: vertex for rel_dir, vertex in zip(rel_directions, self.vertices)
        }
        self.vertices = dict(sorted(self.vertices.items()))
        self.loc_vertices = {
            rel_dir: vertex for rel_dir, vertex in zip(rel_directions, self.loc_vertices)
        }
        self.loc_vertices = dict(sorted(self.loc_vertices.items()))
    
    @property
    def length(self):
        result = []
        for sign in itr.product([+1, 1], repeat=2):
            diff = self.loc_vertices[(1, *sign)][0] - self.loc_vertices[(-1, *sign)][0]
            result.append(abs(diff))
        return np.mean(result)
    
    @property
    def height(self):
        result = []
        for sign in itr.product([+1, 1], repeat=2):
            diff = self.loc_vertices[(sign[0], 1, sign[1])][1] - self.loc_vertices[(sign[0], -1, sign[1])][1]
            result.append(abs(diff))
        return np.mean(result)

    @property
    def thickness(self):
        result = []
        for sign in itr.product([+1, 1], repeat=2):
            diff = self.loc_vertices[(*sign, 1)][2] - self.loc_vertices[(*sign, -1)][2]
            result.append(abs(diff))
        return np.mean(result)
    
    def remove_pyrex(self, inplace=False):
        if not self.contain_pyrex:
            raise Exception('Pyrex has already been removed')

        new_loc_vertices = [v - self.pyrex_thickness * np.array(iv) for iv, v in self.loc_vertices.items()]
        inv_pca_matrix = np.linalg.inv(self.pca.components_)
        new_vertices = [np.matmul(inv_pca_matrix, v) + self.pca.mean_ for v in new_loc_vertices]

        if inplace:
            self.__init__(new_vertices, contain_pyrex=False)
        else:
            return Bar(new_vertices, contain_pyrex=False)
    
    def add_pyrex(self, inplace=False):
        if self.contain_pyrex:
            raise Exception('Pyrex has already been added')
        
        new_loc_vertices = [v + self.pyrex_thickness * np.array(iv) for iv, v in self.loc_vertices.items()]
        inv_pca_matrix = np.linalg.inv(self.pca.components_)
        new_vertices = [np.matmul(inv_pca_matrix, v) + self.pca.mean_ for v in new_loc_vertices]

        if inplace:
            self.__init__(new_vertices, contain_pyrex=True)
        else:
            return Bar(new_vertices, contain_pyrex=True)
    
    def flatten(self, inplace=True):
        """To make the bar horizontally flat.

        For some unknown reasons, the raw readings from the Inventor file do not
        describe bar with exact flatness. It turns out that by rotating the bar
        by a small degree (< 1.0 degree) such that it aligns back the local
        y-axis to the lab y-axis (pointing upward), we can recover the flatness.
        Of course, there are some inevitable round-up errors due to
        floating-point arithmetic. So the final results are rounded up to force
        coordinates that should have the same numerical values to be exactly
        identical, e.g. there should only be two distinct sets of y-coordinates,
        one corresponds to the top surface, another corresponds to the bottom
        surface.

        Parameters:
            inplace : bool, default True
                If `True`, flattening will be applied to the current `Bar`
                object.  If `False`, a flattened `Bar` object will be returned.
        
        Returns:
            When arugment inplace is set to `False`, a flattened `Bar` object
            will be returned.
        """
        # construct rotation
        origin = np.array([0.0] * 3)
        current_y = self.pca.components_[1]
        target_y = np.array([0.0, 1.0, 0.0])
        rot_angle = float(sp.Line(origin, current_y).angle_between(sp.Line(origin, target_y)))
        rot_vec = np.cross(current_y, target_y)
        rot_vec /= np.linalg.norm(rot_vec)
        self.rot_matrix = Rotation.from_rotvec(rot_angle * rot_vec).as_matrix()

        # apply rotation
        rotated_vertices = dict()
        for key, vertex in self.vertices.items():
            rotated_vertices[key] = np.matmul(self.rot_matrix, vertex - self.pca.mean_)
            rotated_vertices[key] += self.pca.mean_

        # force numerical values to be exactly equal by taking average of close-enough groups
        key_func = lambda num: round(num, 2)
        rounded_vertices = np.array(list(rotated_vertices.values()))
        for ic in range(3):
            groups = {
                key: np.mean(list(val))
                for key, val in itr.groupby(rounded_vertices[:, ic], key=key_func)
            }
            rounded_vertices[:, ic] = [groups[key_func(ele)] for ele in rounded_vertices[:, ic]]

        # finalize result
        flattened_vertices = rounded_vertices
        if inplace:
            self.__init__(flattened_vertices)
        else:
            return Bar(flattened_vertices)

class Wall:
    def __init__(self, AB, refresh_from_inventor_readings=False, flatten=True):
        # initialize class parameters
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.path_inventor_readings = _database_dir / f'inventor_readings_NW{self.AB}.dat'
        self.path_raw = _database_dir / f'NW{self.AB}_raw.dat'
        self.path_flattened = _database_dir / f'NW{self.AB}_flattened.dat'

        # if True, read in again from raw inventor readings
        self._refresh_from_inventor_readings = refresh_from_inventor_readings
        if self._refresh_from_inventor_readings:
            bars_vertices = self.read_from_inventor_readings()
            self.process_and_save_to_database(bars_vertices)

        # read in from database
        path = self.path_flattened if flatten else self.path_raw
        self.database = pd.read_csv(path, comment='#', delim_whitespace=True)
        index_names = [f'nw{self.ab}-bar', 'dir_x', 'dir_y', 'dir_z']
        self.database.set_index(index_names, drop=True, inplace=True)

        # construct a dictionary of bar objects
        bar_nums = self.database.index.get_level_values(f'nw{self.ab}-bar')
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
                vertex = [None] * 3
            
            # append and reset if all 8 vertices of a bar have been read in
            if len(vertices) == 8:
                bars_vertices.append(vertices)
                vertices = []
        
        return bars_vertices
    
    def process_and_save_to_database(self, bars_vertices):
        # construct bar objects from vertices and sort
        bar_objects = [Bar(vertices) for vertices in bars_vertices]
        bar_objects = sorted(bar_objects, key=lambda bar: bar.pca.mean_[1]) # sort from bottom to top

        # collect all vertices from all bars and save into a dataframe
        df, df_flattened = [], []
        for bar_num, bar_obj in enumerate(bar_objects, start=1):
            flattened_bar_obj = bar_obj.flatten(inplace=False)
            for sign, vertex in bar_obj.vertices.items():
                df.append([bar_num, *sign, *vertex])

                flattened_vertex = flattened_bar_obj.vertices[tuple(sign)]
                df_flattened.append([bar_num, *sign, *flattened_vertex])
                
        columns = [
            f'nw{self.ab}-bar',
            'dir_x', 'dir_y', 'dir_z',
            'x', 'y', 'z',
        ]
        df = pd.DataFrame(df, columns=columns)
        df_flattened = pd.DataFrame(df_flattened, columns=columns)

        # save to database
        kwargs = dict(
            comment='# measurement unit: cm',
            floatfmt=[
                '.0f',
                '.0f', '.0f', '.0f',
                '.4f', '.4f', '.4f',
            ],
        )
        tables.to_fwf(df, self.path_raw, **kwargs)
        tables.to_fwf(df_flattened, self.path_flattened, **kwargs)
