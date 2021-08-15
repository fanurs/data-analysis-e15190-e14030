import itertools as itr

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from e15190 import PROJECT_DIR
from e15190.utilities import tables

_database_dir = PROJECT_DIR / 'database/neutron_wall/geometry'

class Bar:
    def __init__(self, vertices):
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
        self.vertices = dict(sorted(self.vertices.items(), reverse=True))
        self.loc_vertices = {
            rel_dir: vertex for rel_dir, vertex in zip(rel_directions, self.loc_vertices)
        }
        self.loc_vertices = dict(sorted(self.loc_vertices.items(), reverse=True))

class Wall:
    def __init__(self, AB, refresh_from_inventor_readings=False):
        # initialize class parameters
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.path_inventor_readings = _database_dir / f'inventor_readings_NW{self.AB}.dat'
        self.path_database = _database_dir / f'NW{self.AB}.dat'

        # if True, read in again from raw inventor readings
        self._refresh_from_inventor_readings = refresh_from_inventor_readings
        if self._refresh_from_inventor_readings:
            bars_vertices = self.read_from_inventor_readings()
            self.process_and_save_to_database(bars_vertices)
        
        # read in from database
        self.database = pd.read_csv(self.path_database, comment='#', delim_whitespace=True)
        index_names = [f'nw{self.ab}-bar', 'dir_x', 'dir_y', 'dir_z']
        self.database.set_index(index_names, drop=True, inplace=True)

        # construct a dictionary of bar objects
        bar_nums = self.database.index.get_level_values(f'nw{self.ab}-bar')
        self.bars = {b: Bar(self.database.loc[b][['x', 'y', 'z']]) for b in bar_nums}
        self._simple_check = True
        for index, coordinate in self.database.iterrows():
            bar = index[0]
            if tuple(self.bars[bar].vertices[index[1:]]) == tuple(coordinate):
                pass
            else:
                self._simple_check = False
                break
    
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
        signs = list(itr.product([+1, -1], repeat=3))
        df = []
        for bar_num, bar_obj in enumerate(bar_objects, start=1):
            for sign in signs:
                vertex = bar_obj.vertices[tuple(sign)]
                df.append([bar_num, *sign, *vertex])
        columns = [
            f'nw{self.ab}-bar',
            'dir_x', 'dir_y', 'dir_z',
            'x', 'y', 'z',
        ]
        df = pd.DataFrame(df, columns=columns)

        # save to database
        tables.to_fwf(df, self.path_database, comment='# measurement unit: cm')
