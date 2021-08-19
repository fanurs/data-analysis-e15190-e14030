import itertools as itr

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import sympy as sp

from e15190 import PROJECT_DIR
from e15190.utilities import tables
from e15190.utilities import ray_triangle_intersection as rti

_database_dir = PROJECT_DIR / 'database/neutron_wall/geometry'

def spherical_to_cartesian(radius, polar, azimuth):
    """Convert coordinates from spherical system to Cartesian system.

    All angles are presented with the unit of radian.

    Parameters:
        radius : scalar or array-like
            Distance from the origin.
        polar : scalar or array-like
            Angle measured from the +z-axis.
        azimuth : sclar or array-like
            Counterclockwise angle measured from the +x-axis on the xy-plane.

    Returns:
        If all arguments are scalars, returns `(x, y, z)`. If all arguments are
        arrays, returns `np.vstack([x, y, z])`.
    """
    is_array = any(map(lambda x: hasattr(x, '__len__'), (radius, polar, azimuth)))
    radius, polar, azimuth = map(lambda _x: np.array(_x), (radius, polar, azimuth))

    sin_polar = np.sin(polar)
    result = np.vstack([
        radius * sin_polar * np.cos(azimuth),
        radius * sin_polar * np.sin(azimuth),
        radius * np.cos(polar),
    ])
    return result if is_array else tuple(result.ravel())

def cartesian_to_spherical(x, y, z):
    """Convert coordinates from Cartesian system to spherical system.

    This function follow the convention in most physics literature that place
    polar angle before azimuthal angle, i.e. it returns tuples (radius, polar,
    azimuth). The polar angle is given within the range of $[0, 2\pi]$. The
    azimuthal angle, following the convetion of `atan2()`, is given within the
    range of $(-\pi, \pi]$.

    Parameters:
        x : scalar or array-like
        y : scalar or array-like
        z : sclar or array-like

    Returns:
        If all arguments are scalars, returns `(radius, polar, azimuth)`. If all
        arguments are arrays, returns `np.vstack([radius, polar azimuth])`.
    """
    is_array = any(map(lambda _x: hasattr(_x, '__len__'), (x, y, z)))
    x, y, z = map(lambda _x: np.array(_x), (x, y, z))

    rho2 = x**2 + y**2
    result = np.vstack([
        np.sqrt(rho2 + z**2),
        np.arctan2(np.sqrt(rho2), z),
        np.arctan2(y, x),
    ])
    return result if is_array else tuple(result.ravel())

def angle_between_vectors(u, v, directional=False):
    u, v = map(lambda x: np.array(x), (u, v))
    dot_prod = np.dot(v, u) if u.ndim < v.ndim else np.dot(u, v)
    angle = dot_prod / np.sqrt(np.square(u).sum(axis=-1) * np.square(v).sum(axis=-1))
    angle = np.arccos(angle.clip(-1, 1)) # clip to account for floating-point error
    angle *= np.sign(np.cross(u, v)) if directional else 1.0
    return angle

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
        current_y = self.pca.components_[1]
        target_y = np.array([0.0, 1.0, 0.0])
        rot_angle = angle_between_vectors(current_y, target_y)
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

    def construct_plotly_mesh3d(self):
        key_index = {key: index for index, key in enumerate(self.vertices)}
        tri_indices = []
        for xyz, sign in itr.product(range(3), [-1, +1]):
            triangle_1 = np.array([[-1, -1], [-1, +1], [+1, -1]])
            triangle_2 = np.array([[+1, +1], [-1, +1], [+1, -1]])

            triangle_1 = np.insert(triangle_1, xyz, [sign] * 3, axis=1)
            triangle_2 = np.insert(triangle_2, xyz, [sign] * 3, axis=1)

            tri_index_1 = [key_index[tuple(vertex)] for vertex in triangle_1]
            tri_index_2 = [key_index[tuple(vertex)] for vertex in triangle_2]

            tri_indices.extend([tri_index_1, tri_index_2])
        
        vertices = np.array(list(self.vertices.values()))
        self.triangle_mesh = rti.TriangleMesh(vertices, tri_indices)
    
    def simple_simulation(self, n_rays=10, random_seed=None):
        if 'triangle_mesh' not in self.__dict__:
            self.construct_plotly_mesh3d()
        rng = np.random.default_rng(random_seed)
        origin = np.array([0.0] * 3)

        # identify the minimal region in space to emit rays
        # The region is orthogonal in the (polar, azimuth) space, i.e. some
        # rectangle, so users can later easily recover the cross section from
        # the isotropy of simulation rays.
        vertices = np.array(list(self.vertices.values()))
        xy_vecs = vertices[:, [0, 1]]
        zrho_vecs = np.vstack([vertices[:, 2], np.linalg.norm(xy_vecs, axis=1)]).T
        def identify_angle_range(vecs):
            mean_vec = np.mean(vecs, axis=0)
            angles = angle_between_vectors(mean_vec, vecs, directional=True)
            angles += np.arctan2(mean_vec[1], mean_vec[0])
            return np.array([angles.min(), angles.max()])
        azimuth_range, polar_range = map(identify_angle_range, (xy_vecs, zrho_vecs))
        def slight_widening(range_, fraction=0.01):
            delta_width = fraction * (range_[1] - range_[0])
            range_[0] -= delta_width
            range_[1] += delta_width
            return range_
        azimuth_range, polar_range = map(slight_widening, (azimuth_range, polar_range))

        # simulate
        polars = np.arccos(rng.uniform(*np.cos(polar_range), size=n_rays).clip(-1, 1))
        azimuths = rng.uniform(*azimuth_range, size=n_rays)
        rays = spherical_to_cartesian(1.0, polars, azimuths).T
        triangles = self.triangle_mesh.get_triangles()
        intersections = rti.moller_trumbore(origin, rays, triangles)

        # save results
        self.simulation_result = dict()
        var_names = ['origin', 'azimuth_range', 'polar_range', 'intersections']
        for var in var_names:
            self.simulation_result[var] = locals()[var]
        return self.simulation_result
    
    def analyze_hit_positions(
        self,
        out_coordinates='local',
        hit_t='uniform',
        random_seed=None,
        tol=1e-9,
    ):
        sim_res = self.simulation_result # shorthand

        # shift to make origin at zero
        hits = sim_res['intersections'] - sim_res['origin']

        # for each ray, collect a pair of incoming and outgoing vertices
        # i.e. shape of (n_rays, 2, 3)
        norm2 = np.square(hits).sum(axis=-1)
        norm2[norm2 < tol] = 0.0 # account for floating-point errors
        ii = np.argsort(norm2, axis=0)
        ii = np.tile(ii[:, :, None], reps=(1, 1, 3))
        hits = np.take_along_axis(hits, ii, axis=0)[-2:]
        hits = np.swapaxes(hits, 0, 1)

        # determine the points of interaction
        # i.e. some points on the line segments bounded by the in-and-out vertices.
        n_hits = len(hits)
        if hit_t == 'uniform':
            rng = np.random.default_rng(random_seed)
            t = rng.uniform(size=n_hits)
        elif callable(hit_t):
            t = hit_t(n_hits)
        else: # assume scalar
            t = hit_t * np.ones(shape=n_hits)
        hits[:, 0] *= (1 - t)[:, None]
        hits[:, 1] *= t[:, None]
        hits = np.sum(hits, axis=1)

        # throw away non-interacting hits
        hits = hits[np.sum(norm2, axis=0) > 0]

        # convert coordinates
        if out_coordinates == 'global':
            pass
        elif out_coordinates == 'local':
            hits = np.matmul(self.pca.components_, (hits - self.pca.mean_).T).T

        return hits

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
