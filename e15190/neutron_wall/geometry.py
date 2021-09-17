import copy
import itertools as itr
import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from e15190 import PROJECT_DIR
from e15190.utilities import fast_histogram as fh
from e15190.utilities import geometry as geom
from e15190.utilities import ray_triangle_intersection as rti
from e15190.utilities import tables

class Bar:
    def __init__(self, vertices, contain_pyrex=True, check_pca_orthogonal=True):
        """Construct a Neutron Wall bar, either from Wall A or Wall B.

        Parameters:
            vertices : ndarray of shape (8, 3)
                Exactly eight (x, y, z) lab coordinates measured in centimeter
                (cm). As long as all eight vertices are distincit, i.e. can
                properly define a cuboid, the order of vertices does not matter
                because PCA will be applied to automatically identify the 1st,
                2nd and 3rd principal axes of the bar. These vertices are always
                assumed to have included the Pyrex thickness
            contain_pyrex : bool, default True
                The input `vertices` are always assumed to have included the
                Pyrex thickness. If this function is supplied with the raw
                readings from the Inventor files or database, then no extra care
                is needed as those measurements always include the Pyrex
                thickness. If `True`, the Pyrex will not be removed; if `False`,
                the Pyrex will be removed, leaving only the scintillation
                material.
            check_pca_orthogonal : boo, default True
                If `True`, an Exception is raised whenever the PCA matrix is not
                orthogonal; if `False`, no check would be done.

                PCA is used to extract the principal axes of the bar.
                Mathematically, we would expect the PCA matrix to be orthogonal,
                but in actual implementation, the matrix is often not exactly
                orthogonal due to floating-point errors. Hence, we may want to
                check if the PCA matric we compute is "nearly orthogonal".
        """
        self.contain_pyrex = True # always starts with True
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
        
        # check if the resultant pca is an orthogonal matrix
        pca_mat = self.pca.components_ # shorthand
        if check_pca_orthogonal and not np.allclose(np.identity(3), np.matmul(pca_mat.T, pca_mat)):
            raise(f'The PCA matrix is not orthogonal:\n{pca_mat}')

        # save the vertices in local coordinates
        self.loc_vertices = self.to_local_coordinates(vertices)

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

        # remove Pyrex if requested
        if not contain_pyrex:
            self.remove_pyrex()
        else:
            # database always have included Pyrex, so no action is required
            pass
    
    def dimension(self, index=None):
        if index is None:
            return tuple(self.dimension(index=i) for i in range(3))

        if isinstance(index, str):
            index = 'xyz'.find(index.lower())

        result = []
        for sign in itr.product([+1, 1], repeat=2):
            pos, neg = list(sign), list(sign)
            pos.insert(index, +1)
            neg.insert(index, -1)
            diff = self.loc_vertices[tuple(pos)][index] - self.loc_vertices[tuple(neg)][index]
            result.append(abs(diff))
        return np.mean(result)
    
    @property
    def length(self):
        return self.dimension(index=0)
    
    @property
    def height(self):
        return self.dimension(index=1)

    @property
    def thickness(self):
        return self.dimension(index=2)
    
    def modify_pyrex(self, mode):
        if mode == 'remove':
            scalar = -1
        elif mode == 'add':
            scalar = +1
        
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
        if not self.contain_pyrex:
            raise Exception('Pyrex has already been removed')
        self.modify_pyrex('remove')
    
    def add_pyrex(self):
        if self.contain_pyrex:
            raise Exception('Pyrex has already been added')
        self.modify_pyrex('add')

    @staticmethod
    def _deco_numpy_ndim(func):
        def inner(x, *args, **kwargs):
            x = np.array(x)
            ndim = x.ndim
            if ndim == 1:
                x = np.array([x])
            result = func(x, *args, **kwargs)
            return result if ndim == 2 else np.squeeze(result)
        return inner

    def to_local_coordinates(self, lab_coordinates):
        return self._deco_numpy_ndim(self.pca.transform)(lab_coordinates)

    def to_lab_coordinates(self, local_coordinates):
        return self._deco_numpy_ndim(self.pca.inverse_transform)(local_coordinates)

    def is_inside(self, coordinates, frame='local', tol=5e-4):
        """Check if the given coordinates are inside the bar.

        Parameters
            coordinates : 3-tuple or list of 3-tuples
                The coordinates to be checked. Only support Cartesian
                coordinates.
            frame : 'local' or 'lab', default 'local'
                The frame of the coordinates.
            tol : float, default 5e-4
                The tolerance of the boundary check in the unit of centimeters.
        
        Returns:
            bool or an array of bool
                `True` if the coordinates are inside the bar, otherwise `False`.
        """
        # convert into 2d numpy array of shape (n_points, 3)
        coordinates = np.array(coordinates, dtype=float)
        
        # convert to local frame
        if frame == 'local':
            pass
        else:
            coordinates = self.to_local_coordinates(coordinates)

        # convert into 2d numpy array of shape (n_points, 3)
        if coordinates.ndim == 1:
            coordinates = np.array([coordinates])

        # check if the coordinates are inside the bar
        dimension = self.dimension()
        result = np.array([True] * len(coordinates))
        for i in range(3):
            i_components = coordinates[:, i]
            result = result & (i_components >= -0.5 * dimension[i] - tol)
            result = result & (i_components <= +0.5 * dimension[i] + tol)
        return result if len(result) > 1 else result[0]

    def randomize_from_local_x(
        self,
        local_x,
        return_frame='lab',
        local_ynorm=[-0.5, 0.5],
        local_znorm=[-0.5, 0.5],
        random_seed=None,
    ):
        """Returns randomized point(s) from the given local x coordinate(s).

        This is useful for getting a uniform hit distribution within the bulk of
        the bar. Experimentally, we can only determine the local x coordinate
        for each hit.

        Parameters:
            local_x : float or list of floats
                The local x coordinate(s) of the point(s) in centimeters.
            return_frame : 'lab' or 'local', default 'lab'
                The frame of the returned point(s) in Cartesian coordinates.
            local_ynorm : float or 2-tuple of floats, default [-0.5, 0.5]
                The range of randomization in the local y coordinate. If float,
                no randomization is performed. Center is at `0.0`; the top
                surface is `+0.5`; the bottom surface is `-0.5`. Values outside
                this range will still be calculated, but those points will be
                outside the bar.
            local_znorm : float or 2-tuple of floats, default [-0.5, 0.5]
                The range of randomization in the local z coordinate. If float,
                no randomization is performed. Center is at `0.0`; the front
                surface is `+0.5`; the back surface is `-0.5`. Values outside
                this range will still be calculated, but those points will be
                outside the bar.
            random_seed : int, default None
                The random seed to be used. If None, randomization is
                non-reproducible.
        
        Returns:
            3-tuple or list of 3-tuples
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
        """A simple ray simulation on the neutron wall bar.

        Simple simulation means that there is no scattering, reflection,
        refraction, or any other actual physics consideration. Instead, we are
        only calculating geometrical intersections of random rays with the
        neutron wall bar geometry.

        Since the solid angle of the neutron wall bar is very small, the
        function would first identify the smallest rectangular region in the
        (theta, phi) space that fully contains the neutron wall bar. Then, it
        would randomly emit rays toward this minimal region. Both theta range
        and phi range will be saved, so that user can calculate the actual solid
        angle later on.

        Parameters
            n_rays : int, default 10
                Number of rays to be emitted. Not all of them are guaranteed to
                interact.
            random_seed : int, default None
                Random seed to be used for random number generation. If None, a
                time-based seed will be used, i.e. non-reproducible.
        
        Returns
            A dictionary with the following keys and array shapes:
                * origin : shape (3, )
                * azimuth_range : shape (2, )
                * polar_range : shape (2, )
                * intersections : shape (12, n_rays, 3)
            The first component of intersections has a size of 12, which comes
            from the 12 triangles that made up the neutron wall bar (we are
            using triangular mesh). Other functions, `self.get_hit_positions()`
            can be used to further simplify the intersection points.
        """
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
            angles = geom.angle_between(mean_vec, vecs, directional=True)
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
        rays = np.transpose(geom.spherical_to_cartesian(1.0, polars, azimuths))
        triangles = self.triangle_mesh.get_triangles()
        intersections = rti.moller_trumbore(origin, rays, triangles)

        # save results
        self.simulation_result = {
            'origin': origin,
            'azimuth_range': azimuth_range,
            'polar_range': polar_range,
            'intersections': intersections,
        }
        return self.simulation_result
    
    def get_hit_positions(
        self,
        hit_t='uniform',
        frame='local',
        coordinate='cartesian',
        random_seed=None,
        tol=1e-9,
    ):
        """Return the hit positions of the simulation.

        This function should be called after `self.simulation_result` is filled,
        e.g. by `self.simple_simulation()`.

        Parameters
            hit_t: scalar or callable or 'uniform', default 'uniform'
                If scalar, its value should be within [0, 1], with 0 being at
                the incident point at the surface, 1 being at the exit point,
                and the rest being somewhere in between. If callable, it should
                take in an integer `n_rays`, and return an array of size
                `n_rays` that collect the `hit_t` values in the range of [0, 1].
                If 'uniform', the `hit_t` values will be uniformly distributed
                over [0, 1].
            frame: 'local' or 'lab', default 'local'
                The coordinate frame of the returned hit positions.
            coordinate: 'cartesian' or 'spherical', default 'cartesian'
                The coordinate system of the returned hit positions.
            random_seed: int, default None
                The random seed used to generate the `hit_t` values. If None, a
                time-based seed will be used, i.e. non-reproducible.
            tol: float, default 1e-9
                The tolerance used to filter out hit points that are too close
                to the origin, i.e. there were no intersections. The default
                simulation setting is such that if there were no intersections,
                the hit position will be the origin. Here, we simply put in a
                non-zero tolerance to account for any floating point errors.
        
        Returns
            hit_positions: ndarray, shape (n_rays, 3)
                The hit positions of the simulation.
        """
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
        if len(hits) == 0:
            return hits

        # convert frame and coordinates
        if frame == 'local':
            hits = self.to_local_coordinates(hits)
        if coordinate == 'spherical':
            hits = geom.cartesian_to_spherical(hits)
        return hits
    
    def draw_hit_pattern2d(
        self,
        hits,
        ax,
        frame='lab',
        coordinate='spherical',
        cartesian_coordinates=('x', 'y'),
        cmap='jet',
    ):
        if isinstance(cmap, str):
            cmap = copy.copy(plt.get_cmap(cmap))
        else:
            cmap = copy.copy(cmap)
        cmap.set_under('white')

        if frame == 'lab' and coordinate == 'spherical':
            hist = fh.plot_histo2d(
                ax.hist2d,
                np.degrees(hits[:, 1]), np.degrees(hits[:, 2]),
                range=[[25, 55], [-30, 30]],
                bins=[250, 250],
                cmap=cmap,
                vmin=1,
            )

            ax.set_xlabel(r'Polar $\theta$ (deg)')
            ax.set_ylabel(r'Azimuth $\phi$ (deg)')

        elif frame == 'local' and coordinate == 'cartesian':
            cart = cartesian_coordinates # shorthand
            j = {'x': 0, 'y': 1, 'z': 2}
            ranges = dict(x=[-120, 120], y=[-4.5, 4.5], z=[-4.5, 4.5])
            bins = dict(x=200, y=100, z=100)

            hist = fh.plot_histo2d(
                ax.hist2d,
                hits[:, j[cart[0]]], hits[:, j[cart[1]]],
                range=[ranges[cart[0]], ranges[cart[1]]],
                bins=[bins[cart[0]], bins[cart[1]]],
                cmap=cmap,
                vmin=1,
            )

            ax.set_xlabel(r'$%s$ (cm)' % cart[0])
            ax.set_ylabel(r'$%s$ (cm)' % cart[1])
        else:
            raise ValueError('Frame-coordinate pair undefined yet')

        return hist

class Wall:
    database_dir = PROJECT_DIR / 'database/neutron_wall/geometry'

    def __init__(self, AB, contain_pyrex=True, refresh_from_inventor_readings=False):
        # initialize class parameters
        self.AB = AB.upper()
        self.ab = self.AB.lower()
        self.path_inventor_readings = self.database_dir / f'inventor_readings_NW{self.AB}.txt'
        self.path_vertices = self.database_dir / f'NW{self.AB}_vertices.dat'
        self.path_pca = self.database_dir / f'NW{self.AB}_pca.dat'

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
        bar_nums = self.database.index.get_level_values(f'nw{self.ab}-bar')
        self.bars = {
            b: Bar(self.database.loc[b][['x', 'y', 'z']], contain_pyrex=contain_pyrex)
            for b in bar_nums
        }
    
    @staticmethod
    def read_from_inventor_readings(filepath):
        with open(filepath, 'r') as file:
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
        
        # create Bar objects
        bars = [Bar(vertices) for vertices in bars_vertices]
        bars = sorted(bars, key=lambda bar: bar.pca.mean_[1]) # sort from bottom to top
        return bars
    
    @staticmethod
    def save_vertices_to_database(AB, filepath, bars):
        """
        Parameters
            AB : 'A' or 'B'
                Neutron wall A or B.
            filepath : str or pathlib.Path
                Path to the database file.
            bars : list of Bar objects
                Sorted from bottom to top. The bottommost bar is numbered 0; the
                topmost bar is numbered 24.
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
        """
        Parameters
            AB : str
                Neutron wall A or B.
            filepath : str or pathlib.Path
                Path to the database file.
            bars : list of Bar objects
                Sorted from bottom to top. The bottommost bar is numbered 0; the
                topmost bar is numbered 24.
        """
        # collect all PCA components and means from all bars and save into a dataframe
        df = []
        for bar_num, bar_obj in enumerate(bars):
            df.append([bar_num, 'L', *bar_obj.pca.mean_])
            for ic, component in enumerate(bar_obj.pca.components_):
                df.append([bar_num, 'XYZ'[ic], *component])

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
                #   - X, Y, Z are NW bar's principal components in lab frame w.r.t. to L
            '''),
            floatfmt=[
                '.0f',
                's',
                '.5f', '.5f', '.5f',
            ],
        )
