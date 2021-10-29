"""A submodule that deals with 2D and 3D geometry.

Surprisingly, by the time of this writing, there does not seem to be any
libraries that can do several common geometrical manipulations well, in the
sense that they are pythonic, vectorized, and match the convention in physics.

Hence, this submodule is written.
"""
import copy
import functools
import itertools as itr

from alphashape import alphashape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
from sklearn.decomposition import PCA

from e15190.utilities import fast_histogram as fh
from e15190.utilities import ray_triangle_intersection as rti

class CoordinateConversion:
    @staticmethod
    def _spherical_to_cartesian(radius, polar, azimuth):
        """Convert spherical coordinates to Cartesian coordinates.

        This a private method that contains the actual mathematics of the
        conversion. It is meant to be used by other wrapper functions that
        accommondate various input formats and output formats.

        Parameters
        ----------
        radius : scalar or 1D numpy array
            The distance from the origin. Accepts any real numbers. Negative
            radius is simply interpreted as the opposite direction, e.g. (x, y,
            z) = (1, 2, 3) would become (-1, -2, -3).
        polar : scalar or 1D numpy array
            The polar angle of the point in radians. Commonly denoted as theta
            in physics. Accepts any real numbers, but expect identical result
            for any theta with the same :math:`\mathrm{mod}(\\theta, 2\pi)` value.
        azimuth : scalar or 1D numpy array
            The azimuthal angle of the point in radians. Commonly denoted as phi
            in physics. Accepts any real numbers, but expect identical result
            for any phi with the same :math:`\mathrm{mod}(\phi, 2\pi)` value.

        Returns
        -------
        x : scalar or 1D numpy array
            The x-coordinate of the point in Cartesian coordinates.
        y : scalar or 1D numpy array
            The y-coordinate of the point in Cartesian coordinates.
        z : scalar or 1D numpy array
            The z-coordinate of the point in Cartesian coordinates.    
        """
        x = radius * np.sin(polar) * np.cos(azimuth)
        y = radius * np.sin(polar) * np.sin(azimuth)
        z = radius * np.cos(polar)
        return x, y, z
    
    @staticmethod
    def _cartesian_to_spherical(x, y, z):
        """Convert Cartesian coordinates to spherical coordinates.

        This a private method that contains the actual mathematics of the
        conversion. It is meant to be used by other wrapper functions that
        accommondate various input formats and output formats.

        Parameters
        ----------
        x : scalar or 1D numpy array
            The x-coordinate of the point in Cartesian coordinates.
        y : scalar or 1D numpy array
            The y-coordinate of the point in Cartesian coordinates.
        z : scalar 1D numpy array
            The z-coordinate of the point in Cartesian coordinates.

        Returns
        -------
        radius : scalar or 1D numpy array
            The radius of the point in spherical coordinates. Has a range of
            :math:`[0, \infty)`.
        polar : scalar or 1D numpy array
            The polar angle of the point in radians. Commonly denoted as theta
            in physics. Has a range of :math:`[0, \pi)`.
        azimuth : scalar or 1D numpy array
            The azimuthal angle of the point in radians. Commonly denoted as phi
            in physics. Has a range of :math:`[-\pi, \pi)`.
        """
        # A trick to avoid "-0.00", which will lead to very different results
        # when fed into np.arctan2() in some edge cases:
        # >>> np.arctan2(+0.0, -1.0) = np.pi
        # >>> np.arctan2(-0.0, -1.0) = -np.pi
        x = x + 0.0
        y = y + 0.0
        z = z + 0.0

        # the conversion
        rho2 = x**2 + y**2
        radius = np.sqrt(rho2 + z**2)
        polar = np.arctan2(np.sqrt(rho2), z)
        azimuth = np.arctan2(y, x)
        return radius, polar, azimuth

def deco_to_2darray(func):
    """A decorator that turn functions into accepting row vectors.

    The argument `func` is assumed to be a function that does the following mapping:

    .. math::
        f: \\mathbb{R}^k \\rightarrow \\mathbb{R}^k \ , 
        (x_1, x_2, ..., x_k) \mapsto (y_1, y_2, ..., y_k)

    This decorator will turn the function into accepting 2D array of shape
    `(n_vectors, k)`. The result is equivalent to looping over the rows of the
    array and calling the undecorated function.
    """
    def inner(vecs):
        return np.vstack(func(*vecs.T)).T
    return inner

def spherical_to_cartesian(*args):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    args : 3-tuple of scalars, 3-tuple of 1D numpy arrays or a 2D numpy array
        If the input is a 3-tuple of scalars or a 3-tuple of 1D numpy arrays,
        they will be interpreted as :math:`(r, \\theta, \phi)`.

        If the input is a 2D numpy array of shape `(n_vectors, 3)`, it will be
        interpreted as an array of row vectors :math:`(r, \\theta, \phi)`.
               
        :math:`r` is the distance from the origin. Accepts any real numbers.
        Negative radius is simply interpreted as the opposite direction, e.g.
        :math:`(1, 2, 3)` becomes :math:`(-1, -2, -3)`.
        
        :math:`\\theta` is the polar angle of the point in radians. Accepts any
        real numbers, but expect identical result for any theta with the same
        :math:`\mathrm{mod}(\\theta, 2\pi)` value.
        
        :math:`\phi` is the azimuthal angle of the point in radians. Accepts
        any real numbers, but expect identical result for any phi with the same
        :math:`\mathrm{mod}(\phi, 2\pi)` value.

    Returns
    -------
    cartesian_coordinates : same as ``args``
        Cartesian coordinates of the inputs.
    """
    func = CoordinateConversion._spherical_to_cartesian
    if len(args) > 1:
        return func(*args)
    else:
        return deco_to_2darray(func)(args[0])

def cartesian_to_spherical(*args):
    """Convert Cartesian cordinates to spherical coordinates.

    Parameters
    ----------
    args : tuple of scalars, tuple of 1D numpy arrays or a 2D numpy array
        If the input is a tuple of scalars or a tuple of 1D numpy arrays,
        they will be interpreted as :math:`(x, y, z)`.
        
        If the input is a 2D numpy array of shape `(n_vectors, 3)`, it will be
        interpreted as an array of row vectors :math:`(x, y, z)`.
        
    Returns
    -------
    spherical_coordinates : same as ``args``
        The returned ranges are :math:`r \in [0, \infty)` for radius,
        :math:`\\theta \in [0, \pi]` for polar angle and :math:`\phi \in (-\pi,
        \pi]` for azimuthal angle.
    """
    func = CoordinateConversion._cartesian_to_spherical
    if len(args) > 1:
        return func(*args)
    else:
        return deco_to_2darray(func)(args[0])

def angle_between(v1, v2, directional=False, zero_vector=None):
    """Compute the angle between vectors.

    Parameters
    ----------
    v1 : 1D array-like or 2D numpy array of shape `(n_vectors, n_dim)`
        The first vector or vectors. The dimension of vectors must be consistent
        with ``v2``. If ``v1`` is a 2D numpy array, the length must also be
        consistent with the length of ``v2``.
    v2 : 1D array-like or 2D numpy array of shape `(n_vectors, n_dim)`
        The second vector or vectors. The dimension of vectors must be
        consistent with ``v1``. If ``v2`` is a 2D numpy array, the length must also
        be consistent with the length of ``v1``.
    directional : bool
        Whether to return the directional angle. Only support for vectors of
        two-dimension, otherwise it will be ignored. If `True`, the angle will
        be in the range :math:`(-\pi, \pi]`. If `False`, the angle will be in
        the range :math:`[0, \pi]`.
    zero_vector : None, 'raise' or scalar
        To specify what to do when ``v1`` or ``v2`` contains zero vectors, i.e.
        vectors with zero norms. If `None`, nothing will be done. Whatever
        warnings or errors show up during the calculation will be shown unless
        otherwise being suppressed. If 'raise', a ``RunTimeError`` will be
        raised. If a scalar, the value will be assigned as the angle. The last
        option is sometimes useful for later identifying zero vectors.

    Returns
    -------
    angles : float or 1D array-like
        The angle(s) between the two vectors in radians.
    """
    v1, v2 = np.array(v1), np.array(v2)

    # compute the dot products
    if v1.ndim == 1:
        dot_prod = np.dot(v2, v1)
    elif v2.ndim == 1:
        dot_prod = np.dot(v1, v2)
    elif v1.ndim == 2 and v2.ndim == 2 and len(v1) == len(v2):
        dot_prod = np.sum(v1 * v2, axis=1)
    else:
        raise ValueError('Cannot broadcast v1 and v2.')

    # calculate the angle
    norm_prod = np.sqrt(np.square(v1).sum(axis=-1) * np.square(v2).sum(axis=-1))
    if np.any(norm_prod == 0) and zero_vector == 'raise':
        raise RuntimeError('The norm of v1 or v2 is zero.')
    elif zero_vector is None or zero_vector == 'warn':
        # regular division
        angle = dot_prod / norm_prod
        angle = np.clip(angle, -1, 1) # clip to account for floating-point errors
        angle = np.arccos(angle)
    else:
        # division with special replacement at (norm == 0)
        mask = (norm_prod == 0)
        angle = np.divide(
            dot_prod, norm_prod,
            out=np.zeros_like(norm_prod),
            where=(~mask),
        )
        angle = np.clip(angle, -1, 1)
        angle = np.arccos(angle)
        angle = np.where(mask, zero_vector, angle)

    # determine direction from v1 to v2
    if directional and v1.shape[-1] == 2:
        sign = np.sign(np.cross(v1, v2))
        angle = angle * (1 * (sign >= 0) - 1 * (sign < 0))
    return angle

class RectangularBar:
    def __init__(self, vertices, calculate_local_vertices=True, make_vertices_dict=True):
        """Construct a rectangular bar.

        Parameters
        ----------
        vertices : ndarray of shape (8, 3)        
            Exactly eight tuples of (x, y, z) in lab coordinates. As long as all
            eight vertices are distinct, i.e. can properly define a cuboid, the
            order of vertices does not matter because PCA will be applied to
            automatically identify the 1st, 2nd and 3rd principal axes of the
            bar.
        calculate_local_vertices : bool, default True
            Whether to calculate the local vertices. Local vertices are the
            coordinates with respect to the bar's x, y and z axes, defined as
            the 1st, 2nd and 3rd principal axes.
        make_vertices_dict : bool, default True
            Whether to make the vertices dictionary. See more at
            :py:func:`_make_vertices_dict`.
        """
        self.vertices = np.array(vertices)
        """dict : The vertices of the bar in lab frame (unit cm)

        A length-8 dictionary of the form ``{(i, j, k): (x, y, z)}``
        """

        self.loc_vertices = None
        """dict : The vertices of the bar in local frame (unit cm)

        A length-8 dictionary of the form ``{(i, j, k): (x', y', z')}``
        """

        self.pca = PCA(n_components=3, svd_solver='full')
        """``sklearn.decomposition.PCA`` : The PCA of the bar vertices in lab frame"""
        self.pca.fit(self.vertices)

        # update the vertices in local coordinates
        if calculate_local_vertices:
            self.loc_vertices = self.to_local_coordinates(vertices)

        # update both the vertices and the local vertices into dictionaries
        if make_vertices_dict:
            self.vertices_dict = self._make_vertices_dict()
    
    def _make_vertices_dict(self):
        """Returns a dictionary of directions mapped onto vertices.

        Attributes ``self.vertices`` and ``self.loc_vertices`` will be updated.

        A rectangular bar has eight vertices. To distinguish the vertices, we
        use 3-tuples of -1 and +1 to indicate the directions in local (bar)
        frame. For example, ``(-1, +1, -1)`` implies the vertex, when observed
        in the local frame, is in the negative x direction, positive y
        direction, and negative z direction. Each vertex's coordinates would be
        stored as numpy array of length 3, representing :math:`(x, y, z)`.
        """
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
    
    def dimension(self, index=None):
        """Returns the dimension(s) of the bar.
        
        Parameters
        ----------
        index : int, default None
            The index of the direction. If `None`, all three dimensions will be
            returned. Indices 1, 2 and 3 correspond to the x, y and z in the
            local (bar) frame.
        
        Returns
        -------
        dimension : float or 3-tuple of floats
            The dimension(s) of the bar.
        """
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
    
    @staticmethod
    def _deco_numpy_ndim(func):
        """Decorator for preserving numpy array's dimensionality.

        Some functions only accepts 2D arrays, i.e. rows of 1D arrays. To apply
        this kind of functions on 1D arrays, we may use this decoratord. The
        decorated functions can take in either a 2D array or a 1D array. If the
        input is a 2D array, the output will be a 2D array; if the input is a 1D
        array, the output will be kept as a 1D array.

        Parameters
        ----------
        func : function
            The function to decorate.

        Returns
        -------
        func : function
            The decorated function.
        """
        def inner(x, *args, **kwargs):
            x = np.array(x)
            ndim = x.ndim
            if ndim == 1:
                x = np.array([x])
            result = func(x, *args, **kwargs)
            return result if ndim == 2 else np.squeeze(result)
        return inner

    def to_local_coordinates(self, lab_coordinates):
        """Converts coordinates from lab frame to local frame.

        Always assume Cartesian coordinates in both frames.

        Parameters
        ----------
        lab_coordinates : (N, 3) or (3, ) array_like
            The lab coordinates :math:`(x, y, z)` to be converted.

        Returns
        -------
        local_coordinates : (N, 3) or (3, ) array_like
            The local coordinates :math:`(x', y', z')`.
        """
        return self._deco_numpy_ndim(self.pca.transform)(lab_coordinates)

    def to_lab_coordinates(self, local_coordinates):
        """Converts coordinates from local frame to lab frame.

        Always assume Cartesian coordinates in both frames.

        Parameters
        ----------
        local_coordinates : (N, 3) or (3, ) array_like
            The local coordinates :math:`(x', y', z')` to be converted.

        Returns
        -------
        lab_coordinates : (N, 3) or (3, ) array_like
            The lab coordinates :math:`(x, y, z)`.
        """
        return self._deco_numpy_ndim(self.pca.inverse_transform)(local_coordinates)

    def is_inside(self, coordinates, frame='local', tol=5e-4):
        """Check if the given coordinates are inside the bar.

        Parameters
        ----------
        coordinates : 3-tuple or list of 3-tuples
            The coordinates to be checked. Only support Cartesian coordinates.
        frame : 'local' or 'lab', default 'local'
            The frame of the coordinates.
        tol : float, default 5e-4
            The tolerance of the boundary check.
        
        Returns
        -------
        inside : bool or an array of bool
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

    def construct_plotly_mesh3d(self):
        """Update bar's attribute ``self.triangle_mesh``.

        Construct a
        :py:class:`TriangleMesh <e15190.utilities.ray_triangle_intersection.TriangleMesh>`
        object, i.e. a triangular mesh for the geometry of the bar, and save as
        ``self.triangle_mesh``.
        """
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

    def simple_simulation(
        self,
        n_rays=10,
        random_seed=None,
        polar_range=None,
        azimuth_range=None,
        save_result=True,
    ):
        """A simple ray simulation on the bar.

        Simple simulation means that there is no scattering, reflection,
        refraction, or any other actual physics consideration. Instead, we are
        only calculating geometrical intersections of random rays with the bar
        geometry.

        By default, this function would first identify the smallest rectangular
        region in the (theta, phi) space that fully contains the entire bar
        geometry.  Then, it would randomly emit rays toward this minimal region.
        Both theta range and phi range will be saved, so that user can calculate
        the actual solid angle later on.

        Parameters
        ----------
        n_rays : int, default 10
            Number of rays to be emitted. Not all of them are guaranteed to
            interact.
        random_seed : int, default None
            Random seed to be used for random number generation. If None, a
            time-based seed will be used, i.e. non-reproducible.
        polar_range : 2-tuple of floats or None, default None
            The polar range in radians. If None, the algorithm will use the
            minimal range determined by the bar vertices. This allows more
            efficient simulation.
        azimuth_range : 2-tuple of floats or None, default None
            The azimuth range in radians. If None, the algorithm will use the
            minimal range determined by the bar vertices. This allows more
            efficient simulation.
        save_result : bool, default True
            Whether to save the result to this object as
            ``self.simulation_result``.
        
        Returns
        -------
        simulation_result : dict
            A dictionary with the following keys and array shapes:
                * ``origin`` : shape (3, )
                * ``azimuth_range`` : shape (2, )
                * ``polar_range`` : shape (2, )
                * ``intersections`` : shape (12, ``n_rays``, 3)
            The first component of intersections has a size of 12, which comes
            from the 12 triangles that made up the rectangular bar (triangle
            mesh). Other functions like :py:func:`get_hit_positions` can be used
            to further simplify the intersection points.
        """
        if 'triangle_mesh' not in self.__dict__:
            self.construct_plotly_mesh3d()
        origin = np.array([0.0] * 3)

        # identify the minimal region in space to emit rays
        # The region is orthogonal in the (polar, azimuth) space, i.e. some
        # rectangle, so users can later easily recover the cross section from
        # the isotropy of simulation rays.
        if polar_range is None and azimuth_range is None:
            vertices = np.array(list(self.vertices.values()))
            xy_vecs = vertices[:, [0, 1]]
            zrho_vecs = np.vstack([vertices[:, 2], np.linalg.norm(xy_vecs, axis=1)]).T

            def identify_angle_range(vecs):
                mean_vec = np.mean(vecs, axis=0)
                angles = angle_between(mean_vec, vecs, directional=True)
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
        rays = rti.emit_isotropic_rays(
            n_rays,
            polar_range=polar_range, azimuth_range=azimuth_range,
            random_seed=random_seed,
        )
        triangles = self.triangle_mesh.get_triangles()
        intersections = rti.moller_trumbore(origin, rays, triangles)

        # save results
        simulation_result = {
            'origin': origin,
            'azimuth_range': azimuth_range,
            'polar_range': polar_range,
            'intersections': intersections,
        }
        if save_result:
            self.simulation_result = simulation_result
        return simulation_result
    
    def get_hit_positions(
        self,
        hit_t='uniform',
        frame='local',
        coordinate='cartesian',
        simulation_result=None,
        random_seed=None,
        tol=1e-9,
    ):
        """Return the hit positions of the simulation.

        This function should be called after ``self.simulation_result`` is
        updated, e.g. by calling
        :py:func:`simple_simulation <simple_simulation>`.

        Parameters
        ----------
        hit_t : scalar or callable or 'uniform', default 'uniform'
            If scalar, its value should be within [0, 1], with 0 being at the
            incident point at the surface, 1 being at the exit point, and the
            rest being somewhere in between. If callable, it should take in an
            integer ``n_rays``, and return an array of size ``n_rays`` that
            collect the ``hit_t`` values in the range of [0, 1].  If 'uniform',
            the ``hit_t`` values will be uniformly distributed over [0, 1].
        frame : 'local' or 'lab', default 'local'
            The coordinate frame of the returned hit positions.
        coordinate : 'cartesian' or 'spherical', default 'cartesian'
            The coordinate system of the returned hit positions.
        simulation_result : dict, default None
            The simulation result to be used. If None, the function uses
            ``self.simulation_result``.
        random_seed : int, default None
            The random seed used to generate the ``hit_t`` values. If None, a
            time-based seed will be used, i.e. non-reproducible.
        tol : float, default 1e-9
            The tolerance used to filter out hit points that are too close to
            the origin, i.e. there were no intersections. The default simulation
            setting is such that if there were no intersections, the hit
            position will be the origin. Here, we simply put in a non-zero
            tolerance to account for any floating point errors.
        
        Returns
        -------
        hit_positions : ndarray, shape (n_rays, 3)
            The hit positions of the simulation.
        """
        if simulation_result is None:
            sim_res = self.simulation_result # shorthand
        else:
            sim_res = simulation_result

        # shift to make origin at zero
        hits = sim_res['intersections'] - np.array(sim_res['origin'])

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
            hits = cartesian_to_spherical(hits)
        return hits
    
    def get_total_solid_angle_monte_carlo(
        self,
        n_rays=1_000_000,
        random_seed=None,
        return_error=False,
        verbose=False,
    ):
        """Calculate the total solid angle of the bar by Monte Carlo simulation.

        Parameters
        ----------
        n_rays : int, default 1_000_000
            The number of rays to simulate.
        random_seed : int, default None
            The random seed of the Monte Carlo simulation. If None, a time-based
            seed will be used, i.e. non-reproducible.
        return_error : bool, default False
            Whether to return the error (uncertainty) of the calculation.
        verbose : bool, default False
            Whether to print out the progress of the calculation.
        
        Returns
        -------
        total_solid_angle : float
            The total solid angle of the bar in steradians.
        error : float, optional
            The uncertainty of the calculation. Only returned if
            ``return_error`` is True.
        """
        # get the solid angle of minimal region
        sim_result = self.simple_simulation(save_result=False, random_seed=random_seed)
        region_solid_angle = np.ptp(sim_result['azimuth_range'])
        region_solid_angle *= np.ptp(np.cos(sim_result['polar_range']))

        n_rays_per_sim = int(2e5)
        n_rays_list = [n_rays_per_sim] * (n_rays // n_rays_per_sim) + [n_rays % n_rays_per_sim]
        n_hits = 0
        n_simulated = 0
        for n in n_rays_list:
            if verbose:
                print(f'\rSimulated {n_simulated:,d} rays', end='')
                n_simulated += n
            sim_res = self.simple_simulation(
                n_rays=n,
                polar_range=sim_result['polar_range'],
                azimuth_range=sim_result['azimuth_range'],
                save_result=False,
            )
            n_hits += len(self.get_hit_positions(hit_t=0, simulation_result=sim_res))
        if verbose:
            print(f'\rSimulated {n_rays:,d} rays')
        
        solid_angle = (n_hits / n_rays) * region_solid_angle
        if return_error:
            error = (np.sqrt(n_hits) / n_rays) * region_solid_angle
            return solid_angle, error
        else:
            return solid_angle
    
    def get_theta_phi_alphashape(self, delta=1.0, alpha=10.0, cut=None):
        """Calculate the alphashape of the bar in (theta, phi) coordinates.

        Parameters
        ----------
        delta : int or array of size 3, default 1.0
            Grid spacing in cm.
        alpha : float, default 10.0
            The alphashape parameter. The larger the value, the greater the
            curvature of the alphashape can be. When alpha is zero, the
            alphashape reduces into a convex hull; when alpha is too large, the
            alphashape could be "overfitted" to data points.
        cut : str, default None
            The cut of the alphashape. Available variables include 'x', 'y' and
            'z' in local frame. If None, no cut will be applied.
        
        Returns
        -------
        alpha_shape_xy : ndarray, shape (n, 2)
            Vertices that define the alphashape in (theta, phi) coordinates.
        """
        if isinstance(delta, (int, float)):
            delta = [delta] * 3
        n = [max(int(np.ceil(self.dimension(i) / delta[i])), 3) for i in range(3)]

        perms = np.array(list(itr.product([0, 1], repeat=2)))
        w = np.empty((0, 3))

        x_grid = np.linspace(0, 1, n[0])
        x_perms = np.tile(perms, (1, len(x_grid))).reshape(-1, 2)
        pts = np.concatenate([np.tile(x_grid, 4)[:, None], x_perms], axis=1)
        pts = pts[:, [0, 1, 2]]
        w = np.concatenate([w, pts])

        y_grid = np.linspace(0, 1, n[1])
        y_perms = np.tile(perms, (1, len(y_grid))).reshape(-1, 2)
        pts = np.concatenate([np.tile(y_grid, 4)[:, None], y_perms], axis=1)
        pts = pts[:, [2, 0, 1]]
        w = np.concatenate([w, pts])

        z_grid = np.linspace(0, 1, n[2])
        z_perms = np.tile(perms, (1, len(z_grid))).reshape(-1, 2)
        pts = np.concatenate([np.tile(z_grid, 4)[:, None], z_perms], axis=1)
        pts = pts[:, [1, 2, 0]]
        w = np.concatenate([w, pts])

        corner0 = self.loc_vertices[(-1, -1, -1)]
        corner1 = self.loc_vertices[(1, 1, 1)]
        coords = pd.DataFrame({
            'x': w[:, 0] * corner0[0] + (1 - w[:, 0]) * corner1[0],
            'y': w[:, 1] * corner0[1] + (1 - w[:, 1]) * corner1[1],
            'z': w[:, 2] * corner0[2] + (1 - w[:, 2]) * corner1[2],
        })
        if cut is not None:
            coords = coords.query(cut)
        coords = self.to_lab_coordinates(coords)

        coords = cartesian_to_spherical(coords)
        ashape = alphashape(coords[:, [1, 2]], alpha)
        return np.transpose(ashape.exterior.coords.xy)
    
    @staticmethod
    def _split_theta_phi_alphashape_to_upper_and_lower(ashape):
        """Split the alphashape into upper and lower parts.
    
        After splitting, both parts should form a function of theta. This is a
        function written mainly to help calculating the azimuthal width
        :math:`\delta\phi` as a function of theta, which then is used to give
        the geometry efficiency.

        Parameters
        ----------
        ashape : ndarray, shape (n, 2)
            Vertices that define the alphashape in (theta, phi) coordinates.
        
        Returns 
        -------
        split_ashape : dict
            The upper and lower parts of the alphashape. The keys are 'upper'
            and 'lower', the values are ndarrays of shape (n, 2).
        """
        i_min = np.argmin(ashape[:, 0])
        i_max = np.argmax(ashape[:, 0])
        i_min, i_max = min(i_min, i_max), max(i_min, i_max)
        upper_ashape = ashape[i_min:i_max + 1]
        lower_ashape = ashape[np.mod(range(i_max, i_min + len(ashape) + 1), len(ashape))]
        if np.max(lower_ashape[:, 1]) > np.max(upper_ashape[:, 1]):
            upper_ashape, lower_ashape = lower_ashape, upper_ashape
        return {
            'upper': np.array(sorted(upper_ashape, key=lambda x: x[0])),
            'lower': np.array(sorted(lower_ashape, key=lambda x: x[0])),
        }
    
    def get_total_solid_angle_alphashape(self, return_error=False):
        """Calculate the total solid angle of the bar using the alphashape.

        Parameters
        ----------
        return_error : bool, default False
            Whether to return the error (uncertainty) of the calculation.
        
        Returns
        -------
        total_solid_angle : float
            The total solid angle of the bar in steradians.
        error : float, optional
            The uncertainty of the calculation. Only returned if
            ``return_error`` is True.
        """
        ashape = self.get_theta_phi_alphashape()
        delta_phi = self.get_geometry_efficiency_alphashape()

        # integrate to get the total solid angle
        theta_range = [np.min(ashape[:, 0]), np.max(ashape[:, 0])]
        integrate = scipy.integrate.quadrature
        integrand = lambda theta: delta_phi(theta) * np.sin(theta)
        total_solid_angle, err = integrate(integrand, *theta_range)
        if return_error:
            return total_solid_angle, err
        else:
            return total_solid_angle
    
    @functools.lru_cache(maxsize=5)
    def get_total_solid_angle(
        self,
        method='alphashape',
        **kwargs,
    ):
        """Return the total solid angle of the bar.

        A wrapper function for the different methods of calculating the total
        solid angle.

        Parameters
        ----------
        method : 'alphashape' or 'monte_carlo'
            The method to use to calculate the total solid angle.
        kwargs : dict()
            Keyword arguments to pass to the method.
        
        Returns
        -------
        total_solid_angle : float
            The total solid angle of the bar in steradians.
        """
        method = method.lower()
        method = ''.join(c for c in method if c.isalpha())
        if method == 'alphashape':
            return self.get_total_solid_angle_alphashape(**kwargs)
        elif method == 'montecarlo':
            return self.get_total_solid_angle_monte_carlo(**kwargs)
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_geometry_efficiency_alphashape(self, cut=None):
        """Calculate the geometry efficiency of the bar using the alphashape.

        The geometry efficiency is defined as the effective azimuthal coverage
        ratio, i.e.
        :math:`\delta\phi/(2\pi)`, as a function of theta.

        Parameters
        ----------
        cut : str or list of str, default None
            Cut to apply to the dataframe of the bar. Available variables include
            'x', 'y' and 'z' in local frame. If None, no cut is applied.

            Users should make sure that the every cut string must only split the
            geometry of the bar into one volume, so cut like ``(-10 < x & x <
            10)`` is okay, but ``(-10 < x & x < 10) & (15 < x & x < 25)`` is not
            because there is a gap from x = 10 to x = 15, resulting in two
            unconnected volumes. To apply cuts that result in multiple volumes,
            supply a list of cuts instead. So
            ``cut = ['(-10 < x & x < 10)', '(15 < x & x < 25)']``
            would be okay.

        Returns
        -------
        delta_phi : callable
            A function that takes theta (radian) as an argument and returns the
            geometry efficiency.
        """
        if isinstance(cut, list):
            geom_eff = [
                self.get_geometry_efficiency_alphashape(cut=single_cut)
                for single_cut in cut
            ]
            def result(theta):
                nonlocal geom_eff
                return np.sum([single_geom_eff(theta) for single_geom_eff in geom_eff], axis=0)
            return result

        ashape = self.get_theta_phi_alphashape(cut=cut)
        ashape = self._split_theta_phi_alphashape_to_upper_and_lower(ashape)

        # find the delta-phi as a function of theta using interpolation
        interp = lambda x, y: scipy.interpolate.interp1d(
            x, y,
            kind='linear', fill_value=0.0, bounds_error=False,
        )
        upper_line = interp(ashape['upper'][:, 0], ashape['upper'][:, 1])
        lower_line = interp(ashape['lower'][:, 0], ashape['lower'][:, 1])
        delta_phi = lambda theta: upper_line(theta) - lower_line(theta)
        return delta_phi

    def draw_hit_pattern2d(
        self,
        hits,
        ax,
        frame='lab',
        coordinate='spherical',
        cartesian_coordinates=('x', 'y'),
        cmap='jet',
    ):
        """Draw the hit pattern of the bar with 2D histogram.

        Parameters
        ----------
        hits : ndarray, shape (n, 3)
            The hits to draw.
        ax : matplotlib.axes.Axes
            The axes to draw the hits on.
        frame : 'lab' or 'local'
            The frame of reference to use for the hits.
        coordinate : 'spherical' or 'cartesian'
            The coordinate system to use for the hits.
        cartesian_coordinates : tuple of str
            The Cartesian coordinates to use for the hits. If ``coordinate`` is
            not 'cartesian', this argument has no effect.
        cmap : str or matplotlib.colors.Colormap
            The colormap to use for plotting the hits in 2D histogram.

        Returns
        -------
        hist : 2D ndarray
            The histogram counts of the hits.
        """
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