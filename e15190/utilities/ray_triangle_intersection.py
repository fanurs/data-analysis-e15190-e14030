import functools
import itertools

import numpy as np
import plotly.graph_objects as go
import sympy as sp

from e15190.utilities import geometry as geom

def moller_trumbore(ray_origin, ray_vectors, triangles, mode='einsum', tol=1e-9):
    """Implementation of Moller-Trumbore ray-triangle intersection algorithm
    To read more about the mathematical derivation, visit
    https://www.scratchapixel.comhttps://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.

    Parameters:
        ray_origin : array of shape (3, )
            This implementation only allows all rays to have one common origin.
        ray_vectors : array of shape (n_rays, 3)   
            The ray vectors are only used to tell the directions. No
            normalization is needed.
        triangles : array of shape (n_triangles, 3, 3)
            An array of 3 x 3 arrays. Each 3 x 3 array consists of three
            vertices that specify the triangle. For example, the y-coordinate of
            the first vertex in the fifth triangle would be
            `triangles[4][0][1]`.
        mode : str, default `'einsum'`
            To select different implementations of the algorithm. In 'einsum'
            mode, the `numpy.einsum()` function is heavily used to optimize the
            calculations. Otherwise, we will use the implementation without
            'einsum'. Our simple test suggests, for 1 million rays and 12
            triangles, 'einsum' mode performs at least two times faster than the
            mode without 'einsum'.
        tol : float, default 1e-9
            Tolerance for checking the determinant in the algorithm. If the
            determinant is smaller than the tolerance, then the ray would be
            assumed parallel to the plane of the triangle, hence no attempt
            would be made to determine the intersection point.
    
    Returns:
        A numpy.ndarray of intersections with shape (n_triangles, n_rays, 3).
    """
    args = (ray_origin, ray_vectors, triangles, tol)
    if mode == 'einsum':
        return _moller_trumbore_with_einsum(*args)
    else:
        return _moller_trumbore_with_loop_over_triangles(*args)

def _moller_trumbore_with_einsum(ray_origin, ray_vectors, triangles, tol):
    ray_origin, ray_vectors, triangles = map(lambda x: np.array(x), [ray_origin, ray_vectors, triangles])

    eijk = [sp.LeviCivita(*ijk) for ijk in itertools.product(range(1, 4), repeat=3)]
    eijk = np.reshape(eijk, (3, 3, 3)).astype(float)
    einsum = functools.partial(np.einsum, optimize='optimal')

    edge_01 = triangles[:, 1] - triangles[:, 0] # (tri, 3)
    edge_02 = triangles[:, 2] - triangles[:, 0] # (tri, 3)
    translation = ray_origin - triangles[:, 0] # (tri, 3)

    # einsum allows us to skip intermediate vectors, vec_p and vec_q
    # which improves the performance significantly.
    # For future reference, here are how vec_p and vec_q can be calculated:
    #    >> vec_p = einsum('ijk,rj,tk->tri', eijk, ray_vectors, edge_02)
    #    >> vec_q = einsum('ijk,tj,tk->ti', eijk, translation, edge_01)

    # denominator = vec_p cross edge_01
    denominator = einsum( # shape = (tri, ray, 3)
        'ijk,rj,tk,ti->tr',
        eijk, ray_vectors, edge_02, edge_01 # ==> vec_p, edge_01
    )
    non_parallel = (np.abs(denominator) > tol) # (tri, ray, 3)
    denominator[~non_parallel] = tol # to avoid divide by zero warning
    scalar = 1 / denominator

    # u = vec_p dot translation / denominator
    u = einsum( # shape = (tri, ray)
        'ijk,rj,tk,ti,tr->tr',
        eijk, ray_vectors, edge_02, translation, scalar, # ==> vec_p, translation, scalar
    )

    # v = vec_q dot ray_vectors / denominator
    v = einsum( # shape = (tri, ray)
        'ijk,tj,tk,ri,tr->tr',
        eijk, translation, edge_01, ray_vectors, scalar, # ==> vec_q, ray_vectors, scalar
    )

    # t = vec_q dot edge_02 / denominator
    t = einsum( # shape = (tri, ray)
        'ijk,tj,tk,ti,tr->tr',
        eijk, translation, edge_01, edge_02, scalar, # ==> vec_q, edge_02, scalar
    )
    intersected = (0 < u) & (u < 1) & (v > 0) & (u + v < 1)
    t[(~intersected) | (t <= tol)] = 0.0

    return ray_origin + einsum('tr,ri->tri', t, ray_vectors) # shape = (tri, ray, 3)

def _moller_trumbore_with_loop_over_triangles(ray_origin, ray_vectors, triangles, tol):
    ray_origin, ray_vectors, triangles = map(lambda x: np.array(x), [ray_origin, ray_vectors, triangles])

    dot = lambda x, y: np.sum(np.multiply(x, y), axis=1) # to allow broadcasting

    intersections = []
    for triangle in triangles:
        edge_01 = triangle[1] - triangle[0]
        edge_02 = triangle[2] - triangle[0]
        translation = ray_origin - triangle[0]

        vec_p = np.cross(ray_vectors, edge_02)
        vec_q = np.cross(translation, edge_01)

        denominator = dot(vec_p, edge_01)
        non_parallel = (np.abs(denominator) > tol)
        denominator[~non_parallel] = tol # to avoid divide by zero warning

        u = dot(vec_p, translation) / denominator
        v = dot(vec_q, ray_vectors) / denominator

        intersected = (0 < u) & (u < 1) & (v > 0) & (u + v < 1)
        t = np.where(intersected, np.dot(vec_q, edge_02) / denominator, -1e9)[:, None]
        intersections.append(ray_origin + t * ray_vectors * (t > tol))

    return np.array(intersections)

def emit_isotropic_rays(
    n_rays,
    polar_range=[0, np.pi],
    azimuth_range=[-np.pi, np.pi],
    random_seed=None,
    frame='cartesian',
):
    """Return rays with random directions (isotropic emission).

    Parameters:
        n_rays : int
            Number of rays to be emitted.
        polar_range : 2-tuple or 2-list, default [0, np.pi]
            The range of polar angles in radians.
        azimuth_range : 2-tuple or 2-list, default [-np.pi, np.pi]
            The range of azimuth angles in radians.
        random_seed : int, default None
            Random seed for the random number generator. If `None`, then the
            randomization is non-reproducible.
        frame : 'cartesian' or 'spherical', default 'cartesian'
            If 'cartesian', the rays are returned as rows of `(x, y, z)`; if
            'spherical', the rays are returned as rows of `(1, theta, phi)`, in
            radians.

    Returns:
        A numpy.ndarray of rays with shape (n_rays, 3).
    """
    rng = np.random.default_rng(random_seed)
    polars = np.arccos(rng.uniform(*np.cos(polar_range)[::-1], size=n_rays).clip(-1, 1))
    azimuths = rng.uniform(*azimuth_range, size=n_rays)
    if frame == 'cartesian':
        rays = np.column_stack(geom.spherical_to_cartesian(1.0, polars, azimuths))
    else:
        rays = np.column_stack([polars, azimuths])
        rays = np.insert(rays, 0, 1.0, axis=1)
    return rays

class TriangleMesh:
    def __init__(self, vertices, tri_indices):
        self.vertices = np.array(vertices, dtype=float)
        self.tri_indices = np.array(tri_indices, dtype=int)

    def get_triangles(self):
        return self.vertices[self.tri_indices]
    
    def plotly_trace(self, **kwargs):
        kwargs.update({c: self.vertices[:, i] for i, c in enumerate('xyz')})
        kwargs.update({c: self.tri_indices[:, i] for i, c in enumerate('ijk')})
        kwargs.setdefault('flatshading', True)
        return go.Mesh3d(**kwargs)