import numpy as np
import plotly.graph_objects as go

def moller_trumbore(ray_origin, ray_vectors, triangles, tol=1e-9):
    """Implementation of Moller-Trumbore ray-triangle intersection algorithm
    To read more about the mathematical derivation, visit
    https://www.scratchapixel.comhttps://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.
    In this implementation, we have made some light optimization. When looping
    over the triangles, we use pure Python for-loop. Inside the loop, we use
    only NumPy functions to manage all the ray_vectors, which are much faster
    than looping over with pure Python loops. This optimization, of course,
    works the best when dealing with a large number of ray vectors (> 1,000,000)
    rather than numerous triangles.

    Parameters:
        ray_origin : array of shape (3, )
            This implementation assumes all rays have one common origin.
        ray_vectors : array of shape (n_rays, 3)   
            The ray vectors are only used to tell the directions. No
            normalization is needed.
        triangles : array of shape (n_triangles, 3, 3)
            An array of 3 x 3 arrays. Each 3 x 3 array consists of three
            vertices that specify the triangle. For example, the y-coordinate of
            the first vertex in the fifth triangle would be
            `triangles[4][0][1]`.
        tol : float, default 1e-8
            Tolerance for checking the determinant in the algorithm. If the
            determinant is smaller than the tolerance, then the ray would be
            assumed parallel to the plane of the triangle, hence no attempt
            would be made to determine the intersection point.
    
    Returns:
        A numpy.ndarray of intersections with shape (n_triangles, n_rays, 3).
    """
    ray_origin, ray_vectors, triangles = map(lambda x: np.array(x), [ray_origin, ray_vectors, triangles])

    dot = lambda x, y: np.sum(np.multiply(x, y), axis=1) # allowing broadcasting

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

        u = np.where(non_parallel, dot(vec_p, translation) / denominator, -1e9)
        v = np.where(non_parallel, dot(vec_q, ray_vectors) / denominator, -1e9)

        intersected = (0 < u) & (u < 1) & (v > 0) & (u + v < 1)
        t = np.where(intersected, np.dot(vec_q, edge_02) / denominator, -1e9)[:, None]
        intersections.append(ray_origin + t * ray_vectors * (t > tol))

    return np.array(intersections)

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