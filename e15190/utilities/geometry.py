"""A submodule that deals with 2D and 3D geometry.

Surprisingly, by the time of this writing, there does not seem to be any
libraries that can do several common geometrical manipulations well, in the
sense that they are pythonic, vectorized, and match the convention in physics.

Hence, this submodule is written.
"""
import numpy as np

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
