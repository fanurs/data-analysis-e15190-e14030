import pytest

import numpy as np

from e15190.utilities import geometry as geo
from e15190.utilities import ray_triangle_intersection as rti

class TestCoordinateConversion:
    def test__spherical_to_cartesian(self):
        func = geo.CoordinateConversion._spherical_to_cartesian

        io_pairs = np.array([
            [
                [1.0,    0.0 * np.pi,    0.0 * np.pi], # input: [r, theta, phi]
                [0.0,    0.0,    1.0],                 # expected output: [x, y, z]
            ],

            # changing radii
            [
                [2.0,    0.0 * np.pi,    0.0 * np.pi],
                [0.0,    0.0,    2.0],
            ],
            [
                [0.0,    1.2 * np.pi,    3.4 * np.pi],
                [0.0,    0.0,    0.0],
            ],
            [
                [-1.0,   0.0 * np.pi,    0.0 * np.pi],
                [0.0,    0.0,   -1.0],
            ],

            # changing theta (polar angle) at 90-degree intervals
            [
                [1.0,    0.5 * np.pi,    0.0 * np.pi],
                [1.0,    0.0,    0.0],
            ],
            [
                [1.0,    1.0 * np.pi,    0.0 * np.pi],
                [0.0,    0.0,   -1.0],
            ],
            [
                [1.0,    1.5 * np.pi,    0.0 * np.pi],
                [-1.0,   0.0,    0.0],
            ],
            [
                [1.0,    2.0 * np.pi,    0.0 * np.pi],
                [0.0,   0.0,    1.0],
            ],

            # check phi (azimuthal angle) should have no effect when theta = 0
            [
                [1.0,    0.0 * np.pi,    0.5 * np.pi],
                [0.0,    0.0,    1.0],
            ],

            # changing phi at 90-degree intervals
            [
                [1.0,    0.5 * np.pi,    0.5 * np.pi],
                [0.0,    1.0,    0.0],
            ],
            [
                [1.0,    0.5 * np.pi,    1.0 * np.pi],
                [-1.0,   0.0,    0.0],
            ],
            [
                [1.0,    0.5 * np.pi,    1.5 * np.pi],
                [0.0,   -1.0,    0.0],
            ],
            [
                [1.0,    0.5 * np.pi,    2.0 * np.pi],
                [1.0,    0.0,    0.0],
            ],

            # negative theta
            [
                [1.0,   -0.5 * np.pi,    0.0 * np.pi],
                [-1.0,   0.0,    0.0],
            ],

            # negative phi
            [
                [1.0,    0.5 * np.pi,   -0.5 * np.pi],
                [0.0,   -1.0,    0.0],
            ]
        ])
        sph_coords = io_pairs[:, 0]
        expected_cart_coords = io_pairs[:, 1]

        # test as numpy arrays
        outputs = func(*sph_coords.T)
        outputs = np.vstack(outputs).T
        assert np.allclose(outputs, expected_cart_coords)

        # test as tuple of scalars
        for io_pair in io_pairs:
            sph_coord = io_pair[0]
            outputs = func(sph_coord[0], sph_coord[1], sph_coord[2])
            expected_cart_coord = tuple(io_pair[1])
            assert outputs == pytest.approx(expected_cart_coord)
    
    def test__cartesian_to_spherical(self):
        func = geo.CoordinateConversion._cartesian_to_spherical

        io_pairs = np.array([
            [
                [0.0,    0.0,    1.0],                 # nput: [x, y, z]
                [1.0,    0.0 * np.pi,    0.0 * np.pi], # expected output: [r, theta, phi]
            ],
            [ # scale up z
                [0.0,    0.0,    2.0],
                [2.0,    0.0 * np.pi,    0.0 * np.pi],
            ],
            [ # zero vector
                [0.0,    0.0,    0.0],
                [0.0,    0.0 * np.pi,    0.0 * np.pi],
            ],
            [ # always gives positive radius
                [0.0,    0.0,   -1.0],
                [1.0,    1.0 * np.pi,    0.0 * np.pi],
            ],

            # rotating vector on xy-plane at 90-degree intervals
            [
                [1.0,    0.0,    0.0],
                [1.0,    0.5 * np.pi,    0.0 * np.pi],
            ],
            [
                [0.0,    1.0,    0.0],
                [1.0,    0.5 * np.pi,    0.5 * np.pi],
            ],
            [
                [-1.0,   0.0,    0.0],
                [1.0,    0.5 * np.pi,    1.0 * np.pi],
            ],
            [ # principal value of arctan2() ranges from (-pi, pi]
                [0.0,   -1.0,    0.0],
                [1.0,    0.5 * np.pi,   -0.5 * np.pi],
            ],

            # on trying to get phi = -pi, which should be impossible
            [
                [-1.0,  -0.0,    0.0],
                [1.0,    0.5 * np.pi,    1.0 * np.pi],
            ],
        ])
        cart_coords = io_pairs[:, 0]
        expected_sph_coords = io_pairs[:, 1]

        # test as numpy arrays
        outputs = func(*cart_coords.T)
        outputs = np.vstack(outputs).T
        assert np.allclose(outputs, expected_sph_coords)

        # test as tuple of scalars
        for io_pair in io_pairs:
            cart_coord = io_pair[0]
            outputs = func(cart_coord[0], cart_coord[1], cart_coord[2])
            expected_sph_coord = tuple(io_pair[1])
            assert outputs == pytest.approx(expected_sph_coord)

def test_deco_to_2darray():
    func = lambda x, y: (x + y, x - y)
    deco_func = geo.deco_to_2darray(func)
    x = np.array([1, 3, 6.5])
    y = np.array([2, 4, 5.5])
    xy = np.vstack([x, y]).T

    outputs = func(x, y)
    deco_outputs = deco_func(xy)
    assert outputs[0] == pytest.approx(deco_outputs[:, 0])
    assert outputs[1] == pytest.approx(deco_outputs[:, 1])

def test_spherical_to_cartesian():
    func = geo.spherical_to_cartesian
    n_pts = 20
    radius = np.random.uniform(-3, 3, size=n_pts)
    theta = np.random.uniform(-2 * np.pi, 2 * np.pi, size=n_pts)
    phi = np.random.uniform(-2 * np.pi, 2 * np.pi, size=n_pts)
    sph_pts = np.vstack([radius, theta, phi]).T

    # test as numpy arrays
    outputs = func(radius, theta, phi)
    deco_outputs = func(sph_pts)
    for i in range(3):
        assert outputs[i] == pytest.approx(deco_outputs[:, i])
    
    # test as scalars
    for n in range(n_pts):
        output = func(radius[n], theta[n], phi[n])
        for i in range(3):
            assert output[i] == pytest.approx(deco_outputs[n, i])
    
    # test as a mixed of scalars and numpy arrays
    outputs = np.array(func(radius[0], theta, phi)).T
    for n, output in enumerate(outputs):
        assert output == pytest.approx(func(radius[0], theta[n], phi[n]))
    outputs = np.array(func(radius, theta[0], phi)).T
    for n, output in enumerate(outputs):
        assert output == pytest.approx(func(radius[n], theta[0], phi[n]))
    outputs = np.array(func(radius, theta, phi[0])).T
    for n, output in enumerate(outputs):
        assert output == pytest.approx(func(radius[n], theta[n], phi[0]))

def test_cartesian_to_spherical():
    func = geo.cartesian_to_spherical
    n_pts = 20
    x = np.random.uniform(-3, 3, size=n_pts)
    y = np.random.uniform(-3, 3, size=n_pts)
    z = np.random.uniform(-3, 3, size=n_pts)
    cart_pts = np.vstack([x, y, z]).T

    # test as numpy arrays
    outputs = func(x, y, z)
    deco_outputs = func(cart_pts)
    for i in range(3):
        assert outputs[i] == pytest.approx(deco_outputs[:, i])
    
    # test as scalars
    for n in range(n_pts):
        output = func(x[n], y[n], z[n])
        for i in range(3):
            assert output[i] == pytest.approx(deco_outputs[n, i])
    
    # test as a mixed of scalars and numpy arrays
    outputs = np.array(func(x[0], y, z)).T
    for n, output in enumerate(outputs):
        assert output == pytest.approx(func(x[0], y[n], z[n]))
    outputs = np.array(func(x, y[0], z)).T
    for n, output in enumerate(outputs):
        assert output == pytest.approx(func(x[n], y[0], z[n]))
    outputs = np.array(func(x, y, z[0])).T
    for n, output in enumerate(outputs):
        assert output == pytest.approx(func(x[n], y[n], z[0]))

def test_angle_between():
    func = geo.angle_between

    # always np.pi, never -np.pi
    assert func([1, 0], [-1, 0]) == pytest.approx(np.pi)
    assert func([-2, 0, 0], [2, 0, 0]) == pytest.approx(np.pi)
    assert func([3, 0], [-3, 0], directional=True) == pytest.approx(np.pi)
    assert func([-4, 0, 0], [4, 0, 0], directional=True) == pytest.approx(np.pi)

    # zero vector treatment
    with pytest.raises(RuntimeError) as err:
        func([0, 0], [0, 0], zero_vector='raise')
        func([1, 0], [0, 0], zero_vector='raise')
    with pytest.warns(RuntimeWarning) as warn:
        func([0, 0], [0, 0], zero_vector=None)
        func([1, 0], [0, 0], zero_vector=None)
    assert func([0, 0], [0, 0], zero_vector=99) == 99
    assert func([1, 0], [0, 0], zero_vector=-99) == -99

    # general cases in two-dimension
    io_triplets_2d = np.array([
        [
            [1.0, 0.0], [0.0, 1.0],     # inputs: v1, v2
            [0.5 * np.pi, 0.5 * np.pi], # outputs: angle, directional angle
        ],
        [
            [0.0, 1.0], [1.0, 0.0],
            [0.5 * np.pi, -0.5 * np.pi],
        ],
        [
            [1.0, 0.0], [-1.0, 0.0],
            [1.0 * np.pi, 1.0 * np.pi],
        ],
        [
            [-1.0, 0.0], [1.0, 0.0],
            [1.0 * np.pi, 1.0 * np.pi],
        ],
        [
            [1.0, 0.0], [1.0, 1.0],
            [0.25 * np.pi, 0.25 * np.pi],
        ],
        [
            [1.0, 0.0], [1.0, -1.0],
            [0.25 * np.pi, -0.25 * np.pi],
        ],
    ])

    v1, v2 = io_triplets_2d[:, 0], io_triplets_2d[:, 1]
    expected_angles = io_triplets_2d[:, 2, 0]
    expected_dir_angles = io_triplets_2d[:, 2, 1]
    assert func(v1, v2) == pytest.approx(expected_angles)
    assert func(v1, v2, directional=True) == pytest.approx(expected_dir_angles)

    # general cases in three-dimension
    io_triplets_3d = np.array([
        [
            [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], # inputs: v1, v2
            [0.5 * np.pi, 0, 0],              # outputs: angle, dummy, dummy
        ],
        [
            [0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
            [0.5 * np.pi, 0, 0],
        ],
        [
            [0.0, 0.0, 1.0], [-1.2, 3.4, 0.0],
            [0.5 * np.pi, 0, 0],
        ],
        [
            [0.0, 0.0, 1.0], [0.0, 0.0, 1.0],
            [0.0 * np.pi, 0, 0],
        ],
        [
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [1.0 * np.pi, 0, 0],
        ],
    ])

    v1, v2 = io_triplets_3d[:, 0], io_triplets_3d[:, 1]
    expected_angles = io_triplets_3d[:, 2, 0]
    assert func(v1, v2) == pytest.approx(expected_angles)
    assert func(v1, v2, directional=True) == pytest.approx(func(v1, v2))

@pytest.fixture
def sample_bar_vertices():
    """A sample bar vertices.

    Center at (0.9, 1.5, 2.1).
    Dimensions in x, y, z are 1.0, 2.0, 3.0, respectively.
    """
    return [
        [0.4, 0.5, 0.6],
        [1.4, 0.5, 0.6],
        [0.4, 2.5, 0.6],
        [0.4, 0.5, 3.6],
        [1.4, 2.5, 0.6],
        [1.4, 0.5, 3.6],
        [0.4, 2.5, 3.6],
        [1.4, 2.5, 3.6],
    ]

class TestRectangularBar:
    def test___init__(self, sample_bar_vertices):
        vertices = sample_bar_vertices

        bar = geo.RectangularBar(
            vertices,
            calculate_local_vertices=False,
            make_vertices_dict=False,
        )
        assert not isinstance(bar.vertices, dict)
        assert np.mean(bar.vertices, axis=0) == pytest.approx([0.9, 1.5, 2.1])
        assert bar.loc_vertices is None
        assert bar.pca.n_components_ == 3
        assert bar.pca.components_.T @ bar.pca.components_ == pytest.approx(np.eye(3))

        bar = geo.RectangularBar(
            vertices,
            calculate_local_vertices=True,
            make_vertices_dict=False,
        )
        assert not isinstance(bar.vertices, dict)
        assert isinstance(bar.loc_vertices, np.ndarray)
        assert bar.loc_vertices.shape == (8, 3)
        assert np.mean(bar.loc_vertices, axis=0) == pytest.approx([0.0, 0.0, 0.0])
    
    def test__get_vertices_dict(self):
        loc_vertices = [
            [-1.0, -2.0, -3.0],
            [+1.0, -2.0, -3.0],
            [-1.0, +2.0, -3.0],
            [-1.0, -2.0, +3.0],
            [+1.0, +2.0, -3.0],
            [+1.0, -2.0, +3.0],
            [-1.0, +2.0, +3.0],
            [+1.0, +2.0, +3.0],
        ]
        vertices = loc_vertices # values of vertices are irrelevant here
        loc_vertices, _ = geo.RectangularBar._get_vertices_dict(loc_vertices, vertices)
        assert set(loc_vertices.keys()) == set([
            (-1, -1, -1),
            (+1, -1, -1),
            (-1, +1, -1),
            (-1, -1, +1),
            (+1, +1, -1),
            (+1, -1, +1),
            (-1, +1, +1),
            (+1, +1, +1),
        ])
    
    def test__make_vertices_dict(self, sample_bar_vertices):
        bar = geo.RectangularBar(
            sample_bar_vertices,
            calculate_local_vertices=True,
            make_vertices_dict=False,
        )
        isinstance(bar.loc_vertices, np.ndarray)
        isinstance(bar.vertices, np.ndarray)

        bar._make_vertices_dict()
        isinstance(bar.loc_vertices, dict)
        isinstance(bar.vertices, dict)
    
    def test_dimension(self, sample_bar_vertices):
        bar = geo.RectangularBar(sample_bar_vertices)
        assert bar.dimension(index=0) == pytest.approx(3.0)
        assert bar.dimension(index=1) == pytest.approx(2.0)
        assert bar.dimension(index=2) == pytest.approx(1.0)
    
    def test__deco_numpy_ndim(self):
        f = lambda X: [np.square(x) for x in X]
        deco_f = geo.RectangularBar._deco_numpy_ndim(f)
        assert deco_f(np.array([[1, 2, 3], [4, 5, 6]])) == pytest.approx(
            np.array([[1, 4, 9], [16, 25, 36]])
        )
        assert deco_f(np.array([1, 2, 3])) == pytest.approx(np.array([1, 4, 9]))
    
    def test_to_local_coordinates(self, sample_bar_vertices):
        bar = geo.RectangularBar(sample_bar_vertices)
        func = bar.to_local_coordinates
        assert np.allclose(np.abs(func(np.array([0, 0, 0]))), [2.1, 1.5, 0.9])
        assert np.allclose(np.abs(func(np.array([[0, 0, 0]] * 5))), [[2.1, 1.5, 0.9]] * 5)
    
    def test_to_lab_coordinates(self, sample_bar_vertices):
        bar = geo.RectangularBar(sample_bar_vertices)
        func = bar.to_lab_coordinates
        assert np.allclose(np.abs(func(np.array([0, 0, 0]))), [0.9, 1.5, 2.1])
        assert np.allclose(np.abs(func(np.array([[0, 0, 0]] * 5))), [[0.9, 1.5, 2.1]] * 5)
    
    def test_is_inside(self, sample_bar_vertices):
        bar = geo.RectangularBar(sample_bar_vertices)
        assert bar.is_inside([0.9, 1.5, 2.1]) # lab frame
        assert bar.is_inside([0.0, 0.0, 0.0], frame='local')
        assert not bar.is_inside([1e9, 0.0, 0.0], frame='lab')
        assert not bar.is_inside([1e9, 0.0, 0.0], frame='local')
        assert bar.is_inside([0.5, 0.0, 0.0], frame='local')

        # testing the tolerance
        dim = bar.dimension()
        assert bar.is_inside([0.5 * dim[0], 0.0, 0.0], frame='local', tol=1e-3)
        assert bar.is_inside([0.5 * dim[0] + 0.9e-3, 0.0, 0.0], frame='local', tol=1e-3)
        assert not bar.is_inside([0.5 * dim[0] + 1.1e-3, 0.0, 0.0], frame='local', tol=1e-3)

        assert bar.is_inside([0.0, 0.5 * dim[1], 0.0], frame='local', tol=1e-3)
        assert bar.is_inside([0.0, 0.5 * dim[1] + 0.9e-3, 0.0], frame='local', tol=1e-3)
        assert not bar.is_inside([0.0, 0.5 * dim[1] + 1.1e-3, 0.0], frame='local', tol=1e-3)

        assert bar.is_inside([0.0, 0.0, 0.5 * dim[2]], frame='local', tol=1e-3)
        assert bar.is_inside([0.0, 0.0, 0.5 * dim[2] + 0.9e-3], frame='local', tol=1e-3)
        assert not bar.is_inside([0.0, 0.0, 0.5 * dim[2] + 1.1e-3], frame='local', tol=1e-3)
    
    def test_construct_plotly_mesh3d(self, sample_bar_vertices):
        bar = geo.RectangularBar(sample_bar_vertices)
        assert 'triangle_mesh' not in bar.__dict__
        bar.construct_plotly_mesh3d()
        assert isinstance(bar.triangle_mesh, rti.TriangleMesh)
        assert np.allclose(list(bar.vertices.values()), bar.triangle_mesh.vertices)

    def test_simple_simulation(self, sample_bar_vertices):
        bar = geo.RectangularBar(sample_bar_vertices)
        bar.construct_plotly_mesh3d()

        # smoke test
        assert 'simulation_result' not in bar.__dict__
        n_rays = 10
        bar.simple_simulation(
            n_rays=n_rays,
            random_seed=0,
        )
        assert 'simulation_result' in bar.__dict__
        sim = bar.simulation_result
        assert np.allclose(sim['origin'], [0, 0, 0])
        assert isinstance(sim['azimuth_range'], np.ndarray) and sim['azimuth_range'].shape == (2,)
        assert isinstance(sim['polar_range'], np.ndarray) and sim['polar_range'].shape == (2,)
        assert isinstance(sim['intersections'], np.ndarray)
        assert sim['intersections'].shape[0] == 12 # no. of triangles to form a bar
        assert sim['intersections'].shape[1] == n_rays
        assert sim['intersections'].shape[2] == 3 # (x, y, z) coordinates