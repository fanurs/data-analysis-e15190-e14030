import pytest

import numpy as np

from e15190.utilities import geometry as geo

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
