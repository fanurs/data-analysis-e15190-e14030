import pytest

import numpy as np

from e15190.neutron_wall import geometry

def test_spherical_to_cartesian():
    assert_ = lambda x, y: np.testing.assert_allclose(
        geometry.spherical_to_cartesian(*x),
        y,
        atol=1e-10,
    )

    assert_([0, 0, 0], [0, 0, 0])

    assert_([1, 0, 0.0 * np.pi], [0, 0, 1])
    assert_([1, 0, 0.5 * np.pi], [0, 0, 1])
    assert_([1, 0, 1.0 * np.pi], [0, 0, 1])
    assert_([1, 0, 1.5 * np.pi], [0, 0, 1])

    assert_([1, 0.5 * np.pi, 0.0], [1, 0, 0])
    assert_([1, 1.0 * np.pi, 0.0], [0, 0, -1])

    assert_([1, 0.5 * np.pi, -0.5 * np.pi], [0, -1, 0])
    assert_([1, 0.5 * np.pi, 0.5 * np.pi], [0, 1, 0])
    assert_([1, 0.5 * np.pi, 1.0 * np.pi], [-1, 0, 0])
    assert_([1, 0.5 * np.pi, 1.5 * np.pi], [0, -1, 0])

def test_cartesian_to_spherical():
    assert_ = lambda x, y: np.testing.assert_allclose(
        geometry.cartesian_to_spherical(*x),
        y,
        atol=1e-10,
    )

    assert_([0, 0, 0], [0, 0, 0])

    assert_([1, 0, 0], [1, 0.5 * np.pi, 0])
    assert_([0, 1, 0], [1, 0.5 * np.pi, 0.5 * np.pi])
    assert_([0, 0, 1], [1, 0, 0])
    assert_([-1, 0, 0], [1, 0.5 * np.pi, 1.0 * np.pi])
    assert_([0, -1, 0], [1, 0.5 * np.pi, -0.5 * np.pi])
    assert_([0, 0, -1], [1, 1.0 * np.pi, 0])

def test_angle_between_vectors():
    # directional = False
    func = lambda u, v: geometry.angle_between_vectors(u, v, directional=False)
    assert func([1, 0], [1, 0]) == pytest.approx(0.0)
    assert func([1, 0], [0, 1]) == pytest.approx(0.5 * np.pi)
    assert func([1, 0], [-1, 0]) == pytest.approx(1.0 * np.pi)
    assert func([1, 0], [0, -1]) == pytest.approx(0.5 * np.pi)

    # directional = True
    func = lambda u, v: geometry.angle_between_vectors(u, v, directional=True)
    assert func([1, 0], [1, 0]) == pytest.approx(0.0)
    assert func([1, 0], [0, 1]) == pytest.approx(0.5 * np.pi)
    assert func([1, 0], [-1, 0]) == pytest.approx(1.0 * np.pi)
    assert func([1, 0], [0, -1]) == pytest.approx(-0.5 * np.pi)
