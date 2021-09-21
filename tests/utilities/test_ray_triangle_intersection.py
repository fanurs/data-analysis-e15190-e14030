import pytest

import numpy as np
from scipy import stats

from e15190.utilities import geometry as geom
from e15190.utilities import ray_triangle_intersection as rti

@pytest.fixture()
def synthetic_data():
    rng = np.random.default_rng()

    ray_origins = rng.uniform(low=-1, high=1, size=(10, 3))

    n_rays = 200
    ray_vectors = rng.uniform(low=-1, high=1, size=(n_rays, 3))
    triangles = rng.uniform(low=-1, high=1, size=(n_rays, 3, 3))

    return {
        'ray_origins': ray_origins,
        'ray_vectors': ray_vectors,
        'triangles': triangles,
    }

def test_consistency_between_modes(synthetic_data):
    for ray_origin in synthetic_data['ray_origins']:
        result_einsum = rti.moller_trumbore(
            ray_origin,
            synthetic_data['ray_vectors'],
            synthetic_data['triangles'],
            mode='einsum',
        )
        result_raw = rti.moller_trumbore(
            ray_origin,
            synthetic_data['ray_vectors'],
            synthetic_data['triangles'],
            mode='raw',
        )
        print(result_einsum)
        assert np.allclose(result_einsum, result_raw)

def test_emit_isotropic_rays():
    # check shape
    n_rays = np.random.randint(low=5, high=15)
    rays = rti.emit_isotropic_rays(n_rays)
    assert rays.shape == (n_rays, 3)

    # check random_seed
    cart_rays_s_ref = rti.emit_isotropic_rays(10, random_seed=0)
    cart_rays_s0 = rti.emit_isotropic_rays(10, random_seed=0)
    cart_rays_s1 = rti.emit_isotropic_rays(10, random_seed=1)
    assert np.allclose(cart_rays_s_ref, cart_rays_s0)
    assert not np.allclose(cart_rays_s_ref, cart_rays_s1)

    # the rest are all checks for isotropy
    n_rays = int(1e5) # must be sufficient to pass all statistical tests
    pvalue = 0.01
    uniform_cdf = lambda low, upp: stats.uniform(loc=low, scale=(upp - low)).cdf
    kstest = stats.kstest # alias

    # check isotropy in cartesian coordinates
    cart_rays = rti.emit_isotropic_rays(n_rays, frame='cartesian')
    for i in range(3):
        assert kstest(cart_rays[:, i], uniform_cdf(-1, 1)).pvalue > pvalue
    
    # check isotropy in spherical coordinates
    sphr_rays = rti.emit_isotropic_rays(n_rays, frame='spherical')
    assert np.allclose(sphr_rays[:, 0], 1.0)
    assert kstest(sphr_rays[:, 1], lambda x: 0.5 * (1 - np.cos(x))).pvalue > pvalue
    assert kstest(sphr_rays[:, 2], uniform_cdf(-np.pi, np.pi)).pvalue > pvalue

    # check isotropy in some subregion
    cart_rays = rti.emit_isotropic_rays(n_rays, polar_range=[0, np.pi / 3])
    sphr_rays = geom.cartesian_to_spherical(cart_rays)
    assert kstest(cart_rays[:, 0], uniform_cdf(-1, 1)).pvalue < pvalue # not uniform
    assert kstest(cart_rays[:, 1], uniform_cdf(-1, 1)).pvalue < pvalue # not uniform
    assert kstest(cart_rays[:, 2], uniform_cdf(0.5, 1)).pvalue > pvalue
    assert kstest(sphr_rays[:, 1], lambda x: 2 * (1 - np.cos(x))).pvalue > pvalue
    assert kstest(sphr_rays[:, 2], uniform_cdf(-np.pi, np.pi)).pvalue > pvalue

    cart_rays = rti.emit_isotropic_rays(n_rays, azimuth_range=[-0.5 * np.pi, 0.5 * np.pi])
    sphr_rays = geom.cartesian_to_spherical(cart_rays)
    assert kstest(cart_rays[:, 0], uniform_cdf(0, 1)).pvalue > pvalue
    assert kstest(cart_rays[:, 1], uniform_cdf(-1, 1)).pvalue > pvalue
    assert kstest(cart_rays[:, 2], uniform_cdf(-1, 1)).pvalue > pvalue
    assert kstest(sphr_rays[:, 1], lambda x: 0.5 * (1 - np.cos(x))).pvalue > pvalue
    assert kstest(sphr_rays[:, 2], uniform_cdf(-0.5 * np.pi, 0.5 * np.pi)).pvalue > pvalue
