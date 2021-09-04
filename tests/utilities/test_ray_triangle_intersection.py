import pytest

import numpy as np

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
