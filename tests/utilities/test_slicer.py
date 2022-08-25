import pytest

import numpy as np

from e15190.utilities.slicer import *

class TestSlicer:
    def test_create_ranges(self):
        assert np.allclose(
            create_ranges(0, 4, 1),
            np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
            ])
        )

        assert np.allclose(
            create_ranges(low=0, upp=4, width=1, step=0.5),
            np.array([
                [0.0, 1.0],
                [0.5, 1.5],
                [1.0, 2.0],
                [1.5, 2.5],
                [2.0, 3.0],
                [2.5, 3.5],
                [3.0, 4.0],
            ])
        )

        assert np.allclose(
            create_ranges(low=0, upp=4, width=0.8, step=0.5),
            np.array([
                [0.0, 0.8],
                [0.5, 1.3],
                [1.0, 1.8],
                [1.5, 2.3],
                [2.0, 2.8],
                [2.5, 3.3],
                [3.0, 3.8],
            ])
        )

        assert np.allclose(
            create_ranges(low=0, upp=4, width=2, n_steps=3),
            np.array([
                [0, 2],
                [1, 3],
                [2, 4],
            ])
        )

        assert np.allclose(
            create_ranges(low=0, upp=4, width=2, n_steps=6),
            np.array([
                [0.0, 2.0],
                [0.4, 2.4],
                [0.8, 2.8],
                [1.2, 3.2],
                [1.6, 3.6],
                [2.0, 4.0],
            ])
        )

        assert np.allclose( # step is ignored when n_steps is specified
            create_ranges(low=0, upp=4, width=2, step=1.0, n_steps=6),
            np.array([
                [0.0, 2.0],
                [0.4, 2.4],
                [0.8, 2.8],
                [1.2, 3.2],
                [1.6, 3.6],
                [2.0, 4.0],
            ])
        )