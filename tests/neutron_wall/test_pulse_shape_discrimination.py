import pytest

import numpy as np

from e15190.neutron_wall import pulse_shape_discrimination as psd

class TestFastTotalRansacEstimator:
    def test___init__(self):
        psd.FastTotalRansacEstimator()

    def test_smoke(self):
        estimator = psd.FastTotalRansacEstimator()
        estimator.model = lambda x, p0, p1: p0 + p1 * x
        x = np.linspace(0, 1, 100)
        p0_true, p1_true = 0.5, -2.2
        y = p0_true + p1_true * x
        y += np.random.normal(0, 0.05, len(y))
        estimator.fit(x[:, None], y)
        assert estimator.par[0] == pytest.approx(p0_true, rel=2e-2)
        assert estimator.par[1] == pytest.approx(p1_true, rel=2e-2)
        score = estimator.score(x[:, None], y)
        assert score <= 1
        assert score == pytest.approx(1, rel=2e-2)
    
class TestFastTotalRansacEstimatorGamma:
    def test___init__(self):
        psd.FastTotalRansacEstimatorGamma()
    
    def test_smoke(self):
        estimator = psd.FastTotalRansacEstimatorGamma()
        x = np.linspace(0, 1, 100)
        p0_true, p1_true = 0.5, -2.2
        y = p0_true + p1_true * x
        y += np.random.normal(0, 0.05, len(y))
        estimator.fit(x[:, None], y)
        assert estimator.par[0] == pytest.approx(p0_true, rel=2e-2)
        assert estimator.par[1] == pytest.approx(p1_true, rel=2e-2)
        score = estimator.score(x[:, None], y)
        assert score <= 1
        assert score == pytest.approx(1, rel=2e-2)

class TestFastTotalRansacEstimatorNeutron:
    def test_x_switch(self):
        x_switch = psd.FastTotalRansacEstimatorNeutron.x_switch
        assert x_switch > 500.0 and x_switch < 3000.0
    
    def test___init__(self):
        psd.FastTotalRansacEstimatorNeutron()
    
    def test_model(self):
        estimator = psd.FastTotalRansacEstimatorNeutron()
        a0, a1, a2 = 1.0, 2.0, 3.0
        x = np.array([500.0, 1000.0, 1500.0, 2000.0, estimator.x_switch])
        y = estimator.model(x, a0, a1, a2)
        for x_val, y_val in zip(x, y):
            if x_val < estimator.x_switch:
                assert y_val == pytest.approx(a0 + a1 * x_val, a2 * x_val**2)
            else:
                b1 = a1 + 2 * a2 * estimator.x_switch
                b0 = a0 + a1 * estimator.x_switch + a2 * estimator.x_switch**2 - b1 * estimator.x_switch
                assert y_val == pytest.approx(b0 + b1 * x_val)
        