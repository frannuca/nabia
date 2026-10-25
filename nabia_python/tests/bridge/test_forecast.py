import numpy as np
import pytest

from nabia_python_bridge import forecast


def test_ses():
    model = forecast.SimpleExponentialSmoothing(0.5)
    with pytest.raises(RuntimeError):
        model.forecast(1)
    model.fit(np.array([1.0, 2.0, 3.0]))
    f = model.forecast(4)
    assert f.horizon == 4
    assert np.all(f.mean == f.mean[0])
    assert np.all(f.lower(0.95) < f.mean) and np.all(f.upper(0.95) > f.mean)
    assert model.is_fitted and model.sse > 0


def test_kalman_local_level_tracks_a_constant(rng):
    y = 3.0 + 0.1 * rng.normal(size=200)
    y[50] = np.nan  # missing observation
    kf = forecast.KalmanFilter(forecast.KalmanFilterParams.local_level(1e-4, 0.01))
    kf.fit(y)
    assert kf.state_dim == 1
    assert kf.forecast(5).mean == pytest.approx(np.full(5, 3.0), abs=0.05)
    assert kf.filtered_states().shape == (1, 200)
    assert kf.smoothed_states().shape == (1, 200)
    assert len(kf.filtered_covariances()) == 200
    assert np.isnan(kf.innovations()[50])
    assert np.isfinite(kf.log_likelihood())


def test_kalman_custom_model():
    p = forecast.KalmanFilterParams(
        transition=np.eye(1), observation=np.ones((1, 1)), process_noise=np.eye(1) * 0.1,
        observation_noise=1.0, initial_state=np.zeros(1), initial_covariance=np.eye(1))
    kf = forecast.KalmanFilter(p)
    kf.fit(np.ones(10))
    assert kf.parameters.observation_noise == 1.0


def test_kernels_compose():
    se = forecast.SquaredExponentialKernel(2.0, 1.5)
    assert se(0.0, 0.0) == pytest.approx(2.0)
    per = forecast.PeriodicKernel(1.0, 1.0, 7.0)
    s = se + per
    prod = forecast.ProductKernel(se, per)
    assert isinstance(s, forecast.SumKernel) and isinstance(se * per, forecast.ProductKernel)
    assert s(0.3, 1.1) == pytest.approx(se(0.3, 1.1) + per(0.3, 1.1))
    assert prod(0.3, 1.1) == pytest.approx(se(0.3, 1.1) * per(0.3, 1.1))
    assert s.num_hyperparameters == 5
    t = np.linspace(0, 3, 4)
    assert s.matrix(t, t).shape == (4, 4)
    m = forecast.Matern52Kernel(1.0, 1.0)
    m.set_log_hyperparameters(np.log([4.0, 2.0]))
    assert m.variance == pytest.approx(4.0)


def test_gaussian_process():
    t = np.arange(40.0)
    y = np.sin(2 * np.pi * t / 10)
    params = forecast.GaussianProcessParams(kernel=forecast.PeriodicKernel(1.0, 1.0, 10.0), noise_variance=1e-3)
    gp = forecast.GaussianProcess(params)
    gp.fit(y)
    f = gp.forecast(10)
    assert f.mean == pytest.approx(np.sin(2 * np.pi * np.arange(40, 50) / 10), abs=0.1)
    assert isinstance(gp.kernel(), forecast.PeriodicKernel)
    assert gp.predict(np.array([0.5])).mean.shape == (1,)
    assert isinstance(gp.parameters.kernel, forecast.PeriodicKernel)
    with pytest.raises(ValueError):
        forecast.GaussianProcess(forecast.GaussianProcessParams())  # no kernel
