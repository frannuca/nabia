import math

import numpy as np
import pytest

from nabia_python_bridge import signal, stats


def test_descriptive_matches_numpy(rng):
    x = rng.normal(size=200)
    assert stats.mean(x) == pytest.approx(x.mean())
    assert stats.variance(x) == pytest.approx(x.var(ddof=1))
    assert stats.stddev(x, ddof=0) == pytest.approx(x.std())
    c = x - x.mean()
    assert stats.autocorrelation(x, 1) == pytest.approx((c[1:] * c[:-1]).sum() / (c * c).sum())
    with pytest.raises(ValueError):
        stats.mean(np.array([]))


def test_normal_is_vectorized():
    assert stats.normal_pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi))
    p = stats.normal_cdf(np.array([-1.0, 0.0, 1.0]))
    assert p.shape == (3,)
    assert stats.normal_quantile(p) == pytest.approx([-1.0, 0.0, 1.0])


def test_smoothing():
    x = np.arange(10.0)
    assert signal.moving_average(x, 3) == pytest.approx(np.convolve(x, np.ones(3) / 3, mode="valid"))
    assert signal.diff(x, 2) == pytest.approx(np.zeros(8))
    ema = signal.exponential_moving_average(x, 0.5)
    assert ema[1] == pytest.approx(0.5 * 1 + 0.5 * ema[0])


def test_ewma_streaming_equals_batch(rng):
    x = rng.normal(size=50)
    f = signal.EWMA(signal.EWMAParams(alpha=0.2))
    streamed = np.array([f.process(v) for v in x])
    assert f.apply(x) == pytest.approx(streamed)
    f.reset()
    assert f.process(x[0]) == pytest.approx(x[0])
    assert signal.EWMA.alpha_from_span(9) == pytest.approx(0.2)
    assert f.parameters.alpha == 0.2


def test_bandpass_removes_the_level():
    params = signal.ButterworthBandPassParams(order=2, sample_rate=1.0, low_cutoff=0.02, high_cutoff=0.2)
    y = signal.ButterworthBandPass(params).apply(np.full(2000, 5.0))
    assert abs(y[-1]) < 1e-6
    assert signal.ButterworthBandPass.max_order == 16
    with pytest.raises(ValueError):
        signal.ButterworthBandPass(order=2, low_cutoff=0.3, high_cutoff=0.2)


def test_params_reject_unknown_fields():
    with pytest.raises(AttributeError):
        signal.EWMAParams(alfa=0.1)
    assert "alpha=0.3" in repr(signal.EWMAParams(alpha=0.3))
