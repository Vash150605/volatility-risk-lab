"""Tests for VaR and Expected Shortfall estimators."""

import numpy as np
from volrisk.var_models import (
    var_es_historical,
    var_es_normal,
    var_es_student_t,
    var_es_fhs,
)


def test_var_es_on_standard_normal():
    """On large iid N(0,1) sample, VaR_99 ~ 2.326 and ES_99 ~ 2.665."""
    rng = np.random.default_rng(0)
    w = rng.standard_normal(100_000)

    # Parametric normal
    var_n, es_n = var_es_normal(w, 0.99)
    np.testing.assert_allclose([var_n, es_n], [2.326, 2.665], atol=0.03)

    # Empirical historical
    var_h, es_h = var_es_historical(w, 0.99)
    np.testing.assert_allclose([var_h, es_h], [2.326, 2.665], atol=0.05)

    # Parametric Student-t
    var_t, es_t = var_es_student_t(w, 0.99)
    np.testing.assert_allclose([var_t, es_t], [2.326, 2.665], atol=0.08)


def test_fhs_es_exceeds_var():
    """Expected Shortfall must always be greater than or equal to VaR."""
    rng = np.random.default_rng(42)
    w = rng.standard_normal(500)
    var_val, es_val = var_es_fhs(w, 0.99, params=[0.0, 0.05, 0.05, 0.90])

    assert var_val > 0.0
    assert es_val >= var_val