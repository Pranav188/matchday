import numpy as np
import pytest

from premier_league_predictor.evaluation import classifier_probabilities


class _Float32ProbabilityEstimator:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, features):
        return np.array([[0.7, 0.2, 0.1000001]], dtype=np.float32)


def test_classifier_probabilities_normalizes_float32_outputs():
    probabilities = classifier_probabilities(_Float32ProbabilityEstimator(), np.zeros((1, 1)))

    assert probabilities.sum(axis=1)[0] == pytest.approx(1.0, abs=1e-15)
