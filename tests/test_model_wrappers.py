from __future__ import annotations

import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from rtt_predictor.model_wrappers import apply_sigmoid_calibrator


class SigmoidCalibratorCompatibilityTests(unittest.TestCase):
    def test_direct_binary_probability_matches_sklearn(self) -> None:
        raw = np.array([0.05, 0.25, 0.5, 0.8, 0.97])
        y = np.array([0, 0, 0, 1, 1])
        calibrator = LogisticRegression(max_iter=1000).fit(
            np.log(raw / (1.0 - raw)).reshape(-1, 1),
            y,
        )
        expected = calibrator.predict_proba(
            np.log(raw / (1.0 - raw)).reshape(-1, 1)
        )[:, 1]
        actual = apply_sigmoid_calibrator(raw, calibrator)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_does_not_call_version_specific_predict_proba(self) -> None:
        class SavedBinaryCalibrator:
            coef_ = np.array([[2.0]])
            intercept_ = np.array([-0.25])
            classes_ = np.array([0, 1])

            def predict_proba(self, _):
                raise AttributeError("multi_class")

        actual = apply_sigmoid_calibrator(np.array([0.25, 0.75]), SavedBinaryCalibrator())
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertLess(actual[0], actual[1])


if __name__ == "__main__":
    unittest.main()
