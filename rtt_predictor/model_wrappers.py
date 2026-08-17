from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


EPS = 1e-6


def clip_prob(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), EPS, 1.0 - EPS)


def probability_logit(probabilities) -> np.ndarray:
    p = clip_prob(probabilities)
    return np.log(p / (1.0 - p))


def fit_sigmoid_calibrator(raw_probabilities, y_true) -> LogisticRegression:
    """Fit Platt/sigmoid calibration on one-dimensional model logits."""
    x = probability_logit(raw_probabilities).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(x, y)
    return calibrator


def apply_sigmoid_calibrator(raw_probabilities, calibrator: LogisticRegression) -> np.ndarray:
    x = probability_logit(raw_probabilities).reshape(-1, 1)
    # The project bundle may be opened by an older Jupyter kernel.  sklearn
    # 1.8 removed LogisticRegression.multi_class, while sklearn 1.4 expects
    # that attribute inside predict_proba.  A fitted binary Platt calibrator is
    # fully described by coef_ and intercept_, so calculate its positive-class
    # probability directly instead of depending on version-specific internals.
    coef = np.asarray(getattr(calibrator, "coef_", []), dtype=float)
    intercept = np.asarray(getattr(calibrator, "intercept_", []), dtype=float).reshape(-1)
    classes = np.asarray(getattr(calibrator, "classes_", []))
    if (
        coef.shape == (1, x.shape[1])
        and intercept.size == 1
        and classes.size == 2
        and np.array_equal(classes, np.array([0, 1]))
    ):
        scores = x @ coef[0] + intercept[0]
        calibrated = np.empty_like(scores, dtype=float)
        nonnegative = scores >= 0
        calibrated[nonnegative] = 1.0 / (1.0 + np.exp(-scores[nonnegative]))
        exp_scores = np.exp(scores[~nonnegative])
        calibrated[~nonnegative] = exp_scores / (1.0 + exp_scores)
        return clip_prob(calibrated)
    return clip_prob(calibrator.predict_proba(x)[:, 1])


class ProbabilityCalibratedModel:
    """
    Thin, joblib-friendly wrapper around a fitted binary classifier.

    The base model keeps its own predict_proba API. The wrapper applies a
    fitted probability calibrator to the positive-class probability and returns
    the usual two-column probability matrix.
    """

    def __init__(
        self,
        base_model,
        calibrator=None,
        calibration_method: str = "none",
        calibration_metadata: dict | None = None,
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.calibration_method = calibration_method
        self.calibration_metadata = dict(calibration_metadata or {})
        self.classes_ = getattr(base_model, "classes_", np.array([0, 1]))

    def predict_proba(self, x):
        raw = clip_prob(self.base_model.predict_proba(x)[:, 1])
        if self.calibration_method == "sigmoid" and self.calibrator is not None:
            calibrated = apply_sigmoid_calibrator(raw, self.calibrator)
        else:
            calibrated = raw
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def unwrap_base_model(model):
    return getattr(model, "base_model", model)
