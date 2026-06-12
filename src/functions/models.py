from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

class ClippedLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression with predictions clipped at zero."""

    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClippedLinearRegression":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X)
        return np.maximum(y_pred, 0.0)


