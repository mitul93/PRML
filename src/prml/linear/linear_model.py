import numpy as np
from numpy.typing import NDArray


class LinearModel:
    """Linear model trained using least squares."""

    def __init__(self) -> None:
        self.w: NDArray[np.float64] | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        t: NDArray[np.float64],
        lam: float = 0.0,
    ) -> LinearModel:
        """Fit linear model using L2 regularization.

        Solves min_w ||X w - t||^2 + lam * ||w[1:]||^2 via an augmented
        least-squares system, which avoids forming X.T @ X.

        Args:
            X: Design matrix, shape (N, D).
            t: Target values, shape (N,).
            lam: Regularization strength (lambda), must be >= 0.

        Returns:
            Fitted model.
        """
        if lam < 0:
            raise ValueError("lam must be non-negative.")

        D = X.shape[1]

        # sqrt(lam) * L, where L is the identity without the bias entry.
        reg = np.sqrt(lam) * np.eye(D)
        reg[0, 0] = 0.0  # Do not regularize bias w_0.

        # Augmented system: [X; sqrt(lam) L] w ≈ [t; 0].
        X_aug = np.vstack([X, reg])
        t_aug = np.concatenate([t, np.zeros(D)])

        self.w, *_ = np.linalg.lstsq(X_aug, t_aug, rcond=None)

        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict target values and calculate predictive variance.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            A tuple containing predictions and predictive variances.
        """
        if self.w is None:
            raise RuntimeError("Model must be fitted before prediction.")

        y = X @ self.w
        y_var = np.zeros_like(y)

        return y, y_var

    def params(self) -> NDArray[np.float64] | None:
        if self.w is None:
            raise RuntimeError("Model must be fitted before accessing parameters.")

        return self.w
