import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function.

    sigmoid(z) = 1 / (1 + exp(-z))
    """

    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using NumPy.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

        self.weights = None
        self.bias = 0.0

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted probabilities.

        z = XW + b
        y_hat = sigmoid(z)
        """

        z = X @ self.weights + self.bias

        return sigmoid(z)

    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute binary cross-entropy loss.
        """

        epsilon = 1e-15

        y_pred = np.clip(
            y_pred,
            epsilon,
            1 - epsilon
        )

        loss = -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return loss
    
    def _compute_gradients(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.

        dW = (1/m) * X.T @ (y_pred - y_true)
        db = mean(y_pred - y_true)
        """

        m = X.shape[0]

        error = y_pred - y_true

        dw = (1 / m) * (X.T @ error)

        db = np.mean(error)

        return dw, db

    def _update_parameters(
        self,
        dw: np.ndarray,
        db: float
    ) -> None:
        """
        Update weights and bias using gradient descent.
        """

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Train logistic regression using gradient descent.
        """

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        self.loss_history = []

        for iteration in range(self.n_iterations):

            # Forward pass
            y_pred = self._forward(X)

            # Compute loss
            loss = self._compute_loss(y, y_pred)

            # Compute gradients
            dw, db = self._compute_gradients(
                X,
                y,
                y_pred
            )

            # Update parameters
            self._update_parameters(dw, db)

            # Store loss
            self.loss_history.append(loss)

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Return predicted probability of the positive class.
        """

        return self._forward(X)

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Convert probabilities into binary predictions.
        """

        probabilities = self.predict_proba(X)

        return (probabilities >= threshold).astype(int)    
    