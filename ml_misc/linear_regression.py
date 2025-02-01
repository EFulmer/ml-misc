"""Basic linear regression, optimized with gradient descent."""
from typing import Optional

import numpy as np

from ml_misc.random_data_gen import rand_data


def gradient_descent_mse(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: Optional[float]=0.1,
    max_iterations: Optional[int]=1_000,
    epsilon: Optional[float]=1e-6,
) -> np.ndarray:
    """Perform gradient descent, with a cost function J of mean squared
    error (MSE).

    Args:
        X: Input features.
        y: Ground truth / results.
        learning_rate: The learning rate, alpha, used in gradient
            descent.
        max_iterations: The maximum number of iterations to perform.
        epsilon: Threshold, under which, conversion is considered to
            have been achieved.

    Returns:
        Weights learned through gradient descent.
    """
    # Current assumption: one feature.
    # TODO: expand.
    m = X.shape[0]
    theta = np.zeros((2, 1))
    X_with_intercept = add_intercept(X)
    previous_cost = float("inf")

    for iteration in range(max_iterations):
        # Predict:
        h = X_with_intercept @ theta

        # Compute gradients:
        gradient_0 = (1 / m) * np.sum(h - y)
        gradient_1 = (1 / m) * (X_with_intercept[:, 1] @ (h - y))
        gradient = np.vstack((gradient_0, gradient_1))

        # Obviously the parens aren't needed, I just like them for
        # denoting the same precedence of operations that you would
        # also expect to see in a mathematics textbook.
        theta_new = theta - (learning_rate * gradient)

        # Compute MSE:
        cost = (1 / (2 * m)) * np.sum((h - y) ** 2)

        # Convergence checks:
        # 1. Cost-based:
        if abs(previous_cost - cost) < epsilon:
            print(
                f"Converged (delta J less than epsilon) after {iteration} iterations."
            )
            break

        # 2. Theta-based:
        if np.linalg.norm(theta_new - theta) < epsilon:
            print(
                f"Converged (theta change small) after {iteration} iterations."
            )
            break

        # Update variables for next iteration.
        theta = theta_new
        previous_cost = cost

    if iteration == max_iterations:
        print(f"Ran through max iterations.")

    return theta


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Adds a column of ones to X for the intercept term.

    Args:
        X: Input features.

    Returns:
        A new ndarray of shape (X.shape[0], X.shape[0]+1), where the
        first column is all ones, and the remaining columns are the
        columns of X.
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def main():
    """Basic demo/driver.
    Generates random data, prints the of a default hypothesis of zeros,
    computes GD, then prints the optimized theta.
    """
    # Generate data:
    shape = (10, 1)
    m = shape[0]
    X, y = rand_data(shape=shape)

    # Compute the "default" MSE:
    theta_default = np.zeros((2, 1))
    X_with_intercept = add_intercept(X)
    h_default = X_with_intercept @ theta_default
    mse_initial = (1 / (2 * m)) * np.sum((h_default - y) ** 2)

    print(f"Initial MSE (hypothesis is a vector of zeroes) = {mse_initial}")
    print("Computing gradient descent now.")
    theta_final = gradient_descent_mse(X=X, y=y)
    h_final = X_with_intercept @ theta_final
    mse_final = (1 / (2 * m)) * np.sum((h_final - y) ** 2)
    print(f"Final MSE (learned theta) {mse_final}")


if __name__ == "__main__":
    main()
