"""Linear regression, change of basis, gradient descent, robust regression and RBFs
(CPSC 340 L11-L14, L17).

Least squares
    f(w) = 1/2 ||Xw - y||^2,  grad f(w) = X^T (Xw - y).
    Setting the gradient to zero gives the normal equations X^T X w = X^T y.
    Solving them costs O(n d^2 + d^3).

Change of basis (non-linear regression)
    Replace X by Z = phi(X), e.g. polynomial features [1, x, x^2, ..., x^p], and do
    least squares on Z. The model is still linear in w.

Gradient descent
    w <- w - alpha grad f(w). Cost O(n d) per iteration, so it beats the normal
    equations when d is large. For a convex f, any stationary point is global.
    For least squares, alpha = 1/L with L = largest eigenvalue of X^T X converges.

Robust regression
    Squared error makes big residuals dominate, so outliers pull the fit. The L1 loss
    ||Xw - y||_1 is robust but non-smooth; the Huber loss is quadratic near 0 and
    linear in the tails, so it is smooth and robust, and gradient descent works.

Radial basis functions (RBFs)
    z_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)): one Gaussian bump per training point.
    A non-parametric basis: the number of features grows with n. Needs L2
    regularization (see 05) because there are as many features as examples.

Run:  python 04-linear-regression/linear_regression.py
"""

from __future__ import annotations

import numpy as np


def least_squares(Z, y):
    """Solve the normal equations Z^T Z w = Z^T y."""
    return np.linalg.solve(Z.T @ Z, Z.T @ y)


def poly_basis(x, p):
    return np.vander(np.ravel(x), p + 1, increasing=True)


def gradient_descent(grad, w0, alpha, max_iter=1000, tol=1e-8):
    w = w0.copy()
    for t in range(max_iter):
        g = grad(w)
        if np.linalg.norm(g, np.inf) < tol:
            break
        w = w - alpha * g
    return w, t + 1


def huber_grad(r, eps=1.0):
    """Derivative of the Huber loss h(r) = r^2/2 if |r| <= eps else eps(|r| - eps/2)."""
    return np.clip(r, -eps, eps)


def robust_regression(Z, y, eps=1.0, max_iter=5000):
    L = np.linalg.eigvalsh(Z.T @ Z).max()
    w0 = np.zeros(Z.shape[1])
    return gradient_descent(lambda w: Z.T @ huber_grad(Z @ w - y, eps), w0, 1 / L, max_iter)[0]


def rbf_basis(X, centers, sigma):
    d2 = (X ** 2).sum(1)[:, None] + (centers ** 2).sum(1)[None] - 2 * X @ centers.T
    return np.exp(-np.maximum(d2, 0) / (2 * sigma ** 2))


def fit_rbf(X, y, sigma, lam):
    Z = rbf_basis(X, X, sigma)
    return np.linalg.solve(Z.T @ Z + lam * np.eye(len(X)), Z.T @ y)


def demo():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 80)
    y = 1 + 0.5 * x + 0.3 * x ** 2 + 0.5 * rng.standard_normal(80)
    xt = np.linspace(-3, 3, 500)
    yt = 1 + 0.5 * xt + 0.3 * xt ** 2
    print("Change of basis on a quadratic: test MSE vs. the noiseless curve")
    for p in [1, 2, 5, 10]:
        w = least_squares(poly_basis(x, p), y)
        print(f"  degree {p:2d}: {np.mean((poly_basis(xt, p) @ w - yt) ** 2):.4f}")

    Z = poly_basis(x, 2)
    L = np.linalg.eigvalsh(Z.T @ Z).max()
    w_gd, iters = gradient_descent(lambda w: Z.T @ (Z @ w - y), np.zeros(3), 1 / L, 100000, 1e-6)
    print(f"Gradient descent (alpha = 1/L) reaches the normal-equation solution in {iters} iterations: "
          f"max |w_gd - w_ne| = {np.abs(w_gd - least_squares(Z, y)).max():.1e}")

    # robust regression: corrupt 10% of the labels
    xr = rng.uniform(0, 1, 100)
    yr = 2 * xr + 0.1 * rng.standard_normal(100)
    yr[:10] += 8
    Zr = poly_basis(xr, 1)
    print("Slope with 10% outliers (true slope 2):"
          f" least squares {least_squares(Zr, yr)[1]:.2f}, Huber {robust_regression(Zr, yr, 0.2)[1]:.2f}")

    # RBF regression on a wiggly function
    X1 = rng.uniform(-3, 3, (100, 1))
    y1 = np.sin(3 * X1[:, 0]) + 0.1 * rng.standard_normal(100)
    Xt1 = np.linspace(-3, 3, 400)[:, None]
    print("Gaussian RBF regression on sin(3x), test MSE:")
    for sigma, lam in [(0.05, 1e-3), (0.3, 1e-3), (3.0, 1e-3)]:
        w = fit_rbf(X1, y1, sigma, lam)
        pred = rbf_basis(Xt1, X1, sigma) @ w
        print(f"  sigma={sigma:4}: {np.mean((pred - np.sin(3 * Xt1[:, 0])) ** 2):.4f}")


if __name__ == "__main__":
    demo()
