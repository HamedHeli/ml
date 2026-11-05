"""Training, validation, test and k-fold cross-validation (CPSC 340 L4-L5).

The fundamental trade-off: E_test = E_train + E_gap. A more complex model drives
E_train down and E_gap up. We cannot see E_test, so we estimate it on held-out
data. This file implements k-fold cross-validation by hand and uses it to
choose the depth of a polynomial fit and the k of a k-nearest-neighbour model.
It also shows *optimization bias*: the best of many validation scores is an
optimistic estimate of test error.

Run:  python 00-learning-fundamentals/cross_validation.py
"""

from __future__ import annotations

import numpy as np


def kfold_indices(n: int, k: int, rng: np.random.Generator):
    """Yield (train_idx, valid_idx) for k folds of a random permutation of range(n)."""
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    for i in range(k):
        valid = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, valid


def cross_val_error(fit, predict, loss, X, y, k=10, seed=0) -> float:
    """Average validation loss over k folds.

    fit(X, y) -> model, predict(model, X) -> yhat, loss(yhat, y) -> float.
    """
    rng = np.random.default_rng(seed)
    errs = []
    for tr, va in kfold_indices(len(y), k, rng):
        model = fit(X[tr], y[tr])
        errs.append(loss(predict(model, X[va]), y[va]))
    return float(np.mean(errs))


# --- two tiny models to cross-validate -------------------------------------------------

def poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x.ravel(), degree + 1, increasing=True)


def fit_poly(degree: int):
    def fit(X, y):
        Z = poly_features(X, degree)
        return np.linalg.lstsq(Z, y, rcond=None)[0]

    def predict(w, X):
        return poly_features(X, degree) @ w

    return fit, predict


def mse(yhat, y):
    return float(np.mean((yhat - y) ** 2))


def demo():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(-1, 1, n)
    y = np.sin(3 * x) + 0.3 * rng.standard_normal(n)
    x_test = rng.uniform(-1, 1, 5000)
    y_test = np.sin(3 * x_test) + 0.3 * rng.standard_normal(5000)

    print("Fundamental trade-off: polynomial degree vs. error (noise variance = 0.09)")
    print(f"{'degree':>6} {'E_train':>9} {'E_cv(10)':>9} {'E_test':>9}")
    rows = []
    for deg in range(0, 13):
        fit, pred = fit_poly(deg)
        w = fit(x, y)
        e_tr = mse(pred(w, x), y)
        e_cv = cross_val_error(fit, pred, mse, x, y, k=10)
        e_te = mse(pred(w, x_test), y_test)
        rows.append((deg, e_tr, e_cv, e_te))
        print(f"{deg:6d} {e_tr:9.4f} {e_cv:9.4f} {e_te:9.4f}")
    best = min(rows, key=lambda r: r[2])
    print(f"CV picks degree {best[0]} (test MSE {best[3]:.4f}).")

    # Optimization bias: try many random "models" and keep the best validation score.
    print("\nOptimization bias: 1000 random classifiers on random labels, 50 validation points")
    yv = rng.integers(0, 2, 50)
    accs = [(rng.integers(0, 2, 50) == yv).mean() for _ in range(1000)]
    print(f"best validation accuracy = {max(accs):.2f}, but every model is a coin flip (true accuracy 0.50)")
    return rows


if __name__ == "__main__":
    demo()
