"""Data leakage: three ways information the model should not have reaches training or validation.

1. Preprocessing leakage: selecting features on *all* the data, then cross-validating, on labels that are pure noise.
2. Target leakage: a feature that is only recorded *after* the outcome (here, "was given antibiotics" when predicting
   infection) looks wonderful offline and is useless at prediction time, where it is not yet known.
3. Temporal leakage: a random split of a drifting time series lets the model interpolate between neighbouring days
   instead of forecasting the future.

All models are plain NumPy (logistic regression by gradient descent, k-nearest-neighbour regression).
"""
import numpy as np


# ------------------------------------------------------------------ small models
def fit_logreg(X, y, lam=1e-2, iters=500, lr=0.5):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        w -= lr * (Xb.T @ (p - y) / len(y) + lam * np.r_[0, w[1:]])
    return w


def predict_logreg(w, X):
    return (np.c_[np.ones(len(X)), X] @ w > 0).astype(int)


def top_k_by_correlation(X, y, k):
    """Univariate filter: the k features most correlated with the label."""
    Xc, yc = X - X.mean(0), y - y.mean()
    corr = np.abs(Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)
    return np.argsort(corr)[-k:]


def kfold(n, k, rng):
    idx = rng.permutation(n)
    return np.array_split(idx, k)


# ------------------------------------------------------------------ 1. preprocessing leakage
def selection_leakage(n=100, d=5000, k=20, folds=5, seed=0):
    """CV accuracy on random labels when feature selection is done outside vs. inside the folds."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, n)                         # labels are independent of X: true accuracy is 50%
    leaky_feats = top_k_by_correlation(X, y, k)       # WRONG: looked at every label, including validation ones
    acc_leaky, acc_honest = [], []
    for val in kfold(n, folds, rng):
        tr = np.setdiff1d(np.arange(n), val)
        w = fit_logreg(X[tr][:, leaky_feats], y[tr])
        acc_leaky.append((predict_logreg(w, X[val][:, leaky_feats]) == y[val]).mean())
        feats = top_k_by_correlation(X[tr], y[tr], k)  # RIGHT: selection is part of training
        w = fit_logreg(X[tr][:, feats], y[tr])
        acc_honest.append((predict_logreg(w, X[val][:, feats]) == y[val]).mean())
    return float(np.mean(acc_leaky)), float(np.mean(acc_honest))


# ------------------------------------------------------------------ 2. target leakage
def make_hospital(n, rng):
    """Infection depends weakly on age and fever; antibiotics are given *because* of a diagnosed infection."""
    age = rng.normal(0, 1, n)
    fever = rng.normal(0, 1, n)
    infected = (rng.random(n) < 1 / (1 + np.exp(-(0.4 * age + 0.8 * fever - 0.5)))).astype(int)
    antibiotics = np.where(infected == 1, rng.random(n) < 0.9, rng.random(n) < 0.05).astype(float)
    return np.c_[age, fever], antibiotics, infected


def target_leakage(seed=0):
    rng = np.random.default_rng(seed)
    X_tr, ab_tr, y_tr = make_hospital(4000, rng)
    X_te, ab_te, y_te = make_hospital(4000, rng)
    w_leak = fit_logreg(np.c_[X_tr, ab_tr], y_tr)
    w_ok = fit_logreg(X_tr, y_tr)
    offline_leak = (predict_logreg(w_leak, np.c_[X_te, ab_te]) == y_te).mean()
    # At admission time nobody has received antibiotics yet: the feature is always 0 when the model is used.
    deployed_leak = (predict_logreg(w_leak, np.c_[X_te, np.zeros(len(X_te))]) == y_te).mean()
    honest = (predict_logreg(w_ok, X_te) == y_te).mean()
    majority = max(y_te.mean(), 1 - y_te.mean())
    return dict(offline_with_leak=float(offline_leak), deployed_with_leak=float(deployed_leak),
                honest=float(honest), majority=float(majority))


# ------------------------------------------------------------------ 3. temporal leakage
def make_series(T, rng):
    """Daily demand: a drifting random-walk level plus weekly seasonality plus noise."""
    level = np.cumsum(rng.normal(0, 1, T))
    return level + 3 * np.sin(2 * np.pi * np.arange(T) / 7) + rng.normal(0, 0.5, T)


def knn_regress(X_tr, y_tr, X_q, k=5):
    d = ((X_q[:, None, :] - X_tr[None, :, :]) ** 2).sum(-1)
    return y_tr[np.argsort(d, axis=1)[:, :k]].mean(1)


def temporal_leakage_once(T, horizon, rng):
    """A flexible model (5-NN on date and day-of-week) evaluated three ways.

    The data we hold are days [0, T); the model will be used on the next `horizon` days. A random split puts
    yesterday and tomorrow in the training set, so validation measures interpolation. A forward split (train on
    the past, validate on the last `horizon` days) measures what deployment will do.
    """
    s = make_series(T + horizon, rng)
    t = np.arange(T + horizon)
    X = np.c_[t / 7.0, 3 * np.sin(2 * np.pi * t / 7), 3 * np.cos(2 * np.pi * t / 7)]
    held, fut = np.arange(T), np.arange(T, T + horizon)
    va = rng.choice(T, horizon * 5, replace=False)
    tr = np.setdiff1d(held, va)
    random_val = np.mean((knn_regress(X[tr], s[tr], X[va]) - s[va]) ** 2)
    tr, va = held[: T - horizon], held[T - horizon:]
    time_val = np.mean((knn_regress(X[tr], s[tr], X[va]) - s[va]) ** 2)
    true_future = np.mean((knn_regress(X[held], s[held], X[fut]) - s[fut]) ** 2)
    return random_val, time_val, true_future


def temporal_leakage(T=1000, horizon=30, n_series=50, seed=0):
    """Mean MSE over many simulated series (one random walk is too noisy to compare)."""
    rng = np.random.default_rng(seed)
    r = np.mean([temporal_leakage_once(T, horizon, rng) for _ in range(n_series)], axis=0)
    return dict(random_split=float(r[0]), time_split=float(r[1]), future=float(r[2]))


def demo():
    leaky, honest = selection_leakage()
    print(f"1. noise labels, 5000 features, top-20 selection: CV accuracy {leaky:.3f} if selected on all data, "
          f"{honest:.3f} if selected inside each fold (truth: 0.5)")
    r = target_leakage()
    print("2. target leakage: offline {offline_with_leak:.3f}, deployed {deployed_with_leak:.3f}, "
          "honest model {honest:.3f}, majority class {majority:.3f}".format(**r))
    r = temporal_leakage()
    print("3. time series MSE: random-split validation {random_split:.2f}, forward-split validation "
          "{time_split:.2f}, actual future {future:.2f}".format(**r))


if __name__ == "__main__":
    demo()
