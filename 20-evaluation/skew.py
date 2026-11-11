"""Training-serving skew: the model is fine, but the features it sees in production differ from those it trained on.

A fraud model is trained on features from an offline (batch) pipeline and served with features from an online
pipeline. Each scenario re-implements one feature differently online, the way it happens in real systems:

- units:       the online service sends `amount` in cents, the batch table had dollars;
- imputation:  missing `velocity` is filled with the training mean offline and with 0 online;
- staleness:   `velocity` is read from a cache refreshed daily, so online it lags the value the label depended on.

Then two detectors: a feature-parity check (compute both pipelines on the same logged requests and compare) and the
population stability index (PSI) between training and serving feature distributions. PSI also fires on genuine
distribution shift, which is the world changing rather than a bug, so the last scenario shifts the population and
leaves the pipeline alone.
"""
import numpy as np

FEATURES = ["amount", "velocity", "account_age"]


def make_requests(n, rng, age_shift=0.0):
    amount = rng.lognormal(3.5, 1.0, n)                    # dollars
    velocity = rng.poisson(3, n).astype(float)             # transactions in the last hour
    account_age = rng.exponential(2.0, n) + age_shift      # years
    logit = 1.2 * (np.log(amount) - 3.5) + 0.7 * (velocity - 3) - 0.8 * account_age - 1.0
    fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    missing = rng.random(n) < 0.2                           # velocity is missing for 20% of requests
    return dict(amount=amount, velocity=velocity, account_age=account_age, missing=missing), fraud


# ------------------------------------------------------------------ the two pipelines
def offline_features(r, train_velocity_mean):
    v = np.where(r["missing"], train_velocity_mean, r["velocity"])
    return np.c_[np.log(r["amount"]), v, r["account_age"]]


def online_features(r, train_velocity_mean, bug=None, rng=None):
    amount = r["amount"] * (100 if bug == "units" else 1)
    v = r["velocity"].copy()
    if bug == "stale":                                      # yesterday's cached value: correlated, not equal
        v = np.where(rng.random(len(v)) < 0.5, v, rng.poisson(3, len(v)))
    v = np.where(r["missing"], 0.0 if bug == "imputation" else train_velocity_mean, v)
    return np.c_[np.log(amount), v, r["account_age"]]


# ------------------------------------------------------------------ model and metrics
def fit_logreg(X, y, iters=2000, lr=0.1):
    mu, sd = X.mean(0), X.std(0)
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        w -= lr * Z.T @ (1 / (1 + np.exp(-Z @ w)) - y) / len(y)
    return w, mu, sd


def predict_proba(model, X):
    w, mu, sd = model
    return 1 / (1 + np.exp(-np.c_[np.ones(len(X)), (X - mu) / sd] @ w))


def auc(y, s):
    """Probability that a random positive outranks a random negative (Mann-Whitney U)."""
    ranks = np.empty(len(s))
    ranks[np.argsort(s)] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def psi(expected, actual, bins=10):
    """Population stability index over quantile bins of the expected (training) distribution.

    Rule of thumb: < 0.1 stable, 0.1-0.25 investigate, > 0.25 significant shift.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(actual, edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def run(seed=0):
    rng = np.random.default_rng(seed)
    req_tr, y_tr = make_requests(20000, rng)
    vmean = req_tr["velocity"][~req_tr["missing"]].mean()
    X_tr = offline_features(req_tr, vmean)
    model = fit_logreg(X_tr, y_tr)

    rows = {}
    req, y = make_requests(20000, rng)
    for bug in [None, "units", "imputation", "stale"]:
        X_on = online_features(req, vmean, bug, rng)
        X_off = offline_features(req, vmean)
        p = predict_proba(model, X_on)
        parity = np.mean(np.any(~np.isclose(X_on, X_off), axis=1))   # share of requests whose features differ
        rows[bug or "no skew"] = dict(auc=auc(y, p), log_loss=log_loss(y, p), mean_score=float(p.mean()),
                                      fraud_rate=float(y.mean()), parity_mismatch=float(parity),
                                      psi=[psi(X_tr[:, j], X_on[:, j]) for j in range(3)])
    req, y = make_requests(20000, rng, age_shift=1.5)                 # real drift: older accounts, less fraud
    X_on = online_features(req, vmean)
    p = predict_proba(model, X_on)
    rows["population drift"] = dict(auc=auc(y, p), log_loss=log_loss(y, p), mean_score=float(p.mean()),
                                    fraud_rate=float(y.mean()), parity_mismatch=0.0,
                                    psi=[psi(X_tr[:, j], X_on[:, j]) for j in range(3)])
    return rows


def demo():
    rows = run()
    print(f"{'scenario':18s} {'AUC':>6s} {'logloss':>8s} {'mean score':>10s} {'fraud rate':>10s} "
          f"{'parity':>7s}  PSI ({', '.join(FEATURES)})")
    for name, r in rows.items():
        print(f"{name:18s} {r['auc']:6.3f} {r['log_loss']:8.3f} {r['mean_score']:10.3f} {r['fraud_rate']:10.3f} "
              f"{r['parity_mismatch']:7.2f}  " + ", ".join(f"{v:.2f}" for v in r["psi"]))


if __name__ == "__main__":
    demo()
