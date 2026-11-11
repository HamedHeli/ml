"""Online evaluation: A/B tests, the peeking problem, and off-policy evaluation from logs.

1. A/B test of a conversion rate: two-proportion z-test, and the sample size needed to detect a given lift.
2. Peeking: checking the p-value every day and stopping at the first p < 0.05 inflates the false-positive rate of an
   A/A test (two identical arms) far above 5%.
3. Off-policy evaluation: estimate how a *new* recommendation policy would do using only logs collected under the old
   one. Averaging the reward on the logged rows where the old policy happened to agree with the new one is biased;
   inverse propensity scoring (IPS) is unbiased when the old policy's action probabilities were logged.
"""
import numpy as np
from scipy.stats import norm


# ------------------------------------------------------------------ 1. A/B test
def two_proportion_z(conv_a, n_a, conv_b, n_b):
    """Two-sided z-test for p_b - p_a with a pooled variance. Returns (lift, z, p-value)."""
    pa, pb = conv_a / n_a, conv_b / n_b
    p = (conv_a + conv_b) / (n_a + n_b)
    se = np.sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se
    return pb - pa, z, 2 * norm.sf(abs(z))


def sample_size_per_arm(p_base, mde, alpha=0.05, power=0.8):
    """Users per arm to detect an absolute lift `mde` with the given power (normal approximation)."""
    p_alt = p_base + mde
    za, zb = norm.isf(alpha / 2), norm.isf(1 - power)
    var = p_base * (1 - p_base) + p_alt * (1 - p_alt)
    return int(np.ceil((za + zb) ** 2 * var / mde ** 2))


def ab_test(p_a=0.10, p_b=0.11, seed=0):
    rng = np.random.default_rng(seed)
    n = sample_size_per_arm(p_a, p_b - p_a)
    ca, cb = rng.binomial(n, p_a), rng.binomial(n, p_b)
    lift, z, p = two_proportion_z(ca, n, cb, n)
    se = np.sqrt(ca / n * (1 - ca / n) / n + cb / n * (1 - cb / n) / n)
    return dict(n_per_arm=n, lift=float(lift), ci=(float(lift - 1.96 * se), float(lift + 1.96 * se)), p=float(p))


def power_by_simulation(p_a=0.10, p_b=0.11, reps=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = sample_size_per_arm(p_a, p_b - p_a)
    ca, cb = rng.binomial(n, p_a, reps), rng.binomial(n, p_b, reps)
    return float(np.mean(two_proportion_z(ca, n, cb, n)[2] < 0.05))


# ------------------------------------------------------------------ 2. peeking
def peeking(p=0.10, users_per_day=1000, days=30, reps=4000, seed=0):
    """False-positive rate of an A/A test analysed once at the end vs. checked daily and stopped at p < 0.05."""
    rng = np.random.default_rng(seed)
    ca = rng.binomial(users_per_day, p, (reps, days)).cumsum(1)
    cb = rng.binomial(users_per_day, p, (reps, days)).cumsum(1)
    n = users_per_day * np.arange(1, days + 1)
    pvals = two_proportion_z(ca, n, cb, n)[2]
    return dict(fixed_horizon=float(np.mean(pvals[:, -1] < 0.05)),
                daily_peeking=float(np.mean((pvals < 0.05).any(1))))


# ------------------------------------------------------------------ 3. off-policy evaluation
def ope(n=200000, n_items=5, seed=0):
    """Users have a type (0 or 1); item preference depends on it. Logs come from an epsilon-greedy old policy."""
    rng = np.random.default_rng(seed)
    ctr = np.array([[0.10, 0.04, 0.03, 0.02, 0.06],     # click probability of each item, user type 0
                    [0.02, 0.03, 0.25, 0.05, 0.04]])    # user type 1
    user = (rng.random(n) < 0.3).astype(int)           # 30% of users are type 1

    def old_policy_probs(u):                           # mostly shows item 0, explores 20% of the time
        probs = np.full((len(u), n_items), 0.2 / n_items)
        probs[:, 0] += 0.8
        return probs

    def new_policy(u):                                 # personalised: item 2 for type-1 users
        return np.where(u == 1, 2, 0)

    probs = old_policy_probs(user)
    shown = (rng.random(n)[:, None] > probs.cumsum(1)).sum(1)
    click = (rng.random(n) < ctr[user, shown]).astype(float)
    propensity = probs[np.arange(n), shown]            # logged at serving time; this is what makes OPE possible

    target = new_policy(user)
    match = shown == target
    naive = click[match].mean()                        # "replay" without weights: over-represents type-0 users
    ips = np.mean(match * click / propensity)
    true_new = np.mean(ctr[user, target])
    return dict(old_policy=float(click.mean()), true_new=float(true_new), naive=float(naive), ips=float(ips))


def demo():
    r = ab_test()
    print(f"1. A/B test 10% vs 11%: {r['n_per_arm']} users per arm for 80% power; observed lift "
          f"{r['lift']:.4f}, 95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}], p = {r['p']:.4f}; "
          f"simulated power {power_by_simulation():.3f}")
    r = peeking()
    print(f"2. A/A test over 30 days: false positives {r['fixed_horizon']:.3f} analysed once, "
          f"{r['daily_peeking']:.3f} when peeking daily")
    r = ope()
    print(f"3. off-policy CTR of the new policy: truth {r['true_new']:.4f}, naive replay {r['naive']:.4f}, "
          f"IPS {r['ips']:.4f} (old policy logged {r['old_policy']:.4f})")


if __name__ == "__main__":
    demo()
