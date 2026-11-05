# 04 Linear regression, gradient descent and robust regression

**CPSC 340 L11, L12, L13, L14, L17 (RBFs)** · [back to the graph](../README.md)

**What is learned:** a weight vector `w` such that `ŷ = wᵀφ(x)` predicts a real-valued target, where `φ` is either the
raw features or a fixed non-linear basis (polynomials, Gaussian bumps).

## Lectures

| Slides | Covers |
| --- | --- |
| `L11 - Linear Modeling.pdf` | least squares in 1-D and d-D, matrix notation |
| `L12 - Non-Linear Modeling.pdf` | the normal equations, change of basis, polynomial regression |
| `L13 - Gradient Descent.pdf` | gradient descent, convex functions and how to show convexity |
| `L14 - Robust Regression.pdf` | L1 loss, Huber loss, "brittle" L∞ regression, finding the "true" model |
| `L17 - Non-parametric.pdf` (first part) | radial basis functions, interpolation vs. extrapolation |

## The models

### Least squares

`f(w) = ½‖Xw − y‖²`, `∇f(w) = Xᵀ(Xw − y)`. Setting the gradient to zero gives the **normal equations**
`XᵀX w = Xᵀy`, solvable in `O(nd² + d³)`. The solution is unique iff `XᵀX` is invertible (no collinear features,
`n ≥ d`); otherwise there are infinitely many, which is one motivation for L2 regularization in
[05](../05-feature-selection-and-regularization/). Add a column of ones for the intercept.

### Change of basis

Replace `X` by `Z = φ(X)`, e.g. `[1, x, x², …, x^p]`, and run least squares on `Z`. The model is non-linear in `x` but
still **linear in `w`**, so everything above still applies. Degree `p` is the complexity knob of
[00](../00-learning-fundamentals/).

### Gradient descent and convexity

`w ← w − α∇f(w)`, `O(nd)` per step instead of `O(d³)` for a solve, so it wins when `d` is large. A function is
**convex** if every chord lies above it (equivalently, `∇²f ⪰ 0`); then every stationary point is a global minimum.
Sums of convex functions, convex functions of linear maps and norms are convex, which covers every loss in this
folder. For least squares, `α = 1/L` with `L = λ_max(XᵀX)` guarantees convergence.

### Robust regression

Squared error lets a few huge residuals dominate the fit. The **L1** loss `‖Xw − y‖₁` grows linearly and ignores
outliers more, but is non-differentiable at 0. The **Huber** loss is quadratic for `|r| ≤ ε` and linear beyond, so it
is both robust and smooth. (L∞ regression, minimizing the *largest* error, is the opposite: maximally sensitive to
outliers, "brittle.")

### Radial basis functions (RBFs)

`z_ij = exp(−‖x_i − x_j‖² / 2σ²)`: one Gaussian bump centred on each training point. The basis grows with `n`, so this
is a **non-parametric** regression; with as many features as examples it must be regularized. `σ` sets the
smoothness. RBF predictions decay to 0 away from the data: fine for interpolation, poor for extrapolation.

## Code in this folder

[`linear_regression.py`](linear_regression.py): normal equations, polynomial basis, gradient descent with
`α = 1/L`, Huber regression by gradient descent, Gaussian RBF regression with a small L2 penalty.

```bash
python 04-linear-regression/linear_regression.py
```

| Result (saved run) | |
| --- | --- |
| quadratic truth, polynomial degree 1 / 2 / 5 / 10, test MSE | 0.65 / **0.0032** / 0.0036 / 0.024 |
| gradient descent vs. normal equations | same `w` to 3e-8 after 720 iterations |
| slope with 10% outliers (truth 2) | least squares 2.36, Huber **2.09** |
| RBF regression of `sin 3x`, σ = 0.05 / 0.3 / 3 | test MSE 0.044 / **0.0034** / 0.41 |

## Knowledge gap

**What the slides leave out**

- **Numerics.** Never form `(XᵀX)⁻¹`; use a QR or Cholesky factorization (or `lstsq`, which uses the SVD).
  `XᵀX` squares the condition number, so the normal equations lose twice the digits.
- **The probabilistic reading.** Least squares is the MLE under Gaussian noise; L1 regression is the MLE under
  Laplace noise; Huber is between. This appears in L24 and [08](../08-gaussians-and-bayesian-learning/) but is the
  real reason the loss should match the noise.
- **Convergence rates.** Gradient descent on an `L`-smooth, `μ`-strongly convex function converges linearly at rate
  `(1 − μ/L)`, so ill-conditioning (`L/μ` large) means slow progress. Momentum / Nesterov improve this to
  `1 − sqrt(μ/L)`; Newton's method uses curvature directly. Line search (Armijo) is used in
  [06](../06-linear-classifiers-and-kernels/) but not derived.
- **Inference on `w`**: standard errors, confidence intervals and hypothesis tests on coefficients (the statistics
  view), and why they break under feature selection.
- Iteratively reweighted least squares, the standard way to solve L1 / Huber / logistic problems.
- RANSAC, the other common robust-fitting method (fit to random subsets, keep the consensus).

**Prerequisites to shore up**

- Matrix calculus: `∇(wᵀAw) = (A + Aᵀ)w`, `∇‖Xw − y‖² = 2Xᵀ(Xw − y)`.
- Eigenvalues of symmetric matrices, positive (semi-)definiteness.

**Open questions / further study**

- Why does gradient descent on an under-determined least-squares problem converge to the **minimum-norm** solution,
  and why does that implicit bias matter for over-parameterized models (double descent in
  [00](../00-learning-fundamentals/))?
- Reading: Boyd & Vandenberghe, *Convex Optimization* ch. 3 and 9; Trefethen & Bau, *Numerical Linear Algebra*,
  lectures 7–11.

## Connections

- → [05 Feature selection and regularization](../05-feature-selection-and-regularization/): RBFs and high-degree bases
  need `λ‖w‖²`; L1 penalties reuse the robust-loss intuition on the weights.
- → [06 Linear classifiers](../06-linear-classifiers-and-kernels/): the same linear model with classification losses;
  the kernel trick makes RBF-style bases infinite.
- → [08 Bayesian linear regression](../08-gaussians-and-bayesian-learning/): the same model with a posterior over `w`
  and error bars.
- → [01 Gradient boosting](../01-decision-trees-and-ensembles/): gradient descent where each step is a tree.
