# ML Reference

A personal map of machine learning concepts — organized so you can quickly recall an idea, re-read how you first learned it, and verify the math with runnable code.

Each entry below lists what the notebook covers and links directly to it. GitHub renders notebooks with all outputs.

---

## Computational Methods

Tools for computing with models: how gradients flow, how solvers work, when memory matters.

| Topic | What's inside | Notebook |
|---|---|---|
| **Automatic Differentiation** | Forward mode (dual numbers), reverse mode (tape), backprop by hand on a 2-step ODE, forward vs reverse cost, what PyTorch does under the hood | [notebooks/computational_methods/automatic_differentiation.ipynb](notebooks/computational_methods/automatic_differentiation.ipynb) |
| **Adjoint Sensitivity** | The adjoint trick (store only the final state, rebuild backward), continuous adjoint equations, where dλ/dt = −λ·∂f/∂x comes from, gradient formula, 2-state example, neural ODE VJPs, backward instability, reversibility, training loop vocab, practical notes for neural ODE projects | [notebooks/computational_methods/adjoint_sensitivity.ipynb](notebooks/computational_methods/adjoint_sensitivity.ipynb) |

---

*More categories coming: Linear Models, Bayesian Methods, …*
