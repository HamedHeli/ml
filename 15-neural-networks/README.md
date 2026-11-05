# 15 Neural networks, backpropagation and deep learning

**CPSC 340 L29, L30, L31 · CPSC 440 lectures 4–5 (neural features)** · [back to the graph](../README.md)

**What is learned:** the features *and* the classifier at once: a stack of learned non-linear transformations
`z_l = h(W_l z_{l−1})` followed by a linear (softmax) output layer, trained end to end by gradient descent.

## Lectures

| Slides | Covers |
| --- | --- |
| `L29 - Neural Networks.pdf` | history, motivation, one hidden layer, sigmoid non-linearity, universal approximation, biases, regression vs. classification outputs, biological motivation |
| `L30 - Deep Learning.pdf` | deep networks as pictures and functions, XOR, why multiple layers (hierarchies of parts), ImageNet history |
| `L31 - Deep Learning II.pdf` | automatic differentiation (reverse mode), backpropagation in vector form, vanishing gradients, ReLU and Swish, skip connections and ResNets, common tricks, missing theory; bonus: initialization, step sizes, momentum, early stopping, dropout |
| `4-cat-discrim.pdf`, `5-discrim.pdf` (CPSC 440) | learning nonlinear feature transformations with deep networks, fitting neural networks, convolutional networks and skip connections as feature engineering |

## The model

A network with `L` hidden layers computes `ŷ = V h(W_L h(… h(W_1 x)))`. With a softmax output and the cross-entropy
loss it is multi-class logistic regression on learned features `z_L`.

- **Why non-linearity**: without `h`, the composition of linear layers is one linear map.
- **Universal approximation**: one wide enough hidden layer can approximate any continuous function on a compact set;
  depth makes some functions exponentially cheaper to represent (hierarchies of parts).
- **Backpropagation** is reverse-mode automatic differentiation: a forward pass stores activations, a backward pass
  applies the chain rule from the loss to every weight, for about the cost of the forward pass. For softmax +
  cross-entropy the output error is `softmax(s) − onehot(y)`; each layer then passes back
  `δ_{l−1} = (δ_l W_lᵀ) ⊙ h'(a_{l−1})`.
- **Vanishing gradients**: the sigmoid's derivative is at most ¼, so gradients shrink geometrically with depth. ReLU
  (`max(0, a)`, derivative 1 when active) and **skip connections** `z_l = z_{l−1} + F(z_{l−1})` (ResNets) give gradients a
  short path.
- **Training is non-convex**: initialization (He / Glorot scaling), step size, momentum, mini-batch SGD, weight decay,
  early stopping and dropout are what make it work in practice, with little theory behind them ("babysitting").

## Code in this folder

[`mlp.py`](mlp.py): a multi-layer perceptron in NumPy with a hand-written backward pass (ReLU or sigmoid, optional
skip connections), a finite-difference gradient check, mini-batch SGD with heavy-ball momentum, L2 weight decay and
early stopping on a validation set.

```bash
python 15-neural-networks/mlp.py
```

| Result (8×8 digits; 1000 train / 300 validation / 497 test) | |
| --- | --- |
| backprop vs. finite differences, with and without skip connections | max relative error 2e-6 / 7e-7 |
| first-layer gradient norm at initialization, 2 / 6 / 12 hidden layers, **sigmoid** | 1.1e-1 / 5.1e-4 / **1.4e-6** |
| same, **ReLU** | 8.3e-1 / 5.9e-1 / 2.5e-1 |
| softmax regression (no hidden layer), test error | 0.034 |
| MLP 64-64-10, test error | 0.030 |
| MLP 64-64-64-10, test error | **0.022** |

With 12 sigmoid layers the first layer's gradient is five orders of magnitude smaller than with ReLU: those weights
would barely move.

## Knowledge gap

**What the slides leave out**

- **Normalization layers**: batch norm and layer norm (layer norm appears only inside Transformers in
  [17](../17-sequence-models-and-transformers/)); they matter as much as ReLU for trainability.
- **Adaptive optimizers**: Adam / AdamW, learning-rate warm-up and cosine schedules. Every JAX file in this repo uses
  Adam; the slides cover only SGD and momentum.
- **Initialization theory**: why variance-preserving (He / Glorot) scaling works, signal propagation, and the edge of
  chaos.
- **Generalization of over-parameterized networks**: they can fit random labels (Zhang et al. 2017) yet generalize on
  real ones; implicit regularization of SGD, flat minima, double descent ([00](../00-learning-fundamentals/)). The
  slides flag this as "missing theory."
- **Loss landscapes**: saddle points rather than bad local minima dominate in high dimension; mode connectivity.
- **Automatic differentiation in full**: forward vs. reverse mode costs, the Jacobian-vector / vector-Jacobian product
  view, checkpointing to trade memory for compute.
- **Regularization zoo**: label smoothing, data augmentation ([16](../16-convolutional-networks/)), mixup.

**Prerequisites to shore up**

- The multivariate chain rule in matrix form; Jacobians.
- SGD and step sizes ([06](../06-linear-classifiers-and-kernels/)).

**Open questions / further study**

- What do hidden units represent, and can we read them (mechanistic interpretability)?
- Reading: Goodfellow, Bengio & Courville, *Deep Learning* ch. 6–8; Prince, *Understanding Deep Learning* ch. 3–9
  (free online); Baydin et al. 2018, "Automatic differentiation in machine learning: a survey."

## Connections

- ← [06 Linear classifiers](../06-linear-classifiers-and-kernels/): the output layer is softmax regression; training is SGD.
- ← [04](../04-linear-regression/), [06](../06-linear-classifiers-and-kernels/): instead of choosing a basis or kernel,
  learn it.
- → [16 CNNs](../16-convolutional-networks/): replace dense layers with convolutions for images.
- → [12 VAEs](../12-variational-inference-and-vaes/), [17 Transformers](../17-sequence-models-and-transformers/),
  [18 Diffusion](../18-diffusion-and-image-generation/), [19 Representation learning](../19-representation-learning-and-llms/):
  every deep model in the course is built from these layers and trained this way.
