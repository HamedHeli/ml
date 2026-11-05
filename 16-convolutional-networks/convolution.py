"""Convolutions, filters, im2col and transposed convolutions (CPSC 340 L32-L33;
CPSC 440 lecture 14).

1D convolution (the ML convention, really cross-correlation):
    z_i = sum_{j=-m..m} w_j x_{i+j}
Each output is a weighted average of a local window, with the same weights at
every position (weight sharing). It is a linear map z = C x with a banded,
Toeplitz C, so it is translation-equivariant: shift x, and z shifts the same way.

Classic hand-designed filters:
    Gaussian blur, finite differences / Sobel (edges), Laplacian of Gaussian (blobs).
A CNN learns the filter weights instead.

im2col: unfold every k x k patch of the image into a row; then convolution with
many filters is one matrix product (how libraries implement it).

Stride s keeps every s-th output (down-sampling). A transposed convolution
multiplies by C^T instead of C: it maps the small output space back to the large
input space, which is how decoders and fully-convolutional segmentation nets
up-sample. It is the adjoint: <C x, z> = <x, C^T z>.

Run:  python 16-convolutional-networks/convolution.py
"""

from __future__ import annotations

import numpy as np


def conv1d(x, w):
    """'valid' cross-correlation, output length len(x) - len(w) + 1."""
    m = len(w)
    return np.array([x[i:i + m] @ w for i in range(len(x) - m + 1)])


def im2col(img, k, stride=1):
    """img (H, W, C) -> patches (H_out * W_out, k * k * C)."""
    H, W, C = img.shape
    Ho, Wo = (H - k) // stride + 1, (W - k) // stride + 1
    cols = np.empty((Ho * Wo, k * k * C))
    for i in range(Ho):
        for j in range(Wo):
            cols[i * Wo + j] = img[i * stride:i * stride + k, j * stride:j * stride + k].ravel()
    return cols, (Ho, Wo)


def conv2d(img, filters, stride=1):
    """img (H, W, C), filters (k, k, C, F) -> (H_out, W_out, F)."""
    k, _, C, F = filters.shape
    cols, (Ho, Wo) = im2col(img, k, stride)
    return (cols @ filters.reshape(k * k * C, F)).reshape(Ho, Wo, F)


def conv2d_transpose(z, filters, out_shape, stride=1):
    """Adjoint of conv2d: scatter each output's contribution back onto the input grid."""
    k, _, C, F = filters.shape
    H, W = out_shape
    img = np.zeros((H, W, C))
    Ho, Wo, _ = z.shape
    Wmat = filters.reshape(k * k * C, F)
    for i in range(Ho):
        for j in range(Wo):
            img[i * stride:i * stride + k, j * stride:j * stride + k] += (Wmat @ z[i, j]).reshape(k, k, C)
    return img


def gaussian_filter(k=5, sigma=1.0):
    a = np.arange(k) - k // 2
    g = np.exp(-(a[:, None] ** 2 + a[None] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def log_filter(k=7, sigma=1.0):
    a = np.arange(k) - k // 2
    r2 = a[:, None] ** 2 + a[None] ** 2
    f = (r2 - 2 * sigma ** 2) / sigma ** 4 * np.exp(-r2 / (2 * sigma ** 2))
    return f - f.mean()


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)


def demo():
    rng = np.random.default_rng(0)
    x = np.zeros(20)
    x[8:12] = 1
    print("1D: [-1, 0, 1] finite-difference filter on a box signal finds its edges:")
    print("  ", conv1d(x, np.array([-1, 0, 1])).astype(int))
    print("Translation equivariance: shift the input by 3, the output shifts by 3:",
          np.allclose(conv1d(np.roll(x, 3), np.ones(3))[3:], conv1d(x, np.ones(3))[:-3]))

    img = np.zeros((16, 16, 1))
    img[4:12, 4:12] = 1.0
    for name, f in [("Gaussian blur", gaussian_filter()), ("Sobel x", SOBEL_X), ("Laplacian of Gaussian", log_filter())]:
        out = conv2d(img, f[:, :, None, None])[..., 0]
        print(f"  {name:>22}: output {out.shape}, min {out.min():+.2f}, max {out.max():+.2f}, "
              f"response in the flat centre of the square {out[out.shape[0] // 2, out.shape[1] // 2]:+.2f}")

    filters = rng.standard_normal((3, 3, 2, 4))
    a = rng.standard_normal((9, 9, 2))
    z = rng.standard_normal((4, 4, 4))
    lhs = np.sum(conv2d(a, filters, stride=2) * z)
    rhs = np.sum(a * conv2d_transpose(z, filters, (9, 9), stride=2))
    print(f"Transposed convolution is the adjoint: <C x, z> = {lhs:.6f}, <x, C^T z> = {rhs:.6f}")
    print(f"  stride-2 conv maps 9x9x2 -> {conv2d(a, filters, 2).shape}; the transpose maps 4x4x4 -> "
          f"{conv2d_transpose(z, filters, (9, 9), 2).shape}")
    print(f"Parameters: 3x3 conv 2->4 channels has {filters.size} weights; a dense layer between the same "
          f"tensors would have {9 * 9 * 2 * 4 * 4 * 4}")


if __name__ == "__main__":
    demo()
