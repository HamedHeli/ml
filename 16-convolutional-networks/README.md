# 16 Convolutions and convolutional neural networks

**CPSC 340 L32, L33 · CPSC 440 lectures 5, 14 (first half)** · [back to the graph](../README.md)

**What is learned:** small filters that are slid across an image (or sequence), stacked into a hierarchy: edges, then
parts, then objects. Transposed convolutions run the hierarchy in reverse to produce images or pixel-wise labels.

## Lectures

| Slides | Covers |
| --- | --- |
| `L32 - Convolutions.pdf` | brain-tumour segmentation motivation, local context, 1-D convolution, boundary handling, filters (Gaussian, derivatives, Laplacian of Gaussian), 2-D and 3-D convolution, FFT implementation (bonus) |
| `L33 - Convolutional Neural Networks.pdf` | multi-channel inputs, stride, translation invariance vs. equivariance, local connections and weight sharing, convolution as a linear map and im2col, 1×1 convolutions, CNNs, receptive fields, max pooling, LeNet, AlexNet, ImageNet |
| `5-discrim.pdf` (CPSC 440) | convolutional networks and skip connections as learned feature engineering |
| `14-deconv-reps.pdf` (first half, CPSC 440) | convolutions and transposed convolutions, pixel-level classification, fully-convolutional networks, up-sampling, semantic segmentation labels |

## The models

### Convolution

`z_i = Σ_j w_j x_{i+j}`: every output is the same weighted combination of a local window (strictly this is
cross-correlation, the convention in ML). It is a **linear map** with a banded, weight-shared matrix, so it is
**translation-equivariant**: shift the input and the output shifts by the same amount. Classic hand-made filters:
Gaussian blur, finite differences and Sobel (edges), Laplacian of Gaussian (blobs, a "Mexican hat"). Multi-channel
filters have shape `k × k × C_in × C_out`; **im2col** unfolds patches so the whole layer is one matrix product; a
**1×1 convolution** mixes channels per pixel.

### Convolutional neural networks

Stack convolution → non-linearity → pooling. Compared with a dense layer, a convolution layer has far fewer
parameters (local connections, weight sharing) and builds in equivariance. **Pooling** (max over a window) and stride
reduce resolution and add some invariance; the **receptive field** of a unit grows with depth. Finish with global
pooling or a dense layer. LeNet (1998) → AlexNet (2012, ReLU + GPUs + dropout) → VGG / ResNet.

### Transposed convolution and fully-convolutional networks

A stride-`s` convolution `z = Cx` maps a large grid to a small one. Its **transpose** `Cᵀ` maps small back to large;
it is the adjoint (`⟨Cx, z⟩ = ⟨x, Cᵀz⟩`) and is how decoders up-sample. A **fully-convolutional network**
(down-sampling encoder, transposed-convolution decoder, often with skip connections as in U-Net) predicts a label for
every pixel (semantic segmentation), and is the default decoder for image VAEs and diffusion models.

## Code in this folder

| File | What it implements |
| --- | --- |
| [`convolution.py`](convolution.py) | 1-D convolution, im2col and multi-channel 2-D convolution with stride, transposed convolution, Gaussian / Sobel / Laplacian-of-Gaussian filters, an adjoint check |
| [`cnn.py`](cnn.py) | a two-layer CNN (conv → pool → conv → global average pool → dense) in JAX vs. an MLP of similar size, with and without shift augmentation |

```bash
python 16-convolutional-networks/convolution.py
python 16-convolutional-networks/cnn.py          # about 20 s on a CPU
```

| Result (saved run) | |
| --- | --- |
| `[−1, 0, 1]` filter on a box signal | ±1 exactly at the two edges, 0 elsewhere |
| `⟨Cx, z⟩` vs. `⟨x, Cᵀz⟩` for a stride-2, 2→4-channel convolution | 2.877129 = 2.877129 |
| parameters of a 3×3 conv (2→4 channels) vs. a dense layer between the same tensors | 72 vs. 10,368 |

Digits, test error on the original test images and on the same images shifted by one pixel:

| model | parameters | test | shifted test |
| --- | --- | --- | --- |
| MLP 64-80-10 | 6010 | **0.017** | 0.868 |
| CNN | 5130 | 0.020 | 0.466 |
| MLP + shift augmentation | 6010 | 0.030 | 0.045 |
| CNN + shift augmentation | 5130 | 0.040 | **0.030** |

A one-pixel shift destroys the MLP (87% error) and badly hurts the CNN too: equivariant convolutions are not an
invariant classifier, and on 8×8 images a shift also pushes strokes off the border. Augmentation teaches the
invariance to both, and the CNN then generalizes best to shifted inputs.

## Knowledge gap

**What the slides leave out**

- **Backpropagation through a convolution** is a convolution with the flipped filter (and through a strided one, a
  transposed convolution); the slides show the forward pass only.
- **Modern architecture components**: batch normalization, residual blocks in CNNs (ResNet), depthwise-separable
  convolutions (MobileNet), dilated convolutions (only mentioned in L35), U-Net skip connections, attention-based vision
  models (ViT, [17](../17-sequence-models-and-transformers/)).
- **Checkerboard artifacts** from transposed convolutions and the resize-then-convolve alternative.
- **Equivariance more broadly**: group-equivariant CNNs (rotations), graph neural networks as convolutions on graphs.
- **Object detection** (anchors, region proposals, YOLO / DETR) and how segmentation is evaluated (IoU).
- **Transfer learning**: fine-tuning ImageNet-pretrained CNNs is how CNNs are used in most applications.
- The FFT implementation of convolution is a bonus slide; when it beats direct convolution (large filters) is not.

**Prerequisites to shore up**

- Indexing and strides of multi-dimensional arrays; linear maps as matrices (for the adjoint view).

**Open questions / further study**

- Are CNNs learning shape or texture? ImageNet CNNs are texture-biased (Geirhos et al. 2019), one reason for the
  non-robustness discussed in [19](../19-representation-learning-and-llms/).
- Reading: Dumoulin & Visin 2016, "A guide to convolution arithmetic for deep learning"; Prince, *Understanding Deep
  Learning* ch. 10–11; Long, Shelhamer & Darrell 2015 (FCN); Ronneberger et al. 2015 (U-Net).

## Connections

- ← [15 Neural networks](../15-neural-networks/): the same training machinery with structured, shared weights.
- → [12 VAEs](../12-variational-inference-and-vaes/) and [18 Diffusion](../18-diffusion-and-image-generation/): decoders and
  U-Net denoisers are fully-convolutional.
- ↔ [14 Graphical models](../14-graphical-models/): fully-convolutional nets plus CRFs for consistent segmentations.
- → [17 Transformers](../17-sequence-models-and-transformers/): dilated convolutions vs. attention for sequences; vision
  Transformers replace convolutions with patch attention.
