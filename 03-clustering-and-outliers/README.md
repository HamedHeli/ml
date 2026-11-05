# 03 Clustering and outlier detection

**CPSC 340 L8, L9, L10** · [back to the graph](../README.md)

**What is learned:** a partition of unlabelled data into groups (cluster means, density-connected regions, or a
hierarchy), and a score for how unusual each point is.

## Lectures

| Slides | Covers |
| --- | --- |
| `L8 - Clustering.pdf` | unsupervised learning, k-means, initialization, k-means++, k-medians / k-medoids (bonus), vector quantization |
| `L9 - More Clustering.pdf` | convexity of k-means clusters, DBSCAN, ensemble clustering and label switching, agglomerative clustering, linkages (bonus) |
| `L10 - Outlier.pdf` | biclustering, graph-based clustering, outlier detection: model-based (z-score), graphical, cluster-based, distance-based (global kNN, local outlierness), supervised |

## The models

### k-means

Minimize `Σ_i ‖x_i − w_{c_i}‖²` by alternating (1) assign each point to its nearest mean and (2) move each mean to
the average of its points. Neither step increases the objective, so it converges, but only to a **local** optimum:
run several random restarts and keep the lowest cost. **k-means++** seeds means by sampling points with
probability proportional to their squared distance to the nearest seed so far, which is provably within
`O(log k)` of optimal in expectation. Clusters are Voronoi cells, hence **convex**; k-means cannot find moons or
rings. Each assignment step costs `O(ndk)`. Vector quantization is k-means used for compression (replace a point by
its mean).

### DBSCAN (density-based)

A point with at least `minNeighbours` points within radius `ε` is a **core** point; clusters are the connected
components of core points plus their non-core neighbours; everything else is noise. It finds non-convex clusters
and does not need `k`, but `ε` is hard to set and a single `ε` fails when clusters have different densities.

### Agglomerative (hierarchical) clustering

Start with every point as its own cluster and repeatedly merge the closest pair. The linkage defines "closest":
**single** (closest members; follows chains, finds moons, but chains through noise), **complete** (farthest
members; compact clusters), **average**. The output is a dendrogram, a whole hierarchy of clusterings. Naively
`O(n³)`.

### Outlier detection

- **Model-based**: fit a model, flag low-probability points. The z-score is the simplest, but outliers inflate the
  mean and standard deviation, and it assumes one mode.
- **Distance-based, global**: average distance to the `k` nearest neighbours. Good for points far from everything.
- **Distance-based, local**: a point's kNN distance divided by its neighbours' kNN distances. Catches a point that is
  far *relative to* a dense neighbourhood, even if its raw distance is ordinary.
- **Supervised**: if you have labelled outliers, it is just classification (with heavy class imbalance).

## Code in this folder

| File | What it implements |
| --- | --- |
| [`clustering.py`](clustering.py) | k-means with random or k-means++ initialization and restarts, DBSCAN by breadth-first search, agglomerative clustering with single / complete / average linkage (Lance–Williams updates), adjusted Rand index |
| [`outliers.py`](outliers.py) | z-score, global kNN outlierness, local outlierness ratio |

```bash
python 03-clustering-and-outliers/clustering.py
python 03-clustering-and-outliers/outliers.py
```

Adjusted Rand index against the true groups (1 = perfect):

| method | 4 Gaussian blobs | 2 moons |
| --- | --- | --- |
| k-means++ (10 restarts) | **0.99** | 0.27 |
| DBSCAN | 0.68 | **1.00** |
| agglomerative, single link | 0.71 | **1.00** |
| agglomerative, complete link | **0.99** | 0.28 |
| agglomerative, average link | 0.97 | 0.33 |

Twenty single random restarts of k-means on the blobs end at costs from 734 to 1758: the local-optimum problem in one
number. Outliers, ranked out of 402 points (1 = most outlying): the z-score ranks the *local* outlier 385th, the
global kNN score 34th, and the local outlierness ratio 4th.

## Knowledge gap

**What the slides leave out**

- **k-means as a probabilistic model.** It is the zero-variance limit of EM for a Gaussian mixture with equal
  spherical covariances; that view (in [10](../10-mixture-models-and-em/)) explains its convex, equal-size bias and
  gives soft assignments and a likelihood for choosing `k`.
- **Choosing `k`**: elbow plots are mentioned; silhouette scores, the gap statistic, BIC on a mixture, and stability
  under resampling are not.
- **Spectral clustering** (graph Laplacian eigenvectors, then k-means) is the principled version of "graph-based
  clustering" and handles moons with a single global objective.
- **HDBSCAN / OPTICS** fix DBSCAN's single-density limitation (OPTICS is a bonus slide only).
- **Evaluation without labels** is genuinely ill-posed (Kleinberg's impossibility theorem for clustering): no
  clustering function can satisfy scale-invariance, richness and consistency at once.
- Isolation forests and one-class SVMs, the two outlier detectors most used in practice, and LOF (the standard
  form of the "local outlierness" idea).

**Prerequisites to shore up**

- Convex sets (why Voronoi cells are convex); graph connected components and BFS.

**Open questions / further study**

- Clustering in learned representation spaces (deep clustering, SwAV) rather than raw features.
- Reading: *ESL* ch. 14.3; von Luxburg 2007, "A tutorial on spectral clustering"; Arthur & Vassilvitskii 2007
  (k-means++).

## Connections

- ← [02 kNN](../02-naive-bayes-and-knn/): distance-based outlier scores reuse nearest-neighbour search.
- → [09 Latent-factor models](../09-latent-factor-models/): k-means is matrix factorization `X ≈ ZW` with each row of
  `Z` forced to be a one-hot vector; PCA relaxes that constraint.
- → [10 Mixtures and EM](../10-mixture-models-and-em/): k-means is hard-assignment EM; mean-shift is clustering by the
  modes of a kernel density estimate.
