# Paper Notes

## Paper 1: Eigenfaces for Recognition
Turk & Pentland (1991), Journal of Cognitive Neuroscience.

**What it is about:** a way to recognize faces by treating each photo as a point
in a huge space and finding the few "directions" along which faces really differ.

### 3 key takeaways
1. **An image is a point in a huge space.** A 64 x 64 photo is a list of 4,096
   numbers, so it is one point in R^4096. Faces share structure (two eyes, a nose,
   a mouth), so they bunch together in a tiny region called "face space" instead
   of filling the whole space.
2. **Eigenfaces are the main directions of face variation.** Take the covariance
   of all the (mean-subtracted) faces. Its eigenvectors are "eigenfaces". Any face
   is approximately the mean face plus a weighted mix of a few eigenfaces, so a
   face becomes a short list of weights.
3. **The snapshot trick makes it cheap.** The direct covariance is 4096 x 4096,
   but with only N = 400 photos at most N - 1 directions carry information. So
   we solve a small N x N problem and convert each answer back to image space with
   u = (X_c)^T v. The paper also uses distance from face space to reject images
   that are not faces, which is the idea behind our Gate 2.

## Paper 2: A Tutorial on Principal Component Analysis
Jonathon Shlens (2014), arXiv:1404.1100.

**What it is about:** the math behind PCA, explained from scratch.

### 3 key takeaways
1. **PCA is a change of viewpoint.** We look for a new set of perpendicular axes
   so the data is described with as little redundancy as possible. Directions with
   large variance are treated as signal, and directions with tiny variance as noise.
2. **Max variance = diagonalizing the covariance matrix.** Redundancy shows up as
   off-diagonal terms in the covariance matrix. The covariance matrix is symmetric,
   so it has perpendicular eigenvectors, and rotating into that basis makes it
   diagonal (no redundancy). The eigenvalue of each eigenvector is the variance
   along it.
3. **SVD gets the same answer directly.** The Singular Value Decomposition of the
   centered data gives the principal directions and their strengths without
   building or inverting a covariance matrix. Squared singular values are
   proportional to the eigenvalues.

The tutorial also lists limits: PCA is linear, assumes high variance means
importance, and works best when the data is roughly bell-shaped.

## The 3 core math principles in plain words
1. **Faces live in a small corner of pixel space**, so we can describe them with
   far fewer numbers than pixels.
2. **The best directions are the eigenvectors of the covariance matrix**, ordered
   by eigenvalue (how much variation each one carries).
3. **Work on the small side.** The N x N snapshot matrix has the same non-zero
   eigenvalues as the huge pixel-space covariance, and its eigenvectors map back
   to the true eigenfaces.

## How this project extends the papers
- Eigenfaces recognition in the original paper compares faces by plain distance.
  Here we add **Mahalanobis distance** in face space, which accounts for how much
  each direction normally varies.
- We add **reconstruction error** as a second gate, because Mahalanobis alone let
  random noise through (noise has a small, ordinary-looking projection).