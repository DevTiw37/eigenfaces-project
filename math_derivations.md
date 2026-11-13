# Math Derivations

Setup: N = 400 images, each flattened to a vector x_i in R^4096.
Stack them as rows of X in R^(400 x 4096).

---

## 1. Centering

Mean face and centered data:

$$\mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad X_c = X - \mathbf{1}\mu^T$$

(The code does this with broadcasting: X - mu subtracts mu from every row.)

**Why center?** Variance and covariance measure spread *around the mean*. If we
skip centering, the biggest "direction" PCA finds is just the direction pointing at
the mean, which tells us nothing about how faces differ. After centering, every
column of X_c has mean 0 (checked in the notebook up to float rounding).

The covariance matrix of the pixels is

$$S = \frac{1}{N} X_c^T X_c \in \mathbb{R}^{4096 \times 4096}.$$

---

## 2. Why maximum variance means eigenvectors

Project every centered face onto a unit direction u. The variance of those
projections is

$$\frac{1}{N}\sum_i (x_{c,i}^T u)^2 = u^T S u.$$

We want the u that maximizes u^T S u subject to u^T u = 1. With a Lagrange
multiplier lambda:

$$\mathcal{L} = u^T S u - \lambda (u^T u - 1) \;\Rightarrow\; 2Su - 2\lambda u = 0 \;\Rightarrow\; S u = \lambda u.$$

So the best direction is an eigenvector of S, and the variance along it is

$$u^T S u = \lambda\, u^T u = \lambda.$$

The largest eigenvalue gives the first eigenface, the next gives the best direction
perpendicular to it, and so on. S is symmetric, so its eigenvectors are
orthonormal and S = U Lambda U^T (this is "diagonalizing the covariance").

---

## 3. Why the snapshot trick works

S is 4096 x 4096, which is large and slow. Instead build the small matrix

$$C = \frac{1}{N} X_c X_c^T \in \mathbb{R}^{400 \times 400}.$$

Let v be an eigenvector of C with eigenvalue lambda > 0:

$$X_c X_c^T v = N\lambda\, v.$$

Multiply both sides on the left by X_c^T:

$$X_c^T X_c \,(X_c^T v) = N\lambda\,(X_c^T v) \;\Rightarrow\; S\,(X_c^T v) = \lambda\,(X_c^T v).$$

So u = X_c^T v is an eigenvector of the big matrix S with the same eigenvalue.

**Normalizing.** The length of u is

$$\|X_c^T v\|^2 = v^T X_c X_c^T v = N\lambda \;\Rightarrow\; \|u\| = \sqrt{N\lambda}.$$

Dividing by this gives a unit-length eigenface. (The notebook checks that the raw
lengths equal sqrt(N * lambda).) Different u's are also perpendicular, because
u_i^T u_j = N lambda_j v_i^T v_j = 0 for i not equal to j.

**Why only 399 eigenfaces?** The rows of X_c sum to zero (we subtracted the mean),
so the rows are linearly dependent and rank(X_c) <= N - 1 = 399. The last
eigenvalue of C is therefore 0 (about 1e-10 in practice) and is dropped.

**Link to SVD.** Write X_c = A Sigma B^T. Then X_c X_c^T = A Sigma^2 A^T, so the
columns of A are the eigenvectors v of the snapshot matrix, lambda_i = sigma_i^2 / N,
and the columns of B are the eigenfaces: B = X_c^T A Sigma^{-1}. This is the same
formula as u = X_c^T v / sqrt(N lambda). SVD gives the answer directly, with no
covariance matrix to build or invert.

**Cost.** Eigendecomposition costs about n^3 operations: 400^3 versus 4096^3, which
is roughly 1,000 times less work. The 400 x 400 matrix also uses about 99 percent
less memory than the 4096 x 4096 one.

---

## 4. Projection, reconstruction and error

With U_k the first k eigenfaces (a 4096 x k matrix with orthonormal columns):

$$z = U_k^T (x - \mu), \qquad \hat{x} = U_k z + \mu.$$

For all faces at once: Z = X_c U_k and X_hat = Z U_k^T + mu.

Because the columns of U_k are orthonormal, the leftover x_c - U_k z is perpendicular
to face space, so by Pythagoras

$$\|x_c\|^2 = \|z\|^2 + \|x_c - U_k z\|^2.$$

Averaging over the dataset: the mean of ||x_c||^2 is the total variance
(the sum of all eigenvalues), and the mean of ||z||^2 is the sum of the first k
eigenvalues. Therefore

$$\text{MSE}_k = \frac{1}{4096}\sum_{i>k} \lambda_i, \qquad
\text{variance lost} = 1 - \frac{\sum_{i \le k}\lambda_i}{\sum_i \lambda_i}.$$

The notebook confirms both numerically. The explained-variance ratio of component
i is lambda_i divided by the sum of all eigenvalues, and its running total is the
scree plot. From the notebook: 90 percent of the variance needs k = 66, 95 percent
needs k = 123, and 99 percent needs k = 260.

---

## 5. Mahalanobis distance generalizes Euclidean distance

For one variable, the z-score says how many standard deviations away a value is:

$$\frac{x - \mu}{\sigma}.$$

For a vector z with mean mu_z and covariance Sigma_z:

$$d_M^2(z) = (z - \mu_z)^T \Sigma_z^{-1} (z - \mu_z).$$

- If Sigma_z = I (every direction has variance 1, no correlation), then
  d_M^2 = ||z - mu_z||^2, which is the squared Euclidean distance.
- If Sigma_z is diagonal, d_M^2 = sum of (z_i - mu_i)^2 / sigma_i^2, which is a
  z-score in each direction added up.
- In general, write Sigma_z = Q Lambda Q^T. Then
  d_M^2 = || Lambda^{-1/2} Q^T (z - mu_z) ||^2, which is plain Euclidean distance
  after rotating and rescaling ("whitening") the data so every direction has
  variance 1.

**In our project** the latent codes are Z = X_c U_k, so their covariance is
exactly diag(lambda_1, ..., lambda_k) (verified in the notebook). So

$$d_M^2(z) = \sum_{i=1}^{k} \frac{z_i^2}{\lambda_i}.$$

Example: moving 3 units along eigenface 1 (variance about 18.8) gives d^2 of about
0.5, but 3 units along eigenface 20 (variance about 0.6) gives about 15. Euclidean
distance calls both "3".

If the codes are roughly Gaussian, d^2 follows a chi-square distribution with k
degrees of freedom: its mean is k, and for k = 20 the 99th percentile is 37.57,
which is our Gate 1 threshold. On the data the model was fitted on, the average d^2
is exactly k, because trace(Sigma^{-1} Sigma) = k.

---

## 6. The two gates

- **Gate 1 (inside face space):** d_M^2 <= 37.57 means the k weights look like a
  normal face.
- **Gate 2 (outside face space):** leftover = (||x_c||^2 - ||z||^2) / 4096 is the
  reconstruction error. A real face rebuilds well, but noise has most of its energy
  outside face space.

A "Valid Face" must pass both. Smooth gradients can still pass because they look
like lighting patterns, which live inside face space. This is a known limit of the
method.