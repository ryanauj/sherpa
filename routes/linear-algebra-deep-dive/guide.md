---
title: Linear Algebra Deep Dive
route_map: /routes/linear-algebra-deep-dive/map.md
paired_sherpa: /routes/linear-algebra-deep-dive/sherpa.md
prerequisites:
  - Linear Algebra Essentials (vectors, matrices, transformations)
topics:
  - Vector Spaces
  - Eigenvalues
  - Eigenvectors
  - SVD
---

# Linear Algebra Deep Dive - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/linear-algebra-deep-dive/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/linear-algebra-deep-dive/map.md` for the high-level overview.

## Overview

This route is optional. You don't need eigenvalues or SVD to train a neural network or build a recommendation system -- the frameworks handle the math for you. But if you're the kind of developer who wants to understand *why* things work, not just *how* to call them, this is for you.

Eigenvalues, eigenvectors, and singular value decomposition show up throughout ML:

- **PCA** (dimensionality reduction) is eigendecomposition of the covariance matrix
- **Recommendation systems** use SVD to find latent factors (why Netflix suggested that movie)
- **Image compression** uses SVD to approximate an image with far fewer numbers
- **Stable Diffusion and generative models** operate in latent spaces that are built on these ideas
- **Understanding model internals** -- when researchers say a model "learned a direction" in activation space, they're talking about eigenvector-like concepts

If none of that interests you, skip this route and move on. If it does, let's go deeper.

## Learning Objectives

By the end of this route, you will be able to:
- Explain what a vector space is and identify whether a set of vectors forms a basis
- Compute eigenvalues and eigenvectors and explain their geometric meaning
- Decompose a matrix using eigendecomposition and verify the result
- Apply SVD to any matrix and reconstruct it from its components
- Compress an image using low-rank SVD approximation
- Connect these concepts to ML applications like PCA and recommendation systems

## Prerequisites

Before starting this route, you should be comfortable with:
- **Vectors**: creation, arithmetic, dot products, cosine similarity
- **Matrices**: multiplication, transpose, the column interpretation (columns = where basis vectors land)
- **Transformations**: scaling, rotation, shear, composing transformations
- **numpy**: `np.array`, `@` operator, `np.linalg.norm`

All of this is covered in [Linear Algebra Essentials](/routes/linear-algebra-essentials/guide.md). If any of it feels shaky, review that route first.

## Setup

You need the same tools as Linear Algebra Essentials: numpy and matplotlib. If you completed that route, you're already set.

```bash
pip install numpy matplotlib
```

**Verify your setup:**

```python
import numpy as np
import matplotlib.pyplot as plt

print(f"numpy version: {np.__version__}")

# Quick check: eigenvalue computation works
A = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues of [[2,1],[1,2]]: {eigenvalues}")
print(f"SVD available: {hasattr(np.linalg, 'svd')}")
```

**Expected output:**

```
numpy version: 1.26.4
Eigenvalues of [[2,1],[1,2]]: [3. 1.]
SVD available: True
```

Your version number may differ -- as long as the script runs and prints two eigenvalues, you're ready.

**Optional**: For the practice project, you'll compress a grayscale image. You can use any `.png` or `.jpg` file, or the guide will show you how to create a synthetic test image. If you want to use a real photo, have one ready.

---

## Section 1: Vector Spaces and Basis

### What Is a Vector Space?

In the essentials route, you worked with vectors as lists of numbers and arrows in 2D/3D space. A vector space is the formal name for "a set of vectors where addition and scalar multiplication work the way you'd expect."

More precisely, a vector space is any set where you can:
1. **Add** any two elements and get another element in the set
2. **Multiply** any element by a scalar and get another element in the set

The 2D plane (all possible `[x, y]` pairs) is a vector space. So is 3D space, or 100-dimensional space. The key property is that you never "leave the space" -- adding two vectors in the space always gives you another vector in the space, and scaling a vector always keeps you in the space.

Why does this matter for ML? Because when you work with word embeddings, image features, or model activations, you're working in high-dimensional vector spaces. Understanding the structure of these spaces -- their "shape" and "directions" -- is what eigenvalues and SVD reveal.

### Basis Vectors: The Minimum Spanning Set

You already know the standard basis vectors in 2D: `[1, 0]` and `[0, 1]`. Every 2D vector is a combination of these two:

```python
import numpy as np

# [3, 7] = 3 * [1, 0] + 7 * [0, 1]
e1 = np.array([1, 0])
e2 = np.array([0, 1])

v = 3 * e1 + 7 * e2
print(f"3 * {e1} + 7 * {e2} = {v}")
```

**Expected output:**

```
3 * [1 0] + 7 * [0 1] = [3 7]
```

But the standard basis isn't the *only* basis. Any two vectors that aren't parallel can serve as a basis for 2D space:

```python
import numpy as np

# A different basis for 2D space
b1 = np.array([1, 1])
b2 = np.array([1, -1])

# Express [3, 7] in this basis
# We need c1 and c2 such that c1 * b1 + c2 * b2 = [3, 7]
# c1 * [1,1] + c2 * [1,-1] = [3, 7]
# c1 + c2 = 3
# c1 - c2 = 7
# Solving: c1 = 5, c2 = -2

result = 5 * b1 + (-2) * b2
print(f"5 * {b1} + (-2) * {b2} = {result}")
print(f"Same vector, different coordinates: [3, 7] = 5*b1 + (-2)*b2")
```

**Expected output:**

```
5 * [1 1] + (-2) * [1 -1] = [3 7]
Same vector, different coordinates: [3, 7] = 5*b1 + (-2)*b2
```

The vector `[3, 7]` is still the same arrow in space. But its *coordinates* depend on which basis you use. In the standard basis, the coordinates are `(3, 7)`. In the `{b1, b2}` basis, the coordinates are `(5, -2)`.

### What Makes a Valid Basis?

A set of vectors forms a basis for a space if:
1. **They span the space**: you can reach any vector by combining them
2. **They're linearly independent**: none of them is a combination of the others

For 2D, that means exactly 2 non-parallel vectors. For 3D, exactly 3 vectors that don't all lie in the same plane.

The number of basis vectors equals the **dimension** of the space. 2D space needs 2 basis vectors. 100-dimensional embedding space needs 100 basis vectors.

### Changing Basis

You can convert coordinates between bases using a change-of-basis matrix. The columns of this matrix are the new basis vectors expressed in the old basis:

```python
import numpy as np

# New basis vectors (expressed in standard coordinates)
b1 = np.array([1, 1])
b2 = np.array([1, -1])

# Change-of-basis matrix: columns are the new basis vectors
P = np.column_stack([b1, b2])
print(f"Change-of-basis matrix P:\n{P}")

# Convert FROM new basis TO standard basis: multiply by P
coords_in_new_basis = np.array([5, -2])
standard_coords = P @ coords_in_new_basis
print(f"\nCoordinates in new basis: {coords_in_new_basis}")
print(f"Standard coordinates: {standard_coords}")

# Convert FROM standard basis TO new basis: multiply by P^(-1)
P_inv = np.linalg.inv(P)
v = np.array([3, 7])
new_coords = P_inv @ v
print(f"\nStandard coordinates: {v}")
print(f"Coordinates in new basis: {new_coords}")
```

**Expected output:**

```
Change-of-basis matrix P:
[[ 1  1]
 [ 1 -1]]

Coordinates in new basis: [ 5 -2]
Standard coordinates: [3 7]

Standard coordinates: [3 7]
Coordinates in new basis: [ 5. -2.]
```

This is the same idea as converting between Celsius and Fahrenheit, or between RGB and HSL color spaces. Same underlying quantity, different coordinate systems. In ML, PCA is essentially finding a new basis where the coordinates are more useful (they capture the most variance in your data).

### Exercise 1.1: Verify a Basis

**Task:** Determine whether each of the following sets of vectors forms a valid basis for 2D space. Explain your reasoning, then verify with numpy.

```python
set_A = [np.array([2, 0]), np.array([0, 3])]
set_B = [np.array([1, 2]), np.array([2, 4])]
set_C = [np.array([1, 1]), np.array([-1, 1])]
```

**Hints:**

<details>
<summary>Hint 1: What to check</summary>

Two vectors form a basis for 2D if they're linearly independent -- neither is a scalar multiple of the other. If one is just the other scaled up or down, they're parallel and can only reach points along a single line, not the whole plane.
</details>

<details>
<summary>Hint 2: Using numpy to check independence</summary>

You can check if two vectors are linearly independent by putting them as columns of a matrix and computing the determinant. If the determinant is zero, they're dependent (not a valid basis). If non-zero, they're independent (valid basis).

```python
M = np.column_stack([v1, v2])
det = np.linalg.det(M)
```
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

set_A = [np.array([2, 0]), np.array([0, 3])]
set_B = [np.array([1, 2]), np.array([2, 4])]
set_C = [np.array([1, 1]), np.array([-1, 1])]

for name, vectors in [("A", set_A), ("B", set_B), ("C", set_C)]:
    M = np.column_stack(vectors)
    det = np.linalg.det(M)
    is_basis = abs(det) > 1e-10
    print(f"Set {name}: det = {det:.1f}, valid basis = {is_basis}")
```

**Expected output:**

```
Set A: det = 6.0, valid basis = True
Set B: det = 0.0, valid basis = False
Set C: det = 2.0, valid basis = True
```

**Explanation:**
- **Set A**: `[2, 0]` and `[0, 3]` point along different axes. They span the whole plane. Valid basis.
- **Set B**: `[2, 4]` is just `2 * [1, 2]`. They're parallel -- they can only reach points along one line. Not a basis.
- **Set C**: `[1, 1]` and `[-1, 1]` point in different directions (one is upper-right, the other upper-left). They span the plane. Valid basis.
</details>

### Exercise 1.2: Express a Vector in a Different Basis

**Task:** Given the basis `b1 = [2, 1]` and `b2 = [0, 1]`, express the vector `v = [6, 5]` in this basis. That is, find scalars `c1` and `c2` such that `c1 * b1 + c2 * b2 = v`.

<details>
<summary>Hint 1: Setting up the equation</summary>

You need `c1 * [2, 1] + c2 * [0, 1] = [6, 5]`. This gives you two equations:
- `2*c1 + 0*c2 = 6`
- `1*c1 + 1*c2 = 5`

Solve the first equation for `c1`, then substitute into the second.
</details>

<details>
<summary>Hint 2: Using numpy</summary>

Form the change-of-basis matrix `P` with `b1` and `b2` as columns, then compute `P_inv @ v` to get the coordinates in the new basis.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

b1 = np.array([2, 1])
b2 = np.array([0, 1])
v = np.array([6, 5])

# Method 1: Solve by hand
# 2*c1 = 6  =>  c1 = 3
# c1 + c2 = 5  =>  3 + c2 = 5  =>  c2 = 2

# Method 2: Using numpy
P = np.column_stack([b1, b2])
coords = np.linalg.inv(P) @ v
print(f"Coordinates in new basis: c1 = {coords[0]}, c2 = {coords[1]}")

# Verify: reconstruct v from these coordinates
reconstructed = coords[0] * b1 + coords[1] * b2
print(f"Verification: {coords[0]} * {b1} + {coords[1]} * {b2} = {reconstructed}")
print(f"Matches original: {np.allclose(reconstructed, v)}")
```

**Expected output:**

```
Coordinates in new basis: c1 = 3.0, c2 = 2.0
Verification: 3.0 * [2 1] + 2.0 * [0 1] = [6. 5.]
Matches original: True
```

**Explanation:** The vector `[6, 5]` has coordinates `(3, 2)` in the `{b1, b2}` basis. This means it's 3 copies of `b1` plus 2 copies of `b2`. The arrow in space hasn't moved -- only the way we describe it has changed.
</details>

### Checkpoint 1

Before moving on, make sure you can:
- [ ] Explain what a vector space is in your own words
- [ ] Determine whether a set of vectors forms a valid basis (linear independence check)
- [ ] Convert a vector's coordinates from one basis to another using a change-of-basis matrix
- [ ] Explain why PCA is related to choosing a new basis

---

## Section 2: Eigenvalues and Eigenvectors

### The Key Question

When you multiply a vector by a matrix, the result is usually a completely different vector -- different direction, different length. The matrix rotates it, stretches it, skews it, or some combination.

But for certain special vectors, the matrix only changes the *length*, not the *direction*. These are **eigenvectors**.

```python
import numpy as np

A = np.array([
    [3, 1],
    [0, 2]
])

# Most vectors change direction when transformed
v1 = np.array([1, 1])
print(f"A @ {v1} = {A @ v1}")
print(f"  Original direction: {v1 / np.linalg.norm(v1)}")
print(f"  Result direction:   {(A @ v1) / np.linalg.norm(A @ v1)}")
print(f"  Direction changed: {not np.allclose(v1 / np.linalg.norm(v1), (A @ v1) / np.linalg.norm(A @ v1))}")

print()

# But THIS vector only gets scaled, not rotated
v2 = np.array([1, 0])
print(f"A @ {v2} = {A @ v2}")
print(f"  Original direction: {v2 / np.linalg.norm(v2)}")
print(f"  Result direction:   {(A @ v2) / np.linalg.norm(A @ v2)}")
print(f"  Direction changed: {not np.allclose(v2 / np.linalg.norm(v2), (A @ v2) / np.linalg.norm(A @ v2))}")
```

**Expected output:**

```
A @ [1 1] = [4 2]
  Original direction: [0.70710678 0.70710678]
  Result direction:   [0.89442719 0.4472136 ]
  Direction changed: True

A @ [1 0] = [3 0]
  Original direction: [1. 0.]
  Result direction:   [1. 0.]
  Direction changed: False
```

The vector `[1, 0]` went in pointing right and came out still pointing right, just 3 times longer. It's an eigenvector of `A` with eigenvalue 3.

### The Formal Definition: A*v = lambda*v

An eigenvector **v** of matrix **A** is a non-zero vector that satisfies:

```
A @ v = lambda * v
```

where `lambda` (the Greek letter lambda) is the **eigenvalue** -- the factor by which the eigenvector gets scaled.

Read this equation as: "applying the matrix A to v produces the same result as simply multiplying v by the scalar lambda." The matrix's entire effect on this particular vector is just stretching (or shrinking, or flipping).

```python
import numpy as np

A = np.array([
    [3, 1],
    [0, 2]
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors (as columns):")
print(eigenvectors)

# Verify: A @ v = lambda * v for each eigenvalue/eigenvector pair
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]  # i-th column is the i-th eigenvector

    Av = A @ v
    lam_v = lam * v

    print(f"\nEigenvector {i+1}: {np.round(v, 4)}, eigenvalue: {lam}")
    print(f"  A @ v     = {np.round(Av, 4)}")
    print(f"  lambda * v = {np.round(lam_v, 4)}")
    print(f"  Equal: {np.allclose(Av, lam_v)}")
```

**Expected output:**

```
Eigenvalues: [3. 2.]
Eigenvectors (as columns):
[[ 1.         -0.70710678]
 [ 0.          0.70710678]]

Eigenvector 1: [1. 0.], eigenvalue: 3.0
  A @ v     = [3. 0.]
  lambda * v = [3. 0.]
  Equal: True

Eigenvector 2: [-0.7071  0.7071], eigenvalue: 2.0
  A @ v     = [-1.4142  1.4142]
  lambda * v = [-1.4142  1.4142]
  Equal: True
```

Note: numpy returns eigenvectors as **columns** of the eigenvector matrix, and it normalizes them to unit length. The direction is what matters, not the length -- `[1, 0]` and `[2, 0]` are the same eigenvector (one is just a scaled version of the other).

### Geometric Meaning: Natural Axes of a Transformation

Eigenvectors reveal the "natural axes" of a transformation -- the directions along which the matrix simply stretches or compresses, without any rotation or shear.

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [2, 1],
    [1, 2]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

# Create a circle of unit vectors
theta = np.linspace(0, 2 * np.pi, 100)
circle = np.array([np.cos(theta), np.sin(theta)])

# Transform the circle
transformed = A @ circle

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original circle with eigenvectors
ax1.plot(circle[0], circle[1], 'b-', alpha=0.5)
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    ax1.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(v[0]*1.2, v[1]*1.2, f'v{i+1}', color='red', fontsize=12)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Original (circle + eigenvectors)')

# Transformed ellipse with transformed eigenvectors
ax2.plot(transformed[0], transformed[1], 'b-', alpha=0.5)
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    tv = A @ v  # = eigenvalue * v
    ax2.annotate('', xy=tv, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(tv[0]*1.1, tv[1]*1.1,
             f'v{i+1} (x{eigenvalues[i]:.0f})', color='red', fontsize=10)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('Transformed (ellipse + stretched eigenvectors)')

plt.tight_layout()
plt.savefig('eigen_visualization.png', dpi=100)
plt.show()

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvector 1: {np.round(eigenvectors[:, 0], 4)}")
print(f"Eigenvector 2: {np.round(eigenvectors[:, 1], 4)}")
```

**Expected output:**

```
Eigenvalues: [3. 1.]
Eigenvector 1: [0.7071 0.7071]
Eigenvector 2: [-0.7071  0.7071]
```

The matrix `[[2,1],[1,2]]` turns a circle into an ellipse. The eigenvectors point along the axes of the ellipse -- the directions of maximum and minimum stretching. The eigenvalues (3 and 1) are the stretch factors along those axes.

### Two Illustrative Examples

**A scaling matrix has eigenvectors along the coordinate axes:**

```python
import numpy as np

S = np.array([
    [3, 0],
    [0, 2]
])

eigenvalues, eigenvectors = np.linalg.eig(S)
print(f"Scaling matrix:\n{S}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{np.round(eigenvectors, 4)}")
```

**Expected output:**

```
Scaling matrix:
[[3 0]
 [0 2]]
Eigenvalues: [3. 2.]
Eigenvectors:
[[1. 0.]
 [0. 1.]]
```

The eigenvectors are `[1, 0]` and `[0, 1]` -- the standard basis vectors. This makes intuitive sense: a diagonal scaling matrix just stretches each axis independently, so vectors along the axes don't change direction.

**A rotation matrix has no real eigenvectors:**

```python
import numpy as np

# 90-degree rotation
R = np.array([
    [0, -1],
    [1,  0]
])

eigenvalues, eigenvectors = np.linalg.eig(R)
print(f"Rotation matrix:\n{R}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Are eigenvalues real? {np.all(np.isreal(eigenvalues))}")
```

**Expected output:**

```
Rotation matrix:
[[ 0 -1]
 [ 1  0]]
Eigenvalues: [0.+1.j 0.-1.j]
Are eigenvalues real? False
```

The eigenvalues are complex numbers (`j` is numpy's notation for the imaginary unit `i`). This makes intuitive sense: a rotation changes the direction of *every* vector, so no real vector stays pointing the same way. There are no "natural axes" for a pure rotation.

### Exercise 2.1: Find and Verify Eigenvalues

**Task:** For each matrix below, use `np.linalg.eig()` to find the eigenvalues and eigenvectors. Then verify the equation `A @ v = lambda * v` for each pair.

```python
A = np.array([[4, 2], [1, 3]])
B = np.array([[1, 0], [0, 1]])  # The identity matrix
C = np.array([[0, 1], [1, 0]])  # What transformation is this?
```

<details>
<summary>Hint 1: Verification pattern</summary>

For each eigenvalue/eigenvector pair, compute both sides of `A @ v = lambda * v` and check they match:

```python
eigenvalues, eigenvectors = np.linalg.eig(matrix)
v = eigenvectors[:, i]       # i-th eigenvector
lam = eigenvalues[i]         # i-th eigenvalue
print(np.allclose(matrix @ v, lam * v))
```
</details>

<details>
<summary>Hint 2: Think about what B and C do</summary>

Matrix B is the identity -- it doesn't change any vector. What does that tell you about its eigenvalues?

Matrix C swaps the x and y coordinates: `[a, b]` becomes `[b, a]`. Which vectors survive a coordinate swap without changing direction?
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

matrices = {
    'A': np.array([[4, 2], [1, 3]]),
    'B': np.array([[1, 0], [0, 1]]),
    'C': np.array([[0, 1], [1, 0]]),
}

for name, M in matrices.items():
    eigenvalues, eigenvectors = np.linalg.eig(M)
    print(f"Matrix {name}:\n{M}")
    print(f"  Eigenvalues: {eigenvalues}")

    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        lam = eigenvalues[i]
        check = np.allclose(M @ v, lam * v)
        print(f"  v{i+1} = {np.round(v, 4)}, lambda = {lam:.4f}, "
              f"A@v = lambda*v: {check}")
    print()
```

**Expected output:**

```
Matrix A:
[[4 2]
 [1 3]]
  Eigenvalues: [5. 2.]
  v1 = [0.8944 0.4472], lambda = 5.0000, A@v = lambda*v: True
  v2 = [-0.7071  0.7071], lambda = 2.0000, A@v = lambda*v: True

Matrix B:
[[1 0]
 [0 1]]
  Eigenvalues: [1. 1.]
  v1 = [1. 0.], lambda = 1.0000, A@v = lambda*v: True
  v2 = [0. 1.], lambda = 1.0000, A@v = lambda*v: True

Matrix C:
[[0 1]
 [1 0]]
  Eigenvalues: [ 1. -1.]
  v1 = [0.7071 0.7071], lambda = 1.0000, A@v = lambda*v: True
  v2 = [-0.7071  0.7071], lambda = -1.0000, A@v = lambda*v: True
```

**Explanation:**
- **Matrix A**: Eigenvalues 5 and 2. The matrix stretches space by a factor of 5 along one direction and 2 along another.
- **Matrix B**: The identity matrix. Every vector is an eigenvector with eigenvalue 1 (nothing changes). Numpy picks `[1,0]` and `[0,1]` but any pair of independent vectors would work.
- **Matrix C**: This is a reflection across the line `y = x` (it swaps coordinates). Vectors along `y = x` (like `[1, 1]`) are unchanged (eigenvalue 1). Vectors perpendicular to that line (like `[-1, 1]`) get flipped (eigenvalue -1).
</details>

### Exercise 2.2: Visualize Eigenvectors Under Transformation

**Task:** Create a visualization that shows:
1. A set of random unit vectors (arrows from the origin)
2. Those same vectors after being transformed by a matrix
3. The eigenvectors highlighted in a different color

Use the matrix `A = [[2, 1], [1, 2]]` and show that the eigenvectors stay along their original direction (just getting longer or shorter), while other vectors change direction.

<details>
<summary>Hint 1: Generating random unit vectors</summary>

```python
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])
```
</details>

<details>
<summary>Hint 2: Plotting arrows</summary>

Use `ax.quiver()` or `ax.annotate()` to draw arrows. For side-by-side comparison, use `plt.subplots(1, 2)`.

```python
for v in vectors:
    ax.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1))
```
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1], [1, 2]])
eigenvalues, eigenvectors = np.linalg.eig(A)

# Generate 12 unit vectors evenly spaced around the circle
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
vectors = np.array([[np.cos(a), np.sin(a)] for a in angles])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for v in vectors:
    # Original vectors in blue
    ax1.annotate('', xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1, alpha=0.5))

# Highlight eigenvectors in red (on the original plot)
for i in range(len(eigenvalues)):
    ev = eigenvectors[:, i]
    ax1.annotate('', xy=ev, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(ev[0]*1.15, ev[1]*1.15, f'eigenvector {i+1}', color='red', fontsize=9)

ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title('Before transformation')

for v in vectors:
    tv = A @ v
    ax2.annotate('', xy=tv, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1, alpha=0.5))

# Highlight transformed eigenvectors in red
for i in range(len(eigenvalues)):
    ev = eigenvectors[:, i]
    tev = A @ ev  # This equals eigenvalue * ev
    ax2.annotate('', xy=tev, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.text(tev[0]*1.05, tev[1]*1.05,
             f'eigenvector {i+1} (x{eigenvalues[i]:.0f})', color='red', fontsize=9)

ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_title('After transformation by A')

plt.tight_layout()
plt.savefig('eigenvector_visualization.png', dpi=100)
plt.show()
```

**Key observation:** In the right plot, the blue arrows have all changed direction -- they point somewhere different than they did in the left plot. But the red arrows (eigenvectors) still point in the same direction. They've gotten longer (scaled by their eigenvalue), but they haven't rotated. That's exactly what makes them eigenvectors.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Explain in plain English what an eigenvector is (a direction the matrix doesn't rotate)
- [ ] Use `np.linalg.eig()` to compute eigenvalues and eigenvectors
- [ ] Verify that `A @ v = lambda * v` holds for each eigenvalue/eigenvector pair
- [ ] Explain why a rotation matrix has no real eigenvectors
- [ ] Describe eigenvectors as the "natural axes" of a transformation

---

## Section 3: Eigendecomposition

### Decomposing a Matrix

Eigendecomposition breaks a matrix into three parts:

```
A = P @ D @ P^(-1)
```

where:
- **P** is a matrix whose columns are the eigenvectors of A
- **D** is a diagonal matrix with the eigenvalues on the diagonal
- **P^(-1)** is the inverse of P

What this formula says in plain English: "Any diagonalizable matrix is equivalent to (1) changing to the eigenvector basis, (2) scaling along each eigenvector axis, and (3) changing back to the original basis."

```python
import numpy as np

A = np.array([
    [2, 1],
    [1, 2]
])

# Step 1: Get eigenvalues and eigenvectors
eigenvalues, P = np.linalg.eig(A)
print(f"Matrix A:\n{A}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nP (eigenvectors as columns):\n{np.round(P, 4)}")

# Step 2: Build the diagonal matrix D
D = np.diag(eigenvalues)
print(f"\nD (diagonal of eigenvalues):\n{D}")

# Step 3: Compute P^(-1)
P_inv = np.linalg.inv(P)
print(f"\nP^(-1):\n{np.round(P_inv, 4)}")

# Step 4: Verify A = P @ D @ P^(-1)
reconstructed = P @ D @ P_inv
print(f"\nP @ D @ P^(-1):\n{np.round(reconstructed, 10)}")
print(f"\nMatches original A: {np.allclose(A, reconstructed)}")
```

**Expected output:**

```
Matrix A:
[[2 1]
 [1 2]]

Eigenvalues: [3. 1.]

P (eigenvectors as columns):
[[ 0.7071 -0.7071]
 [ 0.7071  0.7071]]

D (diagonal of eigenvalues):
[[3. 0.]
 [0. 1.]]

P^(-1):
[[ 0.7071  0.7071]
 [-0.7071  0.7071]]

P @ D @ P^(-1):
[[2. 1.]
 [1. 2.]]

Matches original A: True
```

### What Eigendecomposition Reveals

The decomposition `A = P @ D @ P^(-1)` tells you that the matrix A is "really" just a scaling operation -- but in a rotated coordinate system. The three steps are:

1. **P^(-1)**: Convert from standard coordinates to the eigenvector coordinate system
2. **D**: Scale each axis by its eigenvalue (this is the "real work" the matrix does)
3. **P**: Convert back to standard coordinates

This is a powerful insight. A matrix that looks complicated in standard coordinates might be dead simple in the right coordinate system.

### When Eigendecomposition Fails

Not every matrix can be eigendecomposed. Two common cases where it breaks down:

**Complex eigenvalues** (rotation matrices, for example):

```python
import numpy as np

R = np.array([[0, -1], [1, 0]])  # 90-degree rotation
eigenvalues, eigenvectors = np.linalg.eig(R)
print(f"Rotation matrix eigenvalues: {eigenvalues}")
print(f"These are complex numbers (contain 'j')")
```

**Expected output:**

```
Rotation matrix eigenvalues: [0.+1.j 0.-1.j]
These are complex numbers (contain 'j')
```

The decomposition still "works" mathematically with complex numbers, but it's harder to interpret geometrically. For ML purposes, the matrices you'll encounter (covariance matrices, positive semi-definite matrices) almost always have real eigenvalues.

**Non-diagonalizable matrices** (defective matrices):

```python
import numpy as np

# This matrix has a repeated eigenvalue but only one eigenvector direction
J = np.array([[2, 1], [0, 2]])
eigenvalues, eigenvectors = np.linalg.eig(J)
print(f"Matrix:\n{J}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{np.round(eigenvectors, 4)}")

# Check if P is invertible
det = np.linalg.det(eigenvectors)
print(f"\nDeterminant of eigenvector matrix: {det:.6f}")
print(f"Close to zero means eigenvectors are nearly parallel (not a proper basis)")
```

**Expected output:**

```
Matrix:
[[2 1]
 [0 2]]
Eigenvalues: [2. 2.]
Eigenvectors:
[[ 1.0000e+00 -1.0000e+00]
 [ 0.0000e+00  1.1102e-16]]

Determinant of eigenvector matrix: 0.000000
Close to zero means eigenvectors are nearly parallel (not a proper basis)
```

The eigenvalue 2 repeats, but there's only one independent eigenvector direction (`[1, 0]`). The matrix can't be fully diagonalized. This is rare in practice for ML applications, but good to know about.

### Connection to ML: PCA

Principal Component Analysis (PCA) is eigendecomposition applied to a covariance matrix. Here's the connection:

1. You have data points in high-dimensional space
2. Compute the covariance matrix (which measures how features vary together)
3. Eigendecompose the covariance matrix
4. The eigenvectors are the "principal components" -- the directions of maximum variance in your data
5. The eigenvalues tell you how much variance each component captures

The largest eigenvalue corresponds to the direction where your data varies the most. The smallest eigenvalue corresponds to the direction with the least variation. PCA keeps the top-k eigenvectors and throws away the rest, compressing your data while preserving the most important structure.

```python
import numpy as np

# Simulate some 2D data that's stretched along a diagonal
np.random.seed(42)
data = np.random.randn(100, 2) @ np.array([[3, 1], [1, 2]])

# Compute covariance matrix
cov = np.cov(data.T)
print(f"Covariance matrix:\n{np.round(cov, 2)}")

# Eigendecompose the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov)

print(f"\nEigenvalues: {np.round(eigenvalues, 2)}")
print(f"Eigenvectors (principal components):\n{np.round(eigenvectors, 4)}")
print(f"\nVariance explained by PC1: {eigenvalues[0] / eigenvalues.sum():.1%}")
print(f"Variance explained by PC2: {eigenvalues[1] / eigenvalues.sum():.1%}")
```

**Expected output (approximate, due to random data):**

```
Covariance matrix:
[[10.37  4.35]
 [ 4.35  4.67]]

Eigenvalues: [13.14  1.9 ]
Eigenvectors (principal components):
[[ 0.8469 -0.5317]
 [ 0.5317  0.8469]]

Variance explained by PC1: 87.4%
Variance explained by PC2: 12.6%
```

The first principal component captures most of the variance. If you projected the data onto just that one direction, you'd preserve about 87% of the information while halving the dimensionality.

### Exercise 3.1: Eigendecompose and Verify

**Task:** Eigendecompose the matrix below. Verify that `A = P @ D @ P^(-1)` by reconstructing A from its components.

```python
A = np.array([
    [4, 1],
    [2, 3]
])
```

<details>
<summary>Hint: Step-by-step process</summary>

1. Call `np.linalg.eig(A)` to get eigenvalues and eigenvectors
2. Build `D` using `np.diag(eigenvalues)`
3. `P` is the eigenvector matrix (already returned by `eig`)
4. Compute `P_inv = np.linalg.inv(P)`
5. Verify `np.allclose(A, P @ D @ P_inv)`
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

A = np.array([
    [4, 1],
    [2, 3]
])

# Eigendecompose
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

print(f"Matrix A:\n{A}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nP (eigenvectors):\n{np.round(P, 4)}")
print(f"\nD (diagonal):\n{np.round(D, 4)}")
print(f"\nP^(-1):\n{np.round(P_inv, 4)}")

# Reconstruct
reconstructed = P @ D @ P_inv
print(f"\nReconstructed A:\n{np.round(reconstructed, 10)}")
print(f"Matches original: {np.allclose(A, reconstructed)}")
```

**Expected output:**

```
Matrix A:
[[4 1]
 [2 3]]

Eigenvalues: [5. 2.]

P (eigenvectors):
[[ 0.7071 -0.4472]
 [ 0.7071  0.8944]]

D (diagonal):
[[5. 0.]
 [0. 2.]]

P^(-1):
[[ 0.9428  0.4714]
 [-0.7454  0.7454]]

Reconstructed A:
[[4. 1.]
 [2. 3.]]
Matches original: True
```

**Explanation:** The matrix A has eigenvalues 5 and 2. In the eigenvector coordinate system, A is just the diagonal matrix `[[5, 0], [0, 2]]` -- it stretches by 5 in one direction and by 2 in another. The P and P^(-1) matrices handle the coordinate system conversion.
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Write the eigendecomposition formula: `A = P @ D @ P^(-1)`
- [ ] Explain what each of the three matrices represents
- [ ] Perform eigendecomposition in numpy and verify the result
- [ ] Name at least one case where eigendecomposition doesn't work
- [ ] Explain the connection between eigendecomposition and PCA

---

## Section 4: Singular Value Decomposition (SVD)

### Why SVD?

Eigendecomposition has limitations: it only works for square matrices, and it fails for some matrices (defective or complex eigenvalues). SVD is more general -- it works for **any** matrix, even rectangular ones.

```
A = U @ Sigma @ V^T
```

where:
- **U** is an m x m orthogonal matrix (columns are "output directions")
- **Sigma** is an m x n diagonal matrix (the "singular values" on the diagonal, always non-negative)
- **V^T** is an n x n orthogonal matrix (rows are "input directions")

The intuition is similar to eigendecomposition, but it separates the input and output spaces. For a non-square matrix (like a 100x50 matrix mapping 50-dimensional inputs to 100-dimensional outputs), eigendecomposition doesn't even apply, but SVD works perfectly.

### Computing SVD

```python
import numpy as np

A = np.array([
    [3, 2, 2],
    [2, 3, -2]
])

U, sigma, Vt = np.linalg.svd(A)

print(f"Matrix A ({A.shape[0]}x{A.shape[1]}):\n{A}")
print(f"\nU ({U.shape[0]}x{U.shape[1]}):\n{np.round(U, 4)}")
print(f"\nSingular values: {np.round(sigma, 4)}")
print(f"\nV^T ({Vt.shape[0]}x{Vt.shape[1]}):\n{np.round(Vt, 4)}")
```

**Expected output:**

```
Matrix A (2x3):
[[ 3  2  2]
 [ 2  3 -2]]

U (2x2):
[[-0.7071 -0.7071]
 [-0.7071  0.7071]]

Singular values: [5. 3.]

V^T (3x3):
[[-0.7071 -0.7071  0.    ]
 [-0.2357  0.2357 -0.9428]
 [ 0.6667 -0.6667 -0.3333]]
```

Note: numpy returns `sigma` as a 1D array of singular values, not as a full diagonal matrix. To reconstruct A, you need to build the full Sigma matrix.

### Reconstructing A from SVD

```python
import numpy as np

A = np.array([
    [3, 2, 2],
    [2, 3, -2]
])

U, sigma, Vt = np.linalg.svd(A)

# Build the full Sigma matrix (m x n with singular values on diagonal)
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, sigma)

print(f"Sigma matrix ({Sigma.shape[0]}x{Sigma.shape[1]}):\n{np.round(Sigma, 4)}")

# Reconstruct A
reconstructed = U @ Sigma @ Vt
print(f"\nU @ Sigma @ V^T:\n{np.round(reconstructed, 10)}")
print(f"\nMatches original: {np.allclose(A, reconstructed)}")
```

**Expected output:**

```
Sigma matrix (2x3):
[[5. 0. 0.]
 [0. 3. 0.]]

U @ Sigma @ V^T:
[[ 3.  2.  2.]
 [ 2.  3. -2.]]

Matches original: True
```

### The Three Components: What Each One Means

Think of SVD as breaking a transformation into three steps:

1. **V^T** (rotation/reflection in the input space): Rotate the input vector to align with the "natural input directions"
2. **Sigma** (scaling): Scale each component by the corresponding singular value
3. **U** (rotation/reflection in the output space): Rotate the result to align with the "natural output directions"

Every matrix -- no matter how complicated -- is just a rotation, then a stretch, then another rotation. SVD makes this explicit.

```python
import numpy as np

A = np.array([
    [2, 1],
    [1, 2]
])

U, sigma, Vt = np.linalg.svd(A)

v = np.array([1, 0])

# Step by step:
step1 = Vt @ v                      # Rotate input
step2 = sigma * step1[:len(sigma)]   # Scale by singular values
step3 = U @ step2                    # Rotate output

print(f"Input vector: {v}")
print(f"Step 1 (V^T @ v): {np.round(step1, 4)}")
print(f"Step 2 (scale):   {np.round(step2, 4)}")
print(f"Step 3 (U @ ...): {np.round(step3, 4)}")
print(f"Direct A @ v:     {A @ v}")
print(f"Match: {np.allclose(step3, A @ v)}")
```

**Expected output:**

```
Input vector: [1 0]
Step 1 (V^T @ v): [-0.7071  0.7071]
Step 2 (scale):   [-2.1213  0.7071]
Step 3 (U @ ...): [2. 1.]
Direct A @ v:     [2 1]
Match: True
```

### Low-Rank Approximation: The Power of SVD

Here's where SVD becomes truly useful: you can approximate a matrix by keeping only the largest singular values and setting the rest to zero. This gives you the best possible approximation of a given rank.

```python
import numpy as np

# A matrix with some structure
A = np.array([
    [1, 2, 3, 4],
    [2, 4, 6, 8],
    [3, 5, 7, 9],
    [4, 6, 8, 10]
])

U, sigma, Vt = np.linalg.svd(A)

print(f"Original matrix A:\n{A}")
print(f"\nSingular values: {np.round(sigma, 4)}")
print(f"Notice how they decrease rapidly")

# Rank-1 approximation: keep only the first singular value
k = 1
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]
A_approx_1 = U_k @ sigma_k @ Vt_k

print(f"\nRank-1 approximation:\n{np.round(A_approx_1, 2)}")
error_1 = np.linalg.norm(A - A_approx_1)
print(f"Error (Frobenius norm): {error_1:.4f}")

# Rank-2 approximation: keep the top 2 singular values
k = 2
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]
A_approx_2 = U_k @ sigma_k @ Vt_k

print(f"\nRank-2 approximation:\n{np.round(A_approx_2, 2)}")
error_2 = np.linalg.norm(A - A_approx_2)
print(f"Error (Frobenius norm): {error_2:.4f}")

print(f"\nOriginal storage: {A.size} numbers")
print(f"Rank-1 storage: {U_k.size + sigma[:k].size + Vt_k.size} numbers")
print(f"Rank-2 storage: {U[:, :2].size + sigma[:2].size + Vt[:2, :].size} numbers")
```

**Expected output:**

```
Original matrix A:
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  5  7  9]
 [ 4  6  8 10]]

Singular values: [20.2285  1.0484  0.      0.    ]
Notice how they decrease rapidly

Rank-1 approximation:
[[ 1.15  2.15  3.15  4.15]
 [ 2.3   4.3   6.3   8.3 ]
 [ 2.88  5.07  7.26  9.45]
 [ 3.45  5.84  8.23 10.61]]
Error (Frobenius norm): 1.0484

Rank-2 approximation:
[[ 1.  2.  3.  4.]
 [ 2.  4.  6.  8.]
 [ 3.  5.  7.  9.]
 [ 4.  6.  8. 10.]]
Error (Frobenius norm): 0.0000

Original storage: 16 numbers
Rank-1 storage: 9 numbers
Rank-2 storage: 14 numbers
```

The rank-2 approximation perfectly reconstructs the matrix because the original matrix only has rank 2 (the third and fourth singular values are zero). Even the rank-1 approximation is close -- it captures most of the matrix's structure with just 9 numbers instead of 16.

### How Much Information Each Singular Value Captures

The singular values tell you how important each component is. You can compute the fraction of total "energy" (variance) captured by the top-k singular values:

```python
import numpy as np

A = np.array([
    [5, 4, 3, 2, 1],
    [4, 4, 3, 2, 1],
    [3, 3, 3, 2, 1],
    [2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1]
])

U, sigma, Vt = np.linalg.svd(A)

print(f"Singular values: {np.round(sigma, 4)}")

# Energy captured by each singular value (squared, normalized)
total_energy = np.sum(sigma ** 2)
cumulative_energy = np.cumsum(sigma ** 2) / total_energy

print(f"\nEnergy captured by top-k singular values:")
for k in range(1, len(sigma) + 1):
    print(f"  Top {k}: {cumulative_energy[k-1]:.4f} ({cumulative_energy[k-1]:.1%})")
```

**Expected output:**

```
Singular values: [12.2066  1.7638  0.6052  0.192   0.0439]

Energy captured by top-k singular values:
  Top 1: 0.9753 (97.5%)
  Top 2: 0.9957 (99.6%)
  Top 3: 0.9981 (99.8%)
  Top 4: 0.9998 (100.0%)
  Top 5: 1.0000 (100.0%)
```

The first singular value captures 97.5% of the matrix's energy. This is typical in practice -- most real-world data matrices have a few large singular values and many small ones, meaning you can approximate them well with a low-rank decomposition.

### Exercise 4.1: SVD a Small Matrix and Reconstruct

**Task:** Compute the SVD of the matrix below. Reconstruct it from `U @ Sigma @ V^T` and verify the result matches the original.

```python
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
```

<details>
<summary>Hint: Building the Sigma matrix</summary>

For a 3x2 matrix, `np.linalg.svd` returns `sigma` as a 1D array of length 2. You need to build a 3x2 Sigma matrix:

```python
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, sigma)
```
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

U, sigma, Vt = np.linalg.svd(A)

print(f"Matrix A ({A.shape}):\n{A}")
print(f"\nU ({U.shape}):\n{np.round(U, 4)}")
print(f"\nSingular values: {np.round(sigma, 4)}")
print(f"\nV^T ({Vt.shape}):\n{np.round(Vt, 4)}")

# Build Sigma matrix
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, sigma)
print(f"\nSigma ({Sigma.shape}):\n{np.round(Sigma, 4)}")

# Reconstruct
reconstructed = U @ Sigma @ Vt
print(f"\nReconstructed:\n{np.round(reconstructed, 10)}")
print(f"Matches original: {np.allclose(A, reconstructed)}")
```

**Expected output:**

```
Matrix A ((3, 2)):
[[1 2]
 [3 4]
 [5 6]]

U ((3, 3)):
[[-0.2298  0.8835  0.4082]
 [-0.5247  0.2408 -0.8165]
 [-0.8196 -0.4019  0.4082]]

Singular values: [9.5255 0.5143]

V^T ((2, 2)):
[[-0.6196 -0.7849]
 [-0.7849  0.6196]]

Sigma ((3, 2)):
[[9.5255 0.    ]
 [0.     0.5143]
 [0.     0.    ]]

Reconstructed:
[[1. 2.]
 [3. 4.]
 [5. 6.]]
Matches original: True
```

**Explanation:** SVD works on this 3x2 matrix even though eigendecomposition would not (it's not square). U is 3x3, Sigma is 3x2, and V^T is 2x2. The shapes work out: `(3x3) @ (3x2) @ (2x2) = (3x2)`, matching the original matrix.
</details>

### Exercise 4.2: Low-Rank Approximation with Error Measurement

**Task:** Create a 5x5 matrix, compute its SVD, and produce rank-1 and rank-2 approximations. For each, compute the Frobenius norm of the error (`np.linalg.norm(A - A_approx)`) and the percentage of total energy captured.

```python
A = np.array([
    [9, 8, 7, 6, 5],
    [8, 7, 6, 5, 4],
    [7, 6, 5, 4, 3],
    [6, 5, 4, 3, 2],
    [5, 4, 3, 2, 1]
])
```

<details>
<summary>Hint 1: Building a rank-k approximation</summary>

```python
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
Vt_k = Vt[:k, :]
A_approx = U_k @ sigma_k @ Vt_k
```
</details>

<details>
<summary>Hint 2: Energy calculation</summary>

The energy captured by the top k singular values is `sum(sigma[:k]**2) / sum(sigma**2)`.
</details>

**Solution:**

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

A = np.array([
    [9, 8, 7, 6, 5],
    [8, 7, 6, 5, 4],
    [7, 6, 5, 4, 3],
    [6, 5, 4, 3, 2],
    [5, 4, 3, 2, 1]
])

U, sigma, Vt = np.linalg.svd(A)

print(f"Singular values: {np.round(sigma, 4)}")
total_energy = np.sum(sigma ** 2)

for k in [1, 2]:
    U_k = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    Vt_k = Vt[:k, :]
    A_approx = U_k @ sigma_k @ Vt_k

    error = np.linalg.norm(A - A_approx)
    energy = np.sum(sigma[:k] ** 2) / total_energy

    print(f"\nRank-{k} approximation:")
    print(np.round(A_approx, 2))
    print(f"  Error: {error:.4f}")
    print(f"  Energy captured: {energy:.4%}")
    print(f"  Storage: {U_k.size + sigma[:k].size + Vt_k.size} numbers "
          f"(original: {A.size})")
```

**Expected output:**

```
Singular values: [25.4368  1.7228  0.      0.      0.    ]

Rank-1 approximation:
[[8.85 7.69 6.54 5.38 4.23]
 [7.69 6.69 5.69 4.69 3.69]
 [6.54 5.69 4.84 3.99 3.14]
 [5.38 4.69 3.99 3.29 2.59]
 [4.23 3.69 3.14 2.59 2.04]]
  Error: 1.7228
  Energy captured: 99.5420%
  Storage: 11 numbers (original: 25)

Rank-2 approximation:
[[ 9.  8.  7.  6.  5.]
 [ 8.  7.  6.  5.  4.]
 [ 7.  6.  5.  4.  3.]
 [ 6.  5.  4.  3.  2.]
 [ 5.  4.  3.  2.  1.]]
  Error: 0.0000
  Energy captured: 100.0000%
  Storage: 17 numbers (original: 25)
```

**Explanation:** This matrix has only two non-zero singular values, meaning it has rank 2. The rank-2 approximation perfectly reconstructs it. Even the rank-1 approximation captures 99.5% of the energy using just 11 numbers instead of 25 -- a significant compression. In high-dimensional data (like images or embeddings), this compression ratio becomes dramatic.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Write the SVD formula: `A = U @ Sigma @ V^T`
- [ ] Explain what U, Sigma, and V^T each represent
- [ ] Compute SVD with `np.linalg.svd()` and reconstruct the original matrix
- [ ] Build a rank-k approximation by keeping only the top k singular values
- [ ] Calculate how much energy (information) a rank-k approximation captures

---

## Practice Project: Image Compression with SVD

### Project Description

You'll compress a grayscale image using SVD. A grayscale image is just a matrix of pixel values -- each element is a brightness value between 0 and 255. By computing the SVD and keeping only the top k singular values, you can reconstruct an approximation of the image using far fewer numbers. The tradeoff between k (how many singular values you keep) and image quality is the heart of this project.

This project ties together everything in this route: the SVD decomposition, low-rank approximation, and the idea that a few large singular values capture most of the information.

### Requirements

Build a script that:
1. Loads a grayscale image as a matrix (or creates a synthetic test image)
2. Computes the SVD
3. Reconstructs the image using k = 1, 5, 10, 20, 50, and full rank
4. Plots the original and compressed images side by side
5. Calculates the compression ratio and reconstruction error for each k

### Getting Started

**Step 1: Get an image**

You can use a real photo or create a synthetic test image:

```python
import numpy as np

# Option A: Create a synthetic test image (no file needed)
def create_test_image(size=100):
    """Create a test image with gradients, shapes, and edges."""
    img = np.zeros((size, size))

    # Diagonal gradient
    for i in range(size):
        for j in range(size):
            img[i, j] = (i + j) / (2 * size) * 255

    # Add a bright rectangle
    img[20:40, 30:70] = 200

    # Add a dark circle
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = (x - center)**2 + (y - 60)**2 < 15**2
    img[mask] = 50

    return img

# Option B: Load a real image
# from matplotlib.image import imread
# img = imread('your_photo.png')
# if img.ndim == 3:  # Convert RGB to grayscale
#     img = np.mean(img, axis=2)
```

**Step 2: Compute SVD and create approximations**

```python
U, sigma, Vt = np.linalg.svd(image, full_matrices=False)
# full_matrices=False gives compact SVD, more memory-efficient

# Rank-k approximation
k = 10
approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
```

**Step 3: Plot and compare**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# Plot original and approximations at different k values
```

### Hints and Tips

<details>
<summary>Hint 1: Compression ratio calculation</summary>

The original image stores `m * n` numbers. A rank-k SVD approximation stores `m*k + k + k*n` numbers (the truncated U, sigma, and V^T). The compression ratio is:

```python
original_size = m * n
compressed_size = m * k + k + k * n
ratio = original_size / compressed_size
```
</details>

<details>
<summary>Hint 2: Clipping pixel values</summary>

After SVD reconstruction, some pixel values might go below 0 or above 255. Clip them:

```python
approx = np.clip(approx, 0, 255)
```
</details>

<details>
<summary>Hint 3: Reconstruction error</summary>

Use the Frobenius norm to measure error, and normalize by the norm of the original:

```python
error = np.linalg.norm(image - approx) / np.linalg.norm(image)
print(f"Relative error: {error:.4%}")
```
</details>

<details>
<summary>Hint 4: Plotting images</summary>

Use `plt.imshow()` with a grayscale colormap:

```python
ax.imshow(image, cmap='gray', vmin=0, vmax=255)
ax.set_title(f'k = {k}')
ax.axis('off')
```
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def create_test_image(size=100):
    """Create a test image with gradients, shapes, and edges."""
    img = np.zeros((size, size))

    # Diagonal gradient
    for i in range(size):
        for j in range(size):
            img[i, j] = (i + j) / (2 * size) * 255

    # Bright rectangle
    img[20:40, 30:70] = 200

    # Dark circle
    y, x = np.ogrid[:size, :size]
    mask = (x - 50)**2 + (y - 60)**2 < 15**2
    img[mask] = 50

    # Bright diagonal stripe
    for i in range(size):
        for j in range(max(0, i-2), min(size, i+3)):
            img[i, j] = max(img[i, j], 220)

    return img

# Create or load image
image = create_test_image(100)
m, n = image.shape
print(f"Image size: {m}x{n} = {m*n} pixels")

# Compute SVD
U, sigma, Vt = np.linalg.svd(image, full_matrices=False)
print(f"Number of singular values: {len(sigma)}")
print(f"Top 10 singular values: {np.round(sigma[:10], 1)}")

# Plot energy distribution
total_energy = np.sum(sigma ** 2)
cumulative = np.cumsum(sigma ** 2) / total_energy

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(sigma) + 1), cumulative, 'b-o', markersize=3)
plt.xlabel('Number of singular values (k)')
plt.ylabel('Cumulative energy captured')
plt.title('How many singular values do we need?')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% energy')
plt.axhline(y=0.99, color='g', linestyle='--', alpha=0.5, label='99% energy')
plt.legend()
plt.tight_layout()
plt.savefig('svd_energy.png', dpi=100)
plt.show()

# Create approximations at different k values
k_values = [1, 5, 10, 20, 50, min(m, n)]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, k in enumerate(k_values):
    # Rank-k approximation
    approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
    approx = np.clip(approx, 0, 255)

    # Compression metrics
    original_size = m * n
    compressed_size = m * k + k + k * n
    ratio = original_size / compressed_size
    rel_error = np.linalg.norm(image - approx) / np.linalg.norm(image)
    energy = np.sum(sigma[:k] ** 2) / total_energy

    # Plot
    axes[idx].imshow(approx, cmap='gray', vmin=0, vmax=255)
    title = f'k = {k}'
    if k < min(m, n):
        title += f'\nRatio: {ratio:.1f}x, Error: {rel_error:.2%}'
    else:
        title += '\n(Full rank — original)'
    axes[idx].set_title(title, fontsize=10)
    axes[idx].axis('off')

    print(f"k = {k:>3}: compression = {ratio:>5.1f}x, "
          f"error = {rel_error:.4%}, energy = {energy:.4%}, "
          f"storage = {compressed_size} numbers")

plt.suptitle('Image Compression with SVD', fontsize=14)
plt.tight_layout()
plt.savefig('svd_compression.png', dpi=100)
plt.show()
```

**Key points in this solution:**
- `full_matrices=False` gives the compact SVD (more memory-efficient for large images)
- `np.clip` prevents pixel values from going out of range
- The compression ratio tells you how much smaller the compressed representation is
- With a real photo, you'll typically find that k=20-50 produces a recognizable image at 5-10x compression

**What to look for in the output:**
- k=1 is a blurry smear -- just the dominant direction of brightness change
- k=5 starts to show the main shapes
- k=10-20 looks recognizable, with most structural detail preserved
- k=50 is nearly indistinguishable from the original in most images
- The singular values drop off quickly, so a small k captures most of the information
</details>

### Extending the Project

If you want to go further, try:
- Use a real photograph and compare compression quality at different k values
- Compress an RGB image by applying SVD to each color channel separately
- Compare SVD compression to JPEG at similar file sizes
- Plot the singular value spectrum for different types of images (photos vs. drawings vs. noise) and see how the decay rate differs
- Animate the reconstruction: show the image building up as you add one singular value at a time

---

## Summary

### Key Takeaways

- **Vector spaces** are sets where addition and scalar multiplication work. A **basis** is the minimum set of vectors that spans the whole space. The standard basis isn't the only basis -- choosing a good basis is the whole point of PCA.
- **Eigenvectors** are the special directions that a matrix doesn't rotate -- it only stretches them. The stretch factor is the **eigenvalue**. Together, they reveal the "natural axes" of a transformation.
- **Eigendecomposition** (`A = P @ D @ P^(-1)`) says every diagonalizable matrix is just scaling in a rotated coordinate system. PCA is eigendecomposition of the covariance matrix.
- **SVD** (`A = U @ Sigma @ V^T`) works for any matrix, even non-square ones. It decomposes a transformation into rotation, scaling, and another rotation.
- **Low-rank approximation** via SVD lets you keep only the most important components, compressing data while preserving most of the structure.

### Skills You've Gained

You can now:
- Determine whether a set of vectors forms a basis and convert between coordinate systems
- Compute eigenvalues and eigenvectors and explain their geometric meaning
- Decompose a matrix into its eigenvalue components and verify the result
- Apply SVD to any matrix and reconstruct it from the three components
- Build low-rank approximations that trade accuracy for compression
- Explain how PCA, recommendation systems, and image compression use these ideas

### Self-Assessment

Take a moment to reflect:
- Could you explain to a colleague why eigenvectors are the "natural axes" of a matrix?
- If someone showed you a matrix's singular values `[100, 50, 0.1, 0.01]`, what would you say about using a rank-2 approximation?
- Can you see why a recommendation system might use SVD on a user-item rating matrix?
- Does the connection between eigendecomposition and PCA make sense?

---

## Next Steps

### Continue Learning

**Apply these concepts directly:**
- **PCA (Principal Component Analysis)** -- eigendecomposition of covariance matrices to reduce dimensionality. This is the most direct application of what you learned here.
- **Latent spaces in generative models** -- when a diffusion model or VAE works in a "latent space", it's operating in a low-dimensional subspace discovered through SVD-like decompositions.
- **Spectral methods** -- graph neural networks and spectral clustering use eigenvalues of graph Laplacian matrices to find community structure.

**Return to the main learning path:**
- The main ML route doesn't require this material, so continue where you left off.

### Additional Resources

**Videos:**
- 3Blue1Brown's *Essence of Linear Algebra*, episodes on eigenvectors and change of basis -- the best visual explanations available
- Steve Brunton's SVD lectures on YouTube -- excellent for the data science perspective

**Books:**
- *Linear Algebra and Its Applications* by Gilbert Strang -- chapters on eigenvalues and SVD
- *Mathematics for Machine Learning* by Deisenroth, Faisal, Ong (free online) -- Chapter 4 covers eigendecomposition, Chapter 10 covers PCA

**Interactive:**
- setosa.io/ev/eigenvectors-and-eigenvalues -- interactive eigenvalue visualization
- numpy.org documentation for `np.linalg.eig` and `np.linalg.svd`

---

## Quick Reference

### Numpy Cheat Sheet for Eigenvalues and SVD

```python
import numpy as np

# --- Eigenvalues and Eigenvectors ---
eigenvalues, eigenvectors = np.linalg.eig(A)  # A must be square
# eigenvalues: 1D array of eigenvalues
# eigenvectors: columns are eigenvectors (eigenvectors[:, i] pairs with eigenvalues[i])

# Verify: A @ v = lambda * v
v = eigenvectors[:, 0]
lam = eigenvalues[0]
np.allclose(A @ v, lam * v)  # Should be True

# --- Eigendecomposition ---
P = eigenvectors                  # Eigenvector matrix
D = np.diag(eigenvalues)          # Diagonal eigenvalue matrix
P_inv = np.linalg.inv(P)          # Inverse of eigenvector matrix
# A = P @ D @ P_inv

# --- Singular Value Decomposition ---
U, sigma, Vt = np.linalg.svd(A)               # Full SVD
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)  # Compact SVD

# Build Sigma matrix from sigma vector
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, sigma)
# A = U @ Sigma @ Vt

# --- Low-Rank Approximation ---
k = 10
A_approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

# --- Error and Energy ---
error = np.linalg.norm(A - A_approx)                    # Frobenius norm of error
rel_error = error / np.linalg.norm(A)                    # Relative error
energy = np.sum(sigma[:k]**2) / np.sum(sigma**2)         # Fraction of energy captured

# --- Useful Checks ---
np.linalg.det(M)          # Determinant (0 means singular/dependent)
np.linalg.inv(M)          # Matrix inverse
np.linalg.matrix_rank(M)  # Rank of a matrix
```

---

## Glossary

- **Basis**: A minimal set of linearly independent vectors that spans a vector space. Every vector in the space can be uniquely expressed as a combination of basis vectors.
- **Change of basis**: Converting a vector's coordinates from one basis to another. Achieved by multiplying by a change-of-basis matrix (or its inverse).
- **Diagonalizable**: A matrix that can be written as `P @ D @ P^(-1)` where D is diagonal. Most matrices encountered in ML are diagonalizable.
- **Dimension**: The number of basis vectors needed to span a vector space. 2D space has dimension 2, a 768-dimensional embedding space has dimension 768.
- **Eigendecomposition**: Factoring a square matrix as `A = P @ D @ P^(-1)`, where P contains eigenvectors and D contains eigenvalues. Reveals the "natural axes" of the transformation.
- **Eigenvalue**: The scalar lambda in `A @ v = lambda * v`. Tells you how much an eigenvector gets stretched (or compressed, or flipped) by the matrix.
- **Eigenvector**: A non-zero vector whose direction is preserved by a matrix transformation. Only its length changes (by the eigenvalue factor).
- **Frobenius norm**: A measure of matrix "size", computed as the square root of the sum of all squared elements. Used to measure approximation error.
- **Linear independence**: A set of vectors is linearly independent if no vector in the set can be written as a combination of the others.
- **Low-rank approximation**: Approximating a matrix by one with fewer non-zero singular values. Keeps the most important structure while reducing storage.
- **Orthogonal matrix**: A square matrix whose columns (and rows) are orthonormal vectors. Its inverse equals its transpose: `Q^(-1) = Q^T`.
- **Rank**: The number of linearly independent rows (or columns) in a matrix. Equals the number of non-zero singular values.
- **Singular value**: The diagonal entries of Sigma in the SVD. Always non-negative. Measure how much the matrix stretches along each singular direction.
- **Singular Value Decomposition (SVD)**: Factoring any matrix as `A = U @ Sigma @ V^T`. Works for all matrices, including non-square ones.
- **Span**: The set of all vectors reachable by combining a given set of vectors (using addition and scalar multiplication).
- **Vector space**: A set of vectors closed under addition and scalar multiplication. The "arena" where linear algebra takes place.
