---
title: Linear Algebra Essentials
route_map: /routes/linear-algebra-essentials/map.md
paired_sherpa: /routes/linear-algebra-essentials/sherpa.md
prerequisites:
  - Python basics (variables, loops, functions)
  - Comfort running Python scripts
topics:
  - Vectors
  - Matrices
  - Dot Products
  - Linear Transformations
---

# Linear Algebra Essentials - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/linear-algebra-essentials/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/linear-algebra-essentials/map.md` for the high-level overview.

## Overview

Linear algebra is the math behind machine learning. Not "one of the maths" -- *the* math. When a neural network processes an image, it multiplies matrices. When a recommendation engine finds similar products, it computes dot products. When a language model generates text, it transforms vectors through dozens of layers of matrix operations.

The good news: if you've worked with arrays, lists, and dictionaries in code, you already have the intuitions you need. A vector is a list of numbers. A matrix is a function that transforms one list into another. Dot products are a way to measure how similar two lists are. That's most of it.

This guide teaches linear algebra through code first and equations second. Every concept starts with numpy, and every equation gets a plain-English explanation before it appears. You'll build visual intuition for what these operations actually *do* to data before you worry about the formulas.

By the end, you'll have a working 2D transformation sandbox and enough linear algebra to read ML papers without glazing over at the math.

## Learning Objectives

By the end of this route, you will be able to:
- Represent data as vectors and perform arithmetic on them using numpy
- Compute and interpret dot products as similarity measures
- Describe matrices as transformations that warp space
- Perform matrix-vector and matrix-matrix multiplication by hand and in code
- Predict what a transformation matrix will do to a set of points
- Build a 2D transformation sandbox that visualizes matrix operations

## Prerequisites

Before starting this route, you should be comfortable with:
- **Python basics**: Variables, loops, functions, lists
- **Running Python scripts**: From the command line or a notebook
- **High school algebra** (helpful): Variables, simple equations

No prior linear algebra knowledge is assumed. If you've heard terms like "vector" or "matrix" before but they feel fuzzy, that's fine -- this guide starts from scratch.

## Setup

You need two Python packages: numpy for array operations and matplotlib for visualization.

```bash
pip install numpy matplotlib
```

**Verify your setup:**

Create a file called `la_setup_test.py` and run it:

```python
import numpy as np
import matplotlib
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")

# Quick sanity check
v = np.array([3, 4])
print(f"Test vector: {v}")
print(f"Its length: {np.linalg.norm(v)}")
```

```bash
python la_setup_test.py
```

**Expected output:**

```
numpy version: 1.26.4
matplotlib version: 3.9.2
Test vector: [3 4]
Its length: 5.0
```

Your version numbers may differ -- that's fine as long as the script runs without errors. If you got `5.0` for the length, you're ready.

---

## Section 1: Vectors as Arrows and Data

### What Is a Vector?

If you've used a Python list or a database row, you've already worked with vectors. A vector is just an ordered list of numbers:

```python
import numpy as np

# This is a vector
v = np.array([3, 1])
print(v)
```

**Expected output:**
```
[3 1]
```

That's it. A numpy array with some numbers in it.

But a vector is also something *geometric* -- an arrow in space with a direction and a length. The vector `[3, 1]` points 3 units to the right and 1 unit up from the origin. This dual nature (a list of numbers *and* an arrow) is the key insight that makes linear algebra useful for ML.

Think of it this way:
- **As data**: A vector is a row in your database. A user might be `[age, income, num_purchases]` = `[32, 75000, 12]`.
- **As geometry**: That same vector is a point (or arrow) in 3D space. Users with similar profiles are *literally close together* in that space.

ML exploits this connection constantly. When you represent data as vectors, geometric operations (distance, angle, projection) become meaningful data operations (similarity, relevance, recommendation).

### Vector Arithmetic

Vectors support the same arithmetic you'd expect from numbers, but applied element-by-element.

#### Addition

```python
import numpy as np

a = np.array([3, 1])
b = np.array([1, 2])
c = a + b
print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {c}")
```

**Expected output:**
```
a = [3 1]
b = [1 2]
a + b = [4 3]
```

Each element adds independently: `[3+1, 1+2]` = `[4, 3]`.

Geometrically, vector addition is "tip-to-tail": place the tail of `b` at the tip of `a`, and the result `c` points from the origin to where `b` ends up. If `a` is "walk 3 east and 1 north" and `b` is "walk 1 east and 2 north", then `a + b` is "walk 4 east and 3 north".

#### Subtraction

```python
d = a - b
print(f"a - b = {d}")
```

**Expected output:**
```
a - b = [2 -1]
```

Subtraction gives you the vector *from* `b` *to* `a`. This is how you compute the difference or "displacement" between two data points.

#### Scalar Multiplication

Multiplying a vector by a single number (a "scalar") stretches or shrinks it:

```python
a = np.array([3, 1])
print(f"2 * a = {2 * a}")
print(f"0.5 * a = {0.5 * a}")
print(f"-1 * a = {-1 * a}")
```

**Expected output:**
```
2 * a = [6 2]
0.5 * a = [1.5 0.5]
-1 * a = [-3 -1]
```

- Multiplying by 2 doubles the length, same direction.
- Multiplying by 0.5 halves the length, same direction.
- Multiplying by -1 flips the direction.

The term "scalar" just means a single number (as opposed to a vector). It "scales" the vector -- makes it bigger or smaller without changing its direction (unless negative, which reverses it).

### Vector Length (Magnitude)

The length of a vector -- how long the arrow is -- comes from the Pythagorean theorem. For a 2D vector `[x, y]`, the length is `sqrt(x^2 + y^2)`. In math notation, the length of vector **v** is written as ||**v**|| (double bars around the name, called the "norm").

```python
a = np.array([3, 4])
length = np.linalg.norm(a)
print(f"Vector: {a}")
print(f"Length: {length}")
```

**Expected output:**
```
Vector: [3 4]
Length: 5.0
```

This works because `sqrt(3^2 + 4^2)` = `sqrt(9 + 16)` = `sqrt(25)` = `5`. The classic 3-4-5 right triangle.

`np.linalg.norm()` works for any number of dimensions -- it's the generalized Pythagorean theorem.

### Unit Vectors

A unit vector has length 1. You can turn any vector into a unit vector by dividing it by its length. This is called "normalizing" a vector:

```python
a = np.array([3, 4])
a_unit = a / np.linalg.norm(a)
print(f"Original: {a}, length: {np.linalg.norm(a)}")
print(f"Unit vector: {a_unit}, length: {np.linalg.norm(a_unit)}")
```

**Expected output:**
```
Original: [3 4], length: 5.0
Unit vector: [0.6 0.8], length: 1.0
```

Why would you want a unit vector? Because it preserves the *direction* of the original vector while discarding the magnitude. This is useful in ML when you care about the direction of a feature vector (what kind of thing is it?) but not its magnitude (how much of it is there?).

> **Visualization**: Run `python tools/ml-visualizations/vectors.py` to see vector addition, subtraction, and scaling animated on a 2D plot.

### Exercise 1.1: Vector Prediction

**Task:** Before running any code, predict the result of each operation. Then verify with numpy.

```python
a = np.array([2, 3])
b = np.array([-1, 4])
```

1. What is `a + b`?
2. What is `a - b`?
3. What is `3 * a`?
4. What is the length of `a`?

<details>
<summary>Hint: How to compute the length</summary>

The length of `[2, 3]` is `sqrt(2^2 + 3^2)` = `sqrt(4 + 9)` = `sqrt(13)`. Use `np.linalg.norm(a)` to verify.
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

a = np.array([2, 3])
b = np.array([-1, 4])

print(f"a + b = {a + b}")         # [1, 7]
print(f"a - b = {a - b}")         # [3, -1]
print(f"3 * a = {3 * a}")         # [6, 9]
print(f"|a| = {np.linalg.norm(a)}")  # 3.6055... (sqrt(13))
```

**Expected output:**
```
a + b = [1 7]
a - b = [ 3 -1]
3 * a = [6 9]
|a| = 3.605551275463989
```

**Explanation:** Each operation works element-by-element. Addition combines corresponding elements. Subtraction finds the difference. Scalar multiplication scales each element. The length uses the Pythagorean theorem: `sqrt(2^2 + 3^2)` = `sqrt(13)` = approximately 3.61.
</details>

### Exercise 1.2: Direction of Sums

**Task:** Given these pairs of vectors, predict which *direction* their sum points (roughly: up, down, left, right, or some combination). Then verify by computing the sum.

```python
pair_1 = (np.array([5, 0]), np.array([0, 5]))    # a points right, b points up
pair_2 = (np.array([3, 3]), np.array([-3, -3]))   # a points upper-right, b points lower-left
pair_3 = (np.array([1, 0]), np.array([0, -4]))    # a points right, b points down
```

<details>
<summary>Hint: Think tip-to-tail</summary>

For pair_1: Start at the origin, walk 5 right (that's `a`), then from there walk 5 up (that's `b`). Where do you end up? That's the direction of the sum.

For pair_2: The two vectors are exact opposites. What happens when you add something to its opposite?
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

pair_1 = (np.array([5, 0]), np.array([0, 5]))
pair_2 = (np.array([3, 3]), np.array([-3, -3]))
pair_3 = (np.array([1, 0]), np.array([0, -4]))

for i, (a, b) in enumerate([pair_1, pair_2, pair_3], 1):
    s = a + b
    print(f"Pair {i}: {a} + {b} = {s}")
```

**Expected output:**
```
Pair 1: [5 0] + [0 5] = [5 5]
Pair 2: [3 3] + [-3 -3] = [0 0]
Pair 3: [1 0] + [0 -4] = [ 1 -4]
```

- **Pair 1**: `[5, 5]` -- points upper-right at a 45-degree angle.
- **Pair 2**: `[0, 0]` -- the zero vector. Opposite vectors cancel out.
- **Pair 3**: `[1, -4]` -- points mostly downward with a slight rightward lean.
</details>

### Checkpoint 1

Before moving on, make sure you can:
- [ ] Create numpy vectors and perform addition, subtraction, and scalar multiplication
- [ ] Explain what a vector represents both as data and as a geometric arrow
- [ ] Compute the length of a vector using `np.linalg.norm()`
- [ ] Predict the approximate direction of a vector sum without running code

---

## Section 2: Dot Products and Similarity

### What Is a Dot Product?

The dot product takes two vectors and returns a single number. You compute it by multiplying corresponding elements and adding the results:

```
[a1, a2] . [b1, b2] = a1*b1 + a2*b2
```

That notation -- the dot between two vectors -- is where the name comes from. In math, you'll see it written as **a** . **b** or sometimes with angle brackets as <**a**, **b**>.

```python
import numpy as np

a = np.array([2, 3])
b = np.array([4, 1])

# Three equivalent ways to compute the dot product
dot1 = np.dot(a, b)
dot2 = a @ b             # The @ operator does matrix/dot products
dot3 = np.sum(a * b)     # Manual: element-wise multiply, then sum

print(f"np.dot(a, b) = {dot1}")
print(f"a @ b = {dot2}")
print(f"sum(a * b) = {dot3}")
```

**Expected output:**
```
np.dot(a, b) = 11
a @ b = 11
sum(a * b) = 11
```

The `@` operator is the preferred way in modern numpy. It reads cleanly and works for both dot products (vector @ vector) and matrix multiplication (matrix @ vector, matrix @ matrix).

Let's verify by hand: `2*4 + 3*1` = `8 + 3` = `11`. That's all a dot product is.

### What Does a Dot Product Tell You?

The number you get from a dot product tells you how much two vectors point in the same direction. More precisely, the dot product is related to the angle between the vectors:

- **Positive** dot product: the vectors point in roughly the same direction (angle less than 90 degrees)
- **Zero** dot product: the vectors are perpendicular (exactly 90 degrees)
- **Negative** dot product: the vectors point in roughly opposite directions (angle greater than 90 degrees)

```python
import numpy as np

right = np.array([1, 0])
up = np.array([0, 1])
upper_right = np.array([1, 1])
left = np.array([-1, 0])

print(f"right . upper_right = {right @ upper_right}")   # Positive: similar direction
print(f"right . up = {right @ up}")                       # Zero: perpendicular
print(f"right . left = {right @ left}")                   # Negative: opposite direction
```

**Expected output:**
```
right . upper_right = 1
right . up = 0
right . left = -1
```

This is the single most important property of the dot product for ML. When you hear "cosine similarity" or "how similar are these vectors?", it's a dot product (with some normalization).

### The Geometric Formula

The dot product has a geometric formula that makes its meaning precise:

```
a . b = |a| * |b| * cos(angle)
```

Read this as: "the dot product of **a** and **b** equals the length of **a** times the length of **b** times the cosine of the angle between them."

You don't need to memorize this formula, but it explains why the sign of the dot product tells you about the angle:
- cos(0 degrees) = 1 (same direction, positive dot product)
- cos(90 degrees) = 0 (perpendicular, zero dot product)
- cos(180 degrees) = -1 (opposite direction, negative dot product)

### Cosine Similarity

In ML, you often want to compare vectors regardless of their length. A long feature vector and a short feature vector might represent the same "kind" of thing -- they just differ in magnitude. Cosine similarity strips out the length:

```
cosine_similarity(a, b) = (a . b) / (|a| * |b|)
```

This always gives a value between -1 and 1:
- 1 means identical direction
- 0 means perpendicular (unrelated)
- -1 means opposite direction

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 4, 6])    # Same direction as a, just doubled
c = np.array([-1, -2, -3]) # Opposite direction from a
d = np.array([3, -1, 0])   # Some unrelated direction

def cosine_similarity(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"a vs b (same direction): {cosine_similarity(a, b):.4f}")
print(f"a vs c (opposite):       {cosine_similarity(a, c):.4f}")
print(f"a vs d (unrelated):      {cosine_similarity(a, d):.4f}")
```

**Expected output:**
```
a vs b (same direction): 1.0000
a vs c (opposite):       -1.0000
a vs d (unrelated):      0.0535
```

Notice that `a` and `b` have cosine similarity of 1 even though `b` is twice as long. Cosine similarity only cares about direction. This is why search engines, recommendation systems, and embedding models use cosine similarity -- it measures *what kind of thing* a vector represents, not *how much* of it there is.

### Exercise 2.1: Dot Products by Hand

**Task:** Compute each dot product by hand first, then verify with numpy.

```python
a = np.array([1, 0, 3])
b = np.array([2, 5, 1])
c = np.array([0, -2, 4])
```

1. `a . b` = ?
2. `b . c` = ?
3. `a . c` = ?

<details>
<summary>Hint: Step by step</summary>

For `a . b`: multiply element-by-element and sum: `1*2 + 0*5 + 3*1` = ?
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

a = np.array([1, 0, 3])
b = np.array([2, 5, 1])
c = np.array([0, -2, 4])

print(f"a . b = {a @ b}")  # 1*2 + 0*5 + 3*1 = 2 + 0 + 3 = 5
print(f"b . c = {b @ c}")  # 2*0 + 5*(-2) + 1*4 = 0 - 10 + 4 = -6
print(f"a . c = {a @ c}")  # 1*0 + 0*(-2) + 3*4 = 0 + 0 + 12 = 12
```

**Expected output:**
```
a . b = 5
b . c = -6
a . c = 12
```

**Explanation:**
- `a . b = 5`: positive, so `a` and `b` point in a generally similar direction.
- `b . c = -6`: negative, so `b` and `c` point in generally opposite directions.
- `a . c = 12`: strongly positive, so `a` and `c` are quite aligned (both have large positive third components).
</details>

### Exercise 2.2: Perpendicular and Similar Vectors

**Task:** For each pair, determine whether the vectors are perpendicular, similar (point roughly the same way), or opposite. Use the dot product to check your answer.

```python
pair_1 = (np.array([1, 2]), np.array([-2, 1]))
pair_2 = (np.array([3, 4]), np.array([6, 8]))
pair_3 = (np.array([1, 1]), np.array([-1, -1]))
pair_4 = (np.array([5, 0]), np.array([0, 3]))
```

<details>
<summary>Hint: What to look for</summary>

Compute the dot product for each pair. If it's 0, the vectors are perpendicular. If positive, they point in a similar direction. If negative, they point in opposite directions. For pair_2, also compute the cosine similarity -- what do you notice?
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

pairs = [
    (np.array([1, 2]), np.array([-2, 1])),
    (np.array([3, 4]), np.array([6, 8])),
    (np.array([1, 1]), np.array([-1, -1])),
    (np.array([5, 0]), np.array([0, 3])),
]

def cosine_similarity(v1, v2):
    return (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

for i, (a, b) in enumerate(pairs, 1):
    dot = a @ b
    cos_sim = cosine_similarity(a, b)
    print(f"Pair {i}: dot = {dot:>3}, cosine = {cos_sim:>7.4f}")
```

**Expected output:**
```
Pair 1: dot =   0, cosine =  0.0000
Pair 2: dot =  50, cosine =  1.0000
Pair 3: dot =  -2, cosine = -1.0000
Pair 4: dot =   0, cosine =  0.0000
```

- **Pair 1**: Perpendicular (dot product is 0). `[1,2]` and `[-2,1]` meet at a right angle.
- **Pair 2**: Identical direction (cosine similarity is 1). `[6,8]` is just `[3,4]` doubled.
- **Pair 3**: Opposite direction (cosine similarity is -1). `[-1,-1]` is `[1,1]` flipped.
- **Pair 4**: Perpendicular (dot product is 0). One points along the x-axis, the other along the y-axis.
</details>

### Checkpoint 2

Before moving on, make sure you can:
- [ ] Compute a dot product by hand and with `a @ b`
- [ ] Explain what the sign of a dot product tells you about two vectors
- [ ] Define cosine similarity and explain why it's useful for comparing data
- [ ] Identify perpendicular vectors using the dot product

---

## Section 3: Matrices as Transformations

### What Is a Matrix?

A matrix is a grid of numbers. In numpy, it's a 2D array:

```python
import numpy as np

M = np.array([
    [2, 0],
    [0, 3]
])
print(M)
```

**Expected output:**
```
[[2 0]
 [0 3]]
```

But here's the mental model that matters: **a matrix is a function**. It takes a vector as input and produces a vector as output. In programming terms, if a function maps inputs to outputs, a matrix does the same thing for vectors.

```python
v = np.array([1, 1])
result = M @ v
print(f"Input:  {v}")
print(f"Output: {result}")
```

**Expected output:**
```
Input:  [1 1]
Output: [2 3]
```

The matrix `M` took the vector `[1, 1]` and produced `[2, 3]`. It stretched the x-component by 2 and the y-component by 3. That's a **scaling transformation**.

### The Identity Matrix: Doing Nothing

The simplest matrix is one that does nothing -- every vector comes out unchanged:

```python
import numpy as np

I = np.eye(2)  # 2x2 identity matrix
print(f"Identity matrix:\n{I}")

v = np.array([3, 7])
print(f"\nI @ {v} = {I @ v}")
```

**Expected output:**
```
Identity matrix:
[[1. 0.]
 [0. 1.]]

I @ [3 7] = [3. 7.]
```

The identity matrix has 1s on the diagonal and 0s everywhere else. It's the matrix equivalent of multiplying by 1, or calling an identity function: `lambda x: x`. In math notation, the identity matrix is written as **I**.

### Scaling

A diagonal matrix (numbers on the diagonal, zeros elsewhere) scales each axis independently:

```python
import numpy as np

# Scale x by 2, y by 0.5
scale = np.array([
    [2,   0],
    [0, 0.5]
])

v = np.array([3, 4])
print(f"Original: {v}")
print(f"Scaled:   {scale @ v}")
```

**Expected output:**
```
Original: [3 4]
Scaled:   [6. 2.]
```

The x-component doubled (3 became 6). The y-component halved (4 became 2). Diagonal matrices are the simplest transformations to understand: each diagonal entry tells you the scale factor for that axis.

### Rotation

A rotation matrix spins vectors around the origin. For an angle theta (in radians), the rotation matrix is:

```
[[cos(theta), -sin(theta)],
 [sin(theta),  cos(theta)]]
```

Don't memorize this formula -- understand what it does:

```python
import numpy as np

# Rotate 90 degrees counterclockwise
theta = np.pi / 2  # 90 degrees in radians
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Round to avoid floating-point noise
R = np.round(R, 10)
print(f"Rotation matrix (90 degrees):\n{R}")

v = np.array([1, 0])  # Points right
rotated = R @ v
print(f"\n[1, 0] rotated 90 degrees = {rotated}")

v2 = np.array([3, 1])
rotated2 = R @ v2
print(f"[3, 1] rotated 90 degrees = {rotated2}")
```

**Expected output:**
```
Rotation matrix (90 degrees):
[[ 0. -1.]
 [ 1.  0.]]

[1, 0] rotated 90 degrees = [0. 1.]
[3, 1] rotated 90 degrees = [-1.  3.]
```

A vector pointing right `[1, 0]` becomes `[0, 1]` (pointing up) after a 90-degree counterclockwise rotation. That's exactly what you'd expect if you drew the arrow and spun it.

Note on radians: Python's math functions use radians, not degrees. 90 degrees = pi/2 radians, 180 degrees = pi radians, 360 degrees = 2*pi radians. You can convert with `np.radians(degrees)`.

### Shear

A shear matrix skews one axis based on the other, like pushing the top of a rectangle sideways to make a parallelogram:

```python
import numpy as np

# Horizontal shear: x gets shifted based on y
shear = np.array([
    [1, 1],
    [0, 1]
])

print(f"Shear matrix:\n{shear}")

points = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
print("\nShearing a unit square:")
for p in points:
    print(f"  {p} -> {shear @ p}")
```

**Expected output:**
```
Shear matrix:
[[1 1]
 [0 1]]

Shearing a unit square:
  [0 0] -> [0 0]
  [1 0] -> [1 0]
  [0 1] -> [1 1]
  [1 1] -> [2 1]
```

Points on the x-axis (`y=0`) didn't move. Points with `y=1` shifted 1 unit to the right. The square became a parallelogram.

### The Key Insight: Columns Tell the Story

Here's the most important idea in this entire guide: **the columns of a matrix tell you where the basis vectors end up**.

The "basis vectors" are the two simplest arrows: `[1, 0]` (pointing right) and `[0, 1]` (pointing up). Every other vector is a combination of these two. If you know where these two arrows land after a transformation, you know everything about that transformation.

```python
import numpy as np

M = np.array([
    [2, -1],
    [1,  3]
])

e1 = np.array([1, 0])  # First basis vector
e2 = np.array([0, 1])  # Second basis vector

print(f"Matrix M:\n{M}")
print(f"\n[1, 0] -> {M @ e1}")  # This is the FIRST COLUMN of M
print(f"[0, 1] -> {M @ e2}")    # This is the SECOND COLUMN of M
```

**Expected output:**
```
Matrix M:
[[ 2 -1]
 [ 1  3]]

[1, 0] -> [2 1]
[0, 1] -> [-1  3]
```

Look at the results: `[1, 0]` maps to `[2, 1]`, which is the first column of `M`. And `[0, 1]` maps to `[-1, 3]`, which is the second column. This always works.

So when you see a matrix, you can immediately read off what it does:
- First column = where `[1, 0]` goes
- Second column = where `[0, 1]` goes
- Everything else follows from linearity (scaling and adding)

> **Visualization**: Run `python tools/ml-visualizations/linear_transforms.py` to see how different matrices warp a grid of points. Watch how the basis vectors move and how the grid follows.

### Exercise 3.1: Reading Transformations from Matrices

**Task:** For each matrix below, predict where `[1, 0]` and `[0, 1]` will land (just read the columns). Then verify with numpy.

```python
A = np.array([[3, 0], [0, 3]])    # Matrix A
B = np.array([[0, -1], [1, 0]])   # Matrix B
C = np.array([[1, 0], [0, -1]])   # Matrix C
```

<details>
<summary>Hint: Reading columns</summary>

The first column is where `[1, 0]` goes. The second column is where `[0, 1]` goes. For matrix A, the first column is `[3, 0]` and the second is `[0, 3]`. What kind of transformation is that?
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

A = np.array([[3, 0], [0, 3]])
B = np.array([[0, -1], [1, 0]])
C = np.array([[1, 0], [0, -1]])

e1 = np.array([1, 0])
e2 = np.array([0, 1])

for name, M in [("A", A), ("B", B), ("C", C)]:
    print(f"Matrix {name}:\n{M}")
    print(f"  [1,0] -> {M @ e1}")
    print(f"  [0,1] -> {M @ e2}")
    print()
```

**Expected output:**
```
Matrix A:
[[3 0]
 [0 3]]
  [1,0] -> [3 0]
  [0,1] -> [0 3]

Matrix B:
[[ 0 -1]
 [ 1  0]]
  [1,0] -> [0 1]
  [0,1] -> [-1  0]

Matrix C:
[[ 1  0]
 [ 0 -1]]
  [1,0] -> [1 0]
  [0,1] -> [ 0 -1]
```

**Explanation:**
- **Matrix A**: Uniform scaling by 3. Both basis vectors triple in length, same direction. Every point moves 3x farther from the origin.
- **Matrix B**: 90-degree counterclockwise rotation. `[1,0]` (right) becomes `[0,1]` (up), and `[0,1]` (up) becomes `[-1,0]` (left).
- **Matrix C**: Reflection across the x-axis. `[1,0]` stays put, `[0,1]` flips to `[0,-1]`. Top becomes bottom.
</details>

### Exercise 3.2: Building a Rotation Matrix

**Task:** Construct a rotation matrix for 45 degrees. Apply it to the vector `[1, 0]` and verify that the result points in the expected direction (upper-right at 45 degrees).

<details>
<summary>Hint: Degrees to radians</summary>

Use `np.radians(45)` or `np.pi / 4` to convert 45 degrees to radians. Then plug into the rotation formula: `[[cos, -sin], [sin, cos]]`.
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

theta = np.radians(45)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

print(f"Rotation matrix (45 degrees):")
print(np.round(R, 4))

v = np.array([1, 0])
rotated = R @ v
print(f"\n[1, 0] rotated 45 degrees = {np.round(rotated, 4)}")
print(f"Length of result: {np.linalg.norm(rotated):.4f}")
```

**Expected output:**
```
Rotation matrix (45 degrees):
[[ 0.7071 -0.7071]
 [ 0.7071  0.7071]]

[1, 0] rotated 45 degrees = [0.7071 0.7071]
Length of result: 1.0000
```

**Explanation:** The result `[0.7071, 0.7071]` has equal x and y components, confirming a 45-degree angle. The length is still 1.0 -- rotation preserves lengths (it just spins, it doesn't stretch).
</details>

### Checkpoint 3

Before moving on, make sure you can:
- [ ] Apply a matrix to a vector using `M @ v`
- [ ] Identify what the identity, scaling, rotation, and shear matrices do
- [ ] Read the columns of a matrix to predict where basis vectors land
- [ ] Construct a rotation matrix for a given angle

---

## Section 4: Matrix Operations

### Matrix-Vector Multiplication (Row Perspective)

We've been using `M @ v` to transform vectors. Let's look at the mechanics of how the multiplication works, one row at a time.

Each row of the matrix produces one element of the output by computing a dot product with the input vector:

```python
import numpy as np

M = np.array([
    [2, 3],
    [1, -1]
])
v = np.array([4, 2])

# The full multiplication
result = M @ v
print(f"M @ v = {result}")

# What's happening row by row:
row_0_dot = M[0] @ v   # [2, 3] . [4, 2] = 2*4 + 3*2 = 14
row_1_dot = M[1] @ v   # [1, -1] . [4, 2] = 1*4 + (-1)*2 = 2
print(f"\nRow 0 dot v: {M[0]} . {v} = {row_0_dot}")
print(f"Row 1 dot v: {M[1]} . {v} = {row_1_dot}")
```

**Expected output:**
```
M @ v = [14  2]

Row 0 dot v: [2 3] . [4 2] = 14
Row 1 dot v: [ 1 -1] . [4 2] = 2
```

So matrix-vector multiplication is just "dot product each row of the matrix with the vector." The first row's dot product becomes the first element of the result, and so on.

### Matrix-Matrix Multiplication: Composing Transformations

When you multiply two matrices together, you get a new matrix that represents *doing both transformations in sequence*. This is composition, the same idea as composing functions: `f(g(x))`.

```python
import numpy as np

# Scale by 2
S = np.array([
    [2, 0],
    [0, 2]
])

# Rotate 90 degrees
R = np.array([
    [0, -1],
    [1,  0]
])

# Compose: scale first, then rotate
RS = R @ S

v = np.array([1, 0])

# Two ways to get the same result:
step_by_step = R @ (S @ v)    # Scale v, then rotate the result
composed = RS @ v              # Use the composed matrix directly

print(f"Step by step: S @ v = {S @ v}, then R @ that = {step_by_step}")
print(f"Composed (RS @ v): {composed}")
print(f"Same result: {np.array_equal(step_by_step, composed)}")
```

**Expected output:**
```
Step by step: S @ v = [2 0], then R @ that = [0 2]
Composed (RS @ v): [0 2]
Same result: True
```

The composed matrix `R @ S` is a single matrix that scales AND rotates in one operation. This is powerful: no matter how many transformations you chain together, the result is always just one matrix. Neural networks exploit this heavily -- a layer of a neural network is essentially a matrix multiplication.

### Order Matters

Matrix multiplication is **not commutative**: `A @ B` is generally not the same as `B @ A`. This makes intuitive sense -- rotating then scaling gives a different result than scaling then rotating.

```python
import numpy as np

S = np.array([[2, 0], [0, 1]])   # Scale x by 2, leave y alone
R = np.array([[0, -1], [1, 0]])  # Rotate 90 degrees

RS = R @ S   # Scale first, then rotate
SR = S @ R   # Rotate first, then scale

v = np.array([1, 0])

print(f"Scale then rotate: {RS @ v}")
print(f"Rotate then scale: {SR @ v}")
print(f"Same? {np.array_equal(RS @ v, SR @ v)}")
```

**Expected output:**
```
Scale then rotate: [0 2]
Rotate then scale: [0 1]
Same? False
```

Reading `A @ B @ v` right to left: first apply `B` to `v`, then apply `A` to the result. The rightmost matrix acts first.

### Transpose

Transposing a matrix swaps its rows and columns. Row 0 becomes column 0, row 1 becomes column 1, and so on. In math notation, the transpose of matrix **A** is written as **A**^T (superscript T).

```python
import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(f"A (2x3):\n{A}")
print(f"\nA transposed (3x2):\n{A.T}")
```

**Expected output:**
```
A (2x3):
[[1 2 3]
 [4 5 6]]

A transposed (3x2):
[[1 4]
 [2 5]
 [3 6]]
```

A 2x3 matrix becomes a 3x2 matrix. The first row `[1, 2, 3]` became the first column, and the second row `[4, 5, 6]` became the second column.

Transpose comes up constantly in ML because data is often stored in a different orientation than an algorithm expects. If your data has samples as rows but the formula wants samples as columns, you transpose.

A symmetric matrix is one that equals its own transpose (**A** = **A**^T). These have special properties that ML algorithms exploit -- covariance matrices, for instance, are always symmetric.

```python
# Symmetric matrix example
sym = np.array([
    [4, 2, 1],
    [2, 5, 3],
    [1, 3, 6]
])
print(f"Symmetric? {np.array_equal(sym, sym.T)}")
```

**Expected output:**
```
Symmetric? True
```

### Putting It Together

Here's a full example that chains several operations:

```python
import numpy as np

# Start with a set of 2D points forming a triangle
triangle = np.array([
    [0, 0],
    [1, 0],
    [0.5, 1]
]).T  # Transpose so each column is a point (2x3 matrix)

print(f"Original triangle (columns are points):\n{triangle}")

# Step 1: Scale x by 2
S = np.array([[2, 0], [0, 1]])

# Step 2: Rotate 45 degrees
theta = np.radians(45)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

# Compose the transformations
T = R @ S  # Scale first, then rotate

# Apply to all points at once
transformed = T @ triangle
print(f"\nTransformed triangle:\n{np.round(transformed, 4)}")
```

**Expected output:**
```
Original triangle (columns are points):
[[0.  1.  0.5]
 [0.  0.  1. ]]

Transformed triangle:
[[ 0.      1.4142  0.    ]
 [ 0.      1.4142  1.4142]]
```

Notice how we applied the transformation to all three points at once with a single matrix multiplication. This is why ML uses matrices -- you can transform entire datasets in one operation.

### Exercise 4.1: Matrix Multiplication by Hand

**Task:** Multiply these matrices by hand, then verify with numpy.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
```

Compute `A @ B`. Remember: each element of the result is the dot product of a row from A with a column from B.

<details>
<summary>Hint: Step by step</summary>

Result[0,0] = row 0 of A dotted with column 0 of B = `1*5 + 2*7` = ?
Result[0,1] = row 0 of A dotted with column 1 of B = `1*6 + 2*8` = ?
Result[1,0] = row 1 of A dotted with column 0 of B = `3*5 + 4*7` = ?
Result[1,1] = row 1 of A dotted with column 1 of B = `3*6 + 4*8` = ?
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# By hand:
# [0,0]: 1*5 + 2*7 = 5 + 14 = 19
# [0,1]: 1*6 + 2*8 = 6 + 16 = 22
# [1,0]: 3*5 + 4*7 = 15 + 28 = 43
# [1,1]: 3*6 + 4*8 = 18 + 32 = 50

print(f"A @ B =\n{A @ B}")
```

**Expected output:**
```
A @ B =
[[19 22]
 [43 50]]
```

**Explanation:** Each element at position `[i, j]` is the dot product of row `i` from A with column `j` from B. This is the definition of matrix multiplication.
</details>

### Exercise 4.2: Order Matters

**Task:** Demonstrate that rotating then scaling gives a different result than scaling then rotating.

1. Create a scaling matrix that scales x by 3 and y by 1.
2. Create a rotation matrix for 30 degrees.
3. Apply "scale then rotate" to the vector `[1, 1]`.
4. Apply "rotate then scale" to the same vector.
5. Are the results the same?

<details>
<summary>Hint: Setting up the matrices</summary>

Scaling: `[[3, 0], [0, 1]]`. Rotation: use `np.radians(30)` and the rotation formula. "Scale then rotate" means `R @ S @ v` (rightmost applies first). "Rotate then scale" means `S @ R @ v`.
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np

# Scaling matrix: x by 3, y by 1
S = np.array([[3, 0], [0, 1]])

# Rotation matrix: 30 degrees
theta = np.radians(30)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

v = np.array([1, 1])

scale_then_rotate = R @ S @ v
rotate_then_scale = S @ R @ v

print(f"Scale then rotate: {np.round(scale_then_rotate, 4)}")
print(f"Rotate then scale: {np.round(rotate_then_scale, 4)}")
print(f"Same? {np.allclose(scale_then_rotate, rotate_then_scale)}")
```

**Expected output:**
```
Scale then rotate: [1.7981 2.366 ]
Rotate then scale: [1.0981 1.366 ]
Same? False
```

**Explanation:** The results differ. When you scale first, the x-component gets tripled *before* the rotation mixes x and y together. When you rotate first, the original x and y get mixed, and *then* only the resulting x-component gets tripled. The order of operations changes the outcome.
</details>

### Checkpoint 4

Before moving on, make sure you can:
- [ ] Explain matrix-vector multiplication as "dot product each row with the vector"
- [ ] Compute matrix-matrix multiplication by hand
- [ ] Explain why `A @ B` does not equal `B @ A` in general
- [ ] Transpose a matrix and explain when transpose is useful
- [ ] Read `A @ B @ v` right to left: B applies first, then A

---

## Practice Project: 2D Transformation Sandbox

### Project Description

Build a Python script that visualizes how matrices transform 2D points. You'll create a grid of points, apply matrix transformations, and display the before-and-after using matplotlib. The sandbox should support composing multiple transformations.

This project ties together everything you've learned: vectors as points, matrices as transformations, matrix multiplication as composition, and the column interpretation of matrices.

### Requirements

Build a script (`transform_sandbox.py`) that:
1. Creates a grid of 2D points (a square grid works well)
2. Lets the user pick a transformation from presets (scale, rotate, shear) or enter a custom 2x2 matrix
3. Displays the original grid and the transformed grid side by side
4. Supports composing multiple transformations sequentially
5. Shows where the basis vectors `[1,0]` and `[0,1]` end up after the transformation

### Getting Started

**Step 1: Create the grid**

Start by generating a set of 2D points. A simple approach is a grid of points with some extra points along the axes to make the transformation visible.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a grid of points
x = np.linspace(-2, 2, 9)
y = np.linspace(-2, 2, 9)
xx, yy = np.meshgrid(x, y)
points = np.vstack([xx.ravel(), yy.ravel()])  # 2 x N matrix, each column is a point
```

**Step 2: Define transformation presets**

Create a dictionary of named transformations:

```python
presets = {
    "identity": np.eye(2),
    "scale_2x": np.array([[2, 0], [0, 2]]),
    # Add more...
}
```

**Step 3: Plot before and after**

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(points[0], points[1])
ax1.set_title("Original")
# ... apply transformation and plot on ax2
```

### Hints and Tips

<details>
<summary>Hint 1: Structuring the script</summary>

Break it into functions:
- `make_grid(n, bounds)` -- returns a 2xN matrix of grid points
- `plot_points(ax, points, title)` -- plots a set of points with axis labels and grid lines
- `apply_transform(matrix, points)` -- returns `matrix @ points`
- `compose(matrices)` -- multiplies a list of matrices together (left to right means last applied first)
</details>

<details>
<summary>Hint 2: Showing basis vectors</summary>

Plot the basis vectors as arrows using `ax.quiver()` or `ax.annotate()`. The original basis vectors are `[1,0]` and `[0,1]`. After transformation, they become the columns of the matrix.

```python
# Draw basis vectors as arrows from the origin
ax.annotate('', xy=(1, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
```
</details>

<details>
<summary>Hint 3: Composing transformations</summary>

To compose transformations, multiply the matrices. If you have a list `[A, B, C]` meaning "apply A first, then B, then C", the composed matrix is `C @ B @ A`. You can use a loop:

```python
from functools import reduce
composed = reduce(lambda a, b: b @ a, matrices)
```

Or iterate manually:

```python
result = np.eye(2)
for M in matrices:
    result = M @ result
```
</details>

<details>
<summary>Hint 4: User input for custom matrices</summary>

Keep it simple. Ask for four numbers (a, b, c, d) and construct `[[a, b], [c, d]]`. Or just hardcode a sequence of transformations for the first version and add interactivity later.
</details>

### Example Solution

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def make_grid(n=9, bound=2):
    """Generate a 2D grid of points as a 2xN matrix."""
    x = np.linspace(-bound, bound, n)
    y = np.linspace(-bound, bound, n)
    xx, yy = np.meshgrid(x, y)
    return np.vstack([xx.ravel(), yy.ravel()])

def plot_points(ax, points, title, bound=5):
    """Plot a set of 2D points with grid lines and basis vectors."""
    ax.scatter(points[0], points[1], s=10, c='blue', alpha=0.6)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(title)

def draw_basis(ax, matrix=None):
    """Draw basis vectors (or transformed basis vectors) as arrows."""
    if matrix is None:
        e1, e2 = np.array([1, 0]), np.array([0, 1])
    else:
        e1, e2 = matrix[:, 0], matrix[:, 1]
    ax.annotate('', xy=e1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.annotate('', xy=e2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.annotate(f'e1={np.round(e1, 2)}', xy=e1, fontsize=8, color='red')
    ax.annotate(f'e2={np.round(e2, 2)}', xy=e2, fontsize=8, color='green')

# Preset transformations
presets = {
    "identity":   np.eye(2),
    "scale_2x":   np.array([[2, 0], [0, 2]]),
    "scale_xy":   np.array([[2, 0], [0, 0.5]]),
    "rotate_45":  np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4)],
        [np.sin(np.pi/4),  np.cos(np.pi/4)]
    ]),
    "rotate_90":  np.array([[0, -1], [1, 0]]),
    "shear_x":    np.array([[1, 1], [0, 1]]),
    "reflect_x":  np.array([[1, 0], [0, -1]]),
    "squish":     np.array([[1, 0], [0, 0]]),
}

def compose(matrix_list):
    """Compose a list of transformations (first in list applies first)."""
    result = np.eye(2)
    for M in matrix_list:
        result = M @ result
    return result

# Choose transformations to apply (edit this list to experiment)
transforms = ["rotate_45", "scale_xy"]

# Build the composed matrix
matrices = [presets[name] for name in transforms]
composed = compose(matrices)

print(f"Transformations: {' -> '.join(transforms)}")
print(f"Composed matrix:\n{np.round(composed, 4)}")

# Generate grid and transform
grid = make_grid()
transformed = composed @ grid

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

plot_points(ax1, grid, "Original")
draw_basis(ax1)

plot_points(ax2, transformed, f"After: {' then '.join(transforms)}")
draw_basis(ax2, composed)

plt.tight_layout()
plt.savefig("transform_sandbox_output.png", dpi=100)
plt.show()

print("\nBasis vector destinations:")
print(f"  [1, 0] -> {np.round(composed[:, 0], 4)}")
print(f"  [0, 1] -> {np.round(composed[:, 1], 4)}")
```

**Key points in this solution:**
- Points are stored as columns of a 2xN matrix so `M @ points` transforms all of them at once.
- `compose()` builds a single matrix from a sequence of transformations.
- Basis vectors are drawn as arrows so you can see where `[1,0]` and `[0,1]` land.
- Edit the `transforms` list to try different combinations.
</details>

### Extending the Project

If you want to go further, try:
- Add animation: show the grid morphing from original to transformed using matplotlib's animation tools
- Support 3D: extend to 3x3 matrices and 3D points using matplotlib's 3D plotting
- Add a "determinant display" showing how the area of a unit square changes under each transformation
- Build an interactive version where arrow keys rotate and +/- keys scale

---

## Summary

### Key Takeaways

- **Vectors** are lists of numbers that also represent arrows in space. ML uses them to encode data points, features, and embeddings.
- **Dot products** measure how much two vectors point in the same direction. A dot product of zero means perpendicular; positive means similar; negative means opposite. Cosine similarity normalizes this to a [-1, 1] scale.
- **Matrices** are functions that transform vectors. You can read what a matrix does by looking at its columns -- they tell you where the basis vectors land.
- **Matrix multiplication** composes transformations. `A @ B` means "apply B first, then A." Order matters.
- **Transpose** swaps rows and columns, which comes up constantly when reshaping data.

### Skills You've Gained

You can now:
- Create and manipulate vectors using numpy
- Compute and interpret dot products and cosine similarity
- Apply matrices to vectors and understand the geometric result
- Multiply matrices by hand and in code
- Read a matrix's columns to predict its transformation
- Build visualizations of linear transformations

### Self-Assessment

Take a moment to reflect:
- Can you look at a 2x2 matrix and immediately describe what it does to a grid?
- Could you explain cosine similarity to a colleague?
- If someone said "matrix multiplication is function composition," would that make sense?
- Do you understand why ML frameworks spend so much effort optimizing matrix operations?

---

## Next Steps

### Continue Learning

Ready for more? Here are your next options:

**Build on this topic:**
- [Calculus for ML](/routes/calculus-for-ml/map.md) -- Derivatives, gradients, and gradient descent. This is the natural next step before diving into how ML models actually learn.
- [Linear Algebra Deep Dive](/routes/linear-algebra-deep-dive/map.md) -- Eigenvalues, singular value decomposition, and vector spaces. Useful for understanding PCA, dimensionality reduction, and the theory behind many ML algorithms.

### Additional Resources

**Books:**
- *Linear Algebra and Its Applications* by Gilbert Strang -- the classic textbook
- *Mathematics for Machine Learning* by Deisenroth, Faisal, Ong -- freely available online, tailored for ML

**Videos:**
- 3Blue1Brown's *Essence of Linear Algebra* -- the best visual explanation of these concepts, highly recommended
- Gilbert Strang's MIT OpenCourseWare lectures -- the full course for free

**Interactive Tools:**
- NumPy documentation (numpy.org) -- reference for all array operations
- Desmos (desmos.com/calculator) -- free online graphing calculator for quick visualizations

---

## Quick Reference

### Numpy Cheat Sheet for Vectors and Matrices

```python
import numpy as np

# --- Vectors ---
v = np.array([1, 2, 3])           # Create a vector
v + w                              # Element-wise addition
v - w                              # Element-wise subtraction
3 * v                              # Scalar multiplication
np.linalg.norm(v)                  # Length (magnitude)
v / np.linalg.norm(v)              # Unit vector (normalize)

# --- Dot Products ---
np.dot(v, w)                       # Dot product
v @ w                              # Dot product (preferred syntax)

# --- Matrices ---
M = np.array([[1, 2], [3, 4]])     # Create a matrix
np.eye(n)                          # n x n identity matrix
np.zeros((m, n))                   # m x n matrix of zeros

# --- Matrix Operations ---
M @ v                              # Matrix-vector multiplication
A @ B                              # Matrix-matrix multiplication
M.T                                # Transpose
M.shape                            # Dimensions (rows, columns)

# --- Useful Functions ---
np.round(M, decimals)              # Round to n decimal places
np.allclose(A, B)                  # Check if A and B are approximately equal
np.radians(degrees)                # Convert degrees to radians
np.linalg.norm(v)                  # Vector length
```

---

## Glossary

- **Basis vectors**: The standard direction vectors, `[1, 0]` and `[0, 1]` in 2D. Every vector can be expressed as a combination of basis vectors.
- **Cosine similarity**: The dot product of two vectors divided by the product of their lengths. Measures directional similarity on a scale from -1 to 1.
- **Dot product**: An operation that takes two vectors and returns a single number by multiplying corresponding elements and summing. Measures alignment between vectors.
- **Identity matrix**: A square matrix with 1s on the diagonal and 0s elsewhere. Transforms every vector to itself.
- **Linear transformation**: A function (represented by a matrix) that maps vectors to vectors while preserving addition and scalar multiplication. Lines stay straight, the origin stays fixed.
- **Magnitude**: The length of a vector, computed as the square root of the sum of squared components. Also called the norm.
- **Matrix**: A rectangular grid of numbers. In this context, a function that transforms vectors via multiplication.
- **Norm**: Another word for the length (magnitude) of a vector. Written as ||**v**||.
- **Perpendicular**: Two vectors are perpendicular (orthogonal) when they meet at a right angle. Their dot product is zero.
- **Scalar**: A single number, as opposed to a vector or matrix.
- **Shear**: A transformation that skews one axis based on the value of another, turning rectangles into parallelograms.
- **Transpose**: Swapping the rows and columns of a matrix. The transpose of a 2x3 matrix is a 3x2 matrix.
- **Unit vector**: A vector with length 1. Created by dividing a vector by its magnitude.
- **Vector**: An ordered list of numbers that represents both a data point and a geometric arrow in space.
