---
title: Linear Algebra Deep Dive
route_map: /routes/linear-algebra-deep-dive/map.md
paired_guide: /routes/linear-algebra-deep-dive/guide.md
topics:
  - Vector Spaces
  - Eigenvalues
  - Eigenvectors
  - SVD
---

# Linear Algebra Deep Dive - Sherpa (AI Teaching Guide)

**Purpose**: This sherpa guide helps AI assistants teach the deeper linear algebra concepts that appear throughout ML — vector spaces, eigenvalues, eigenvectors, eigendecomposition, and SVD. This is a side quest route: optional but valuable for learners who want to understand *why* ML techniques like PCA and dimensionality reduction work.

**Route Map**: See `/routes/linear-algebra-deep-dive/map.md` for the high-level overview of this route.
**Paired Guide**: The human-focused content is at `/routes/linear-algebra-deep-dive/guide.md`.

---

## Teaching Overview

### Learning Objectives
By the end of this session, the learner should be able to:
- Understand vector spaces, basis, and dimension
- Compute eigenvalues and eigenvectors and explain their geometric meaning
- Decompose matrices using eigendecomposition
- Apply SVD to compress an image
- Connect these concepts to ML applications (PCA, latent spaces)

### Prior Sessions
Before starting, check `.sessions/index.md` and `.sessions/linear-algebra-deep-dive/` for prior session history. If the learner has completed previous sessions on this route, review the summaries to understand what they've covered and pick up where they left off.

### Prerequisites to Verify
Before starting, verify the learner has:
- Completed the linear-algebra-essentials route (vectors, matrices, transformations, dot products)
- Comfort with numpy for vector and matrix operations (`np.array`, `@` operator, `np.linalg.norm`)
- Intuition for matrices as transformations (can read matrix columns to predict what a matrix does)
- numpy and matplotlib installed

**Helpful but not required**:
- Calculus for ML route (helps with optimization context, not strictly needed here)

**How to verify prerequisites:**
1. Ask: "Can you describe what a matrix does to a vector in geometric terms?"
2. Ask: "If I give you a 2x2 matrix, can you tell me what transformation it represents by reading its columns?"
3. Ask: "Have you used `np.linalg` functions before?"

**If prerequisites are missing**: This route builds directly on linear-algebra-essentials. If the learner can't describe matrices as transformations or isn't comfortable with numpy, suggest they complete that route first. Don't try to cover the prerequisite material here — it needs its own space.

### Audience Context
The target learner is a backend developer who has completed linear-algebra-essentials. They know vectors, matrices, and transformations, and they want to go deeper. They're motivated by ML curiosity — they've heard terms like "eigenvalue" and "SVD" and want to understand what they mean.

Key teaching leverage:
- They already think of matrices as transformations — build on that
- They've used `np.linalg.norm` and `@` — introduce `np.linalg.eig` and `np.linalg.svd` the same way
- They've seen the `linear_transforms.py` visualization — reference it when showing eigenvector directions
- Programming analogies still work: eigendecomposition is like factoring a function into simpler parts

This is a side quest. Remind them: nothing here is required for the main path. They're here because they're curious, and that's the best reason to learn anything.

### Learner Preferences Configuration

Learners can configure their preferred learning style by creating a `.sherpa-config.yml` file in the repository root (gitignored by default). Configuration options include:

**Teaching Style:**
- `tone`: objective, encouraging, humorous (default: objective and respectful)
- `explanation_depth`: concise, balanced, detailed
- `pacing`: learner-led, balanced, structured

**Assessment Format:**
- `quiz_type`: multiple_choice, explanation, mixed (default: mixed)
- `quiz_frequency`: after_each_section, after_major_topics, end_of_route
- `feedback_style`: immediate, summary, detailed

**Example `.sherpa-config.yml`:**
```yaml
teaching:
  tone: encouraging
  explanation_depth: balanced
  pacing: learner-led

assessment:
  quiz_type: mixed
  quiz_frequency: after_major_topics
  feedback_style: immediate
```

If no configuration file exists, use defaults (objective tone, mixed assessments, balanced pacing).

### Assessment Strategies

Use a combination of assessment types to verify understanding:

**Multiple Choice Questions:**
- Present 3-4 answer options
- Include one correct answer and plausible distractors
- Good for checking factual knowledge and catching misconceptions about eigenvalues and SVD
- Example: "If Av = 3v, what is the eigenvalue? A) v B) 3 C) A D) 3v"

**Explanation Questions:**
- Ask learner to explain concepts in their own words
- Assess geometric intuition (critical for eigenvectors and SVD)
- Example: "In your own words, what does it mean for a vector to be an eigenvector of a matrix?"

**Prediction Questions:**
- Show a matrix and ask what will happen before running code
- Builds intuition for decomposition behavior
- Example: "If the top singular value is 100 and the second is 2, what does that tell you about the matrix?"

**Computation Questions:**
- Have them work through small examples by hand
- Reinforces the mechanics behind the abstractions
- Example: "Find the eigenvalues of `[[2, 1], [0, 3]]` by solving det(A - λI) = 0"

**Mixed Approach (Recommended):**
- Use multiple choice for quick concept checks
- Use explanation questions for geometric interpretations
- Use prediction questions before running numpy code
- Use computation for 2x2 examples only (larger matrices get tedious without insight)

---

## Teaching Flow

### Introduction

**What to Cover:**
- This is a side quest — optional but powerful
- These concepts show up everywhere in ML: PCA, recommendation systems, dimensionality reduction, image compression, latent spaces in generative models
- By the end, they'll understand eigenvalues, eigenvectors, and SVD — and use SVD to compress an actual image
- Everything builds on the transformation intuition from linear-algebra-essentials

**Opening Framing:**
"You've learned that matrices are transformations — they stretch, rotate, and shear space. Now we're going to ask a deeper question: can we take a transformation apart and understand its fundamental structure? What are the 'natural axes' of a transformation? What directions does it prefer? These questions lead to eigenvalues, eigenvectors, and SVD — and they're the reason PCA, image compression, and recommendation systems work."

**Opening Questions to Assess Level:**
1. "Have you encountered eigenvalues or eigenvectors before — in school, in ML papers, anywhere?"
2. "Have you heard of PCA or SVD? Do you know roughly what they do?"
3. "What motivated you to take this side quest?"

**Adapt based on responses:**
- If they've seen eigenvalues before: Quickly review the definition, spend more time on geometric intuition and decomposition
- If they know PCA conceptually: Great — you can use PCA as a motivating thread throughout
- If purely curious with no prior exposure: Go slower on the eigenvalue definition, use lots of 2x2 examples, lean on geometry
- If they mention generative models or latent spaces: Connect SVD to those concepts when you get there

---

### Setup Verification

**Check numpy and matplotlib:**
```bash
python -c "import numpy as np; print(np.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"
```

**If not installed:**
```bash
pip install numpy matplotlib
```

**Check Pillow (needed for the image compression project):**
```bash
python -c "from PIL import Image; print('Pillow OK')"
```

**If not installed:**
```bash
pip install Pillow
```

**Reference visualization scripts:**
This route reuses `/tools/ml-visualizations/linear_transforms.py` from linear-algebra-essentials. Verify it's accessible. The learner will also write their own visualization code for eigenvectors and SVD.

---

### Section 1: Vector Spaces and Basis

**Core Concept to Teach:**
A vector space is a set of vectors you can add together and scale, and the result stays in the set. A basis is the minimum set of vectors that can build every vector in the space. Dimension is how many basis vectors you need. Changing basis means describing the same vector in a different coordinate system.

**Why This Matters:**
"Before we can talk about eigenvectors, we need the vocabulary of vector spaces. Eigenvectors turn out to form a special basis — the 'natural coordinate system' for a transformation. But first, what IS a basis?"

**How to Explain:**

1. Start with what they know: "In 2D, the standard basis is `[1, 0]` and `[0, 1]` — the x and y axes. Every 2D vector is a combination of these two. `[3, 4]` is 3 times `[1, 0]` plus 4 times `[0, 1]`."

2. The key idea: "But the standard basis isn't the only option. You could use `[1, 1]` and `[1, -1]` as your basis. Every 2D vector can still be built from these two — you'd just need different coefficients."

3. Programming analogy: "A basis is like an encoding scheme. UTF-8 and UTF-16 can both represent the same text — they're different bases for the space of text. The text doesn't change, just how you describe it."

4. Show it:

```python
import numpy as np

# Standard basis
e1 = np.array([1, 0])
e2 = np.array([0, 1])

# The vector [3, 4] in standard coordinates
v = 3 * e1 + 4 * e2
print(v)  # [3, 4]

# Alternative basis
b1 = np.array([1, 1])
b2 = np.array([1, -1])

# The same vector [3, 4] in the new basis
# We need to find coefficients c1, c2 such that c1*b1 + c2*b2 = [3, 4]
# c1 + c2 = 3 and c1 - c2 = 4 → c1 = 3.5, c2 = -0.5
c1, c2 = 3.5, -0.5
print(c1 * b1 + c2 * b2)  # [3.0, 4.0] — same vector!
```

5. Key phrase: "The vector hasn't changed. It's still the same arrow in space. We just described it using different building blocks."

**What Makes a Valid Vector Space:**
"A vector space has two rules: (1) you can add any two vectors and the result is still in the space, and (2) you can scale any vector and the result is still in the space. R^2 (all 2D vectors) is a vector space. The set of vectors that lie on a single line through the origin is a vector space. The set of vectors in the first quadrant is NOT a vector space — scale by -1 and you leave the set."

**Basis and Dimension:**
"A basis is the smallest set of vectors that can build everything in the space. For R^2, you need exactly 2 basis vectors. For R^3, you need 3. That count IS the dimension."

"Requirements for a basis:
- The vectors must be linearly independent (no vector is a combination of the others)
- They must span the space (you can reach every vector by combining them)"

```python
import numpy as np

# This is a valid basis for R^2
b1 = np.array([1, 0])
b2 = np.array([0, 1])

# This is also a valid basis for R^2
b1 = np.array([2, 1])
b2 = np.array([-1, 3])

# This is NOT a basis — the vectors are parallel (linearly dependent)
b1 = np.array([1, 2])
b2 = np.array([2, 4])  # just 2 * b1
```

**Changing Basis:**
"Changing basis is one of the most important operations in linear algebra. It's what eigendecomposition actually does — it finds a basis where the matrix looks simple."

```python
import numpy as np

# Change-of-basis matrix: columns are the new basis vectors
P = np.array([[1, 1],
              [1, -1]])

# A vector in standard coordinates
v_standard = np.array([3, 4])

# To get the coordinates in the new basis, solve P @ v_new = v_standard
v_new_basis = np.linalg.solve(P, v_standard)
print(v_new_basis)  # [3.5, -0.5]

# Verify: go back to standard coordinates
print(P @ v_new_basis)  # [3., 4.]
```

**Common Misconceptions:**
- Misconception: "The standard basis is special or 'correct'" → Clarify: "It's just the most familiar. Other bases can be more useful for specific problems. Eigenvectors form a basis that makes a matrix look as simple as possible."
- Misconception: "Changing basis changes the vector" → Clarify: "The vector (the arrow in space) stays the same. Only its description (the coordinates) changes. Like translating a sentence to a different language — same meaning, different words."
- Misconception: "Any two vectors form a basis for R^2" → Clarify: "Only if they're linearly independent (not parallel). Two parallel vectors can only reach vectors along one line."

**Verification Questions:**
1. "What are the two requirements for a set of vectors to be a basis?"
2. Multiple choice: "How many basis vectors does R^3 need? A) 1 B) 2 C) 3 D) It depends on which vectors you pick"
3. "If I change from the standard basis to a different basis, does the actual vector change?"
4. "Is `{[1, 2], [3, 6]}` a valid basis for R^2? Why or why not?"

**Good answer indicators:**
- They understand that linearly independent + spanning = basis
- They can answer C (3 basis vectors for R^3 — that's the dimension)
- They know changing basis doesn't change the vector, just its coordinates
- They recognize `[3, 6] = 3 * [1, 2]` so those vectors are dependent and NOT a basis

**If they struggle:**
- Stay concrete: "In 2D, you need exactly 2 non-parallel arrows. That's it."
- Use the encoding analogy: "RGB and CMYK are different bases for color space. Same color, different numbers."
- Don't formalize linear independence with determinants yet — keep it geometric: "Can these vectors reach every point, or are they stuck on a line?"
- Defer the change-of-basis matrix until they're comfortable with the concept

**Exercise 1.1:**
"Determine whether each set of vectors is a valid basis for R^2. Explain why or why not:
- `{[1, 0], [0, 1]}`
- `{[1, 1], [-1, 1]}`
- `{[2, 4], [1, 2]}`
- `{[1, 0], [1, 1], [0, 1]}`"

**How to Guide Them:**
1. "For each set, ask: are the vectors linearly independent? Can they reach every point in R^2?"
2. If stuck: "Check if any vector is a scalar multiple of another. If so, they're dependent."
3. For the last one: "How many vectors do you need for a basis of R^2? What happens if you have too many?"

**Solution:**
- `{[1, 0], [0, 1]}` — Valid. Standard basis. Independent, spans R^2.
- `{[1, 1], [-1, 1]}` — Valid. Independent (not parallel), spans R^2.
- `{[2, 4], [1, 2]}` — Not valid. `[2, 4] = 2 * [1, 2]` — they're parallel (dependent).
- `{[1, 0], [1, 1], [0, 1]}` — Not a basis. Three vectors for a 2D space — too many. A basis has exactly `dim` vectors.

**Exercise 1.2:**
"Given the basis `B = {[1, 1], [1, -1]}`, find the coordinates of `[5, 3]` in this basis. Verify by reconstructing the vector from those coordinates."

**How to Guide Them:**
1. "You need to find c1 and c2 such that c1 * [1, 1] + c2 * [1, -1] = [5, 3]"
2. "That gives you two equations: c1 + c2 = 5 and c1 - c2 = 3"
3. "Or use numpy: `np.linalg.solve(P, v)` where P has the basis vectors as columns"

**Solution:**
```python
import numpy as np

P = np.array([[1, 1],
              [1, -1]])
v = np.array([5, 3])

coords = np.linalg.solve(P, v)
print(coords)  # [4., 1.]

# Verify
print(4 * np.array([1, 1]) + 1 * np.array([1, -1]))  # [5, 3]
```

**After exercises, transition:**
"You now know what a basis is. Next we'll find the most interesting basis a matrix can have — its eigenvectors. These are the directions where the matrix does nothing but stretch."

---

### Section 2: Eigenvalues and Eigenvectors

**Core Concept to Teach:**
An eigenvector of a matrix A is a vector that doesn't change direction when multiplied by A — it only gets scaled. The scaling factor is the eigenvalue. Eigenvectors reveal the "natural axes" of a transformation.

**Why This is THE Key Concept:**
"This is the single most important idea in this route. Everything else — eigendecomposition, SVD, PCA — builds on it. If you get eigenvectors, you get everything."

**How to Explain:**

1. Start with the question: "Most vectors change direction when you multiply by a matrix. A rotation matrix rotates them. A shear matrix tilts them. But some special vectors ONLY get stretched or shrunk — they keep pointing in the same direction. Those are eigenvectors."

2. The formal definition: "If Av = λv, then v is an eigenvector and λ (lambda) is the eigenvalue. The matrix A, applied to v, gives back the same vector v scaled by λ."

3. Geometric meaning: "Imagine a matrix that stretches the x-axis by 3 and the y-axis by 2. The eigenvectors are along the x and y axes — those are the directions that don't rotate, they just stretch. The eigenvalues are 3 and 2 — the stretch factors."

4. Show it:

```python
import numpy as np

# A diagonal matrix: stretches x by 3, y by 2
A = np.array([[3, 0],
              [0, 2]])

# [1, 0] is an eigenvector with eigenvalue 3
v1 = np.array([1, 0])
print(A @ v1)  # [3, 0] = 3 * [1, 0] ✓

# [0, 1] is an eigenvector with eigenvalue 2
v2 = np.array([0, 1])
print(A @ v2)  # [0, 2] = 2 * [0, 1] ✓

# [1, 1] is NOT an eigenvector — it changes direction
v3 = np.array([1, 1])
print(A @ v3)  # [3, 2] — not a scalar multiple of [1, 1]
```

5. Now a non-diagonal example:

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]])

# Use numpy to find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)    # [3., 1.]
print("Eigenvectors (columns):")
print(eigenvectors)
# Column 0: [0.707, 0.707] ≈ [1, 1] normalized → eigenvalue 3
# Column 1: [-0.707, 0.707] ≈ [-1, 1] normalized → eigenvalue 1
```

6. Verify geometrically:

```python
# Verify: A @ eigenvector should equal eigenvalue * eigenvector
v = eigenvectors[:, 0]  # first eigenvector
lam = eigenvalues[0]    # first eigenvalue

print(A @ v)       # Should be...
print(lam * v)     # ...the same!
print(np.allclose(A @ v, lam * v))  # True
```

**Important numpy Note:**
"In numpy, `np.linalg.eig()` returns eigenvectors as COLUMNS of the matrix. So `eigenvectors[:, 0]` is the first eigenvector, `eigenvectors[:, 1]` is the second, and so on. This trips people up — the columns, not the rows."

**Visualizing Eigenvectors:**
"Let's see what these eigenvectors look like. Reference the `linear_transforms.py` visualization from linear-algebra-essentials — eigenvectors are the directions where the grid lines stay on the same line (they just stretch or shrink, they don't rotate)."

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1],
              [1, 2]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# Plot eigenvectors and their transformed versions
fig, ax = plt.subplots(figsize=(8, 8))

colors = ['blue', 'red']
for i in range(2):
    v = eigenvectors[:, i]
    Av = A @ v
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
              color=colors[i], alpha=0.5, linewidth=2,
              label=f'eigenvector {i+1}')
    ax.quiver(0, 0, Av[0], Av[1], angles='xy', scale_units='xy', scale=1,
              color=colors[i], linewidth=2, linestyle='dashed',
              label=f'A @ eigenvector {i+1} (λ={eigenvalues[i]:.1f})')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title("Eigenvectors: same direction, just scaled")
plt.show()
```

"Notice: the transformed eigenvectors point in exactly the same direction as the originals. They're just longer (or shorter). That's the defining property."

**Computing Eigenvalues by Hand (2x2 only):**
"For a 2x2 matrix, you can find eigenvalues by solving det(A - λI) = 0:"

```
A = [[a, b],
     [c, d]]

det(A - λI) = (a - λ)(d - λ) - bc = 0
λ² - (a + d)λ + (ad - bc) = 0
```

"That's a quadratic equation in λ. The solutions are the eigenvalues."

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]])

# By hand: det(A - λI) = (2-λ)(2-λ) - 1*1 = λ² - 4λ + 3 = (λ-3)(λ-1) = 0
# Eigenvalues: λ = 3 and λ = 1

# Verify with numpy
eigenvalues, _ = np.linalg.eig(A)
print(eigenvalues)  # [3., 1.] ✓
```

"For matrices larger than 2x2, always use numpy. The hand computation gets unwieldy fast and doesn't add insight."

**Common Misconceptions:**
- Misconception: "Eigenvalues and eigenvectors are just abstract math" → Clarify: "They reveal the fundamental structure of a transformation. The eigenvectors are the 'natural axes' — the directions the matrix prefers. The eigenvalues tell you how much it stretches along each axis."
- Misconception: "Every vector is an eigenvector" → Clarify: "Most vectors change direction when you multiply by a matrix. Eigenvectors are special — they're the rare vectors that only get scaled."
- Misconception: "Eigenvalues are always positive" → Clarify: "Eigenvalues can be negative (the eigenvector flips direction), zero (the eigenvector gets collapsed to the origin), or even complex (for rotation-like transformations)."
- Misconception: "A matrix has one eigenvector" → Clarify: "An n×n matrix has up to n linearly independent eigenvectors. A 2×2 matrix typically has 2, a 3×3 has up to 3, etc."

**Verification Questions:**
1. "In your own words, what does it mean for a vector to be an eigenvector?"
2. "If Av = 5v, what is the eigenvalue?" (Answer: 5)
3. Multiple choice: "An eigenvector with eigenvalue -1 will: A) Stay the same B) Flip to the opposite direction C) Rotate 90 degrees D) Shrink to zero"
4. "Can the zero vector be an eigenvector?" (Answer: No — by convention, eigenvectors must be nonzero)

**Good answer indicators:**
- They can explain that eigenvectors don't change direction, only get scaled
- They immediately say 5 for the eigenvalue question
- They can answer B (eigenvalue -1 means the vector reverses)
- They know the zero vector is excluded by definition

**If they struggle:**
- Use the diagonal matrix example first — eigenvectors are obviously along the axes
- "Forget the formula. Picture a matrix stretching space. Some directions just get longer or shorter without turning. Those are eigenvectors."
- Have them try random vectors with `A @ v` and check if the output is parallel to the input
- Stay in 2x2 — the geometric picture is clear and numpy handles the computation

**Exercise 2.1:**
"For each matrix, predict the eigenvectors intuitively (before computing). Then verify with `np.linalg.eig()`:
```python
A = np.array([[5, 0], [0, 2]])    # Diagonal
B = np.array([[0, -1], [1, 0]])   # What does this one do?
C = np.array([[1, 0], [0, 1]])    # Identity
```"

**How to Guide Them:**
1. For A: "It's diagonal. Which directions just get scaled?"
2. For B: "What transformation is this? (Hint: it's a 90-degree rotation.) Does any vector keep its direction under rotation?"
3. For C: "Every vector stays the same. What does that mean for eigenvectors?"

**Solution:**
```python
import numpy as np

A = np.array([[5, 0], [0, 2]])
eigenvalues_A, eigenvectors_A = np.linalg.eig(A)
print("A eigenvalues:", eigenvalues_A)  # [5, 2]
# Eigenvectors: along x and y axes

B = np.array([[0, -1], [1, 0]])
eigenvalues_B, eigenvectors_B = np.linalg.eig(B)
print("B eigenvalues:", eigenvalues_B)  # [0+1j, 0-1j] — complex!
# 90° rotation has no real eigenvectors — no direction stays the same

C = np.array([[1, 0], [0, 1]])
eigenvalues_C, eigenvectors_C = np.linalg.eig(C)
print("C eigenvalues:", eigenvalues_C)  # [1, 1]
# Every vector is an eigenvector of the identity (eigenvalue 1)
```

"The rotation matrix is the interesting case. It has complex eigenvalues because no real vector survives a 90-degree rotation without changing direction. Complex eigenvalues signal rotation in the transformation."

**Exercise 2.2:**
"Find the eigenvalues of `[[4, 2], [1, 3]]` by hand using the characteristic equation det(A - λI) = 0. Then verify with numpy."

**How to Guide Them:**
1. "Set up A - λI: subtract λ from the diagonal entries"
2. "Compute the determinant: (4-λ)(3-λ) - 2*1"
3. "Expand and solve the quadratic"
4. "Compare to `np.linalg.eig()`"

**Solution:**
```python
# By hand:
# det(A - λI) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = (λ-5)(λ-2) = 0
# Eigenvalues: λ = 5 and λ = 2

import numpy as np
A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)  # [5., 2.]
```

**After exercises, transition:**
"Now you can find eigenvectors and eigenvalues. Next: what happens when you use the eigenvectors as a basis? You get eigendecomposition — and it reveals what a matrix 'really' does."

---

### Section 3: Eigendecomposition

**Core Concept to Teach:**
Eigendecomposition writes a matrix as A = PDP^(-1), where P is the matrix of eigenvectors and D is a diagonal matrix of eigenvalues. This reveals that the matrix "really" just scales along the eigenvector directions.

**Why This Matters:**
"Eigendecomposition says: every (diagonalizable) matrix is secretly just stretching along its eigenvector directions. All the apparent complexity — shearing, skewing, whatever — is just stretching viewed from a rotated perspective."

**How to Explain:**

1. The setup: "We have a matrix A with eigenvectors and eigenvalues. Let's put the eigenvectors as columns of a matrix P, and the eigenvalues on the diagonal of a matrix D."

2. The punchline: "Then A = P @ D @ P^(-1). That's the decomposition."

3. What it means: "P changes to the eigenvector basis. D stretches along each eigenvector direction. P^(-1) changes back to the original basis. So A is: change basis → stretch → change back."

4. Show it:

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]])

# Get eigenvalues and eigenvectors
eigenvalues, P = np.linalg.eig(A)
D = np.diag(eigenvalues)

print("P (eigenvectors as columns):")
print(P)
print("\nD (eigenvalues on diagonal):")
print(D)

# Reconstruct A from the decomposition
A_reconstructed = P @ D @ np.linalg.inv(P)
print("\nA reconstructed from PDP^(-1):")
print(A_reconstructed)
print("\nMatches original?", np.allclose(A, A_reconstructed))  # True
```

5. The interpretation:

```python
# What eigendecomposition reveals:
# Step 1: P^(-1) changes to eigenvector coordinates
# Step 2: D stretches along each eigenvector axis
# Step 3: P changes back to standard coordinates

v = np.array([1, 0])

# The long way:
result_long = A @ v

# The decomposition way:
step1 = np.linalg.inv(P) @ v  # Change to eigenvector basis
step2 = D @ step1              # Stretch (just multiply by eigenvalues!)
step3 = P @ step2              # Change back

print("Direct:", result_long)
print("Decomposed:", step3)
print("Same?", np.allclose(result_long, step3))  # True
```

**Why This is Powerful:**
"In the eigenvector basis, the matrix is diagonal — it just scales each axis independently. Diagonal matrices are trivial to work with. That's the whole point: eigendecomposition finds the coordinate system where A becomes simple."

**Powers of Matrices:**
"Here's a practical payoff: if you need A^100, eigendecomposition makes it easy. A^n = P @ D^n @ P^(-1). And D^n just raises each diagonal entry to the nth power."

```python
import numpy as np

A = np.array([[2, 1],
              [1, 2]])

eigenvalues, P = np.linalg.eig(A)

# A^10 the hard way
A_power_hard = np.linalg.matrix_power(A, 10)

# A^10 the eigendecomposition way
D_power = np.diag(eigenvalues ** 10)
A_power_easy = P @ D_power @ np.linalg.inv(P)

print("Direct A^10:")
print(A_power_hard)
print("\nVia eigendecomposition:")
print(A_power_easy)
print("\nSame?", np.allclose(A_power_hard, A_power_easy))  # True
```

**When Eigendecomposition Doesn't Work:**
"Not every matrix can be eigendecomposed. It fails when:
- The matrix has complex eigenvalues (for real decomposition) — though you can still decompose in the complex numbers
- The matrix is 'defective' — it doesn't have enough linearly independent eigenvectors to form P

The classic example of a defective matrix:"

```python
import numpy as np

# This matrix has eigenvalue 1 with multiplicity 2, but only one eigenvector direction
A = np.array([[1, 1],
              [0, 1]])

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)      # [1., 1.]
print("Eigenvectors:")
print(eigenvectors)
# The eigenvector matrix may be singular — can't invert it cleanly
```

"Don't worry too much about defective matrices. They're rare in practice, and SVD (next section) handles everything."

**Common Misconceptions:**
- Misconception: "Eigendecomposition and SVD are the same thing" → Clarify: "Eigendecomposition requires a square matrix and may not exist. SVD works for any matrix, any shape. SVD is the generalization."
- Misconception: "The decomposition changes what the matrix does" → Clarify: "It doesn't change anything — it reveals the structure. A = PDP^(-1) is still the exact same transformation, just written in a way that shows its anatomy."
- Misconception: "You need to compute P^(-1) manually" → Clarify: "numpy does this for you. And for symmetric matrices (common in ML), P^(-1) = P^T, which is even simpler."

**Verification Questions:**
1. "In A = PDP^(-1), what does each matrix represent?"
2. "Why is eigendecomposition useful for computing A^n?"
3. Multiple choice: "If A has eigenvalues 3 and -1, what are the eigenvalues of A^4? A) 12 and -4 B) 81 and 1 C) 3^4 and (-1)^4 D) Both B and C"
4. Explanation: "Why does eigendecomposition make a matrix 'simpler'?"

**Good answer indicators:**
- They can explain P as change-of-basis, D as stretching, P^(-1) as changing back
- They understand that D^n is trivial (just raise diagonal entries to the nth power)
- They answer D (B and C are the same: 81 and 1)
- They explain that in the eigenvector basis, the matrix is diagonal

**If they struggle:**
- Draw the three-step process: "Change basis → stretch → change back. That's it."
- Use a concrete 2x2 example end-to-end
- Emphasize: "A diagonal matrix just scales each axis. Eigendecomposition finds the axes where A is diagonal."
- If the algebra is overwhelming, focus on the conceptual picture and let numpy handle the computation

**Exercise 3.1:**
"Decompose this matrix, then reconstruct it:
```python
A = np.array([[4, 2],
              [1, 3]])
```
Verify that PDP^(-1) equals A."

**How to Guide Them:**
1. "Use `np.linalg.eig()` to get eigenvalues and eigenvectors"
2. "Build D with `np.diag(eigenvalues)`"
3. "Compute P @ D @ np.linalg.inv(P) and check with `np.allclose()`"

**Exercise 3.2:**
"Using the eigendecomposition from Exercise 3.1, compute A^5 without using `np.linalg.matrix_power`. Verify your answer against the direct computation."

**How to Guide Them:**
1. "If A = PDP^(-1), what's A^5?"
2. "What's D^5 for a diagonal matrix?"
3. "Raise each eigenvalue to the 5th power, keep P and P^(-1) the same"

**Connection to ML:**
"Here's the ML payoff: PCA (Principal Component Analysis) is eigendecomposition of the covariance matrix. The eigenvectors of the covariance matrix are the 'principal components' — the directions of maximum variance in the data. The eigenvalues tell you how much variance each direction captures. Large eigenvalue = important direction. Small eigenvalue = noise you can drop."

---

### Section 4: SVD (Singular Value Decomposition)

**Core Concept to Teach:**
SVD decomposes ANY matrix (not just square ones) into three parts: A = UΣV^T. U gives the output directions, Σ gives the scaling factors (singular values), and V^T gives the input directions. Low-rank approximation keeps only the top k singular values, discarding the least important parts of the matrix.

**Why SVD is the Powerhouse:**
"Eigendecomposition has limitations — it only works for square, diagonalizable matrices. SVD always works. Any matrix, any shape. It's the Swiss Army knife of matrix decompositions. Image compression, recommendation systems, noise reduction, latent semantic analysis — they all use SVD."

**How to Explain:**

1. Start with the formula: "A = U @ Σ @ V^T. Three matrices multiplied together equal A."

2. What each matrix means:
   - "V^T (input directions): rotates/reflects the input space"
   - "Σ (scaling): stretches along each axis by the singular values"
   - "U (output directions): rotates/reflects the output space"

3. The interpretation: "Every matrix transformation can be broken into three steps: rotate the input, stretch, rotate the output. That's what SVD reveals."

4. Show it:

```python
import numpy as np

A = np.array([[3, 2],
              [2, 3]])

U, sigma, Vt = np.linalg.svd(A)

print("U (output directions):")
print(U)
print("\nSingular values:", sigma)
print("\nV^T (input directions):")
print(Vt)

# Reconstruct A
Sigma = np.diag(sigma)
A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed A:")
print(A_reconstructed)
print("Matches?", np.allclose(A, A_reconstructed))  # True
```

5. Non-square matrices:

```python
import numpy as np

# SVD works for non-square matrices too
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix

U, sigma, Vt = np.linalg.svd(A)
print("U shape:", U.shape)       # (2, 2)
print("sigma:", sigma)           # 2 singular values
print("Vt shape:", Vt.shape)     # (3, 3)

# To reconstruct, we need to handle the shape mismatch
# Create the full Sigma matrix (2x3 with singular values on diagonal)
Sigma = np.zeros((2, 3))
Sigma[:2, :2] = np.diag(sigma)

A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed A:")
print(A_reconstructed)
print("Matches?", np.allclose(A, A_reconstructed))  # True
```

**SVD vs. Eigendecomposition:**
"For symmetric matrices, SVD and eigendecomposition give the same information (U = V = eigenvectors, singular values = |eigenvalues|). But SVD is more general:
- Works for any matrix shape (m×n, not just n×n)
- Always exists (no 'defective' cases)
- Singular values are always real and non-negative"

**Low-Rank Approximation — The Practical Payoff:**
"Here's where SVD becomes magical. The singular values are sorted from largest to smallest. The largest singular values capture the most important patterns. The smallest ones are noise or fine detail. If you keep only the top k singular values (and set the rest to zero), you get the best rank-k approximation of the original matrix."

```python
import numpy as np

A = np.array([[3, 2, 2],
              [2, 3, -2]])

U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

print("Singular values:", sigma)
# The first singular value is much larger than the second

# Rank-1 approximation: keep only the first singular value
A_rank1 = sigma[0] * np.outer(U[:, 0], Vt[0, :])
print("\nOriginal A:")
print(A)
print("\nRank-1 approximation:")
print(np.round(A_rank1, 2))
print("\nDifference (what we lost):")
print(np.round(A - A_rank1, 2))
```

"If the top singular value is 10 and the next is 0.5, the rank-1 approximation captures almost all of the matrix's structure. This is why image compression with SVD works — most pixels in an image are highly correlated, so a few singular values capture most of the information."

**Common Misconceptions:**
- Misconception: "You need all singular values for a useful result" → Clarify: "Often the top few capture 90%+ of the information. That's the whole point — SVD separates signal from noise."
- Misconception: "SVD and eigendecomposition are the same" → Clarify: "SVD works for any matrix (including non-square), always exists, and always has real non-negative singular values. Eigendecomposition is more limited."
- Misconception: "Low-rank approximation is a hack or lossy shortcut" → Clarify: "The Eckart-Young theorem proves that the SVD truncation is the BEST possible rank-k approximation in terms of Frobenius norm. You can't do better."

**Verification Questions:**
1. "In A = UΣV^T, what does each matrix represent?"
2. Multiple choice: "SVD works for: A) Only square matrices B) Only symmetric matrices C) Any matrix D) Only invertible matrices"
3. "If a 1000×1000 matrix has singular values [100, 50, 2, 0.1, 0.01, ...], how many singular values would you keep for a good approximation?"
4. "What's the relationship between SVD and eigendecomposition for symmetric matrices?"

**Good answer indicators:**
- They can describe U (output directions), Σ (scaling), V^T (input directions)
- They answer C (SVD works for any matrix)
- They suggest keeping 2-3 singular values (the first two capture most of the information)
- They know that for symmetric matrices, SVD reduces to eigendecomposition

**If they struggle:**
- Simplify: "SVD breaks any transformation into rotate → stretch → rotate. Three simple steps."
- Focus on the singular values first — they're the most intuitive part (bigger = more important)
- Use a 2×2 symmetric matrix where SVD and eigendecomposition are the same, then show a non-square example
- Don't worry about the full_matrices parameter or shape mismatches for non-square matrices — numpy handles it

**Exercise 4.1:**
"Compute the SVD of this matrix:
```python
A = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 0.01]])
```
Look at the singular values. What does this matrix 'mostly' do? What would a rank-2 approximation lose?"

**How to Guide Them:**
1. "Run `np.linalg.svd(A)`. What are the singular values?"
2. "The singular values tell you the importance of each direction. Which directions matter?"
3. "If you drop the smallest singular value (0.01), what information do you lose?"

**Solution:**
```python
import numpy as np

A = np.array([[1, 0, 0],
              [0, 2, 0],
              [0, 0, 0.01]])

U, sigma, Vt = np.linalg.svd(A)
print("Singular values:", sigma)  # [2., 1., 0.01]

# The matrix mostly stretches the first two axes.
# The third axis is barely scaled (0.01).
# A rank-2 approximation loses almost nothing.

# Rank-2 approximation
A_rank2 = np.zeros_like(A)
for i in range(2):
    A_rank2 += sigma[i] * np.outer(U[:, i], Vt[i, :])

print("\nRank-2 approximation error:", np.linalg.norm(A - A_rank2))  # 0.01
```

**Exercise 4.2:**
"Create a random 5×3 matrix, compute its SVD, and reconstruct it. Verify the reconstruction matches the original. Then create a rank-1 approximation and measure the error."

**How to Guide Them:**
1. "`np.random.randn(5, 3)` for a random matrix"
2. "`np.linalg.svd(A, full_matrices=False)` is easier for non-square matrices"
3. "For rank-1: `sigma[0] * np.outer(U[:, 0], Vt[0, :])`"
4. "Error: `np.linalg.norm(A - A_rank1)` (Frobenius norm)"

**After exercises, transition:**
"You've seen SVD decompose matrices and approximate them. Now let's use it for something real — compressing an image."

---

## Practice Project

**Project Introduction:**
"Let's put SVD to work. You'll load a grayscale image, treat it as a matrix of pixel values, compute the SVD, and reconstruct the image using fewer and fewer singular values. You'll see the quality degrade as you use less data — and you'll be surprised how few singular values you need for a recognizable image."

**Requirements:**
Present one at a time:
1. "Load a grayscale image as a numpy matrix (or convert a color image to grayscale)"
2. "Compute the SVD of the image matrix"
3. "Reconstruct the image using only the top k singular values, for several values of k (e.g., 1, 5, 10, 20, 50, 100)"
4. "Display the original and reconstructed images side by side"
5. "Plot the compression ratio vs. image quality (e.g., the reconstruction error or visual comparison)"

**Scaffolding Strategy:**
1. **If they want to try alone**: Let them work. Hint: "The image is just a matrix. SVD it."
2. **If they want guidance**: Build it step by step together
3. **If they're unsure**: Start with loading the image and computing the SVD

**Starter Code (if they want a scaffold):**
```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load a grayscale image
# Option 1: Load your own image
# img = Image.open('photo.jpg').convert('L')

# Option 2: Generate a test image with structure
np.random.seed(42)
# Create an image with clear structure (gradients and shapes)
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
A = (128 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y) + 128).astype(np.float64)

# Compute SVD
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
print(f"Image shape: {A.shape}")
print(f"Number of singular values: {len(sigma)}")
print(f"Top 10 singular values: {sigma[:10].round(1)}")

def reconstruct(U, sigma, Vt, k):
    """Reconstruct matrix using top k singular values."""
    return U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

# Show reconstructions at various ranks
ks = [1, 5, 10, 20, 50, 100]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, k in zip(axes.flat, ks):
    A_approx = reconstruct(U, sigma, Vt, k)
    ax.imshow(A_approx, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f'k={k} singular values')
    ax.axis('off')

plt.suptitle('SVD Image Compression')
plt.tight_layout()
plt.show()

# Plot singular value magnitudes
plt.figure(figsize=(10, 5))
plt.plot(sigma, 'b-')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()

# Compression ratio
for k in ks:
    original_size = A.shape[0] * A.shape[1]
    compressed_size = k * (A.shape[0] + A.shape[1] + 1)
    ratio = compressed_size / original_size
    error = np.linalg.norm(A - reconstruct(U, sigma, Vt, k)) / np.linalg.norm(A)
    print(f"k={k:3d}: compression ratio={ratio:.3f}, relative error={error:.4f}")
```

**Checkpoints During Project:**
- After loading the image: "Can you display the original? What's the matrix shape?"
- After computing SVD: "How many singular values are there? What's the largest? The smallest?"
- After first reconstruction: "Try k=1. Can you see any structure?"
- After multiple reconstructions: "At what k does the image start looking 'good enough'?"
- After compression analysis: "What's the tradeoff? How much data do you save at k=20 vs k=100?"

**Discussion Points:**
- "Look at the singular value spectrum (the plot). Why does it drop off so quickly?"
- "If most singular values are tiny, what does that tell you about the image?"
- "How would this work differently for a purely random image (no structure)?"
- "Where do you think this technique is used in practice?"

**Code Review Approach:**
When reviewing their work:
1. Check they understand what the matrix represents: "Each element is a pixel brightness value"
2. Ask about the singular value spectrum: "Why do the singular values drop off?"
3. Connect to the theory: "The rank-k approximation keeps the k most important 'directions' in pixel space"
4. Push the ML connection: "This is exactly what PCA does to high-dimensional data — keeps the important dimensions, drops the noise"

**If They Get Stuck:**
- "Start with just loading and displaying the image. Then compute the SVD and print the singular values."
- "The reconstruction formula is `U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]` — slice the matrices to keep only the first k components"
- If matplotlib is confusing: "Focus on the numpy part first. We can add plots after."

**Extension Ideas if They Finish Early:**
- "Try it on a real photo. How does the singular value spectrum differ from a synthetic image?"
- "Color images have 3 channels (R, G, B). Apply SVD to each channel separately and recombine."
- "Compare the compression ratio to JPEG at similar visual quality. Which is better?"
- "What happens to the singular values of a matrix of pure random noise? (They're all roughly the same — no structure to exploit.)"

---

## Wrap-Up and Next Steps

**Review Key Takeaways:**
"Let's review what you learned in this side quest:"
- A vector space is a set of vectors closed under addition and scaling. A basis is the minimum set that spans the space.
- Eigenvectors are the special directions that a matrix only scales, not rotates. Eigenvalues are the scale factors.
- Eigendecomposition (A = PDP^(-1)) reveals that a matrix is secretly just stretching along its eigenvector directions.
- SVD (A = UΣV^T) works for any matrix and decomposes it into rotate → stretch → rotate.
- Low-rank approximation with SVD keeps the most important patterns and discards noise — the basis of image compression, PCA, and recommendation systems.

**Ask them to explain key concepts:**
"Can you explain in your own words what an eigenvector is?"
"What's the relationship between eigendecomposition and SVD?"
(These are the two load-bearing ideas — if they can articulate them, the route succeeded.)

**Assess Confidence:**
"On a scale of 1-10, how confident do you feel with eigenvalues, eigenvectors, and SVD?"

**Respond based on answer:**
- 1-4: "These are genuinely deep concepts. The geometric intuition takes time to internalize. Revisit the 2x2 examples and the image compression project — they ground the abstractions."
- 5-7: "Good! You have the conceptual foundation. When you encounter PCA or SVD in ML code, you'll now know what's happening under the hood."
- 8-10: "Excellent. You now have the tools to understand a huge swath of ML techniques at a deeper level than most practitioners."

**ML Connections to Reinforce:**
- "PCA = eigendecomposition of the covariance matrix. The principal components are the eigenvectors. The eigenvalues tell you how much variance each direction captures."
- "Latent spaces in generative models (VAE, diffusion models): the latent dimensions are analogous to the SVD directions — the most important 'factors' that describe the data."
- "Recommendation systems: SVD of the user-item matrix reveals latent factors (e.g., genre preferences) that explain the ratings."
- "Attention weight matrices in transformers: their eigenstructure reveals how information flows through the model."

**Suggest Next Steps:**
Based on their progress and interests:
- "To solidify: Try SVD on different types of images and data matrices. Build intuition for when low-rank approximation works well and when it doesn't."
- "For ML application: Look into how scikit-learn's PCA works — it's eigendecomposition of the covariance matrix. Now you understand every step."
- "For deeper theory: Look into the spectral theorem (symmetric matrices always have real eigenvalues and orthogonal eigenvectors) — it's why PCA works so cleanly."
- "Return to the main path: These concepts will deepen your understanding of neural-network-foundations and training-and-backprop when you get there."

**Encourage Questions:**
"Do you have any questions about anything we covered?"
"Is there a concept that felt shaky or that you want more practice with?"
"Any connections to your own work that you want to explore?"

---

## Adaptive Teaching Strategies

### If Learner is Struggling

**Signs:**
- Can't articulate what an eigenvector is
- Confused by the decomposition formulas (PDP^(-1), UΣV^T)
- Mixing up eigendecomposition and SVD
- Getting lost in the matrix algebra

**Strategies:**
- Return to 2x2 diagonal matrices — eigenvectors are obvious (along the axes)
- Focus on the geometric picture: "An eigenvector is a direction that doesn't rotate"
- Skip the hand computation of eigenvalues — let numpy do it and focus on interpretation
- For decomposition: "Change basis → stretch → change back. Three steps."
- For SVD: "Rotate → stretch → rotate. Three steps."
- Use the image compression project as motivation — it's concrete and visual
- Don't formalize when they need intuition
- Stay on one concept until it clicks before moving to the next

### If Learner is Excelling

**Signs:**
- Completes exercises quickly and asks about generalization
- Connects concepts to ML applications unprompted
- Wants to understand the proofs or deeper theory
- Experiments beyond the exercises

**Strategies:**
- Discuss the spectral theorem for symmetric matrices
- Introduce the connection between SVD and the pseudoinverse (least squares)
- Explore eigenvalues of graph Laplacians (spectral graph theory)
- Discuss condition number (ratio of largest to smallest singular value) and numerical stability
- Show how PCA is implemented from scratch using eigendecomposition
- Challenge: "Can you implement a basic recommendation system using SVD?"
- Discuss the Eckart-Young theorem: SVD truncation is the optimal low-rank approximation

### If Learner Seems Disengaged

**Signs:**
- Short responses, not asking questions
- Rushing through without engaging with the geometry
- "I get it" without being able to explain

**Strategies:**
- Remind them this is a side quest — there's no obligation to finish
- Jump to the image compression project — it's the most engaging part
- Ask what ML application they're most interested in and tailor examples
- Reduce theory, increase experimentation: "Try changing this matrix and see what happens to the eigenvectors"
- Show something surprising: random matrices have universal eigenvalue distributions (semicircle law)

### Different Learning Styles

**Visual learners:**
- Plot eigenvectors before and after transformation — show that they stay on the same line
- Use the `linear_transforms.py` script to show how the grid warps along eigenvector directions
- Image compression project is ideal — they can see SVD working

**Hands-on learners:**
- Jump to code quickly, explain after they see results
- "Compute the SVD, look at the numbers, then I'll explain what they mean"
- Let them experiment with different matrices and discover patterns

**Conceptual learners:**
- Spend time on why eigendecomposition works (the algebra behind A = PDP^(-1))
- Discuss when eigendecomposition fails and why SVD always succeeds
- Explain the Eckart-Young theorem (optimality of SVD truncation)
- They may want to work through the characteristic equation derivation

**Example-driven learners:**
- Start every concept with a concrete 2x2 example
- Show the numpy code first, explain the math after
- Build from specific cases to general principles

---

## Troubleshooting Common Issues

### numpy Not Installed
```bash
pip install numpy matplotlib Pillow
```
Verify: `python -c "import numpy; import matplotlib; from PIL import Image; print('OK')"`

### matplotlib Not Showing Plots
- On macOS, they may need a backend: `pip install pyobjc` or use `matplotlib.use('TkAgg')`
- In a remote session (SSH): plots won't display. Use `plt.savefig('output.png')` instead of `plt.show()`
- In VS Code: plots may appear in a separate window or inline depending on settings

### Complex Eigenvalues
"If `np.linalg.eig()` returns complex numbers, the matrix has rotation-like behavior. For real matrices, complex eigenvalues always come in conjugate pairs (a+bi and a-bi). This means the transformation involves rotation, and there are no real directions that stay fixed."

Reassure them: "Complex eigenvalues aren't an error — they're telling you something about the geometry. Rotation matrices, for example, always have complex eigenvalues (except 180°)."

### Confusion About numpy SVD Output
"`np.linalg.svd()` returns `(U, sigma, Vt)` where:
- `sigma` is a 1D array of singular values, NOT a matrix
- You need `np.diag(sigma)` to make it a matrix
- For non-square matrices, use `full_matrices=False` to get compact SVD
- `Vt` is V-transpose, not V — numpy already transposes it for you"

This is the #1 source of shape-mismatch errors. Address it proactively.

### Reconstruction Doesn't Match
"If `np.allclose(A, U @ np.diag(sigma) @ Vt)` returns False for a non-square matrix, the issue is almost always the shape of the Sigma matrix. Use `full_matrices=False` in `np.linalg.svd()` for the compact form, which avoids the shape-padding problem."

### Concept-Specific Confusion

**If confused about vector spaces and basis:**
- "A basis is just a coordinate system. Standard basis: x and y axes. But you can choose any two non-parallel directions."
- Use the encoding analogy: "Same data, different representation"
- Stay in R^2 — don't generalize to abstract vector spaces

**If confused about eigenvectors:**
- Return to diagonal matrices where eigenvectors are obvious
- "Apply the matrix to random vectors. Most change direction. Eigenvectors are the ones that don't."
- Have them test vectors with `A @ v` and check if the result is parallel to v

**If confused about eigendecomposition:**
- "It's just three steps: change coordinates, stretch, change back."
- Work through a complete 2x2 example from start to finish
- Show that D is diagonal — "In the eigenvector basis, the matrix is trivially simple"

**If confused about SVD:**
- "It's the same idea as eigendecomposition, but it works for everything."
- Focus on singular values first — "Big singular value = important direction. Small = ignorable."
- Jump to the image compression example — seeing is believing

---

## Teaching Notes

**Key Emphasis Points:**
- Eigenvectors are the "natural axes" of a transformation. This geometric framing is more important than the algebra.
- The hierarchy is: eigenvectors → eigendecomposition → SVD. Each builds on the previous.
- SVD is the practical workhorse. Eigendecomposition is the conceptual foundation. Teach both, but SVD is what they'll use.
- The image compression project is the payoff. Make sure they get there with enough time to explore.

**Pacing Guidance:**
- Section 1 (vector spaces) can be relatively quick if they have good intuition from linear-algebra-essentials. Don't belabor it — it's scaffolding for eigenvectors.
- Section 2 (eigenvalues/eigenvectors) is the conceptual heart. Take your time. The geometric meaning must click before moving on.
- Section 3 (eigendecomposition) can be moderate pace if Section 2 was solid. The "change basis → stretch → change back" framing should land cleanly.
- Section 4 (SVD) builds directly on eigendecomposition. Spend time on low-rank approximation — it's the practical insight.
- Allow plenty of time for the practice project — it's where everything comes together.

**Side Quest Framing:**
Remind the learner at the beginning and end: this is optional enrichment. Nothing on the main learning path requires this material. They're here because they're curious, and that's valuable. If they're feeling overwhelmed, it's fine to stop after eigenvectors and come back later.

**Success Indicators:**
You'll know they've got it when they:
- Can explain what an eigenvector is in plain language (a direction the matrix only scales)
- Can look at SVD singular values and judge which ones matter
- Understand that low-rank approximation keeps signal and drops noise
- Connect PCA to eigendecomposition of the covariance matrix
- Ask questions like "what would the eigenvalues of X look like?" (shows they're thinking structurally)
- Enjoy the image compression project and experiment beyond the requirements

**Most Common Confusion Points:**
1. **What eigenvectors mean geometrically**: The central conceptual hurdle. Invest time here.
2. **numpy SVD output shapes**: The sigma-is-1D-not-a-matrix issue. Address proactively.
3. **Eigendecomposition vs. SVD**: When to use which. SVD always works; eigen is for square matrices with enough eigenvectors.
4. **Why low-rank approximation works**: "Most real-world data has structure, so the top singular values capture most of the information."
5. **Complex eigenvalues**: Confusing but not an error. They signal rotation.

**Teaching Philosophy:**
- This route rewards curiosity. The learner chose to be here — honor that by making the material fascinating, not just correct.
- Geometric intuition first, algebra second, numpy always. The code is the verification, not the lesson.
- Connect every concept to ML applications. The learner wants to understand why PCA works, not just how to call `sklearn.decomposition.PCA`.
- If they're struggling, zoom out. The big picture (matrices have natural axes, SVD finds them) is more valuable than any formula.
