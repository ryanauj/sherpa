# ML Visualizations

Interactive visualization scripts for the ML/AI learning path. Each script is self-contained — run it with `python script.py` and a matplotlib window will appear.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts by Route

### Linear Algebra Essentials
- `vectors.py` — Vector addition, subtraction, and scaling in 2D
- `matrices.py` — Matrix-vector multiplication and its geometric effect
- `linear_transforms.py` — Apply a matrix to a grid and watch it warp

### Calculus for ML
- `derivatives.py` — Tangent lines, slopes, and numerical derivatives
- `gradient_descent.py` — Animated gradient descent on a 2D contour plot

### Neural Network Foundations
- `perceptron.py` — A single perceptron's decision boundary
- `activation_functions.py` — Side-by-side plots of sigmoid, ReLU, tanh
- `decision_boundaries.py` — How layers of transformations warp 2D space to create decision boundaries

### Training and Backpropagation
- `loss_surfaces.py` — 3D loss surface and contour plot for a small network
- `backprop_trace.py` — Step-by-step backpropagation animation through a small network

### LLM Foundations
- `embeddings.py` — Word vectors in 2D space, showing similarity relationships
- `attention.py` — Attention weight heatmap between words in a sentence

## Design Principles

- **Self-contained**: Each script runs standalone with `python script.py`
- **Configurable**: Constants at the top of each file control behavior (try changing them!)
- **Well-commented**: Learners will read and modify these scripts as exercises
- **Minimal dependencies**: Only numpy and matplotlib — no ML frameworks
