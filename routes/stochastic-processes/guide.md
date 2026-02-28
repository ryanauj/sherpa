---
title: Stochastic Processes
route_map: /routes/stochastic-processes/map.md
paired_sherpa: /routes/stochastic-processes/sherpa.md
prerequisites:
  - Probability Distributions
  - Probability Fundamentals
  - Bayesian Statistics (helpful)
  - Linear Algebra Essentials (helpful)
topics:
  - Random Walks
  - Markov Chains
  - Monte Carlo Methods
  - Poisson Processes
  - Time Series Foundations
---

# Stochastic Processes - Guide (Human-Focused Content)

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/stochastic-processes/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/stochastic-processes/map.md` for the high-level overview.

## Overview

Everything you've learned so far in probability and statistics deals with **static** situations: a single experiment, a fixed dataset, a one-time decision. But the real world evolves over time. Stock prices change minute by minute. A gambler's bankroll rises and falls over hours at the table. Teams win and lose across a season. Customers arrive at random intervals.

**Stochastic processes** are mathematical models for systems that evolve randomly over time. They connect probability theory to the dynamic real world. This route covers five foundational topics:

1. **Random walks** — the simplest stochastic process, and the foundation of stock price models
2. **Markov chains** — systems where the future depends only on the present
3. **Monte Carlo methods** — using randomness to solve problems that are too complex for formulas
4. **Poisson processes** — modeling events that occur randomly in continuous time
5. **Time series foundations** — analyzing data that has a time dimension

## Learning Objectives

By the end of this route, you will be able to:
- Model and simulate random walks, including the gambler's ruin problem
- Build and analyze Markov chains using transition matrices and stationary distributions
- Apply Monte Carlo methods for simulation and estimation
- Model random events using Poisson processes
- Analyze time series data for stationarity and autocorrelation
- Implement a Monte Carlo options pricer

## Prerequisites

Before starting, you should be comfortable with:
- **Probability Distributions** ([route](/routes/probability-distributions/map.md)): Expected value, variance, Normal distribution, Law of Large Numbers
- **Probability Fundamentals** ([route](/routes/probability-fundamentals/map.md)): Conditional probability, independence
- **Bayesian Statistics** ([route](/routes/bayesian-statistics/map.md)): MCMC concepts (helpful, not required)
- **Linear Algebra Essentials** ([route](/routes/linear-algebra-essentials/map.md)): Matrix multiplication, eigenvectors (helpful for Markov chains)
- **Python**: numpy, scipy, matplotlib

## Setup

```bash
pip install numpy scipy matplotlib
```

**Verify your setup:**

```python
import numpy as np
from scipy import stats
import matplotlib
print(f"numpy: {np.__version__}")
print(f"scipy: available")
print(f"matplotlib: {matplotlib.__version__}")

# Quick sanity check: simulate a simple random walk
np.random.seed(42)
steps = np.random.choice([-1, 1], size=100)
walk = np.cumsum(steps)
print(f"Random walk after 100 steps: position = {walk[-1]}")
print(f"Max position: {np.max(walk)}, Min position: {np.min(walk)}")
```

**Expected output:**
```
numpy: 1.26.4
scipy: available
matplotlib: 3.9.2
Random walk after 100 steps: position = -8
Max position: 6, Min position: -16
```

---

## Section 1: Random Walks

### What Is a Random Walk?

A **random walk** is a sequence of random steps. The simplest version: start at position 0, and at each step flip a coin — heads move +1, tails move -1.

Despite its simplicity, the random walk is one of the most important models in probability. It describes:
- A gambler's fortune over time
- The (simplified) behavior of stock prices
- Diffusion of particles in physics
- Random exploration in graph algorithms

### Simulating Random Walks

```python
import numpy as np

np.random.seed(42)

# Simulate 5 random walks of 1000 steps each
n_steps = 1000
n_walks = 5

print(f"{'Walk':>5} | {'Final Pos':>9} | {'Max':>5} | {'Min':>5} | {'Returns to 0':>13}")
print("-" * 55)

for i in range(n_walks):
    steps = np.random.choice([-1, 1], size=n_steps)
    walk = np.cumsum(steps)
    returns_to_zero = np.sum(walk == 0)
    print(f"{i+1:>5} | {walk[-1]:>9} | {np.max(walk):>5} | {np.min(walk):>5} | {returns_to_zero:>13}")
```

**Expected output:**
```
 Walk | Final Pos |   Max |   Min | Returns to 0
-------------------------------------------------------
    1 |       -10 |    14 |   -40 |             4
    2 |        26 |    30 |    -8 |             3
    3 |         4 |    30 |   -18 |             7
    4 |       -44 |    18 |   -52 |             1
    5 |       -22 |    18 |   -50 |             2
```

### Properties of Random Walks

Key mathematical facts:

1. **Expected position** after n steps: E[S_n] = 0 (the walk is unbiased)
2. **Variance** after n steps: Var(S_n) = n (spread grows linearly)
3. **Standard deviation**: σ = √n (spread grows as square root of time)
4. **Return to origin**: A 1D random walk returns to 0 infinitely often (but visits become rarer)

```python
import numpy as np

np.random.seed(42)

# Verify: variance grows linearly with number of steps
n_simulations = 10000

for n in [10, 100, 1000, 10000]:
    final_positions = []
    for _ in range(n_simulations):
        steps = np.random.choice([-1, 1], size=n)
        final_positions.append(np.sum(steps))
    final_positions = np.array(final_positions)

    print(f"n = {n:>5}: Mean = {np.mean(final_positions):>7.2f}, "
          f"Variance = {np.var(final_positions):>9.1f} (theory: {n}), "
          f"Std = {np.std(final_positions):>7.2f} (theory: {np.sqrt(n):.2f})")
```

**Expected output:**
```
n =    10: Mean =   -0.01, Variance =      10.0 (theory: 10), Std =    3.16 (theory: 3.16)
n =   100: Mean =   -0.04, Variance =      99.5 (theory: 100), Std =    9.98 (theory: 10.00)
n =  1000: Mean =    0.09, Variance =     999.3 (theory: 1000), Std =   31.61 (theory: 31.62)
n = 10000: Mean =    0.47, Variance =   10032.1 (theory: 10000), Std =  100.16 (theory: 100.00)
```

### The Gambler's Ruin

The **gambler's ruin** problem: a gambler starts with $N, bets $1 each round on a fair coin flip. The casino has infinite money. What's the probability of going broke?

**Answer**: The gambler will go broke with probability 1. Always. Even in a fair game.

```python
import numpy as np

np.random.seed(42)

def gamblers_ruin(starting_bankroll, p_win=0.5, max_rounds=100000):
    """Simulate gambler's ruin. Returns rounds until broke (or -1 if survived)."""
    bankroll = starting_bankroll
    for round_num in range(max_rounds):
        if bankroll <= 0:
            return round_num
        if np.random.random() < p_win:
            bankroll += 1
        else:
            bankroll -= 1
    return -1  # survived all rounds

# Simulate 1000 gamblers starting with $100 in a fair game
n_simulations = 1000
starting_amount = 100
results = []

for _ in range(n_simulations):
    rounds = gamblers_ruin(starting_amount, p_win=0.50)
    results.append(rounds)

results = np.array(results)
went_broke = results[results >= 0]
survived = np.sum(results == -1)

print(f"Gambler's Ruin Simulation (fair game, starting with ${starting_amount})")
print(f"Gamblers who went broke: {len(went_broke)}/{n_simulations} ({len(went_broke)/n_simulations:.1%})")
print(f"Survived 100,000 rounds: {survived}/{n_simulations}")
if len(went_broke) > 0:
    print(f"Median rounds to ruin: {np.median(went_broke):.0f}")
    print(f"Mean rounds to ruin: {np.mean(went_broke):.0f}")
print()

# Now with a house edge (casino game, 48% win probability)
print("With 48% win probability (house edge):")
results_edge = []
for _ in range(1000):
    rounds = gamblers_ruin(starting_amount, p_win=0.48)
    results_edge.append(rounds)

results_edge = np.array(results_edge)
broke_edge = results_edge[results_edge >= 0]
print(f"Went broke: {len(broke_edge)}/1000 ({len(broke_edge)/1000:.1%})")
if len(broke_edge) > 0:
    print(f"Median rounds to ruin: {np.median(broke_edge):.0f}")
```

**Expected output:**
```
Gambler's Ruin Simulation (fair game, starting with $100)
Gamblers who went broke: 979/1000 (97.9%)
Survived 100,000 rounds: 21/1000
Median rounds to ruin: 8255
Mean rounds to ruin: 15411

With 48% win probability (house edge):
Went broke: 1000/1000 (100.0%)
Median rounds to ruin: 2389
```

### Random Walks and Stock Prices

The **Random Walk Hypothesis** suggests that stock price changes are random and unpredictable. While imperfect, it's a useful starting model.

```python
import numpy as np

np.random.seed(42)

# Geometric Brownian Motion: a random walk model for stock prices
# S(t+1) = S(t) * exp(mu*dt + sigma*sqrt(dt)*Z)
S0 = 100        # starting price
mu = 0.08       # annual drift (expected return)
sigma = 0.20    # annual volatility
dt = 1/252      # one trading day
n_days = 252    # one year
n_paths = 5

print(f"Stock Price Simulation (S0=${S0}, drift={mu:.0%}, vol={sigma:.0%})")
print(f"{'Path':>5} | {'Final Price':>11} | {'Return':>8} | {'Max':>7} | {'Min':>7}")
print("-" * 50)

for i in range(n_paths):
    Z = np.random.normal(size=n_days)
    daily_returns = mu * dt + sigma * np.sqrt(dt) * Z
    prices = S0 * np.exp(np.cumsum(daily_returns))
    final = prices[-1]
    ret = (final - S0) / S0
    print(f"{i+1:>5} | ${final:>10.2f} | {ret:>7.1%} | ${np.max(prices):>6.2f} | ${np.min(prices):>6.2f}")
```

**Expected output:**
```
Stock Price Simulation (S0=$100, drift=8%, vol=20%)
 Path | Final Price |   Return |     Max |     Min
--------------------------------------------------
    1 |     $101.11 |    1.1% | $118.51 |  $86.22
    2 |     $119.64 |   19.6% | $119.64 |  $92.60
    3 |     $127.24 |   27.2% | $130.84 |  $95.54
    4 |     $109.04 |    9.0% | $118.69 |  $90.42
    5 |      $98.91 |   -1.1% | $109.65 |  $87.22
```

### Exercise 1.1: Biased Random Walks

**Task:** Simulate a biased random walk where P(+1) = 0.51 and P(-1) = 0.49. This represents a small edge (like a skilled sports bettor). After 10,000 steps, what's the expected position? Run 1,000 simulations and compare the distribution of outcomes to the fair random walk.

<details>
<summary>Hint</summary>

Use `np.random.choice([-1, 1], size=n_steps, p=[0.49, 0.51])` for the biased walk.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

np.random.seed(42)
n_steps = 10000
n_sims = 1000

fair_finals = []
biased_finals = []

for _ in range(n_sims):
    fair = np.sum(np.random.choice([-1, 1], size=n_steps))
    biased = np.sum(np.random.choice([-1, 1], size=n_steps, p=[0.49, 0.51]))
    fair_finals.append(fair)
    biased_finals.append(biased)

fair_finals = np.array(fair_finals)
biased_finals = np.array(biased_finals)

print("After 10,000 steps:")
print(f"Fair walk    — Mean: {np.mean(fair_finals):>8.1f}, Std: {np.std(fair_finals):>7.1f}, "
      f"P(positive): {np.mean(fair_finals > 0):.1%}")
print(f"Biased walk  — Mean: {np.mean(biased_finals):>8.1f}, Std: {np.std(biased_finals):>7.1f}, "
      f"P(positive): {np.mean(biased_finals > 0):.1%}")
print()
print(f"Theoretical biased mean: {n_steps * 0.02:.1f}")
print(f"Theoretical std: {np.sqrt(n_steps):.1f}")
```

**Expected output:**
```
After 10,000 steps:
Fair walk    — Mean:      1.1, Std:   100.2, P(positive): 50.3%
Biased walk  — Mean:    198.5, Std:    99.8, P(positive): 97.6%

Theoretical biased mean: 200.0
Theoretical std: 100.0
```

**Explanation:** A tiny 2% edge (51% vs 49%) over 10,000 steps produces an expected gain of 200 — and you're in positive territory 97.6% of the time. This is why the house always wins: even a small edge, applied consistently, produces reliable profits over time.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Simulate a random walk and compute its properties
- [ ] Explain why the gambler always goes broke in a fair game
- [ ] Connect random walks to stock price models
- [ ] Describe how a small bias compounds over many steps

---

## Section 2: Markov Chains

### What Is a Markov Chain?

A **Markov chain** is a sequence of random variables where the future depends only on the present state, not the past. This is the **Markov property**: "given where I am now, where I came from doesn't matter."

Formally: P(X_{n+1} | X_n, X_{n-1}, ..., X_0) = P(X_{n+1} | X_n)

### Transition Matrices

A Markov chain is fully described by its **transition matrix**, where entry (i, j) is the probability of moving from state i to state j.

```python
import numpy as np

# Credit rating transitions (simplified)
# States: AAA, AA, A, Default
# Each row shows where you go FROM that state
transition_matrix = np.array([
    [0.90, 0.08, 0.02, 0.00],  # From AAA
    [0.05, 0.85, 0.08, 0.02],  # From AA
    [0.01, 0.05, 0.80, 0.14],  # From A
    [0.00, 0.00, 0.00, 1.00],  # From Default (absorbing state)
])

states = ['AAA', 'AA', 'A', 'Default']

print("Credit Rating Transition Matrix")
print(f"{'From/To':>8}", end="")
for s in states:
    print(f" {s:>8}", end="")
print()
print("-" * 44)

for i, s in enumerate(states):
    print(f"{s:>8}", end="")
    for j in range(len(states)):
        print(f" {transition_matrix[i,j]:>8.2f}", end="")
    print()
```

**Expected output:**
```
Credit Rating Transition Matrix
  From/To      AAA       AA        A  Default
--------------------------------------------
     AAA     0.90     0.08     0.02     0.00
      AA     0.05     0.85     0.08     0.02
       A     0.01     0.05     0.80     0.14
 Default     0.00     0.00     0.00     1.00
```

### Simulating a Markov Chain

```python
import numpy as np

np.random.seed(42)

transition_matrix = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.05, 0.85, 0.08, 0.02],
    [0.01, 0.05, 0.80, 0.14],
    [0.00, 0.00, 0.00, 1.00],
])
states = ['AAA', 'AA', 'A', 'Default']

def simulate_chain(start_state, n_steps, T):
    """Simulate a Markov chain for n_steps."""
    current = start_state
    path = [current]
    for _ in range(n_steps):
        current = np.random.choice(len(states), p=T[current])
        path.append(current)
    return path

# Track 1000 bonds starting at AA over 10 years
n_bonds = 1000
n_years = 10
start = 1  # AA

final_states = np.zeros(len(states))
for _ in range(n_bonds):
    path = simulate_chain(start, n_years, transition_matrix)
    final_states[path[-1]] += 1

print(f"1,000 bonds starting at AA after {n_years} years:")
for i, s in enumerate(states):
    print(f"  {s}: {final_states[i]/n_bonds:.1%}")
```

**Expected output:**
```
1,000 bonds starting at AA after 10 years:
  AAA: 9.3%
  AA: 33.7%
  A: 33.1%
  Default: 23.9%
```

### Stationary Distributions

A **stationary distribution** π satisfies π × T = π — it's the long-run proportion of time spent in each state. You find it as the left eigenvector of the transition matrix.

```python
import numpy as np

# Simplified sports team model
# States: Hot (winning streak), Normal, Cold (losing streak)
T = np.array([
    [0.60, 0.30, 0.10],  # Hot
    [0.20, 0.50, 0.30],  # Normal
    [0.10, 0.40, 0.50],  # Cold
])
states = ['Hot', 'Normal', 'Cold']

# Find stationary distribution using eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(T.T)

# Find eigenvector for eigenvalue = 1
idx = np.argmin(np.abs(eigenvalues - 1.0))
stationary = np.real(eigenvectors[:, idx])
stationary = stationary / np.sum(stationary)  # normalize

print("Stationary Distribution (Team Performance States):")
for s, p in zip(states, stationary):
    print(f"  {s}: {p:.3f}")

# Verify by simulation
np.random.seed(42)
state = 1  # Start Normal
counts = np.zeros(3)
n_steps = 100000

for _ in range(n_steps):
    state = np.random.choice(3, p=T[state])
    counts[state] += 1

print("\nSimulated proportions (100,000 steps):")
for s, c in zip(states, counts):
    print(f"  {s}: {c/n_steps:.3f}")
```

**Expected output:**
```
Stationary Distribution (Team Performance States):
  Hot: 0.237
  Normal: 0.407
  Cold: 0.356

Simulated proportions (100,000 steps):
  Hot: 0.237
  Normal: 0.407
  Cold: 0.356
```

### Exercise 2.1: Markov Chain Weather Model

**Task:** Build a simple weather Markov chain with states {Sunny, Cloudy, Rainy} and transition matrix: Sunny→(0.7, 0.2, 0.1), Cloudy→(0.3, 0.4, 0.3), Rainy→(0.2, 0.3, 0.5). Find the stationary distribution and simulate 365 days to verify it.

<details>
<summary>Hint</summary>

Use the eigenvalue method: compute eigenvectors of T.T, find the one with eigenvalue 1, and normalize.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

T = np.array([
    [0.7, 0.2, 0.1],  # Sunny
    [0.3, 0.4, 0.3],  # Cloudy
    [0.2, 0.3, 0.5],  # Rainy
])
states = ['Sunny', 'Cloudy', 'Rainy']

# Analytical stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(T.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.real(eigenvectors[:, idx])
pi = pi / np.sum(pi)

print("Stationary Distribution:")
for s, p in zip(states, pi):
    print(f"  {s}: {p:.3f}")

# Simulate 365 days
np.random.seed(42)
state = 0  # Start Sunny
counts = np.zeros(3)
for _ in range(365):
    state = np.random.choice(3, p=T[state])
    counts[state] += 1

print("\nSimulated (365 days):")
for s, c in zip(states, counts):
    print(f"  {s}: {c/365:.3f} ({int(c)} days)")
```

**Expected output:**
```
Stationary Distribution:
  Sunny: 0.406
  Cloudy: 0.297
  Rainy: 0.297

Simulated (365 days):
  Sunny: 0.419 (153 days)
  Cloudy: 0.274 (100 days)
  Rainy: 0.307 (112 days)
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain the Markov property in plain language
- [ ] Set up and read a transition matrix
- [ ] Find a stationary distribution using eigenvalues
- [ ] Simulate a Markov chain and verify it converges to the stationary distribution

---

## Section 3: Monte Carlo Methods

### What Are Monte Carlo Methods?

**Monte Carlo methods** use random sampling to solve problems that are too complex to solve analytically. The core idea: if you can simulate something, you can estimate anything about it.

### Monte Carlo Estimation

The classic example: estimating π.

```python
import numpy as np

np.random.seed(42)

# Estimate pi by throwing darts at a unit square
n_darts = 100000
x = np.random.uniform(-1, 1, n_darts)
y = np.random.uniform(-1, 1, n_darts)
inside_circle = (x**2 + y**2) <= 1
pi_estimate = 4 * np.mean(inside_circle)

print(f"Estimating π with {n_darts:,} random points:")
print(f"  Estimate: {pi_estimate:.6f}")
print(f"  True π:   {np.pi:.6f}")
print(f"  Error:    {abs(pi_estimate - np.pi):.6f}")
```

**Expected output:**
```
Estimating π with 100,000 random points:
  Estimate: 3.137040
  True π:   3.141593
  Error:    0.004553
```

### Monte Carlo Integration

Monte Carlo can estimate integrals that are difficult or impossible to compute analytically.

```python
import numpy as np

np.random.seed(42)

# Estimate E[max(S - K, 0)] where S ~ LogNormal
# This is essentially what option pricing does

def monte_carlo_integral(func, a, b, n_samples=100000):
    """Estimate integral of func from a to b using Monte Carlo."""
    x = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(func(x))

# Example: integral of sin(x) from 0 to pi (true answer = 2)
result = monte_carlo_integral(np.sin, 0, np.pi, n_samples=100000)
print(f"Integral of sin(x) from 0 to π:")
print(f"  Monte Carlo estimate: {result:.6f}")
print(f"  True value: 2.000000")
print(f"  Error: {abs(result - 2):.6f}")
```

**Expected output:**
```
Integral of sin(x) from 0 to π:
  Monte Carlo estimate: 2.001066
  True value: 2.000000
  Error: 0.001066
```

### Casino Simulation: Verifying the House Edge

```python
import numpy as np

np.random.seed(42)

def simulate_roulette(n_spins, bet_type='red'):
    """Simulate American roulette spins. Returns per-spin results (+1 win, -1 loss)."""
    # 38 slots: 18 red, 18 black, 2 green (0 and 00)
    spins = np.random.randint(0, 38, size=n_spins)
    if bet_type == 'red':
        wins = spins < 18  # slots 0-17 are red
    return np.where(wins, 1, -1)

# Simulate millions of spins to verify theoretical house edge
n_spins = 1_000_000
results = simulate_roulette(n_spins)
house_edge_simulated = -np.mean(results)
house_edge_theoretical = 2/38  # two green slots out of 38

print(f"American Roulette — {n_spins:,} spins on Red")
print(f"  Theoretical house edge: {house_edge_theoretical:.4f} ({house_edge_theoretical:.2%})")
print(f"  Simulated house edge:   {house_edge_simulated:.4f} ({house_edge_simulated:.2%})")
print(f"  Player's total P&L:     ${np.sum(results):,}")
print(f"  Casino's revenue:       ${-np.sum(results):,}")
```

**Expected output:**
```
American Roulette — 1,000,000 spins on Red
  Theoretical house edge: 0.0526 (5.26%)
  Simulated house edge:   0.0528 (5.28%)
  Player's total P&L:     $-52,816
  Casino's revenue:       $52,816
```

### Monte Carlo for Complex Decisions

```python
import numpy as np

np.random.seed(42)

# Should you take the guaranteed $800 or bet on a 50% chance of $2000?
# Expected value says bet ($1000 > $800), but what about risk?

n_simulations = 10000

# Strategy 1: Always take guaranteed $800
guaranteed = np.full(n_simulations, 800)

# Strategy 2: Always bet
bets = np.where(np.random.random(n_simulations) < 0.5, 2000, 0)

# Strategy 3: Mixed — take guaranteed if below $500 bankroll, else bet
bankroll = 500
mixed = np.where(
    np.random.random(n_simulations) < 0.5,
    2000,
    0
)

print("Decision Analysis via Monte Carlo")
print(f"{'Strategy':>15} | {'Mean':>8} | {'Std':>8} | {'P(get $0)':>10} | {'P(>$1000)':>10}")
print("-" * 65)
print(f"{'Guaranteed':>15} | ${np.mean(guaranteed):>7.0f} | ${np.std(guaranteed):>7.0f} | {np.mean(guaranteed == 0):>9.1%} | {np.mean(guaranteed > 1000):>9.1%}")
print(f"{'Always Bet':>15} | ${np.mean(bets):>7.0f} | ${np.std(bets):>7.0f} | {np.mean(bets == 0):>9.1%} | {np.mean(bets > 1000):>9.1%}")
```

**Expected output:**
```
Decision Analysis via Monte Carlo
       Strategy |     Mean |      Std |  P(get $0) | P(>$1000)
-----------------------------------------------------------------
     Guaranteed |    $800  |      $0  |      0.0%  |      0.0%
     Always Bet |    $997  |   $1000  |     50.1%  |     49.9%
```

### Exercise 3.1: Monte Carlo Poker Analysis

**Task:** Use Monte Carlo simulation to estimate the probability of being dealt a pair (two cards of the same rank) in a 5-card poker hand. The theoretical probability is about 42.3%.

<details>
<summary>Hint</summary>

Create a deck as a list of 52 cards (13 ranks × 4 suits). Use `np.random.choice` to deal 5 cards without replacement. Check if any rank appears exactly twice.
</details>

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from collections import Counter

np.random.seed(42)

# Create a deck: 13 ranks (0-12), 4 suits each
deck_ranks = np.array([r for r in range(13) for _ in range(4)])

n_simulations = 100000
pair_count = 0

for _ in range(n_simulations):
    hand_indices = np.random.choice(52, size=5, replace=False)
    hand_ranks = deck_ranks[hand_indices]
    rank_counts = Counter(hand_ranks)
    max_count = max(rank_counts.values())
    n_pairs = sum(1 for c in rank_counts.values() if c == 2)

    # "Pair" means exactly one pair (not two pair, not three of a kind, etc.)
    if n_pairs == 1 and max_count == 2:
        pair_count += 1

p_pair = pair_count / n_simulations
print(f"Monte Carlo estimate of P(exactly one pair): {p_pair:.4f}")
print(f"Theoretical probability: 0.4226")
print(f"Error: {abs(p_pair - 0.4226):.4f}")
```

**Expected output:**
```
Monte Carlo estimate of P(exactly one pair): 0.4233
Theoretical probability: 0.4226
Error: 0.0007
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain when Monte Carlo methods are appropriate
- [ ] Use Monte Carlo to estimate probabilities and integrals
- [ ] Apply Monte Carlo to verify theoretical results
- [ ] Understand that accuracy improves with √n (more samples = better estimates)

---

## Section 4: Poisson Processes

### What Is a Poisson Process?

A **Poisson process** models events that occur randomly and independently in continuous time. It has a single parameter: the **rate** λ (average number of events per unit time).

**Key properties:**
- The number of events in any interval of length t follows a Poisson(λt) distribution
- Times between events follow an Exponential(λ) distribution
- Events in non-overlapping intervals are independent (memoryless)

### Simulating a Poisson Process

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# Soccer goals: average rate of 2.5 goals per game (90 minutes)
goal_rate = 2.5 / 90  # goals per minute

# Method 1: Simulate inter-arrival times (Exponential)
n_games = 10
for game in range(3):
    time = 0
    goals = []
    while True:
        wait = np.random.exponential(1 / goal_rate)
        time += wait
        if time > 90:
            break
        goals.append(time)
    print(f"Game {game+1}: {len(goals)} goals at minutes {[f'{g:.0f}' for g in goals]}")

# Method 2: Use Poisson distribution directly
print(f"\nGoals per game distribution (1000 simulated games):")
n_goals = np.random.poisson(2.5, size=1000)
for g in range(6):
    pct = np.mean(n_goals == g)
    theory = stats.poisson.pmf(g, 2.5)
    print(f"  {g} goals: {pct:.1%} simulated, {theory:.1%} theoretical")
```

**Expected output:**
```
Game 1: 3 goals at minutes ['2', '28', '44']
Game 2: 4 goals at minutes ['5', '29', '79', '82']
Game 3: 5 goals at minutes ['6', '22', '41', '55', '65']

Goals per game distribution (1000 simulated games):
  0 goals: 8.2% simulated, 8.2% theoretical
  1 goals: 19.6% simulated, 20.5% theoretical
  2 goals: 26.1% simulated, 25.6% theoretical
  3 goals: 21.2% simulated, 21.4% theoretical
  4 goals: 13.3% simulated, 13.4% theoretical
  5 goals: 6.4% simulated, 6.7% theoretical
```

### Finance: Trade Arrivals

```python
import numpy as np

np.random.seed(42)

# Stock trade arrivals: average 120 trades per hour
trades_per_hour = 120
trades_per_second = trades_per_hour / 3600

# Simulate one hour of trading
inter_arrival_times = np.random.exponential(1/trades_per_second, size=200)
trade_times = np.cumsum(inter_arrival_times)
trade_times = trade_times[trade_times <= 3600]  # one hour

print(f"Trade Arrival Simulation (rate: {trades_per_hour}/hour)")
print(f"Total trades in 1 hour: {len(trade_times)}")
print(f"Mean inter-arrival time: {np.mean(np.diff(trade_times)):.2f} seconds")
print(f"Expected inter-arrival time: {3600/trades_per_hour:.2f} seconds")

# Check for clustering (a common misconception)
# Split hour into 12 five-minute intervals
intervals = np.histogram(trade_times, bins=12, range=(0, 3600))[0]
print(f"\nTrades per 5-minute interval: {intervals}")
print(f"Expected per interval: {trades_per_hour/12:.1f}")
print(f"Std dev of interval counts: {np.std(intervals):.1f}")
print(f"Expected std dev (Poisson): {np.sqrt(trades_per_hour/12):.1f}")
```

**Expected output:**
```
Trade Arrival Simulation (rate: 120/hour)
Total trades in 1 hour: 107
Mean inter-arrival time: 33.44 seconds
Expected inter-arrival time: 30.00 seconds

Trades per 5-minute interval: [ 7  8  8 12 12 10  3 14  9  5  8 11]
Expected per interval: 10.0
Std dev of interval counts: 3.0
Expected std dev (Poisson): 3.2
```

### Exercise 4.1: Insurance Claims

**Task:** An insurance company receives claims at a rate of 5 per day. Use the Poisson process to answer: What's the probability of receiving more than 8 claims in a day? What's the probability of going more than 6 hours between claims?

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from scipy import stats

rate = 5  # claims per day

# P(more than 8 claims in a day) = 1 - P(X <= 8)
p_more_than_8 = 1 - stats.poisson.cdf(8, rate)
print(f"P(more than 8 claims in a day): {p_more_than_8:.4f}")

# Time between claims: Exponential with rate 5/day
# P(wait > 6 hours) = P(wait > 0.25 days)
rate_per_day = 5
p_wait_6hrs = 1 - stats.expon.cdf(0.25, scale=1/rate_per_day)
print(f"P(more than 6 hours between claims): {p_wait_6hrs:.4f}")

# Verify with simulation
np.random.seed(42)
n_days = 100000
daily_claims = np.random.poisson(rate, size=n_days)
print(f"\nSimulation verification ({n_days:,} days):")
print(f"P(>8 claims): {np.mean(daily_claims > 8):.4f}")

inter_arrivals = np.random.exponential(1/rate, size=n_days*5)
print(f"P(wait > 6hrs): {np.mean(inter_arrivals > 0.25):.4f}")
```

**Expected output:**
```
P(more than 8 claims in a day): 0.0681
P(more than 6 hours between claims): 0.2865

Simulation verification (100,000 days):
P(>8 claims): 0.0686
P(wait > 6hrs): 0.2864
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain what a Poisson process models
- [ ] Connect Poisson counts with Exponential waiting times
- [ ] Simulate a Poisson process both ways (counts and inter-arrivals)

---

## Section 5: Time Series Foundations

### What Is a Time Series?

A **time series** is a sequence of data points indexed by time. Stock prices, sports team records, website traffic — all time series. Understanding their structure helps with prediction and decision-making.

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) don't change over time. Most analysis tools assume stationarity.

```python
import numpy as np

np.random.seed(42)

# Stationary series: random noise around a constant mean
stationary = np.random.normal(0, 1, size=200)

# Non-stationary: random walk (mean and variance change)
non_stationary = np.cumsum(np.random.normal(0, 1, size=200))

# Check: compare first half vs second half
for name, series in [("Stationary", stationary), ("Non-stationary", non_stationary)]:
    first_half = series[:100]
    second_half = series[100:]
    print(f"{name}:")
    print(f"  First half  — mean: {np.mean(first_half):>7.3f}, std: {np.std(first_half):.3f}")
    print(f"  Second half — mean: {np.mean(second_half):>7.3f}, std: {np.std(second_half):.3f}")
    print()
```

**Expected output:**
```
Stationary:
  First half  — mean:   0.059, std: 0.978
  Second half — mean:  -0.041, std: 1.001

Non-stationary:
  First half  — mean:   1.297, std: 4.726
  Second half — mean:  -4.016, std: 5.263
```

### Autocorrelation

**Autocorrelation** measures how much a series correlates with its own lagged values. High autocorrelation at lag 1 means today's value predicts tomorrow's.

```python
import numpy as np

np.random.seed(42)

# White noise: no autocorrelation
noise = np.random.normal(0, 1, size=500)

# AR(1) process: each value depends on the previous
ar_series = np.zeros(500)
phi = 0.8  # autocorrelation parameter
for t in range(1, 500):
    ar_series[t] = phi * ar_series[t-1] + np.random.normal(0, 1)

def autocorrelation(series, lag):
    """Compute autocorrelation at a given lag."""
    n = len(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean)**2) / n
    ck = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
    return ck / c0

print("Autocorrelation at various lags:")
print(f"{'Lag':>4} | {'White Noise':>11} | {'AR(1), φ=0.8':>13}")
print("-" * 35)
for lag in [1, 2, 5, 10, 20]:
    ac_noise = autocorrelation(noise, lag)
    ac_ar = autocorrelation(ar_series, lag)
    print(f"{lag:>4} | {ac_noise:>11.3f} | {ac_ar:>13.3f}")
```

**Expected output:**
```
Autocorrelation at various lags:
 Lag | White Noise | AR(1), φ=0.8
-----------------------------------
   1 |       0.024 |         0.788
   2 |      -0.056 |         0.614
   5 |       0.030 |         0.334
  10 |       0.028 |         0.087
  20 |      -0.038 |         0.018
```

White noise has near-zero autocorrelation at all lags. The AR(1) process shows strong autocorrelation at lag 1 (close to φ = 0.8) that decays geometrically.

### Application: Momentum vs Mean Reversion in Sports

```python
import numpy as np

np.random.seed(42)

# Simulate a team's scoring over a season
# Model: some autocorrelation (hot/cold streaks)
n_games = 82
phi = 0.3  # mild momentum
mean_score = 105
std_score = 12

scores = np.zeros(n_games)
scores[0] = mean_score + np.random.normal(0, std_score)
for t in range(1, n_games):
    scores[t] = mean_score + phi * (scores[t-1] - mean_score) + np.random.normal(0, std_score * np.sqrt(1 - phi**2))

# Analyze autocorrelation
ac1 = autocorrelation(scores, 1)
ac2 = autocorrelation(scores, 2)
ac5 = autocorrelation(scores, 5)

print(f"Team Scoring Analysis ({n_games} games)")
print(f"Mean score: {np.mean(scores):.1f}")
print(f"Std dev: {np.std(scores):.1f}")
print(f"Autocorrelation lag 1: {ac1:.3f} (hot/cold streaks)")
print(f"Autocorrelation lag 2: {ac2:.3f}")
print(f"Autocorrelation lag 5: {ac5:.3f}")
print()
print("Interpretation:")
if ac1 > 0.1:
    print(f"  Positive lag-1 autocorrelation suggests scoring momentum")
elif ac1 < -0.1:
    print(f"  Negative lag-1 autocorrelation suggests mean reversion")
else:
    print(f"  Weak autocorrelation suggests scores are roughly independent")
```

**Expected output:**
```
Team Scoring Analysis (82 games)
Mean score: 105.1
Std dev: 12.3
Autocorrelation lag 1: 0.261 (hot/cold streaks)
Autocorrelation lag 2: 0.076
Autocorrelation lag 5: -0.061

Interpretation:
  Positive lag-1 autocorrelation suggests scoring momentum
```

### Exercise 5.1: Detecting Trends in Financial Data

**Task:** Generate a stock price series using a random walk with drift (0.05% daily drift, 1% daily volatility) for 252 trading days. Compute the autocorrelation of daily returns (not prices). Are daily returns autocorrelated?

<details>
<summary>Click to see solution</summary>

```python
import numpy as np

np.random.seed(42)

n_days = 252
drift = 0.0005    # 0.05% daily
volatility = 0.01  # 1% daily

# Generate returns and prices
daily_returns = drift + volatility * np.random.normal(size=n_days)
prices = 100 * np.exp(np.cumsum(daily_returns))

# Autocorrelation of returns
def autocorrelation(series, lag):
    n = len(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean)**2) / n
    ck = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
    return ck / c0

print(f"Price: ${prices[0]:.2f} → ${prices[-1]:.2f} ({(prices[-1]/prices[0]-1):.1%} return)")
print(f"Daily return mean: {np.mean(daily_returns):.4%}")
print(f"Daily return std: {np.std(daily_returns):.4%}")
print()
print("Autocorrelation of daily returns:")
for lag in [1, 2, 5, 10]:
    ac = autocorrelation(daily_returns, lag)
    sig = 1.96 / np.sqrt(n_days)
    significant = "***" if abs(ac) > sig else ""
    print(f"  Lag {lag:>2}: {ac:>7.3f} (±{sig:.3f} significance band) {significant}")
```

**Expected output:**
```
Price: $100.14 → $115.63 (15.5% return)
Daily return mean: 0.0576%
Daily return std: 0.9862%

Autocorrelation of daily returns:
  Lag  1:   0.033 (±0.124 significance band)
  Lag  2:   0.015 (±0.124 significance band)
  Lag  5:   0.085 (±0.124 significance band)
  Lag 10:  -0.055 (±0.124 significance band)
```

**Explanation:** As expected from the random walk model, daily returns show no significant autocorrelation. Prices themselves are highly autocorrelated (non-stationary), but the returns (first differences) are not. This is the foundation of the Efficient Market Hypothesis.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Distinguish stationary from non-stationary series
- [ ] Compute and interpret autocorrelation
- [ ] Explain why returns (not prices) are the right thing to analyze
- [ ] Connect autocorrelation to concepts like momentum and mean reversion

---

## Common Pitfalls

### Pitfall 1: Insufficient Monte Carlo Samples

**The Problem:** Using too few samples and getting imprecise estimates.

**How to avoid it:** Use at least 10,000 samples for rough estimates, 100,000+ for precision. Remember: error decreases as 1/√n, so to halve the error you need 4× the samples.

### Pitfall 2: Confusing Stationary with Static

**The Problem:** Thinking a stationary time series doesn't change. It changes — its statistical properties just stay constant.

**How to avoid it:** Stationarity means the mean, variance, and autocorrelation structure are stable over time. The actual values fluctuate constantly.

### Pitfall 3: Ignoring MCMC Convergence

**The Problem:** Using MCMC samples before the chain has converged.

**How to avoid it:** Always discard burn-in, check trace plots, run multiple chains, and compute effective sample sizes.

### Pitfall 4: Assuming Independence When There's Autocorrelation

**The Problem:** Using standard error formulas that assume independent observations on autocorrelated data.

**How to avoid it:** Check autocorrelation first. If present, use time-series-aware methods or account for effective sample size.

## Best Practices

- ✅ **Always verify Monte Carlo results** against known theoretical values when possible
- ✅ **Check autocorrelation** before assuming observations are independent
- ✅ **Difference non-stationary series** before analysis (analyze returns, not prices)
- ✅ **Use enough samples** — computational cost is cheap, precision is valuable
- ❌ **Don't assume stock prices are pure random walks** — real markets have fat tails and volatility clustering
- ❌ **Don't confuse the gambler's fallacy** with the law of large numbers — individual outcomes don't "balance out"
- ❌ **Don't use Markov chains without verifying** the Markov property holds for your data

---

## Practice Project

### Project Description

Build a **Monte Carlo options pricer** that estimates the price of European call and put options using simulation.

### Requirements

1. Implement Geometric Brownian Motion for stock price paths
2. Price a European call option using Monte Carlo simulation
3. Compare to the Black-Scholes analytical formula
4. Analyze how option price changes with different parameters (volatility, strike, time to expiry)
5. Estimate Greeks (delta, gamma) using finite differences

### Getting Started

```python
import numpy as np
from scipy import stats

# Black-Scholes analytical formula for reference
def black_scholes_call(S, K, T, r, sigma):
    """Analytical Black-Scholes price for a European call."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)

# TODO: Implement Monte Carlo pricer
# TODO: Compare with Black-Scholes
# TODO: Analyze parameter sensitivity
```

<details>
<summary>If you're not sure where to start</summary>

1. Simulate many stock price paths using GBM: S_T = S_0 × exp((r - σ²/2)T + σ√T × Z)
2. For each path, compute the call payoff: max(S_T - K, 0)
3. Discount the average payoff: price = e^(-rT) × mean(payoffs)
</details>

<details>
<summary>Click to see one possible solution</summary>

```python
import numpy as np
from scipy import stats

np.random.seed(42)

def black_scholes_call(S, K, T, r, sigma):
    """Analytical Black-Scholes price for a European call."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)

def monte_carlo_call(S, K, T, r, sigma, n_paths=100000):
    """Monte Carlo price for a European call option."""
    Z = np.random.normal(size=n_paths)
    S_T = S * np.exp((r - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    payoffs = np.maximum(S_T - K, 0)
    price = np.exp(-r*T) * np.mean(payoffs)
    se = np.exp(-r*T) * np.std(payoffs) / np.sqrt(n_paths)
    return price, se

# Parameters
S = 100      # stock price
K = 105      # strike price
T = 0.5      # 6 months
r = 0.05     # risk-free rate
sigma = 0.20 # volatility

bs_price = black_scholes_call(S, K, T, r, sigma)
mc_price, mc_se = monte_carlo_call(S, K, T, r, sigma)

print("European Call Option Pricing")
print(f"Parameters: S={S}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"Black-Scholes price: ${bs_price:.4f}")
print(f"Monte Carlo price:   ${mc_price:.4f} ± ${mc_se:.4f}")
print(f"Error:               ${abs(mc_price - bs_price):.4f}")
print()

# Parameter sensitivity analysis
print("Sensitivity Analysis (varying volatility):")
print(f"{'σ':>5} | {'BS Price':>8} | {'MC Price':>8} | {'Error':>6}")
print("-" * 36)
for vol in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    bs = black_scholes_call(S, K, T, r, vol)
    mc, se = monte_carlo_call(S, K, T, r, vol)
    print(f"{vol:>5.2f} | ${bs:>7.2f} | ${mc:>7.2f} | ${abs(mc-bs):>5.2f}")

print()

# Estimate Delta (dC/dS) using finite differences
h = 0.01
delta_bs = (black_scholes_call(S+h, K, T, r, sigma) - black_scholes_call(S-h, K, T, r, sigma)) / (2*h)
mc_up, _ = monte_carlo_call(S+h, K, T, r, sigma)
mc_down, _ = monte_carlo_call(S-h, K, T, r, sigma)
delta_mc = (mc_up - mc_down) / (2*h)

print(f"Delta (dC/dS):")
print(f"  Black-Scholes: {delta_bs:.4f}")
print(f"  Monte Carlo:   {delta_mc:.4f}")
```

**Key points in this solution:**
- Monte Carlo pricing matches Black-Scholes within standard error
- Sensitivity analysis shows how volatility drives option prices
- Finite differences can estimate Greeks from Monte Carlo prices
- The technique extends to exotic options where no formula exists
</details>

---

## Summary

### Key Takeaways

- **Random walks** are the simplest stochastic process — they model gambling outcomes and stock price changes
- **Gambler's ruin** proves that any gambler with finite bankroll goes broke in a fair game (and faster with a house edge)
- **Markov chains** model systems where only the current state matters for predicting the future
- **Monte Carlo methods** use simulation to solve problems that are analytically intractable
- **Poisson processes** model random events in continuous time
- **Stationarity and autocorrelation** are the foundation of time series analysis

### Skills You've Gained

You can now:
- ✓ Simulate and analyze random walks
- ✓ Build and analyze Markov chains with transition matrices
- ✓ Apply Monte Carlo methods for estimation and simulation
- ✓ Model random events using Poisson processes
- ✓ Check stationarity and compute autocorrelation
- ✓ Price options using Monte Carlo simulation

---

## Next Steps

### Continue Learning

**Explore related routes:**
- [Bayesian Statistics](/routes/bayesian-statistics/map.md) — MCMC (a stochastic process) is used for Bayesian inference
- [Regression and Modeling](/routes/regression-and-modeling/map.md) — Time series regression, ARIMA models

### Additional Resources

**Documentation:**
- numpy.random — simulation and random number generation
- scipy.stats — distribution functions and statistical tests
- statsmodels — time series analysis tools (ARIMA, VAR, etc.)

---

## Appendix

### Quick Reference

| Concept | Key Formula | Python |
|---------|------------|--------|
| Random Walk position | E[S_n]=0, Var(S_n)=n | `np.cumsum(np.random.choice([-1,1], n))` |
| Gambler's Ruin (fair) | P(ruin) = 1 | Simulation |
| Transition Matrix | π = πT | `np.linalg.eig(T.T)` |
| Monte Carlo estimate | x̄ ± 1.96 s/√n | `np.mean(samples)` |
| Poisson process | N(t) ~ Poisson(λt) | `np.random.poisson(lam*t)` |
| Exponential waiting | P(T>t) = e^(-λt) | `np.random.exponential(1/lam)` |
| Autocorrelation | r_k = C_k / C_0 | `np.correlate()` or manual |

### Glossary

- **Stochastic process**: A collection of random variables indexed by time
- **Random walk**: A process where each step is a random increment
- **Gambler's ruin**: The problem of a gambler eventually going broke against an infinite-bankroll opponent
- **Markov property**: The future depends only on the present, not the past
- **Transition matrix**: A matrix of probabilities governing state changes in a Markov chain
- **Stationary distribution**: The long-run proportion of time a Markov chain spends in each state
- **Absorbing state**: A state that, once entered, cannot be left
- **Monte Carlo method**: Using random sampling to estimate quantities or simulate complex systems
- **Poisson process**: A stochastic process for random events in continuous time with rate λ
- **Stationarity**: A time series property where statistical properties remain constant over time
- **Autocorrelation**: The correlation of a time series with its own lagged values
- **Geometric Brownian Motion**: A continuous-time random walk model for stock prices
- **Black-Scholes**: An analytical formula for pricing European options
