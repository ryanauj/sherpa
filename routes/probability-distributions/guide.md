---
title: Probability Distributions
route_map: /routes/probability-distributions/map.md
paired_sherpa: /routes/probability-distributions/sherpa.md
prerequisites:
  - Probability Fundamentals
  - Stats Fundamentals (helpful)
topics:
  - Discrete Distributions
  - Continuous Distributions
  - Expected Value and Variance
  - Law of Large Numbers
  - Central Limit Theorem
---

# Probability Distributions

> **Note for AI assistants**: This guide has a paired sherpa at `/routes/probability-distributions/sherpa.md` that provides structured teaching guidance.
> **Route map**: See `/routes/probability-distributions/map.md` for the high-level overview.

## Overview

A probability tells you the chance of one thing happening. A probability distribution tells you the chance of *everything* that could happen, all at once.

When you roll a die, you don't just want to know the chance of a 6 -- you want the complete picture: each face has probability 1/6. That complete picture is a distribution. Every casino game has one. Every stock has one. Every ML model produces one. Once you describe a random process with a distribution, you can compute anything: expected profit, risk, the chance of going broke, the chance of beating the market.

This route teaches the major probability distributions -- both discrete (countable outcomes like wins, goals, defaults) and continuous (measurable outcomes like returns, heights, times). You will also learn the two great theorems that make statistics work at scale: the Law of Large Numbers (why casinos always win) and the Central Limit Theorem (why the bell curve is everywhere).

## Learning Objectives

By the end of this route, you will be able to:
- Identify which probability distribution models a given real-world scenario
- Compute probabilities, expected values, and variances for each major distribution
- Use scipy.stats in Python to work with any distribution (PMF/PDF, CDF, sampling)
- Explain why the Law of Large Numbers guarantees the house always wins
- Apply the Central Limit Theorem to understand why sample means are normally distributed
- Build Monte Carlo simulations to verify theoretical results with code

## Prerequisites

Before starting, you should be comfortable with:
- **Probability Fundamentals** ([route](/routes/probability-fundamentals/map.md)): sample spaces, probability rules, conditional probability, expected value
- **Basic Python**: variables, loops, functions, numpy arrays
- **Stats Fundamentals** ([route](/routes/stats-fundamentals/map.md)): helpful but not required -- mean, standard deviation, histograms

## Setup

You need scipy, numpy, and matplotlib.

```bash
pip install scipy numpy matplotlib
```

**Verify your setup:**

```python
from scipy import stats
import numpy as np
import matplotlib
print(f"scipy version: {stats.scipy.__version__}")
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")

# Quick test: generate 5 samples from a normal distribution
samples = stats.norm.rvs(loc=0, scale=1, size=5, random_state=42)
print(f"Random samples: {np.round(samples, 4)}")
```

**Expected Output:**
```
scipy version: 1.x.x
numpy version: 1.x.x
matplotlib version: 3.x.x
Random samples: [ 0.4967 -0.1383  0.6477  1.523  -0.2342]
```

---

## Section 1: Discrete Distributions

Discrete distributions assign probabilities to countable outcomes -- things you can list: 0, 1, 2, 3, and so on. How many free throws does a player make out of 10? How many goals are scored in a soccer match? How many trades profit in a week? These are all discrete questions, and each has a named distribution that models it.

### The Bernoulli Distribution

The simplest distribution: a single trial with two outcomes. Success (1) or failure (0). A coin flip. A free throw. A single stock trade that either profits or loses.

**Parameters:** p = probability of success

```python
from scipy import stats
import numpy as np

# A basketball player makes 80% of free throws
ft = stats.bernoulli(p=0.80)

print(f"P(make) = {ft.pmf(1):.2f}")        # 0.80
print(f"P(miss) = {ft.pmf(0):.2f}")        # 0.20
print(f"Expected value: {ft.mean():.2f}")   # 0.80
print(f"Variance: {ft.var():.4f}")          # 0.1600

# Simulate 10 free throws
np.random.seed(42)
shots = ft.rvs(size=10)
print(f"Simulated shots: {shots}")
print(f"Made {shots.sum()} out of {len(shots)}")
```

**Expected Output:**
```
P(make) = 0.80
P(miss) = 0.20
Expected value: 0.80
Variance: 0.1600
Simulated shots: [1 0 1 1 1 1 1 1 1 1]
Made 9 out of 10
```

**What's happening here:**
- `pmf(k)` is the probability mass function -- it returns P(X = k). For Bernoulli, there are only two values: k=0 (failure) and k=1 (success).
- `rvs(size=10)` generates random samples -- it simulates 10 independent free throws.
- The expected value of a Bernoulli is simply p. The variance is p(1-p).

### The Binomial Distribution

The Binomial distribution counts the number of successes in n independent Bernoulli trials. If you shoot 10 free throws at 80%, how many do you make? If a baseball player with a .300 average gets 4 at-bats, how many hits?

**Parameters:** n = number of trials, p = probability of success per trial

```python
from scipy import stats
import numpy as np

# Baseball: .300 hitter with 4 at-bats per game
binom = stats.binom(n=4, p=0.300)

print("Probability of each number of hits:")
for k in range(5):
    bar = "#" * int(binom.pmf(k) * 50)
    print(f"  P({k} hits) = {binom.pmf(k):.4f}  {bar}")

print(f"\nExpected hits per game: {binom.mean():.1f}")
print(f"Standard deviation: {binom.std():.4f}")
print(f"P(at least 1 hit): {1 - binom.pmf(0):.4f}")
print(f"P(at least 3 hits): {1 - binom.cdf(2):.4f}")
```

**Expected Output:**
```
Probability of each number of hits:
  P(0 hits) = 0.2401  ############
  P(1 hit)  = 0.4116  ####################
  P(2 hits) = 0.2646  #############
  P(3 hits) = 0.0756  ###
  P(4 hits) = 0.0081

Expected hits per game: 1.2
Standard deviation: 0.9165
P(at least 1 hit): 0.7599
P(at least 3 hits): 0.0837
```

The most likely outcome is exactly 1 hit (41% chance). Going hitless isn't rare (24%). Going 4-for-4 almost never happens (0.8%). Over a 162-game season, the expected number of hits is 4 * 0.3 * 162 = 194.4.

**Key formulas:**
- E[X] = np
- Var(X) = np(1-p)

**Sports betting application:** If a team wins 60% of games, what's the probability they win at least 50 of 82 games in an NBA season?

```python
season = stats.binom(n=82, p=0.60)
p_50_plus = 1 - season.cdf(49)
print(f"P(at least 50 wins): {p_50_plus:.4f}")  # ~0.5765
```

**Expected Output:**
```
P(at least 50 wins): 0.5765
```

**Finance application:** In a portfolio of 20 bonds, each with a 5% default probability, how many defaults should you expect?

```python
defaults = stats.binom(n=20, p=0.05)
print(f"Expected defaults: {defaults.mean():.1f}")
print(f"P(0 defaults): {defaults.pmf(0):.4f}")
print(f"P(3+ defaults): {1 - defaults.cdf(2):.4f}")
```

**Expected Output:**
```
Expected defaults: 1.0
P(0 defaults): 0.3585
P(3+ defaults): 0.0755
```

### The Poisson Distribution

The Poisson distribution models the number of events that occur in a fixed interval when events happen independently at a constant average rate. Goals per soccer match. Server errors per hour. Market crashes per decade.

**Parameter:** lambda (mu) = average number of events per interval

```python
from scipy import stats

# Premier League: average 2.7 goals per match
goals = stats.poisson(mu=2.7)

print("Goals per match probability:")
for k in range(8):
    bar = "#" * int(goals.pmf(k) * 50)
    print(f"  P({k} goals) = {goals.pmf(k):.4f}  {bar}")

print(f"\nExpected goals: {goals.mean():.1f}")
print(f"Variance: {goals.var():.1f}")
print(f"P(4+ goals, over bet): {1 - goals.cdf(3):.4f}")
print(f"P(0 goals, goalless draw): {goals.pmf(0):.4f}")
```

**Expected Output:**
```
Goals per match probability:
  P(0 goals) = 0.0672  ###
  P(1 goal)  = 0.1815  #########
  P(2 goals) = 0.2450  ############
  P(3 goals) = 0.2205  ###########
  P(4 goals) = 0.1488  #######
  P(5 goals) = 0.0804  ####
  P(6 goals) = 0.0362  #
  P(7 goals) = 0.0140

Expected goals: 2.7
Variance: 2.7
P(4+ goals, over bet): 0.2694
P(0 goals, goalless draw): 0.0672
```

Notice something special: for the Poisson distribution, the mean and variance are both equal to lambda. If you see count data where the mean and variance are very different, it is probably not Poisson.

**Key formulas:**
- E[X] = lambda
- Var(X) = lambda

**ML connection:** Count data in ML -- number of clicks, words, purchases -- often follows Poisson. Poisson regression is built on this distribution.

### The Geometric Distribution

The Geometric distribution counts how many trials you need until your first success. How many hands of poker until you win? How many sales calls until you close a deal?

**Parameter:** p = probability of success on each trial

```python
from scipy import stats

# Probability of winning a poker hand: ~15%
hands = stats.geom(p=0.15)

print(f"Expected hands until first win: {hands.mean():.1f}")
print(f"P(win on 1st hand): {hands.pmf(1):.4f}")
print(f"P(win within 5 hands): {hands.cdf(5):.4f}")
print(f"P(need 10+ hands): {1 - hands.cdf(9):.4f}")
```

**Expected Output:**
```
Expected hands until first win: 6.7
P(win on 1st hand): 0.1500
P(win within 5 hands): 0.5563
P(need 10+ hands): 0.2316
```

The Geometric distribution is **memoryless**: if you have lost 10 hands in a row, the probability of winning the next hand is still 15%. Past losses do not make a win "due." This is the mathematical refutation of the gambler's fallacy.

**Key formulas:**
- E[X] = 1/p
- Var(X) = (1-p)/p^2

### Exercise 1.1: Choosing the Right Distribution

For each scenario, identify the distribution and compute the requested value.

1. A coin is flipped 20 times. What's the probability of exactly 10 heads?
2. A call center gets 8 calls per hour on average. What's the probability of 12+ calls in an hour?
3. A lottery ticket wins 1 time in 200. How many tickets do you expect to buy before winning?
4. A stock goes up on 55% of trading days. What's the probability it goes up on a given day?

<details>
<summary>Hints</summary>

1. Fixed number of trials, counting successes -- which distribution is that?
2. Events occurring at a rate in a fixed interval -- which distribution?
3. Counting trials until first success -- which distribution?
4. Single trial, two outcomes -- which distribution?
</details>

<details>
<summary>Solution</summary>

```python
from scipy import stats

# 1. Binomial(n=20, p=0.5)
print(f"P(10 heads in 20 flips): {stats.binom.pmf(10, n=20, p=0.5):.4f}")
# 0.1762

# 2. Poisson(mu=8)
print(f"P(12+ calls): {1 - stats.poisson.cdf(11, mu=8):.4f}")
# 0.1119

# 3. Geometric(p=1/200)
print(f"Expected tickets: {stats.geom.mean(p=1/200):.0f}")
# 200

# 4. Bernoulli(p=0.55)
print(f"P(up): {stats.bernoulli.pmf(1, p=0.55):.2f}")
# 0.55
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Name the four discrete distributions and their parameters
- [ ] Choose the right distribution for a given scenario
- [ ] Use scipy.stats to compute PMF, CDF, and expected values
- [ ] Explain why the geometric distribution is memoryless

---

## Section 2: Continuous Distributions

Continuous distributions model outcomes that can take any value in a range -- not just whole numbers. Stock returns, heights, sprint times, waiting times. The key difference from discrete distributions: you cannot assign a probability to a single exact value (P(X = 2.37) = 0). Instead, you compute probabilities over intervals.

### Key Concept: PDF vs PMF

For discrete distributions, the **PMF** (probability mass function) gives you P(X = k) directly.

For continuous distributions, the **PDF** (probability density function) gives you the density at a point. The density is NOT a probability. You get probabilities by computing the area under the PDF curve over an interval:

P(a < X < b) = area under PDF from a to b = CDF(b) - CDF(a)

The **CDF** (cumulative distribution function) is your best friend for continuous distributions. CDF(x) = P(X <= x).

### The Uniform Distribution

The simplest continuous distribution: every value in the interval [a, b] is equally likely. A perfectly fair spinner. A random number generator.

**Parameters:** a = lower bound, b = upper bound

```python
from scipy import stats
import numpy as np

# Random number between 0 and 1
uniform = stats.uniform(loc=0, scale=1)  # loc=a, scale=b-a

print(f"Mean: {uniform.mean():.2f}")
print(f"Variance: {uniform.var():.4f}")
print(f"P(0.3 < X < 0.7): {uniform.cdf(0.7) - uniform.cdf(0.3):.2f}")
print(f"P(X < 0.25): {uniform.cdf(0.25):.2f}")
print(f"PDF at x=0.5: {uniform.pdf(0.5):.2f}")  # constant = 1/(b-a)
```

**Expected Output:**
```
Mean: 0.50
Variance: 0.0833
P(0.3 < X < 0.7): 0.40
P(X < 0.25): 0.25
PDF at x=0.5: 1.00
```

**Key formulas:**
- E[X] = (a + b) / 2
- Var(X) = (b - a)^2 / 12

### The Normal (Gaussian) Distribution

The normal distribution -- the bell curve -- is the most important distribution in statistics. It is defined by two parameters: the mean (mu, the center) and standard deviation (sigma, the spread).

**Parameters:** mu = mean, sigma = standard deviation

```python
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# NBA player heights: mean 78.5 inches, std dev 3.5 inches
heights = stats.norm(loc=78.5, scale=3.5)

print(f"P(under 6 ft / 72 in): {heights.cdf(72):.4f}")
print(f"P(over 7 ft / 84 in): {1 - heights.cdf(84):.4f}")
print(f"P(between 75 and 82): {heights.cdf(82) - heights.cdf(75):.4f}")

# The 68-95-99.7 rule
print("\n68-95-99.7 Rule:")
for k in [1, 2, 3]:
    lower = 78.5 - k * 3.5
    upper = 78.5 + k * 3.5
    prob = heights.cdf(upper) - heights.cdf(lower)
    print(f"  P(within {k} std devs): {prob:.4f}")

# Plot the distribution
x = np.linspace(65, 92, 200)
plt.figure(figsize=(10, 6))
plt.plot(x, heights.pdf(x), 'b-', linewidth=2)
plt.fill_between(x, heights.pdf(x), alpha=0.3)
plt.axvline(78.5, color='red', linestyle='--', label='Mean = 78.5"')
plt.xlabel('Height (inches)')
plt.ylabel('Density')
plt.title('Normal Distribution: NBA Player Heights')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Expected Output:**
```
P(under 6 ft / 72 in): 0.0317
P(over 7 ft / 84 in): 0.0582
P(between 75 and 82): 0.6827

68-95-99.7 Rule:
  P(within 1 std devs): 0.6827
  P(within 2 std devs): 0.9545
  P(within 3 std devs): 0.9973
```

The **68-95-99.7 rule** is the single most useful fact about normal distributions: about 68% of values fall within 1 standard deviation of the mean, 95% within 2, and 99.7% within 3.

**Finance application -- stock returns:**

```python
from scipy import stats

# Daily S&P 500 returns: mean ~0.04%, std ~1.1%
returns = stats.norm(loc=0.0004, scale=0.011)

print(f"P(daily loss > 3%): {returns.cdf(-0.03):.6f}")
print(f"P(daily gain > 2%): {1 - returns.cdf(0.02):.6f}")
```

**Expected Output:**
```
P(daily loss > 3%): 0.002872
P(daily gain > 2%): 0.036811
```

**Important caveat -- fat tails:** The normal distribution says a 5% daily drop is nearly impossible. But the S&P 500 has seen drops of 7%, 10%, even 20%. Real financial returns have **fat tails** -- extreme events happen more often than the normal distribution predicts. The normal distribution is a useful approximation for typical days, but it dangerously underestimates the probability of crashes. The 2008 financial crisis was partly caused by models that assumed normality when reality had fat tails.

**ML connection:** Many algorithms assume features are normally distributed: Gaussian Naive Bayes, Linear Discriminant Analysis, and the residuals in linear regression. Understanding when normality holds (and when it doesn't) is critical for model selection.

### The Exponential Distribution

The Exponential distribution models the time between events in a Poisson process. If goals in a soccer match follow Poisson, the time between goals is exponential. If server errors arrive at a Poisson rate, the time between errors is exponential.

**Parameter:** lambda (rate) -- scipy uses scale = 1/lambda

```python
from scipy import stats

# Soccer: average 2.7 goals per 90 minutes
# Average time between goals: 90/2.7 = 33.3 minutes
time_between = stats.expon(scale=33.3)

print(f"Mean time between goals: {time_between.mean():.1f} min")
print(f"P(goal within 10 min): {time_between.cdf(10):.4f}")
print(f"P(no goal for 45+ min): {1 - time_between.cdf(45):.4f}")
print(f"Median wait time: {time_between.median():.1f} min")
```

**Expected Output:**
```
Mean time between goals: 33.3 min
P(goal within 10 min): 0.2592
P(no goal for 45+ min): 0.2584
Median wait time: 23.1 min
```

Like the Geometric distribution, the Exponential distribution is **memoryless**. If no goal has been scored in 30 minutes, the probability of a goal in the next 10 minutes is exactly the same as the probability of a goal in the first 10 minutes.

**Key formulas:**
- E[X] = 1/lambda = scale
- Var(X) = 1/lambda^2 = scale^2

### Exercise 2.1: Working with Continuous Distributions

<details>
<summary>Hint: scipy.stats function reference</summary>

- `stats.uniform(loc=a, scale=b-a)` -- Uniform on [a, b]
- `stats.norm(loc=mu, scale=sigma)` -- Normal with mean mu, std sigma
- `stats.expon(scale=1/lambda)` -- Exponential with rate lambda
- `.cdf(x)` gives P(X <= x)
- `.pdf(x)` gives the density at x
</details>

Solve these problems:

1. You generate a random number between 0 and 10. What is P(3 < X < 7)?
2. Adult male heights have mean 70 inches, std dev 3 inches. What percentage of men are between 5'6" (66 in) and 6'0" (72 in)?
3. Customers arrive at a coffee shop every 4 minutes on average. What is the probability that you wait more than 10 minutes for the next customer?

<details>
<summary>Solution</summary>

```python
from scipy import stats

# 1. Uniform(0, 10)
u = stats.uniform(loc=0, scale=10)
print(f"P(3 < X < 7): {u.cdf(7) - u.cdf(3):.2f}")  # 0.40

# 2. Normal(70, 3)
h = stats.norm(loc=70, scale=3)
print(f"P(66 < height < 72): {h.cdf(72) - h.cdf(66):.4f}")  # 0.6563 = 65.6%

# 3. Exponential(scale=4)
w = stats.expon(scale=4)
print(f"P(wait > 10 min): {1 - w.cdf(10):.4f}")  # 0.0821 = 8.2%
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain the difference between PMF and PDF
- [ ] Use the CDF to compute probabilities for continuous distributions
- [ ] Apply the 68-95-99.7 rule to any normal distribution
- [ ] Explain why the normal distribution is not always appropriate (fat tails)

---

## Section 3: Expected Value and Variance

Every distribution has an expected value (E[X]) and variance (Var(X)). These two numbers summarize a distribution's center and spread. In gambling, E[X] tells you the house edge. In finance, E[X] is expected return and Var(X) is risk.

### Formulas for Each Distribution

| Distribution | E[X] | Var(X) |
|---|---|---|
| Bernoulli(p) | p | p(1-p) |
| Binomial(n,p) | np | np(1-p) |
| Poisson(lambda) | lambda | lambda |
| Geometric(p) | 1/p | (1-p)/p^2 |
| Uniform(a,b) | (a+b)/2 | (b-a)^2/12 |
| Normal(mu, sigma) | mu | sigma^2 |
| Exponential(lambda) | 1/lambda | 1/lambda^2 |

You can always verify these with scipy.stats:

```python
from scipy import stats

dist = stats.binom(n=10, p=0.3)
print(f"E[X] = {dist.mean():.2f}")   # 3.00 = 10 * 0.3
print(f"Var(X) = {dist.var():.2f}")  # 2.10 = 10 * 0.3 * 0.7
print(f"Std(X) = {dist.std():.4f}") # 1.4491
```

**Expected Output:**
```
E[X] = 3.00
Var(X) = 2.10
Std(X) = 1.4491
```

### Computing the House Edge

The house edge is the negative expected value per bet, expressed as a percentage of the bet. It tells you exactly how much the casino expects to make per dollar wagered.

```python
import numpy as np

# American Roulette: bet $1 on red
# 18 red (win $1), 18 black (lose $1), 2 green (lose $1)
p_win = 18/38
p_lose = 20/38

ev = (1 * p_win) + (-1 * p_lose)
print(f"EV per $1 bet on red: ${ev:.4f}")
print(f"House edge: {-ev:.2%}")

# Variance per bet
var = ((1 - ev)**2 * p_win) + ((-1 - ev)**2 * p_lose)
std = np.sqrt(var)
print(f"Std dev per bet: ${std:.4f}")

# Over 1000 bets of $1 each
n = 1000
print(f"\nOver {n} bets:")
print(f"  Expected total profit: ${ev * n:.2f}")
print(f"  Std dev of total: ${std * np.sqrt(n):.2f}")

# European Roulette (single zero): slightly better odds
p_win_eu = 18/37
ev_eu = (1 * p_win_eu) + (-1 * (1 - p_win_eu))
print(f"\nEuropean Roulette EV: ${ev_eu:.4f}")
print(f"European house edge: {-ev_eu:.2%}")
```

**Expected Output:**
```
EV per $1 bet on red: $-0.0526
House edge: 5.26%

Std dev per bet: $0.9986

Over 1000 bets:
  Expected total profit: $-52.63
  Std dev of total: $31.57

European Roulette EV: $-0.0270
European house edge: 2.70%
```

Every dollar bet on American Roulette costs you 5.26 cents on average. Over 1000 bets, you expect to lose about $52.63. The standard deviation of $31.57 means some sessions you might only lose $20, and rarely you might even come out ahead -- but on average, the house wins.

### Portfolio Expected Return and Risk

```python
from scipy import stats
import numpy as np

# Two assets modeled as normal distributions
tech_stock = stats.norm(loc=0.12, scale=0.25)   # 12% return, 25% risk
bond_fund  = stats.norm(loc=0.04, scale=0.05)   # 4% return, 5% risk

# 60/40 portfolio (assuming independence for simplicity)
w_tech, w_bond = 0.6, 0.4
port_return = w_tech * tech_stock.mean() + w_bond * bond_fund.mean()
port_var = (w_tech**2) * tech_stock.var() + (w_bond**2) * bond_fund.var()
port_std = np.sqrt(port_var)

print(f"Portfolio expected return: {port_return:.2%}")
print(f"Portfolio risk (std dev): {port_std:.2%}")

# Probability of losing money
port = stats.norm(loc=port_return, scale=port_std)
print(f"P(negative return): {port.cdf(0):.4f}")
print(f"P(return > 20%): {1 - port.cdf(0.20):.4f}")
```

**Expected Output:**
```
Portfolio expected return: 8.80%
Portfolio risk (std dev): 15.20%
P(negative return): 0.2815
P(return > 20%): 0.2295
```

### Exercise 3.1: House Edge Calculations

Compute the expected value per bet and the house edge for each game.

1. **European Roulette single number bet**: pays 35-to-1, 37 slots (0 through 36)
2. **Craps pass-line bet**: wins with probability 244/495, pays 1-to-1
3. **Sports bet at -110 odds**: you bet $110 to win $100, on a true 50/50 game

<details>
<summary>Hint</summary>

For each game: EV = (payout_if_win * P(win)) + (payout_if_lose * P(lose)). Remember that "payout" is your net gain or loss -- if you bet $110 and lose, you lose $110.
</details>

<details>
<summary>Solution</summary>

```python
# 1. European Roulette: single number
p_win = 1/37
ev = 35 * p_win + (-1) * (1 - p_win)
print(f"Single number EV: ${ev:.4f}, House edge: {-ev:.2%}")
# EV = -$0.0270, House edge = 2.70%

# 2. Craps pass-line
p_win = 244/495
ev = 1 * p_win + (-1) * (1 - p_win)
print(f"Craps pass-line EV: ${ev:.4f}, House edge: {-ev:.2%}")
# EV = -$0.0141, House edge = 1.41%

# 3. Sports bet at -110 on 50/50 game
p_win = 0.50
ev = 100 * p_win + (-110) * (1 - p_win)
ev_per_dollar = ev / 110  # normalize to per dollar wagered
print(f"Sports bet EV: ${ev:.2f} per bet")
print(f"Per dollar wagered: ${ev_per_dollar:.4f}, House edge: {-ev_per_dollar:.2%}")
# EV = -$5.00 per bet, House edge = 4.55%
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Compute E[X] and Var(X) from distribution parameters
- [ ] Calculate the house edge for any gambling game
- [ ] Explain why variance matters (it quantifies uncertainty/risk)
- [ ] Compute expected portfolio return from individual asset returns

---

## Section 4: Law of Large Numbers

The Law of Large Numbers (LLN) is one of the two foundational theorems of probability. It says: as you take more and more independent samples from a distribution, the sample mean converges to the true expected value.

This is simple to state but profound in its consequences:
- It is why casinos are guaranteed to profit over millions of bets
- It is why insurance companies can set premiums without knowing which specific customers will file claims
- It is why you need a large sample of bets to evaluate a sports betting strategy
- It is why diversification reduces risk in a portfolio

### Visualizing Convergence

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate roulette: bet $1 on red each time
p_red = 18/38
n_bets = 10000
ev = (18/38) * 1 + (20/38) * (-1)  # theoretical EV = -0.0526

# Generate outcomes: +1 (win) or -1 (lose)
outcomes = np.where(np.random.random(n_bets) < p_red, 1, -1)

# Compute running average
running_avg = np.cumsum(outcomes) / np.arange(1, n_bets + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(running_avg, linewidth=0.8, color='steelblue')
plt.axhline(y=ev, color='red', linestyle='--', linewidth=2,
            label=f'Theoretical EV = ${ev:.4f}')
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Number of Bets')
plt.ylabel('Average Profit per Bet ($)')
plt.title('Law of Large Numbers: Roulette Running Average')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print convergence at different points
for n in [10, 100, 1000, 5000, 10000]:
    print(f"After {n:>5} bets: avg = ${running_avg[n-1]:+.4f}  "
          f"(error: ${abs(running_avg[n-1] - ev):.4f})")
```

**Expected Output:**
```
After    10 bets: avg = $+0.2000  (error: $0.2526)
After   100 bets: avg = $-0.0400  (error: $0.0126)
After  1000 bets: avg = $-0.0380  (error: $0.0146)
After  5000 bets: avg = $-0.0548  (error: $0.0022)
After 10000 bets: avg = $-0.0512  (error: $0.0014)
```

After 10 bets, the average is wildly off -- the gambler is actually up $0.20 per bet. After 10,000 bets, the average is within a fraction of a penny of the theoretical house edge. Short-run luck is real. Long-run averages are predictable.

### Why Casinos Always Win

This is the key insight: a casino does not know whether you will win or lose tonight. But it knows that across all its tables, all night, all week, the average profit per bet will be very close to the house edge. The casino does not gamble -- it collects a mathematical tax. LLN is its business model.

### Sports Betting Implications

```python
import numpy as np

np.random.seed(42)

# A bettor wins 53% of bets at -110 odds (positive EV!)
# EV per bet = 100*0.53 - 110*0.47 = 53 - 51.7 = +$1.30
p_win = 0.53
n_simulations = 1000

for n_bets in [20, 100, 500, 1000]:
    profits = []
    for _ in range(n_simulations):
        wins = np.random.binomial(n_bets, p_win)
        profit = wins * 100 - (n_bets - wins) * 110
        profits.append(profit)
    profits = np.array(profits)
    pct_profitable = np.mean(profits > 0) * 100
    print(f"After {n_bets:>4} bets: {pct_profitable:.1f}% of bettors profitable, "
          f"avg profit = ${np.mean(profits):+.0f}")
```

**Expected Output:**
```
After   20 bets: 56.5% of bettors profitable, avg profit = $+20
After  100 bets: 61.3% of bettors profitable, avg profit = $+121
After  500 bets: 73.9% of bettors profitable, avg profit = $+636
After 1000 bets: 82.3% of bettors profitable, avg profit = $+1264
```

Even with a positive edge, 43.5% of bettors are in the red after just 20 bets. You need hundreds of bets for LLN to separate genuine skill from noise. This is why professional sports bettors talk about "sample size" -- you cannot evaluate a strategy on 20 bets.

### Common Misconceptions

**LLN does NOT mean results balance out.** If you lose 10 bets in a row, LLN does not mean you will win the next 10. Future bets do not "remember" past ones. LLN says the average converges -- not that individual outcomes compensate for each other.

**LLN does NOT mean you can't win at gambling.** An individual gambler can absolutely win in a session. The key is that the gambler's expected value is negative, and LLN ensures the casino's aggregate average converges to the house edge.

### Exercise 4.1: LLN Simulation

Simulate rolling a fair die 10,000 times. Plot the running average and verify it converges to 3.5. Then repeat with a loaded die where P(6) = 0.3 and all other faces have probability 0.14 each. What does the running average converge to?

<details>
<summary>Hint</summary>

For the loaded die, E[X] = 1(0.14) + 2(0.14) + 3(0.14) + 4(0.14) + 5(0.14) + 6(0.30). Use `np.random.choice` with a probability array.
</details>

<details>
<summary>Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Fair die
n = 10000
fair_rolls = np.random.randint(1, 7, size=n)
fair_avg = np.cumsum(fair_rolls) / np.arange(1, n + 1)

# Loaded die: P(6) = 0.30, others = 0.14 each
probs = [0.14, 0.14, 0.14, 0.14, 0.14, 0.30]
loaded_rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=n, p=probs)
loaded_avg = np.cumsum(loaded_rolls) / np.arange(1, n + 1)

# Theoretical expected values
fair_ev = 3.5
loaded_ev = sum((i+1) * p for i, p in enumerate(probs))
print(f"Fair die E[X] = {fair_ev}")
print(f"Loaded die E[X] = {loaded_ev:.2f}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(fair_avg, linewidth=0.8)
ax1.axhline(y=fair_ev, color='r', linestyle='--', label=f'E[X] = {fair_ev}')
ax1.set_title('Fair Die')
ax1.set_xlabel('Number of Rolls')
ax1.set_ylabel('Running Average')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(loaded_avg, linewidth=0.8)
ax2.axhline(y=loaded_ev, color='r', linestyle='--', label=f'E[X] = {loaded_ev:.2f}')
ax2.set_title('Loaded Die')
ax2.set_xlabel('Number of Rolls')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

The loaded die converges to E[X] = 1(0.14) + 2(0.14) + 3(0.14) + 4(0.14) + 5(0.14) + 6(0.30) = 3.90.
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] Explain the Law of Large Numbers in your own words
- [ ] Explain why casinos are guaranteed to profit in the long run
- [ ] Explain why a small sample of bets is unreliable for evaluating a strategy
- [ ] Simulate LLN convergence in Python

---

## Section 5: Central Limit Theorem

The Central Limit Theorem (CLT) is the second foundational theorem of probability, and many statisticians consider it the most important theorem in all of statistics.

**The claim:** Take any distribution -- Poisson, Exponential, Uniform, anything. Draw samples of size n, compute the mean, and repeat thousands of times. The distribution of those sample means will be approximately normal (bell-shaped), regardless of the shape of the original distribution.

This is surprising. The original distribution can be skewed, multimodal, or flat. But averages of samples from it will form a bell curve.

### Why It Matters

Without the CLT, every statistical method would need to know the exact distribution of the data. With the CLT, you can use the same methods -- confidence intervals, hypothesis tests, regression -- regardless of the underlying distribution, because sample means are always approximately normal. It is a universal approximation theorem for averages.

### Demonstrating the CLT

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Start with an exponential distribution (very skewed, not normal at all)
rate = 2.0
true_mean = 1 / rate  # 0.5
true_std = 1 / rate    # 0.5

n_experiments = 5000
sample_sizes = [1, 5, 30, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, n in zip(axes.flat, sample_sizes):
    # Each experiment: draw n values, compute their mean
    sample_means = np.array([
        np.mean(np.random.exponential(scale=1/rate, size=n))
        for _ in range(n_experiments)
    ])

    # Plot histogram
    ax.hist(sample_means, bins=50, density=True, alpha=0.7,
            edgecolor='black', linewidth=0.5)

    # Overlay the theoretical normal predicted by CLT
    std_error = true_std / np.sqrt(n)
    x = np.linspace(true_mean - 4*std_error, true_mean + 4*std_error, 200)
    ax.plot(x, stats.norm.pdf(x, true_mean, std_error), 'r-', linewidth=2,
            label='CLT prediction')

    ax.axvline(true_mean, color='green', linestyle='--', alpha=0.7, label='True mean')
    ax.set_title(f'Sample size n = {n}')
    ax.set_xlabel('Sample Mean')
    ax.legend(fontsize=8)

plt.suptitle('Central Limit Theorem: Means from Exponential(2)', fontsize=14)
plt.tight_layout()
plt.show()

# Print how the spread decreases
print("Standard error of sample mean:")
for n in sample_sizes:
    se = true_std / np.sqrt(n)
    print(f"  n = {n:>3}: std error = {se:.4f}")
```

**Expected Output:**
```
Standard error of sample mean:
  n =   1: std error = 0.5000
  n =   5: std error = 0.2236
  n =  30: std error = 0.0913
  n = 100: std error = 0.0500
```

**What you should see in the plots:**
- **n=1**: The histogram looks like the original exponential distribution -- very right-skewed. No bell curve at all.
- **n=5**: Starting to look more symmetric, but still skewed.
- **n=30**: Essentially a bell curve. The red line (CLT prediction) fits the histogram well.
- **n=100**: Almost perfectly normal. The CLT approximation is excellent.

### The Key Formula

If the original distribution has mean mu and standard deviation sigma, the CLT says the sample mean (for samples of size n) has:
- **Mean** = mu (same as the original)
- **Standard deviation** = sigma / sqrt(n) (called the "standard error")

Larger samples give tighter estimates. This is why polling organizations survey thousands of people -- the standard error shrinks with sqrt(n).

### CLT with Different Starting Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

n = 50            # sample size
n_experiments = 5000

# Three very different distributions
distributions = {
    'Uniform(0,1)': {
        'sampler': lambda size: np.random.uniform(0, 1, size),
        'mean': 0.5,
        'std': 1/np.sqrt(12)
    },
    'Exponential(1)': {
        'sampler': lambda size: np.random.exponential(1, size),
        'mean': 1.0,
        'std': 1.0
    },
    'Loaded Die': {
        'sampler': lambda size: np.random.choice(
            [1,2,3,4,5,6], size=size, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.2]),
        'mean': 3.0,  # 1*0.4 + 2*0.1 + 3*0.1 + 4*0.1 + 5*0.1 + 6*0.2
        'std': 1.897
    }
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, d) in zip(axes, distributions.items()):
    means = np.array([np.mean(d['sampler'](n)) for _ in range(n_experiments)])

    ax.hist(means, bins=50, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)

    se = d['std'] / np.sqrt(n)
    x = np.linspace(d['mean'] - 4*se, d['mean'] + 4*se, 200)
    ax.plot(x, stats.norm.pdf(x, d['mean'], se), 'r-', linewidth=2)

    ax.set_title(f'{name}\n(n={n} per sample)')
    ax.set_xlabel('Sample Mean')

plt.suptitle('CLT Works Regardless of Original Distribution', fontsize=14)
plt.tight_layout()
plt.show()
```

All three histograms should look like bell curves with the CLT normal overlay fitting well, even though the original distributions are flat (Uniform), skewed (Exponential), and discrete/irregular (Loaded Die).

### Finance Application

A portfolio of many uncorrelated assets: each asset has its own return distribution (possibly skewed, possibly fat-tailed). But the portfolio return is the weighted average of many returns, and by the CLT, this average is approximately normal. This is why portfolio theory uses the normal distribution even though individual stock returns are not perfectly normal.

### ML Application

Many ML algorithms assume features or residuals are normally distributed. The CLT provides the theoretical justification: if a feature is the sum or average of many small, independent factors, it will be approximately normal. Height is the result of many genetic and environmental factors -- the CLT explains why height is normally distributed.

### Common Misconceptions

**"CLT means all data is normally distributed."** No. The CLT says sample *means* are normally distributed, not the raw data. Individual data points can follow any distribution.

**"You always need n >= 30."** The number 30 is a rule of thumb, not a law. For symmetric distributions, n = 10 may suffice. For heavily skewed distributions, you might need n > 100. It depends on how non-normal the original distribution is.

### Exercise 5.1: CLT Verification

Verify the CLT with the Poisson distribution. For Poisson(lambda=3):
1. Draw 5,000 samples of size n=40, compute the mean of each
2. Plot the histogram of sample means
3. Overlay the normal distribution predicted by CLT (mean = 3, std error = sqrt(3)/sqrt(40))
4. Report what percentage of sample means fall within 1 and 2 standard errors of the true mean

<details>
<summary>Solution</summary>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

lam = 3
n = 40
n_experiments = 5000

# Draw sample means
sample_means = np.array([
    np.mean(np.random.poisson(lam, size=n))
    for _ in range(n_experiments)
])

# CLT prediction
se = np.sqrt(lam) / np.sqrt(n)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(lam - 4*se, lam + 4*se, 200)
plt.plot(x, stats.norm.pdf(x, lam, se), 'r-', linewidth=2, label='CLT Normal')
plt.axvline(lam, color='green', linestyle='--', label=f'True mean = {lam}')
plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title(f'CLT: Sample Means from Poisson({lam}), n={n}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Check 68-95 rule
within_1se = np.mean(np.abs(sample_means - lam) <= se) * 100
within_2se = np.mean(np.abs(sample_means - lam) <= 2 * se) * 100
print(f"Within 1 SE: {within_1se:.1f}% (expected ~68%)")
print(f"Within 2 SE: {within_2se:.1f}% (expected ~95%)")
```
</details>

**Self-Check:** Before moving on, make sure you can:
- [ ] State the Central Limit Theorem in your own words
- [ ] Explain why sample means are approximately normal even when the data is not
- [ ] Compute the standard error of a sample mean
- [ ] Explain why CLT is considered the most important theorem in statistics

---

## Practice Project: Monte Carlo Casino Simulation

Now let's put everything together. You will build a Monte Carlo simulation that demonstrates every concept from this route.

### Project Goals

- Simulate three casino games (roulette, simplified blackjack, craps)
- Run many sessions to see LLN and CLT in action
- Visualize convergence and outcome distributions
- Calculate the probability of a gambler walking away with a profit

### Requirements

1. Simulate 1,000 bets per session for each game
2. Run 1,000 sessions (gamblers) for each game
3. Compare simulated results to theoretical expected values
4. Plot LLN convergence (running average over bets)
5. Plot CLT distribution (histogram of session outcomes)
6. Report the percentage of gamblers who finish with a profit

### Getting Started

Start with roulette -- it is the simplest game to simulate.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================
# GAME 1: American Roulette (bet on red)
# ============================================================
# 18 red, 18 black, 2 green. Bet $1 on red each time.
# Win: +$1, Lose: -$1

def simulate_roulette(n_bets):
    """Simulate n_bets on red. Return array of per-bet outcomes (+1 or -1)."""
    p_red = 18/38
    return np.where(np.random.random(n_bets) < p_red, 1, -1)

# Theoretical values
roulette_ev = (18/38) * 1 + (20/38) * (-1)  # -0.0526
roulette_std = np.sqrt((1 - roulette_ev)**2 * 18/38 + (-1 - roulette_ev)**2 * 20/38)

print(f"Roulette EV per bet: ${roulette_ev:.4f}")
print(f"Roulette std per bet: ${roulette_std:.4f}")
```

**Expected Output:**
```
Roulette EV per bet: $-0.0526
Roulette std per bet: $0.9986
```

### Full Implementation

<details>
<summary>If you want to try building it yourself first, skip this section</summary>

Here is one possible approach. Try to write your own version before looking at this.
</details>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ============================================================
# Game Simulators
# ============================================================

def simulate_roulette(n_bets):
    """American Roulette: bet $1 on red. Win +1, Lose -1."""
    p_red = 18/38
    return np.where(np.random.random(n_bets) < p_red, 1, -1)

def simulate_blackjack(n_bets):
    """Simplified Blackjack: bet $1 each hand.
    Win: +1 (P=0.42), Push: 0 (P=0.08), Lose: -1 (P=0.50).
    Approximate probabilities for basic strategy."""
    r = np.random.random(n_bets)
    outcomes = np.where(r < 0.42, 1, np.where(r < 0.50, 0, -1))
    return outcomes

def simulate_craps(n_bets):
    """Craps pass-line bet: $1 each roll.
    Win: +1 (P=244/495), Lose: -1 (P=251/495)."""
    p_win = 244/495
    return np.where(np.random.random(n_bets) < p_win, 1, -1)

# ============================================================
# Theoretical Expected Values
# ============================================================
games = {
    'Roulette (Red)': {
        'simulator': simulate_roulette,
        'ev': (18/38) * 1 + (20/38) * (-1),
        'std': None  # computed below
    },
    'Blackjack (Basic)': {
        'simulator': simulate_blackjack,
        'ev': 0.42 * 1 + 0.08 * 0 + 0.50 * (-1),
        'std': None
    },
    'Craps (Pass)': {
        'simulator': simulate_craps,
        'ev': (244/495) * 1 + (251/495) * (-1),
        'std': None
    }
}

# Compute standard deviations
for name, g in games.items():
    if 'Blackjack' in name:
        g['std'] = np.sqrt(0.42*(1 - g['ev'])**2 + 0.08*(0 - g['ev'])**2
                           + 0.50*(-1 - g['ev'])**2)
    else:
        p_win = (1 + g['ev']) / 2  # solve from ev = 2p - 1
        g['std'] = np.sqrt(p_win*(1 - g['ev'])**2 + (1-p_win)*(-1 - g['ev'])**2)

print("Theoretical Values:")
print(f"{'Game':<20} {'EV/bet':>10} {'Std/bet':>10} {'House Edge':>12}")
print("-" * 55)
for name, g in games.items():
    print(f"{name:<20} ${g['ev']:>+8.4f} ${g['std']:>8.4f} {-g['ev']:>10.2%}")

# ============================================================
# Run Simulations
# ============================================================
n_bets = 1000       # bets per session
n_sessions = 1000   # number of gamblers

print(f"\nSimulating {n_sessions} sessions of {n_bets} bets each...\n")

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

for i, (name, g) in enumerate(games.items()):
    # --- Run all sessions ---
    session_profits = []
    for _ in range(n_sessions):
        outcomes = g['simulator'](n_bets)
        session_profits.append(np.sum(outcomes))
    session_profits = np.array(session_profits)

    # --- Results ---
    sim_ev = np.mean(session_profits) / n_bets
    pct_profitable = np.mean(session_profits > 0) * 100

    print(f"{name}:")
    print(f"  Theoretical EV/bet: ${g['ev']:+.4f}")
    print(f"  Simulated EV/bet:   ${sim_ev:+.4f}")
    print(f"  Gamblers profitable: {pct_profitable:.1f}%")
    print()

    # --- LLN Plot: Running average for one session ---
    single_session = g['simulator'](n_bets)
    running_avg = np.cumsum(single_session) / np.arange(1, n_bets + 1)

    ax_lln = axes[i, 0]
    ax_lln.plot(running_avg, linewidth=0.8, color='steelblue')
    ax_lln.axhline(y=g['ev'], color='red', linestyle='--', linewidth=2,
                   label=f"E[X] = ${g['ev']:.4f}")
    ax_lln.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_lln.set_xlabel('Bet Number')
    ax_lln.set_ylabel('Avg Profit/Bet ($)')
    ax_lln.set_title(f'{name} - LLN Convergence')
    ax_lln.legend()
    ax_lln.grid(True, alpha=0.3)

    # --- CLT Plot: Distribution of session outcomes ---
    session_mean = g['ev'] * n_bets
    session_std = g['std'] * np.sqrt(n_bets)

    ax_clt = axes[i, 1]
    ax_clt.hist(session_profits, bins=50, density=True, alpha=0.7,
                edgecolor='black', linewidth=0.5)

    # Normal overlay (CLT prediction)
    x = np.linspace(session_mean - 4*session_std, session_mean + 4*session_std, 200)
    ax_clt.plot(x, stats.norm.pdf(x, session_mean, session_std), 'r-',
                linewidth=2, label='CLT Normal')
    ax_clt.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Break even')
    ax_clt.axvline(x=session_mean, color='red', linestyle=':', alpha=0.7,
                   label=f'Expected = ${session_mean:.0f}')
    ax_clt.set_xlabel('Session Profit ($)')
    ax_clt.set_ylabel('Density')
    ax_clt.set_title(f'{name} - CLT (Session Outcomes)')
    ax_clt.legend(fontsize=8)
    ax_clt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# Probability of Profit (Theoretical vs Simulated)
# ============================================================
print("\nProbability of Profit (1000-bet session):")
print(f"{'Game':<20} {'Theoretical':>12} {'Simulated':>12}")
print("-" * 45)
for name, g in games.items():
    session_mean = g['ev'] * n_bets
    session_std = g['std'] * np.sqrt(n_bets)
    theoretical_p_profit = 1 - stats.norm.cdf(0, session_mean, session_std)

    session_profits = np.array([np.sum(g['simulator'](n_bets))
                                for _ in range(n_sessions)])
    simulated_p_profit = np.mean(session_profits > 0)

    print(f"{name:<20} {theoretical_p_profit:>11.1%} {simulated_p_profit:>11.1%}")
```

**Expected Output (approximate):**
```
Theoretical Values:
Game                     EV/bet    Std/bet   House Edge
-------------------------------------------------------
Roulette (Red)         $-0.0526   $ 0.9986       5.26%
Blackjack (Basic)      $-0.0800   $ 0.9252       8.00%
Craps (Pass)           $-0.0141   $ 0.9999       1.41%

Simulating 1000 sessions of 1000 bets each...

Roulette (Red):
  Theoretical EV/bet: $-0.0526
  Simulated EV/bet:   $-0.0524
  Gamblers profitable: 4.8%

Blackjack (Basic):
  Theoretical EV/bet: $-0.0800
  Simulated EV/bet:   $-0.0813
  Gamblers profitable: 0.4%

Craps (Pass):
  Theoretical EV/bet: $-0.0141
  Simulated EV/bet:   $-0.0136
  Gamblers profitable: 33.1%

Probability of Profit (1000-bet session):
Game                 Theoretical    Simulated
---------------------------------------------
Roulette (Red)            4.9%         5.0%
Blackjack (Basic)         0.3%         0.4%
Craps (Pass)             32.7%        33.0%
```

### What the Results Show

**LLN in action (left column):** The running average for each game starts noisy and converges to the theoretical expected value. After 1,000 bets, it is very close to the house edge.

**CLT in action (right column):** The distribution of session outcomes across 1,000 gamblers forms a bell curve centered on the expected total loss. The red normal overlay (predicted by CLT) fits the histogram well.

**The gambler's odds:** With a 5.26% house edge (roulette), only about 5% of gamblers walk away with a profit after 1,000 bets. With the tiny 1.41% house edge (craps pass line), about a third of gamblers come out ahead -- but the casino still profits on average.

### Extending the Project

If you want to go further, try:
- Add a game with variable bet sizes (e.g., Martingale strategy in roulette -- double after each loss)
- Simulate a card-counting blackjack strategy and check if the edge flips to the player
- Model a sports bettor with a 53% win rate at -110 odds and determine how many bets they need to be 95% confident of a profit
- Add a "gambler's ruin" simulation: start with $100, bet $1 at a time, and see how often the gambler goes broke before doubling their money

---

## Summary

### Key Takeaways

- **Probability distributions** are complete mathematical models of random processes. They describe every possible outcome and its likelihood.
- **Discrete distributions** (Bernoulli, Binomial, Poisson, Geometric) model countable outcomes like wins, goals, defaults, and trials until success.
- **Continuous distributions** (Uniform, Normal, Exponential) model outcomes on a continuous scale like returns, heights, and waiting times.
- **Expected value** is the long-run average. **Variance** measures the spread around that average. Together they quantify risk and reward.
- **The Law of Large Numbers** guarantees that sample means converge to the true expected value. This is why casinos always win in aggregate and why you need large samples to evaluate strategies.
- **The Central Limit Theorem** guarantees that sample means are approximately normally distributed, regardless of the original distribution. This is why the normal distribution appears everywhere and why so many statistical methods work.

### Skills You've Gained

You can now:
- Identify which distribution models a given scenario
- Compute probabilities using PMF, PDF, and CDF
- Calculate expected values and variances from distribution parameters
- Explain the business model of a casino using LLN
- Explain why sample means form bell curves using CLT
- Build Monte Carlo simulations to verify and explore probabilistic claims

---

## Next Steps

### Continue Learning

**Build on this topic:**
- [Statistical Inference](/routes/statistical-inference/map.md) -- Confidence intervals, hypothesis testing, p-values (builds directly on CLT)

**Explore related routes:**
- [Bayesian Statistics](/routes/bayesian-statistics/map.md) -- Prior and posterior distributions, Bayesian updating
- [Regression and Modeling](/routes/regression-and-modeling/map.md) -- Linear regression, residual distributions, prediction intervals

### Practice More

- Simulate other casino games: poker, baccarat, slot machines
- Download real stock return data and test normality assumptions
- Model sports outcomes (baseball hit rates, soccer goal distributions) with real data
- Build a sports betting bankroll simulator

---

## Appendix: Distribution Cheat Sheet

### When to Use Each Distribution

| Scenario | Distribution | Parameters |
|---|---|---|
| Single yes/no outcome | Bernoulli | p = P(success) |
| Count successes in n trials | Binomial | n = trials, p = P(success) |
| Count events in fixed interval | Poisson | lambda = avg events per interval |
| Trials until first success | Geometric | p = P(success) |
| Equally likely values in a range | Uniform | a = min, b = max |
| Symmetric bell-shaped data | Normal | mu = mean, sigma = std dev |
| Time between Poisson events | Exponential | lambda = rate (scale = 1/lambda) |

### Quick Formula Reference

| Distribution | E[X] | Var(X) | scipy.stats |
|---|---|---|---|
| Bernoulli(p) | p | p(1-p) | `stats.bernoulli(p)` |
| Binomial(n,p) | np | np(1-p) | `stats.binom(n, p)` |
| Poisson(lambda) | lambda | lambda | `stats.poisson(mu)` |
| Geometric(p) | 1/p | (1-p)/p^2 | `stats.geom(p)` |
| Uniform(a,b) | (a+b)/2 | (b-a)^2/12 | `stats.uniform(loc=a, scale=b-a)` |
| Normal(mu,sigma) | mu | sigma^2 | `stats.norm(loc=mu, scale=sigma)` |
| Exponential(lambda) | 1/lambda | 1/lambda^2 | `stats.expon(scale=1/lambda)` |

### Common scipy.stats Methods

```python
from scipy import stats

dist = stats.norm(loc=0, scale=1)  # create a distribution object

dist.pmf(k)        # P(X = k) -- discrete distributions only
dist.pdf(x)        # density at x -- continuous distributions only
dist.cdf(x)        # P(X <= x)
dist.sf(x)         # P(X > x) = 1 - CDF(x)  (survival function)
dist.ppf(q)        # inverse CDF: value x such that P(X <= x) = q
dist.mean()        # expected value
dist.var()         # variance
dist.std()         # standard deviation
dist.rvs(size=n)   # generate n random samples
dist.interval(0.95)  # central 95% interval
```
