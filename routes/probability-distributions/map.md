---
title: Probability Distributions
topics:
  - Discrete Distributions
  - Continuous Distributions
  - Expected Value and Variance
  - Law of Large Numbers
  - Central Limit Theorem
related_routes:
  - probability-fundamentals
  - stats-fundamentals
  - statistical-inference
  - bayesian-statistics
---

# Probability Distributions - Route Map

## Overview

This route moves from individual probabilities to probability distributions -- mathematical models that describe all possible outcomes of a random process and how likely each one is. Where probability fundamentals taught you how to compute the chance of a single event, distributions give you the complete picture: every outcome, every likelihood, all at once. You'll learn the distributions that show up everywhere -- from casino floors to stock trading desks to machine learning pipelines -- and the two theorems (Law of Large Numbers and Central Limit Theorem) that explain why these distributions behave so predictably at scale.

## What You'll Learn

By following this route, you will:
- Understand and apply discrete distributions (Bernoulli, Binomial, Poisson, Geometric) to model count-based outcomes
- Work with continuous distributions (Uniform, Normal, Exponential) and interpret their density functions
- Compute expected values and variances for any distribution and use them to quantify risk and reward
- Apply the Law of Large Numbers to explain why casinos always win and diversification works
- Understand and apply the Central Limit Theorem -- the reason sample averages are normally distributed
- Build Monte Carlo simulations to verify theoretical results with code

## Prerequisites

Before starting this route:
- **Required**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) (sample spaces, conditional probability, Bayes' theorem, independence)
- **Helpful**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) (mean, median, standard deviation, basic data summarization)

## Route Structure

### 1. Discrete Distributions
- Bernoulli: single yes/no trial (free throw make/miss, stock up/down)
- Binomial: number of successes in n independent trials (wins in a season, successful trades in a week)
- Poisson: count of rare events in a fixed interval (goals per soccer match, market crashes per decade)
- Geometric: number of trials until first success (hands until winning at poker, calls until a sale)

### 2. Continuous Distributions
- Uniform: equally likely outcomes in a range (spinning a wheel, random number generation)
- Normal (Gaussian): the bell curve that appears everywhere (player stats, stock returns, measurement error)
- Exponential: time between events (time between goals, time between trades)

### 3. Expected Value and Variance of Distributions
- Expected value as long-run average: E[X] for each distribution
- Variance and standard deviation: quantifying spread and risk
- Computing the house edge in casino games
- Portfolio expected return and risk

### 4. Law of Large Numbers
- Sample mean converges to expected value as sample size grows
- Why casinos are guaranteed to profit over thousands of bets
- Why sports bettors need large samples to evaluate a strategy
- Why diversification reduces risk in finance

### 5. Central Limit Theorem
- Sample means are approximately normal regardless of the underlying distribution
- Why this is called "the most important theorem in statistics"
- Foundation for confidence intervals and hypothesis testing
- Why portfolio returns tend toward normal as you add assets

### 6. Practice Project -- Monte Carlo Casino Simulation
- Simulate thousands of rounds of roulette, blackjack, and craps
- Visualize how outcomes converge to expected values (LLN in action)
- Show the distribution of session outcomes (CLT in action)
- Calculate probability of a gambler going broke vs. walking away with profit

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Python 3 with `scipy.stats` for distribution functions (PMF, PDF, CDF, sampling)
- `numpy` for numerical computation and random simulation
- `matplotlib` for plotting distributions, histograms, and convergence charts

## Next Steps

After completing this route:
- **[Statistical Inference](/routes/statistical-inference/map.md)** -- Confidence intervals, hypothesis testing, p-values (builds directly on CLT)
- **[Bayesian Statistics](/routes/bayesian-statistics/map.md)** -- Prior and posterior distributions, Bayesian updating
- **[Regression and Modeling](/routes/regression-and-modeling/map.md)** -- Linear regression, residual distributions, prediction intervals
