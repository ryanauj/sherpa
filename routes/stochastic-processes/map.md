---
title: Stochastic Processes
topics:
  - Random Walks
  - Markov Chains
  - Monte Carlo Methods
  - Poisson Processes
  - Time Series Foundations
related_routes:
  - probability-distributions
  - bayesian-statistics
  - regression-and-modeling
---

# Stochastic Processes - Route Map

## Overview

This route teaches mathematical models for systems that evolve randomly over time -- from random walks to Markov chains to Monte Carlo simulation. Where probability distributions describe single random outcomes, stochastic processes describe sequences of random outcomes that unfold over time or across steps. You'll learn to model how a gambler's bankroll drifts toward ruin, how credit ratings migrate between states, how random sampling can price financial derivatives, and how events cluster in time. Every concept connects directly to applications in gambling theory, quantitative finance, and machine learning.

## What You'll Learn

By following this route, you will:
- Model random walks and analyze their properties (expected position, variance growth, return probability, gambler's ruin)
- Build and analyze Markov chains using transition matrices, stationary distributions, and absorbing states
- Apply Monte Carlo methods for simulation, estimation, and numerical integration
- Understand Poisson processes for modeling random events in continuous time
- Analyze time series data using stationarity, autocorrelation, and basic AR/MA models
- Implement a Monte Carlo options pricing engine that connects stochastic modeling to quantitative finance

## Prerequisites

Before starting this route:
- **Required**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) (sample spaces, conditional probability, Bayes' theorem, independence)
- **Required**: [Probability Distributions](/routes/probability-distributions/map.md) (discrete and continuous distributions, expected value, variance, Law of Large Numbers, Central Limit Theorem)
- **Helpful**: [Bayesian Statistics](/routes/bayesian-statistics/map.md) (prior/posterior distributions, Bayesian updating -- useful for understanding MCMC)
- **Helpful**: [Linear Algebra Essentials](/routes/linear-algebra-essentials/map.md) (matrix multiplication, eigenvectors -- useful for Markov chain analysis)

## Route Structure

### 1. Random Walks
- Definition: a stochastic process built from a sequence of random steps
- 1D symmetric random walk: coin flips determine +1 or -1 steps
- Properties: expected position, variance that grows linearly with time, probability of return to origin
- Gambler's ruin: a gambler with $N betting $1 per round against a house with infinite bankroll
- Finance connection: random walk hypothesis of stock prices, geometric Brownian motion (conceptual)
- ML connection: random walks on graphs, node2vec and graph embeddings

### 2. Markov Chains
- The Markov property: the future depends only on the present state, not the history
- Transition matrices: encoding all state-to-state probabilities in a single matrix
- Stationary distributions: the long-run equilibrium behavior of a chain
- Absorbing states: states the chain can never leave (gambler's ruin revisited)
- Applications: credit rating transitions, PageRank, sports streaks, Hidden Markov Models

### 3. Monte Carlo Methods
- Core idea: use random sampling to estimate quantities that are hard to compute analytically
- Monte Carlo estimation: estimating pi, computing integrals
- Monte Carlo simulation: modeling complex systems with randomness (casino games, portfolios)
- Importance sampling: focusing computational effort on the regions that matter most
- MCMC connection: using Markov chains to sample from complex distributions

### 4. Poisson Processes
- Modeling events that occur randomly and independently in continuous time
- Properties: memorylessness, independent increments, connection to exponential inter-arrival times
- Applications: goals in soccer matches, trade arrivals in markets, insurance claims
- Compound and non-homogeneous Poisson processes (conceptual)

### 5. Time Series Foundations
- Stationarity: statistical properties that don't change over time
- Autocorrelation: how a value relates to its own past values
- Basic models: autoregressive (AR), moving average (MA), ARMA (conceptual introduction)
- Applications: team performance tracking, stock return analysis, volatility clustering

### 6. Practice Project -- Monte Carlo Options Pricing
- Implement a Monte Carlo simulation to price a European call option
- Model stock prices using geometric Brownian motion
- Analyze how parameters (volatility, time to expiry, strike price) affect option value
- Compare Monte Carlo estimates to the Black-Scholes analytical formula
- Extensions: path-dependent options, confidence intervals on price estimates

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Python 3 with `numpy` for numerical computation, random simulation, and linear algebra
- `scipy.stats` for distribution functions and statistical tests
- `matplotlib` for plotting random walks, distributions, convergence charts, and time series

## Next Steps

After completing this route:
- **[Bayesian Statistics](/routes/bayesian-statistics/map.md)** -- Prior and posterior distributions, MCMC sampling in depth
- **[Regression and Modeling](/routes/regression-and-modeling/map.md)** -- Linear regression, time series regression, prediction intervals
- **Quantitative Finance** -- Deep dive into options pricing, risk modeling, portfolio optimization
- **Reinforcement Learning** -- Markov Decision Processes extend Markov chains with actions and rewards
