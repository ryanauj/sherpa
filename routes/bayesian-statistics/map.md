---
title: Bayesian Statistics
topics:
  - Bayesian vs Frequentist Thinking
  - Prior Distributions
  - Likelihood and Posterior
  - Conjugate Priors
  - Markov Chain Monte Carlo
  - Bayesian Decision Theory
related_routes:
  - probability-fundamentals
  - probability-distributions
  - statistical-inference
  - stochastic-processes
---

# Bayesian Statistics - Route Map

## Overview

Bayesian statistics provides a framework for updating beliefs as new evidence arrives -- the natural way humans think about uncertainty, formalized mathematically. Where frequentist methods treat parameters as fixed unknowns and probabilities as long-run frequencies, Bayesian methods treat probability as a degree of belief and express uncertainty about parameters as full distributions. This route teaches you to build, update, and act on probabilistic models using the Bayesian paradigm, with applications to sports betting, finance, and machine learning.

## What You'll Learn

By following this route, you will:
- Understand the philosophical and practical differences between Bayesian and frequentist paradigms and know when each is appropriate
- Construct prior distributions that encode domain knowledge and update them with observed data to produce posteriors
- Compute posterior distributions analytically using conjugate priors for common data-generating processes
- Apply Markov Chain Monte Carlo (MCMC) methods to sample from complex posterior distributions that lack closed-form solutions
- Make optimal decisions under uncertainty using Bayesian decision theory, loss functions, and expected utility
- Build a complete Bayesian sports rating system that updates team strengths from game results and predicts future outcomes with calibrated uncertainty

## Prerequisites

Before starting this route:
- **Required**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) -- Bayes' theorem, conditional probability, expected value
- **Required**: [Probability Distributions](/routes/probability-distributions/map.md) -- Beta, Normal, Poisson, Gamma distributions and their properties
- **Required**: [Statistical Inference](/routes/statistical-inference/map.md) -- sampling distributions, estimation, hypothesis testing (to understand the frequentist baseline)
- **Helpful**: [Calculus for ML](/routes/calculus-for-ml/map.md) -- derivatives and optimization concepts (useful for understanding MCMC and posterior maximization)

## Route Structure

### 1. Bayesian vs Frequentist Thinking
- Two paradigms: probability as long-run frequency vs. probability as degree of belief
- Parameters are fixed (frequentist) vs. parameters have distributions (Bayesian)
- When each approach is more appropriate: large samples vs. small samples, prior knowledge vs. no prior knowledge
- Sports betting as natural Bayesian reasoning: "I believe this team has a 60% chance of winning"
- Finance and risk assessment with limited historical data
- ML connections: regularization as implicit priors, uncertainty quantification

### 2. Priors, Likelihood, and Posteriors
- The Bayesian update cycle: prior x likelihood is proportional to posterior
- Prior distributions: uninformative, weakly informative, and strongly informative
- Likelihood functions: how well does the data fit each possible parameter value?
- Posterior distributions: the complete answer to "what do I believe after seeing this data?"
- How the prior gets overwhelmed by data as sample size grows
- Sports: updating a prior on team strength as game results arrive
- Finance: prior belief about expected returns, updated with earnings data
- ML: L2 regularization is a Gaussian prior on weights

### 3. Conjugate Priors
- When the prior and posterior belong to the same distribution family
- Beta-Binomial: win rates, conversion rates, free throw percentages
- Normal-Normal: estimating true means with known or unknown variance
- Gamma-Poisson: scoring rates in sports, event counts in finance
- Why conjugacy matters: closed-form updates without simulation
- Practical applications: shrinkage estimators, empirical Bayes, batting average estimation

### 4. Introduction to MCMC
- Why we need MCMC: most real-world posteriors have no closed-form solution
- The core idea: construct a Markov chain whose stationary distribution is the posterior
- Metropolis-Hastings algorithm: propose, accept/reject, iterate
- Diagnostics: trace plots, autocorrelation, effective sample size, convergence checks
- Hierarchical models: partial pooling for team strength estimation
- When MCMC is overkill and when it is essential

### 5. Bayesian Decision Theory
- Decisions under uncertainty: expected utility and loss functions
- Choosing loss functions that match your problem (squared error, absolute error, 0-1 loss)
- Sports betting: the Kelly criterion for optimal bet sizing
- Finance: portfolio allocation under parameter uncertainty
- ML: Bayesian decision boundaries, credible intervals vs. confidence intervals
- When Bayesian and frequentist answers agree -- and when they diverge

### 6. Practice Project -- Bayesian Sports Rating System
- Build a system that rates teams using Bayesian updating
- Start with prior beliefs about team strengths
- Update after each game result using conjugate priors
- Predict future game outcomes with full uncertainty estimates
- Compare Bayesian ratings to simple win-percentage rankings
- Evaluate calibration: are the predicted probabilities reliable?

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Python with `scipy.stats`, `numpy`, and `matplotlib` for computation and visualization
- Conjugate prior formulas for closed-form Bayesian updates
- A from-scratch Metropolis-Hastings implementation for understanding MCMC
- Optionally, `PyMC` for production-grade probabilistic programming (conceptual references only -- all core code uses scipy/numpy)
- Templates from `/techniques/` for route structure

## Next Steps

After completing this route:
- **[Stochastic Processes](/routes/stochastic-processes/map.md)** -- Markov chains, random walks, and time-series models that build on the MCMC foundations here
- **[Regression and Modeling](/routes/regression-and-modeling/map.md)** -- Apply Bayesian methods to regression, model comparison, and prediction
- **[Neural Network Foundations](/routes/neural-network-foundations/map.md)** -- See how Bayesian ideas (priors as regularization, uncertainty quantification) appear in deep learning
