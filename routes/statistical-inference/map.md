---
title: Statistical Inference
topics:
  - Sampling and Estimation
  - Confidence Intervals
  - Hypothesis Testing
  - P-Values and Significance
  - Types of Errors
related_routes:
  - stats-fundamentals
  - probability-distributions
  - regression-and-modeling
  - bayesian-statistics
---

# Statistical Inference - Route Map

## Overview

This route teaches statistical inference -- the art and science of drawing conclusions about populations from samples. It bridges the gap between descriptive statistics (summarizing what you see) and decision-making (acting on what you infer). You'll learn to quantify uncertainty, test claims rigorously, and avoid the reasoning traps that plague sports bettors, data scientists, and financial analysts alike.

## What You'll Learn

By following this route, you will:
- Understand how samples relate to populations and why sampling distributions are the foundation of inference
- Construct and correctly interpret confidence intervals for unknown parameters
- Formulate null and alternative hypotheses and conduct hypothesis tests
- Interpret p-values accurately and recognize when they are misused
- Distinguish Type I and Type II errors and analyze their trade-offs in real decisions
- Apply statistical inference to evaluate sports betting strategies, financial returns, and ML model comparisons

## Prerequisites

Before starting this route:
- **Required**: [Stats Fundamentals](/routes/stats-fundamentals/map.md) (mean, variance, standard deviation, distributions)
- **Required**: [Probability Distributions](/routes/probability-distributions/map.md) (especially the Central Limit Theorem)
- **Helpful**: [Probability Fundamentals](/routes/probability-fundamentals/map.md) (conditional probability, Bayes' theorem)

## Route Structure

### 1. Sampling and Estimation
- Populations vs. samples: the distinction that makes inference necessary
- Point estimates: using sample statistics to estimate population parameters
- Sampling distributions: what happens if you resample many times
- Standard error and its relationship to sample size

### 2. Confidence Intervals
- Constructing confidence intervals using the Central Limit Theorem
- What a 95% CI actually means (and what it does not mean)
- Confidence intervals for means, proportions, and differences
- How sample size and variability affect interval width

### 3. Hypothesis Testing
- Null and alternative hypotheses: framing a testable claim
- Test statistics: z-tests and t-tests
- The decision framework: reject or fail to reject
- One-tailed vs. two-tailed tests

### 4. P-Values and Statistical Significance
- What a p-value actually measures
- The arbitrary nature of significance thresholds
- P-hacking, data dredging, and publication bias
- Effect size: statistical significance vs. practical significance

### 5. Types of Errors and Power
- Type I error (false positive) and Type II error (false negative)
- Statistical power: the probability of detecting a real effect
- Power analysis: determining necessary sample sizes
- The multiple testing problem and corrections

### 6. Practice Project -- A/B Testing a Sports Betting Strategy
- Formulate hypotheses about a betting strategy's profitability
- Compute confidence intervals for the strategy's edge
- Conduct hypothesis tests and interpret p-values
- Assess error trade-offs and account for multiple comparisons

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Python with `scipy.stats`, `numpy`, and `matplotlib` for computation and visualization
- Simulation-based approaches to build intuition before formulas
- Templates from `/techniques/` for route structure

## Next Steps

After completing this route:
- **[Regression and Modeling](/routes/regression-and-modeling/map.md)** -- Apply inference to regression coefficients and model selection
- **[Bayesian Statistics](/routes/bayesian-statistics/map.md)** -- An alternative framework for inference using prior beliefs and updating
- **[Experimental Design](/routes/experimental-design/map.md)** -- How to design studies that produce valid inferences
