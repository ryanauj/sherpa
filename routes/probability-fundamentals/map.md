---
title: Probability Fundamentals
topics:
  - Sample Spaces and Events
  - Probability Rules
  - Conditional Probability
  - Bayes' Theorem
  - Expected Value
related_routes:
  - stats-fundamentals
  - probability-distributions
  - bayesian-statistics
---

# Probability Fundamentals - Route Map

## Overview

This route teaches the core concepts of probability that underpin gambling and casino games, sports betting strategy, risk assessment in finance, and the mathematical foundations of machine learning. You will build a practical, applied understanding of probability — not just formulas, but how to think probabilistically about uncertain outcomes in the real world.

## What You'll Learn

By following this route, you will:
- Understand probability as a framework for quantifying uncertainty on a 0-to-1 scale
- Apply probability rules (addition, multiplication, complement) to compute the likelihood of combined events
- Compute conditional probabilities and understand how new information changes the odds
- Apply Bayes' theorem to update beliefs and predictions when new evidence arrives
- Calculate expected value to make rational decisions under uncertainty

## Prerequisites

Before starting this route:
- **Required**: Basic arithmetic — addition, subtraction, multiplication, division
- **Required**: Comfort with fractions, decimals, and percentages
- **Helpful**: The [stats-fundamentals](/routes/stats-fundamentals/map.md) route for context on data and distributions

## Route Structure

### 1. What Is Probability?
- The meaning of probability: quantifying how likely something is to happen
- Sample spaces and events — the building blocks
- Three interpretations: classical (equally likely outcomes), frequentist (long-run frequency), subjective (degree of belief)
- Examples from dice, cards, coins, and the roulette wheel

### 2. Probability Rules
- Addition rule for mutually exclusive events (heart OR diamond)
- Multiplication rule for independent events (rolling a 6 twice)
- Complement rule (probability something does NOT happen)
- Combining rules to solve poker, craps, and investment scenarios

### 3. Conditional Probability
- Definition of P(A|B) — probability of A given that B has occurred
- How new information shifts the odds in sports betting, finance, and medical testing
- The critical difference between independence and conditional dependence
- Foundation for Naive Bayes classification in machine learning

### 4. Bayes' Theorem
- Updating beliefs with new evidence: prior, likelihood, posterior
- Classic applications: medical testing, spam filtering, sports predictions
- Why humans are bad at this intuitively (base rate neglect)
- Bayesian reasoning as a general decision-making tool

### 5. Expected Value
- The long-run average outcome of a random process
- Why casinos always win: every game has negative expected value for the player
- Finding +EV opportunities in sports betting
- Expected return on investments and risk-adjusted decision-making

### 6. Practice Project
- Build a probability and expected value calculator in Python
- Compute win probabilities and expected values for common casino games
- Evaluate whether a given sports bet has positive expected value

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- Python 3 with the standard library (no external packages required for core concepts)
- `numpy` for the practice project and numerical computations (see `/tools/`)

## Next Steps

After completing this route:
- **[Probability Distributions](/routes/probability-distributions/map.md)** — Learn about binomial, normal, Poisson, and other distributions that model real-world randomness
- **[Bayesian Statistics](/routes/bayesian-statistics/map.md)** — Go deeper into Bayesian inference, priors, posteriors, and Bayesian modeling
- **[Stats Fundamentals](/routes/stats-fundamentals/map.md)** — If you haven't already, build your foundation in descriptive and inferential statistics
