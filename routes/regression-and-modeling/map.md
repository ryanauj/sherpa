---
title: Regression and Modeling
topics:
  - Correlation and Causation
  - Simple Linear Regression
  - Multiple Linear Regression
  - Logistic Regression
  - Model Evaluation
related_routes:
  - stats-fundamentals
  - statistical-inference
  - probability-distributions
  - bayesian-statistics
---

# Regression and Modeling - Route Map

## Overview

This route teaches how to build, interpret, and evaluate statistical models that predict outcomes. You will move from understanding correlations to fitting regression models to classifying binary outcomes -- learning the techniques that power sports prediction systems, financial risk models, and machine learning pipelines. Every concept is grounded in applications from sports betting, casino gambling, finance, and data science so that the math always connects to real decisions.

## What You'll Learn

By following this route, you will:
- Distinguish correlation from causation and recognize spurious relationships, Simpson's paradox, and confounding variables
- Build and interpret simple linear regression models, including slope, intercept, and residual analysis
- Extend to multiple linear regression with several predictors, handle multicollinearity, and perform feature selection
- Apply logistic regression to classify binary outcomes and interpret coefficients as odds ratios
- Evaluate model quality using R-squared, MSE, RMSE, AIC/BIC, and cross-validation
- Avoid common modeling pitfalls including overfitting, p-hacking with features, and mistaking training accuracy for predictive power

## Prerequisites

Before starting this route:
- **Required**: [Statistics Fundamentals](/routes/stats-fundamentals/map.md) -- you need comfort with means, variance, standard deviation, and basic data visualization
- **Required**: [Statistical Inference](/routes/statistical-inference/map.md) -- you need to understand hypothesis testing, confidence intervals, and p-values
- **Helpful**: [Probability Distributions](/routes/probability-distributions/map.md) -- familiarity with the normal distribution and the idea of a probability model
- **Helpful**: Calculus for ML -- derivatives and partial derivatives help you understand how least squares minimization works, but are not strictly required

## Route Structure

### 1. Correlation and Causation
- Pearson correlation coefficient: computing and interpreting r
- Spurious correlations: ice cream sales and drowning, Nicolas Cage films and pool drownings
- Simpson's paradox: a team's batting average can rise in each half of the season but fall overall
- Confounding variables and why "correlation is not causation" is the most important sentence in statistics
- Applications: do home runs predict wins? Correlation matrices for financial portfolios. Feature correlation in ML pipelines.

### 2. Simple Linear Regression
- Fitting y = mx + b to data using ordinary least squares
- Interpreting the slope ("for each additional unit of x, y changes by...") and intercept
- Residuals: what the model gets wrong, and why the pattern of residuals matters
- Applications: predicting points scored from pace of play, the CAPM model (beta as a regression slope), the simplest supervised learning algorithm

### 3. Multiple Linear Regression
- Adding more predictors to the model
- Interpreting coefficients while holding other variables constant
- Feature selection: which variables actually matter?
- Multicollinearity: when predictors are correlated with each other and coefficients become unstable
- Applications: predicting game outcomes from multiple team stats, Fama-French multi-factor models, feature engineering in ML

### 4. Logistic Regression
- When the outcome is binary: win/lose, default/no-default, spam/not-spam
- The sigmoid function and why it maps any input to a probability between 0 and 1
- Odds, log-odds, and interpreting coefficients as odds ratios
- Applications: predicting win probability for sports betting, credit scoring in finance, the foundational classification algorithm in ML

### 5. Model Evaluation
- R-squared and adjusted R-squared: how much variance does the model explain?
- MSE, RMSE, MAE: how big are the errors?
- AIC and BIC: comparing models with a complexity penalty
- Cross-validation: testing on unseen data to detect overfitting
- Overfitting vs underfitting: the bias-variance tradeoff
- Applications: is your sports model better than just picking the home team? Backtesting pitfalls in finance. Train/test splits in ML.

### 6. Practice Project
- Build a sports prediction model using real team/game statistics
- Fit regression models to predict scores and logistic models to predict win/loss
- Evaluate with cross-validation and compare against a naive baseline
- Communicate results with clear metrics and visualizations

## Learning Modes

This route supports three learning modes:

1. **Self-guided**: Read the guide.md file and work through exercises at your own pace
2. **AI-guided**: Work with an AI assistant using the sherpa.md teaching script
3. **Collaborative**: Read guide.md while getting help from AI following sherpa.md

## Tools & Techniques

This route references:
- **Python 3** with **NumPy** and **pandas** for data manipulation
- **scikit-learn** for building and evaluating regression and classification models
- **statsmodels** for detailed statistical output (p-values, confidence intervals, diagnostic tests)
- **matplotlib** for visualization of data, residuals, and model performance

## Next Steps

After completing this route:
- **[Bayesian Statistics](/routes/bayesian-statistics/map.md)** -- Learn to incorporate prior knowledge into your models and update beliefs as new data arrives
- **[Neural Network Foundations](/routes/neural-network-foundations/map.md)** -- See how regression and classification generalize into deep learning architectures
- **[Training and Backprop](/routes/training-and-backprop/map.md)** -- Understand how models learn their parameters through optimization
- **Time Series Analysis** -- Extend regression to data ordered in time, with autocorrelation, seasonality, and forecasting
