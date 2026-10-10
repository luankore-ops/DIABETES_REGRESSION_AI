# Diabetes Regression AI

A machine learning project for biomedical regression, model optimization, explainability, and error analysis using the diabetes dataset included in scikit-learn.

## Project Overview

This project explores how machine learning regression models can estimate a quantitative measure of disease progression from clinical and physiological features.

The workflow covers data exploration, baseline modeling, cross-validation, hyperparameter optimization, explainable AI (XAI), and prediction error analysis.

**Purpose:** Educational and portfolio project. This is not a clinical decision-support or diagnostic tool.

## Objectives

- Explore and visualize a biomedical dataset.
- Establish baseline regression performance.
- Compare linear and tree-based regression models.
- Optimize model hyperparameters using cross-validation.
- Interpret model behavior using Ridge coefficients and SHAP.
- Analyze prediction errors and model limitations.
- Build a reproducible Python workflow.

## Dataset

The project uses the diabetes dataset distributed with scikit-learn.

- **Observations:** 442
- **Input features:** 10
- **Target:** Quantitative measure of diabetes disease progression approximately one year after baseline
- **Training set:** 353 observations
- **Test set:** 89 observations
- **Split:** 80/20, with `random_state=42`

The features include age, sex, body mass index (BMI), blood pressure, and six physiological measurements (`s1`–`s6`).

The features in this dataset are already centered and scaled. The target is a quantitative score, not a probability or a direct measurement of clinical risk.

## Methodology

### 1. Exploratory Data Analysis

- Dataset structure and descriptive statistics
- Missing-value checks
- Target distribution
- Feature distributions
- Correlation analysis

### 2. Baseline Models

A `DummyRegressor` provides a simple reference. Linear Regression establishes the initial predictive baseline.

### 3. Model Comparison

The following algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Five-fold cross-validation was used to compare model performance on the training set.

### 4. Hyperparameter Optimization

`GridSearchCV` optimized Ridge, Random Forest, and Gradient Boosting using mean absolute error (MAE) as the selection metric.

The best hyperparameters were selected using training-set cross-validation. The test set was used for performance reporting.

### 5. Explainable AI

Ridge coefficients were examined to understand the direction and relative magnitude of the model's linear associations.

SHAP (`TreeExplainer`) was used to investigate feature contributions in a reconstructed Gradient Boosting model. Global feature importance and individual prediction explanations were generated.

### 6. Error Analysis

The workflow calculates MAE, RMSE, R², mean prediction error, median absolute error, and maximum absolute error. It also generates actual-versus-predicted plots, residual plots, and a list of the ten largest absolute prediction errors.

## Results

### Test-set performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---: