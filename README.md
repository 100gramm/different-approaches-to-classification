Heart Disease Classification: From Classical Statistics to Modern Machine Learning
Project Overview
This study investigates the evolution of diagnostic classification methods utilizing the CDC Heart Disease Dataset. The objective is to compare modern machine learning architectures (XGBoost, MLP) with classical statistical approaches, evaluating trade-offs between model complexity, interpretability, and predictive performance. Optimization is focused on Recall to minimize False Negatives, reflecting critical medical requirements where diagnostic omission carries high clinical cost.

Evolutionary Stages of Classification
The project evaluates five distinct algorithmic generations:

Classical Approaches
Expert Rules System: Logic-based classification simulating standard medical protocols (e.g., threshold-based risk assessment).

Point Scoring System: Linear model applying weights based on feature-target correlation, mirroring clinical scoring systems.

Naive Bayes: Probabilistic approach calculating symptom frequency within the population, representing early statistical automation.

Modern Approaches
XGBoost: Gradient boosting framework optimized for structured tabular data.

MLP (Multi-Layer Perceptron): Neural network architecture identifying non-linear patterns in medical indicators.

Key Tasks and Objectives
Data Preprocessing: Categorical data encoding, age stratification, and normalization of health metrics.

Threshold Engineering: Implementation of iterative evaluation to achieve a target Recall of 0.95.

Comparative Analysis: Quantification of False Positive rates required to maintain high sensitivity thresholds.

Visualization: Generation of Confusion Matrices and training metrics (Loss/F1 curves).

Technical Stack
Language: Python 3.11

Libraries: pandas, numpy, scikit-learn, xgboost, pytorch, optuna, matplotlib, seaborn

Results
Full research results are pending availability.
