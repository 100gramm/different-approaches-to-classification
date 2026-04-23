Heart Disease Classification: From Classical Statistics to Modern ML

Project Overview
  This project investigates the evolution of diagnostic classification methods using the CDC Heart Disease Dataset. The primary goal is to compare modern machine learning architectures (XGBoost, MLP) against "classical" statistical approaches to determine the trade-offs between model complexity, interpretability, and predictive power.
A specific focus is placed on Recall-driven optimization, simulating a real-world medical requirement where missing a sick patient (False Negative) is significantly more costly than a false alarm.

Evolutionary Stages of Classification
The project implements and compares five distinct generations of algorithms:
  Classical
    Expert Rules System: A rigid, logic-based classifier simulating early medical protocols (e.g., "If Risk Factors  >3, then High Risk").
    Point Scoring System: A linear model where features are weighted by their correlation with the target, mimicking diagnostic scoring sheets used in clinical practice.
    Naive Bayes: A probabilistic approach based on the frequency of symptoms within the population, representing early statistical automation in healthcare.
  Modern
    XGBoost: A state-of-the-art gradient boosting framework optimized for tabular data.
    MLP (Multi-Layer Perceptron): A deep learning approach used to capture non-linear relationships within medical indicators.

Key Tasks & Objectives
  Data Preprocessing: Handling categorical medical data, encoding age categories, and normalizing general health metrics.
  Threshold Engineering: Implementing a custom evaluation loop to find the optimal classification threshold that ensures a Target Recall of 0.95.
  Comparative Analysis: Analyzing the "False Positive penalty" paid by each model to achieve high sensitivity.
  Visualization: Generating Confusion Matrices and training logs (Loss/F1 curves) for each approach.

Technical Stack
  Language: Python 3.11
  Libraries: pandas, numpy, scikit-learn, xgboost, pytorch, optuna, matplotlib, seaborn

Here you can see full results of research: "link will be soon"
