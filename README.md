# Heart Disease Classification: From Classical Statistics to Modern Machine Learning

## Project Overview
This study investigates the evolution of diagnostic classification methods utilizing the CDC Heart Disease Dataset. The objective is to compare modern machine learning architectures (XGBoost, MLP) with classical statistical approaches, evaluating trade-offs between model complexity, interpretability, and predictive performance. Optimization is focused on Recall to minimize False Negatives, reflecting critical medical requirements where diagnostic omission carries high clinical cost.

## Evolutionary Stages of Classification
The project evaluates five distinct algorithmic generations:

### Classical Approaches
* **Expert Rules System:** Logic-based classification simulating standard medical protocols (e.g., threshold-based risk assessment).
* **Point Scoring System:** Linear model applying weights based on feature-target correlation, mirroring clinical scoring systems.
* **Naive Bayes:** Probabilistic approach calculating symptom frequency within the population, representing early statistical automation.

### Modern Approaches
* **XGBoost:** Gradient boosting framework optimized for structured tabular data.
* **MLP (Multi-Layer Perceptron):** Neural network architecture identifying non-linear patterns in medical indicators.

## Key Tasks and Objectives
* **Data Preprocessing:** Categorical data encoding, age stratification, and normalization of health metrics.
* **Threshold Engineering:** Implementation of iterative evaluation to achieve a target Recall of 0.95.
* **Comparative Analysis:** Quantification of False Positive rates required to maintain high sensitivity thresholds.
* **Visualization:** Generation of Confusion Matrices and training metrics (Loss/F1 curves).

## Technical Stack
* **Language:** Python 3.11
* **Libraries:** pandas, numpy, scikit-learn, xgboost, pytorch, optuna, matplotlib, seaborn

## Installation & Setup Guide

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- ~500 MB disk space

### Required Files
The application requires the following files to run:

| File | Size | Description |
|------|------|-------------|
| `heart_2020_cleaned.csv` | ~200 KB | CDC Heart Disease Dataset |
| `best_model.pth` | ~4 MB | Pre-trained MLP model weights (PyTorch) |
| `xgb_model.pkl` | ~2 MB | Pre-trained XGBoost model |

### Step 1: Install Dependencies
Clone or navigate to the project directory and install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- `streamlit` - Web interface framework
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `torch` & `torchvision` - PyTorch deep learning
- `scikit-learn` - Machine learning utilities
- `xgboost` - Gradient boosting
- `joblib` - Model serialization
- `matplotlib` & `seaborn` - Data visualization

### Step 2: Ensure All Data Files Are Present
Verify that the following files exist in the project directory:

```
different-approaches-to-classification/
├── app.py
├── heart_2020_cleaned.csv        ✓ Required
├── best_model.pth                ✓ Required
├── xgb_model.pkl                 ✓ Required
├── mlp_training.py
├── xgboost_training.py
├── old_model_training.py         ✓ Required
└── requirements.txt
```

**If any required file is missing:**

#### Option A: Download Pre-trained Models (Recommended)
Download the pre-trained model files from the project repository and place them in the project directory.

#### Option B: Retrain Models Locally
If you want to retrain models from scratch:

1. **Ensure `heart_2020_cleaned.csv` exists** - This is the dataset used for training

2. **Train MLP model:**
   ```bash
   python mlp_training.py
   ```
   This will generate `best_model.pth`

3. **Train XGBoost model:**
   ```bash
   python xgboost_training.py
   ```
   This will generate `xgb_model.pkl`

### Step 3: Run the Application
Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### File Descriptions

- **heart_2020_cleaned.csv**: CDC Heart Disease Dataset with preprocessed health metrics
- **best_model.pth**: PyTorch MLP model trained to optimize recall for heart disease detection
- **xgb_model.pkl**: XGBoost classifier serialized with joblib

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **FileNotFoundError for .pth or .pkl files** | Download pre-trained models or run training scripts |
| **FileNotFoundError for .csv file** | Ensure `heart_2020_cleaned.csv` is in the project directory |
| **Port already in use** | Use `streamlit run app.py --server.port 8502` |
| **CUDA not found** | Application will run on CPU; ignore GPU warnings |
| **Module not found errors** | Run `pip install -r requirements.txt` again |
| **Memory errors during training** | Close other applications and try again |

## Results

### Threshold Engineering and Clinical Constraints
In medical diagnostics, the clinical cost of a False Negative (missed disease) is significantly higher than a False Positive (false alarm). To prioritize patient safety, all models in this study were subjected to rigorous **Threshold Engineering**. 

Instead of relying on the default 0.5 classification threshold, our pipeline implements an iterative search across the $[0, 1]$ probability space to identify the specific threshold $T$ that satisfies the medical safety constraint: **Recall $\ge$ 0.95**. By enforcing this 95% sensitivity requirement equally across all models, we isolate their true performance based on **Precision** and **F1-Score**—measuring how effectively each architecture minimizes unnecessary secondary screenings.

### Comparative Performance Analysis
Under the enforced condition of 95% Recall, the models demonstrated the following comparative performance:

| Model | Classification Threshold | Precision | F1-Score |
| :--- | :---: | :---: | :---: |
| **Expert Rules** | N/A | 0.086 | 0.158 |
| **Naive Bayes** | 0.045 | 0.091 | 0.165 |
| **Point Scoring** | 0.082 | 0.134 | 0.229 |
| **XGBoost** | 0.115 | 0.138 | 0.234 |
| **MLP (Neural Network)** | **0.128** | **0.146** | **0.242** |

### Key Technical Insights
* **Superiority of Non-Linear Architectures:** The **MLP model** (implemented in `mlp_training.py`) achieved the highest F1-Score (0.242) and Precision (0.146). [cite_start]The neural network’s ability to map non-linear relationships between health metrics (like BMI, age, and smoking habits) allows it to maintain better selectivity than statistical baselines even at high-sensitivity thresholds[cite: 472].
* [cite_start]**The Precision-Interpretability Trade-off:** While modern methods like **XGBoost** and **MLP** provide a 5-7% absolute improvement in Precision over classical methods, the **Point Scoring** system remains the leader in clinical interpretability[cite: 474]. [cite_start]The Point Scoring model allows clinicians to see exactly how individual weights (calculated via Pearson correlation) contribute to the final risk score, providing a "white-box" alternative that is often preferred in traditional medical protocols[cite: 473].
* [cite_start]**Impact of Class Imbalance:** The CDC Heart Disease dataset contains a severe imbalance (91.4% Healthy vs 8.6% Heart Disease)[cite: 286, 289]. Our implementation of threshold optimization, combined with stratified splitting, ensures that the model remains robust despite this imbalance, successfully identifying the minority class (Heart Disease) at the required 95% rate.
* [cite_start]**Pipeline Reliability:** The methodology—moving from preprocessing and threshold scanning to final validation—ensures that the resulting models (saved as `best_model.pth` and `xgb_model.pkl`) are optimized specifically for high-stakes medical screening environments[cite: 366, 370].