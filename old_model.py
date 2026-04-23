import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score, confusion_matrix

data = pd.read_csv('heart_2020_cleaned.csv')

binary_cols = data.select_dtypes(include=['object']).columns
for col in binary_cols:
    if set(data[col].unique()) == {'Yes', 'No'}:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

age_map = {val: i for i, val in enumerate(sorted(data['AgeCategory'].unique()))}
data['AgeCategory'] = data['AgeCategory'].map(age_map)

gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
data['GenHealth'] = data['GenHealth'].map(gen_health_map)

data = data.select_dtypes(include=[np.number])

y = data['HeartDisease']
X = data.drop('HeartDisease', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class OldClassifier:
    def __init__(self):
        self.weights = None
        self.prior_pos = None
        self.feature_probs = None

    # 1. Expert System
    def predict_expert_rules(self, X):
        predictions = []
        for _, row in X.iterrows():
            if row.sum() >= 3: 
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)

    # 2. Point Scoring
    def fit_scoring(self, X, y):
        self.weights = X.corrwith(y).fillna(0)
        
    def get_scoring_probs(self, X):
        scores = X.dot(self.weights)
        if scores.max() == scores.min():
            return np.zeros(len(scores))
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return norm_scores

    # 3. Naive Bayes
    def fit_statistical(self, X, y):
        self.prior_pos = y.mean()
        self.feature_probs = {}
        for col in X.columns:
            self.feature_probs[col] = {
                1: X[y == 1][col].mean() + 1e-6,
                0: X[y == 0][col].mean() + 1e-6
            }

    def predict_statistical_probs(self, X):
        probs = []
        for _, row in X.iterrows():
            p_pos = self.prior_pos
            p_neg = 1 - self.prior_pos
            for col in X.columns:
                val = row[col]
                if val > 0.5:
                    p_pos *= self.feature_probs[col][1]
                    p_neg *= self.feature_probs[col][0]
                else:
                    p_pos *= (1 - self.feature_probs[col][1])
                    p_neg *= (1 - self.feature_probs[col][0])
            
            prob = p_pos / (p_pos + p_neg + 1e-12)
            probs.append(prob)
        return np.array(probs)


def evaluate_soviet_model(y_true, y_probs, name, target_recall=0.95):
    thresholds = np.linspace(0, 1, 1000)
    best_threshold = 0.0
    best_recall = 0.0

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= target_recall:
            best_threshold = t
            best_recall = r
        else:
            break
            
    final_pred = (y_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_true, final_pred)
    f1 = f1_score(y_true, final_pred, zero_division=0)
    
    print(f"--- {name} ---")
    print(f"Порог: {best_threshold:.4f} | Recall: {best_recall:.4f} | F1: {f1:.4f}")
    
    plt.figure(figsize=(6, 5))
    cmap = 'Reds' if 'Старый' in name else ('Oranges' if 'Средний' in name else 'Blues')
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap)
    plt.title(f"{name}\nThreshold = {best_threshold:.3f}")
    plt.show()


soviet_logic = OldClassifier()

# 1. Expert System
y_pred_old = soviet_logic.predict_expert_rules(X_test)
evaluate_soviet_model(y_test, y_pred_old.astype(float), "Старый подход (Правила)")

# 2. Point Scoring
soviet_logic.fit_scoring(X_train, y_train)
y_scores_mid = soviet_logic.get_scoring_probs(X_test)
evaluate_soviet_model(y_test, y_scores_mid, "Средний подход (Баллы)")

# 3. Naive Bayes
soviet_logic.fit_statistical(X_train, y_train)
y_probs_new = soviet_logic.predict_statistical_probs(X_test)
evaluate_soviet_model(y_test, y_probs_new, "Новый подход (Байес)")