import numpy as np
import pandas as pd
import random
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, recall_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

np.random.seed(42)
random.seed(42)

data = pd.read_csv('heart_2020_cleaned.csv')
data = data.dropna()

y = data['HeartDisease']
data = data.drop(columns=['HeartDisease'])
y = LabelEncoder().fit_transform(y)

age_map = {val: i for i, val in enumerate(sorted(data['AgeCategory'].unique()))}
data['AgeCategory'] = data['AgeCategory'].map(age_map)

gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
data['GenHealth'] = data['GenHealth'].map(gen_health_map)

numeric_col = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']
categorical_col = [c for c in data.columns if c not in numeric_col]

for col in numeric_col:
    data[col] = data[col].astype(float)
    upper = data[col].quantile(0.99)
    data.loc[data[col] > upper, col] = upper

for col in categorical_col:
    counts = data[col].value_counts()
    rare = counts[counts < 3].index
    data[col] = data[col].replace(rare, 'other')

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_encoded = encoder.fit_transform(data[categorical_col])

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(data[numeric_col])

X_encoded_scaled = np.hstack([X_cat_encoded, X_num_scaled])
ratio = np.sum(y == 0) / np.sum(y == 1)

scoring = ['accuracy', 'f1', 'roc_auc']
X_train, X_val, y_train, y_val = train_test_split(X_encoded_scaled, y, test_size=0.2, random_state=42, stratify=y)

dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
dval = xgb.QuantileDMatrix(X_val, label=y_val)



def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=25),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),

        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'scale_pos_weight': ratio, 
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    score = f1_score(y_val, preds)
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"\nЛучший F1-результат: {study.best_value:.4f}")

final_params = study.best_params.copy()
final_params.update({
    'device': 'cuda', 
    'tree_method': 'hist',
    'scale_pos_weight': ratio
})

final_params['eval_metric'] = ['logloss', 'auc']
best_model = XGBClassifier(**final_params)

best_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

results = best_model.evals_result()
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
plt.plot(x_axis, results['validation_1']['logloss'], label='Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x_axis, results['validation_0']['auc'], label='Train')
plt.plot(x_axis, results['validation_1']['auc'], label='Validation')
plt.legend()
plt.ylabel('ROC AUC')
plt.title('XGBoost ROC AUC')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

y_probs = best_model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)

target_recall = 0.95
validx_idx = np.where(recalls[:-1] >= target_recall)[0]
best_idx = validx_idx[-1]
threshold = thresholds[best_idx]

y_pred = (y_probs >= threshold).astype(int)

print(f"Новый найденный порог: {threshold:.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")

cm_custom = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Oranges')
plt.title(f'XGBoost (Threshold = {threshold:.3f})')
plt.ylabel('Реальность (True)')
plt.xlabel('Прогноз (Predicted)')
plt.show()