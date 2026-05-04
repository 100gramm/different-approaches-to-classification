import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import joblib
from old_model_training import OldClassifier

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        self.res_block = nn.ModuleDict({
            'linear': nn.Linear(256, 256),
            'bn': nn.BatchNorm1d(256),
            'dropout': nn.Dropout(0.4)
        })
        
        self.final_layers = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        x = self.input_layer(X)
        
        identity = x
        out = self.res_block['linear'](x)
        out = self.res_block['bn'](out)
        out = F.gelu(out)
        out = self.res_block['dropout'](out)
        x = x + out
        return self.final_layers(x)

matplotlib.use('Agg')

@st.cache_data
def load_and_preprocess_data():
    data = pd.read_csv('heart_2020_cleaned.csv')
    data = data.dropna()
    
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

    X_train, X_val_old_full, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    data2 = pd.read_csv('heart_2020_cleaned.csv')
    data2 = data2.dropna()
    age_map2 = {val: i for i, val in enumerate(sorted(data2['AgeCategory'].unique()))}
    data2['AgeCategory'] = data2['AgeCategory'].map(age_map2)

    gen_health_map2 = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
    data2['GenHealth'] = data2['GenHealth'].map(gen_health_map2)

    y2 = LabelEncoder().fit_transform(data2['HeartDisease'])

    cols_to_drop = ['HeartDisease'] 
    data_cleaned = data2.drop(columns=cols_to_drop)

    numeric_col = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']
    categorical_col = [c for c in data_cleaned.columns if c not in numeric_col]

    for col in numeric_col:
        upper = data_cleaned[col].quantile(0.99)
        data_cleaned.loc[data_cleaned[col] > upper, col] = upper

    X_train_raw, X_val_raw, y_train2, y_val2 = train_test_split(
        data_cleaned, y2, test_size=0.2, random_state=42, stratify=y2
    )

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_train = encoder.fit_transform(X_train_raw[categorical_col])
    X_cat_val = encoder.transform(X_val_raw[categorical_col])

    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_train_raw[numeric_col])
    X_num_val = scaler.transform(X_val_raw[numeric_col])

    X_val_encoded = np.hstack([X_cat_val, X_num_val])
    
    return X_val_encoded, X_val_old_full.values, y_val, encoder, scaler, numeric_col, categorical_col

@st.cache_resource
def load_models_and_probs(X_val_encoded, X_val_old, y_val):
    device = torch.device('cpu')
    
    in_features = X_val_encoded.shape[1]
    mlp_model = MLP(in_dim=in_features, num_classes=1).to(device)
    mlp_model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=True))
    mlp_model.eval()

    old_clf = OldClassifier()
    old_clf.fit_scoring(pd.DataFrame(X_val_old), pd.Series(y_val.values))
    old_clf.fit_statistical(pd.DataFrame(X_val_old), pd.Series(y_val.values))
    
    xgb_model = joblib.load('xgb_model.pkl')
    
    models = {
        'MLP': mlp_model,
        'XGBoost': xgb_model,
        'Expert Rules': old_clf,
        'Scoring': old_clf,
        'Naive Bayes': old_clf
    }
    
    probs = {}
    for name, model in models.items():
        probs[name] = get_probs(model, X_val_encoded, X_val_old, name, device)
    
    thresholds = {}
    for name, prob in probs.items():
        thresholds[name] = find_threshold_for_recall(y_val, prob)
    
    return mlp_model, old_clf, xgb_model, device, probs, thresholds

def get_probs(model, X_val_encoded, X_val_old, model_name, device=None):
    if model_name == 'MLP':
        with torch.no_grad():
            logits = model(torch.tensor(X_val_encoded, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
    elif model_name == 'XGBoost':
        X_val_encoded_float32 = X_val_encoded.astype(np.float32)
        probs = model.predict_proba(X_val_encoded_float32)[:, 1]
    elif model_name == 'Expert Rules':
        probs = model.predict_expert_rules(pd.DataFrame(X_val_old)).astype(float)
    elif model_name == 'Scoring':
        probs = model.get_scoring_probs(pd.DataFrame(X_val_old))
    elif model_name == 'Naive Bayes':
        probs = model.predict_statistical_probs(pd.DataFrame(X_val_old))
    return probs

def find_threshold_for_recall(y_true, y_probs, target_recall=0.95):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[-1]
        return thresholds[best_idx]
    else:
        return 0.5

def main():
    st.title("Интерактивный дашборд моделей Heart Disease")
    
    X_val_encoded, X_val_old, y_val, encoder, scaler, numeric_col, categorical_col = load_and_preprocess_data()
    
    mlp_model, old_clf, xgb_model, device, probs, thresholds = load_models_and_probs(X_val_encoded, X_val_old, y_val)
    
    st.sidebar.header("Настройки")
    
    model_choice = st.sidebar.selectbox("Выберите модель", list(probs.keys()))
    
    threshold = st.sidebar.slider(
        "Порог классификации", 
        min_value=0.0, 
        max_value=1.0, 
        value=thresholds[model_choice], 
        step=0.01
    )
    
    y_pred = (probs[model_choice] >= threshold).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    
    st.header(f"Метрики для модели: {model_choice}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("Precision", f"{prec:.4f}")
    
    st.header("Матрица ошибок")
    cm = confusion_matrix(y_val.values, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.3f})', fontsize=12)
    fig.tight_layout()
    
    st.pyplot(fig, width='stretch')
    plt.close(fig)

if __name__ == "__main__":
    main()