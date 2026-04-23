import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, recall_score, precision_recall_curve

data = pd.read_csv('heart_2020_cleaned.csv')

age_map = {val: i for i, val in enumerate(sorted(data['AgeCategory'].unique()))}
data['AgeCategory'] = data['AgeCategory'].map(age_map)

gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
data['GenHealth'] = data['GenHealth'].map(gen_health_map)

y = LabelEncoder().fit_transform(data['HeartDisease'])

cols_to_drop = ['HeartDisease'] 
data_cleaned = data.drop(columns=cols_to_drop)

numeric_col = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime', 'AgeCategory', 'GenHealth']
categorical_col = [c for c in data_cleaned.columns if c not in numeric_col]

for col in numeric_col:
    upper = data_cleaned[col].quantile(0.99)
    data_cleaned.loc[data_cleaned[col] > upper, col] = upper

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    data_cleaned, y, test_size=0.2, random_state=42, stratify=y
)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat_train = encoder.fit_transform(X_train_raw[categorical_col])
X_cat_val = encoder.transform(X_val_raw[categorical_col])

scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_train_raw[numeric_col])
X_num_val = scaler.transform(X_val_raw[numeric_col])

X_train = np.hstack([X_cat_train, X_num_train])
X_val = np.hstack([X_cat_val, X_num_val])

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TabularDataset(X_train, y_train)
val_ds = TabularDataset(X_val, y_val)

class_counts = np.bincount(y_train)
class_weights = 1. / class_counts
sample_weights = torch.from_numpy(class_weights[y_train])
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=1024, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=1024)

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
    
class CalibratedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

def calibrate_model(model, val_loader, device):
    model.eval()
    calibrator = CalibratedModel(model).to(device)
    optimizer = optim.LBFGS([calibrator.temperature], lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    logits_list, labels_list = [], []
    with torch.no_grad():
        for X, y in val_loader:
            logits_list.append(model(X.to(device)))
            labels_list.append(y.to(device).view(-1, 1).float())
    
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / calibrator.temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    print(f'Оптимальная температура: {calibrator.temperature.item():.4f}')
    return calibrator

device = torch.device('cuda')

def eval_epoch(model, loader, criterion):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            loss = criterion(outputs, y.view(-1, 1).float())
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
    
    all_probs = np.array(all_probs).flatten()
    all_labels = np.array(all_labels).flatten()

    roc_auc  =roc_auc_score(all_labels, all_probs)
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    all_preds = (all_probs > best_threshold).astype(int)
    acc = accuracy_score(all_labels, all_preds)

    metrics = {
        'acc': acc,
        'f1': best_f1,
        'roc_auc': roc_auc,
        'best_threshold': best_threshold
    }

    return total_loss / len(loader), metrics

def train_model(model, train_loader, val_loader, epochs=100, patience=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_losses, val_losses = [], []
    history = {'acc':[], 'f1': [], 'roc_auc': [], 'best_threshold': []}

    best_val_loss = float('inf')
    epoch_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
    
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, metrics = eval_epoch(model, val_loader, criterion)
        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        for key in metrics:
            history[key].append(metrics[key])
        
        print(f'Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | '
        f'Acc: {metrics["acc"]:.4f} | F1: {metrics["f1"]:.4f} | '
        f'AUC: {metrics["roc_auc"]:.4f} | T: {metrics["best_threshold"]:.3f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epoch_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epoch_no_improve += 1
            if epoch_no_improve >= patience:
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                model.load_state_dict(torch.load('best_model.pth', weights_only=True))
                break

    return model, train_losses, val_losses, history

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(42)
print(f'\n=== Training MLP (Device: {device}) ===')

in_features = X_train.shape[1]
model = MLP(in_dim=in_features, num_classes=1).to(device)
print(f'Trainable parameters: {count_parameters(model):,}')

model, train_losses, val_losses, history = train_model(
    model, train_loader, val_loader, epochs=30, patience=7
)
final_calibrated_model = calibrate_model(model, val_loader, device)
final_calibrated_model.eval()

all_probs, all_labels = [], []

with torch.no_grad():
    for X, y in val_loader:
        logits = final_calibrated_model(X.to(device))
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs).flatten()
all_labels = np.array(all_labels).flatten()

cost_fn = 3
cost_fp = 1
smart_threshold = cost_fp / (cost_fp + cost_fn)

final_preds = (all_probs > smart_threshold).astype(int)

print(f"\nИспользуем порог: {smart_threshold:.3f}")
print(f"Recall: {recall_score(all_labels, final_preds):.4f}")
print(f"F1 Score: {f1_score(all_labels, final_preds):.4f}")

plt.figure(figsize=(18, 5))
epochs_range = range(len(train_losses))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', lw=2)
plt.plot(epochs_range, val_losses, label='Validation Loss', lw=2)
plt.title('Loss Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, history['acc'], label='Val Accuracy', color='green', lw=2)
plt.plot(epochs_range, history['roc_auc'], label='Val ROC-AUC', color='blue', lw=2)
plt.title('Acc & AUC Metrics', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(epochs_range, history['f1'], label='Val F1 Score', color='red', lw=2)
plt.title('F1 Score Curve', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

best_epoch = np.argmin(val_losses)
print("\n" + "="*30)
print("MLP BEST RESULTS:")
print(f"Best Epoch: {best_epoch + 1}")
print(f"ROC-AUC:    {history['roc_auc'][best_epoch]:.4f}")
print(f"Accuracy:   {history['acc'][best_epoch]:.4f}")
print(f"F1 Score:   {history['f1'][best_epoch]:.4f}")
print("="*30)

final_probs = np.array(all_probs).flatten()
final_labels = np.array(all_labels).flatten()

cm = confusion_matrix(final_labels, final_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Heart Disease', 'Heart Disease'])

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title('Confusion Matrix: MLP')
plt.show()

recall = recall_score(final_labels, final_preds)
print(f"Recall (Чувствительность): {recall:.4f}")
print("Это процент больных, которых модель смогла найти.")