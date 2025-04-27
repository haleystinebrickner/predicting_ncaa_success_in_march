import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import SelectKBest, f_classif

# Add load_and_preprocess_data utility
def load_and_preprocess_data():
    
    years = ['21', '22', '23', '24']
    dfs = []
    col_map = {
        'EFG%': 'EFG_O', 'EFGD%': 'EFG_D', 'EFGD_D': 'EFG_D'
    }
    drop_cols = ['TEAM', 'CONF', 'G', 'W', 'L', 'SEED', 'RK']
    for yr in years:
        file = f'cbb{yr}.csv'
        df = pd.read_csv(file)
        df = df.rename(columns=str.upper).rename(columns=col_map)
        cols = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=cols, errors='ignore')
        if 'POSTSEASON' not in df.columns:
            continue
        df['POSTSEASON'] = df['POSTSEASON'].astype(str)
        df = df[df['POSTSEASON'] != 'nan']
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        if not df.empty:
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    X = combined.drop(columns=['POSTSEASON'])
    y = combined['POSTSEASON']
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.dropna()
    y = y.loc[X.index]
    return X, y

## Load and preprocess data from multiple years
X, y = load_and_preprocess_data()

## Feature engineering: polynomial features and select top features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)
selector = SelectKBest(f_classif, k=min(30, X_poly.shape[1]))
X = selector.fit_transform(X_poly, y)
print(f"After feature engineering, shape: {X.shape}")

## Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

## Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

## Apply Undersampling
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

## SMOTE oversampling for minority classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_res, y_train_res)

## Compute sample weights to handle class imbalance
sample_weights = compute_sample_weight('balanced', y_train_res)

## Initialize TabNet with enhanced hyperparameters
tabnet_model = TabNetClassifier(
    n_d=64, n_a=64, n_steps=5,
    gamma=1.5,
    n_independent=2, n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=30, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax',
    verbose=10
)

## Fit with early stopping and sample weights
tabnet_model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_test, y_test)],
    eval_name=['valid'],
    eval_metric=['accuracy', 'logloss'],
    max_epochs=100,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128,
    weights=sample_weights,
    drop_last=False
)

## Evaluate
y_pred = tabnet_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

## Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)

## Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

## Plot training & validation curves
history_obj = tabnet_model.history
# Extract history dict from History object
history_dict = history_obj.history if hasattr(history_obj, 'history') else vars(history_obj)
train_loss = history_dict['loss']
val_loss = history_dict.get('valid_logloss', history_dict.get('valid_loss', []))
train_acc = history_dict.get('accuracy', history_dict.get('train_accuracy', []))
val_acc = history_dict.get('valid_accuracy', [])
epochs = list(range(1, len(train_loss) + 1))
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val LogLoss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss Curves')
plt.subplot(1,2,2)
plt.plot(epochs, train_acc, label='Train Acc')
plt.plot(epochs, val_acc, label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy Curves')
plt.tight_layout()
plt.show()
