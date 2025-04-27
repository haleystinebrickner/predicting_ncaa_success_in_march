import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import keras_tuner as kt
import os
import glob
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

def load_and_preprocess_data():
    """
    Load and preprocess data from specific CBB years (2021-2024).
    """
    years = ['21', '22', '23', '24']
    dfs = []
    
    # Column name mapping to standardize across years
    column_mapping = {
        'EFG%': 'EFG_O',
        'EFGD%': 'EFG_D',
        'EFGD_D': 'EFG_D'
    }
    
    for year in years:
        file = f'cbb{year}.csv'
        print(f"\nLoading {file}...")
        try:
            df = pd.read_csv(file)
            print(f"Initial shape: {df.shape}")
            print(f"Initial columns: {df.columns.tolist()}")
            
            # Ensure consistent column names
            df = df.rename(columns=str.upper)
            df = df.rename(columns=column_mapping)
            print(f"Columns after renaming: {df.columns.tolist()}")
            
            # Drop unnecessary columns if they exist
            columns_to_drop = ['TEAM', 'CONF', 'G', 'W', 'L', 'SEED', 'RK']
            columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=columns_to_drop, errors='ignore')
            print(f"Columns after dropping: {df.columns.tolist()}")
            
            # Ensure POSTSEASON column exists
            if 'POSTSEASON' not in df.columns:
                print(f"Warning: POSTSEASON column not found in {file}")
                continue
            
            # Convert POSTSEASON to string and handle missing values
            df['POSTSEASON'] = df['POSTSEASON'].astype(str)
            df = df[df['POSTSEASON'] != 'nan']  # Remove rows with 'nan' in POSTSEASON
            print(f"Rows after POSTSEASON cleaning: {len(df)}")
            
            # Handle missing values in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            if not df.empty:
                dfs.append(df)
                print(f"Successfully loaded {len(df)} rows from {file}")
                print(f"Sample data from {file}:\n{df.head()}")
            else:
                print(f"Warning: {file} is empty after preprocessing")
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    if not dfs:
        raise ValueError("No valid data found in any files")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows after combining: {len(combined_df)}")
    print(f"Combined columns: {combined_df.columns.tolist()}")
    
    # Separate features and target
    X = combined_df.drop(columns=['POSTSEASON'])
    y = combined_df['POSTSEASON']
    
    print(f"\nFeature columns: {X.columns.tolist()}")
    print(f"Sample of X data:\n{X.head()}")
    print(f"Sample of y data:\n{y.head()}")
    
    # Ensure all feature columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any remaining rows with NaN values
    X = X.dropna()
    y = y[X.index]
    
    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"\nFeature columns: {X.columns.tolist()}")
    print(f"\nSample of X data:\n{X.head()}")
    print(f"\nSample of y data:\n{y.head()}")
    
    return X, y

# Load and preprocess data from specific years
print("Loading CBB data from 2021-2024...")
X, y = load_and_preprocess_data()

# Print some information about the combined dataset
print(f"\nCombined dataset information:")
print(f"Total number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print("\nClass distribution:")
print(pd.Series(y).value_counts())

# Feature engineering
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_poly, y)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)
class_names = le.classes_

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
)

def lr_schedule(epoch):
    """Learning rate schedule"""
    initial_lr = 0.001
    if epoch < 10:
        return initial_lr
    elif epoch < 20:
        return initial_lr * 0.1
    else:
        return initial_lr * 0.01

def weighted_cross_entropy(alpha=0.25, gamma=2.0):
    """
    Custom weighted cross entropy loss function similar to focal loss
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Add small epsilon to avoid log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal weight
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1. - y_pred, gamma) * y_true
        
        # Apply class weights
        weighted_loss = alpha * weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(weighted_loss, axis=-1))
    return loss

# Globals for HyperModel shapes
input_dim = None
num_classes = None

def build_model(hp):
    """
    Build a model with strong regularization and balanced handling
    """
    # L1L2 regularization
    regularizer = tf.keras.regularizers.L1L2(
        l1=hp.Float('l1', 1e-6, 1e-3, sampling='log'),
        l2=hp.Float('l2', 1e-6, 1e-3, sampling='log')
    )
    
    model = tf.keras.Sequential()
    # Explicit input layer
    model.add(tf.keras.layers.InputLayer(input_shape=(input_dim,)))
    # First Dense layer
    model.add(tf.keras.layers.Dense(
        units=hp.Int('units_input', min_value=128, max_value=512, step=64),
        kernel_regularizer=regularizer
    ))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout1', 0.3, 0.6, step=0.1)))
    
    # Hidden layers 
    prev_units = None
    for i in range(hp.Int("n_layers", 2, 4)):
        units = hp.Int(f'units_{i}', min_value=64, max_value=256, step=32)
        
        # Store for skip connection
        if prev_units is None:
            prev_units = units
            
        # Main path
        model.add(tf.keras.layers.Dense(
            units=units,
            kernel_regularizer=regularizer
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(hp.Float(f'dropout_{i}', 0.2, 0.5, step=0.1)))
        
        prev_units = units
    
    # Output layer with label smoothing
    model.add(tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=regularizer
    ))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Choice('lr', [1e-3, 5e-4, 1e-4]),
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss=weighted_cross_entropy(alpha=0.25, gamma=2.0),
        metrics=['accuracy', 
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    
    return model

def get_callbacks(fold=None):
    """Enhanced callbacks for better training control"""
    callbacks = [
        # Early stopping with more patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001,
            mode='max'
        ),
        # Reduce learning rate when plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            f'best_model{"_fold"+str(fold) if fold else ""}.keras',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    return callbacks

def prepare_data_for_training():
    "Prepare data with class balancing"
    # Load and preprocess data
    print("Loading CBB data from 2021-2024...")
    X, y = load_and_preprocess_data()
    
    # Convert labels to numerical
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Create stratified splits
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Print class distribution
    print("\nClass distribution before balancing:")
    for class_name, count in zip(le.classes_, np.bincount(y_encoded)):
        print(f"{class_name}: {count}")
    
    fold_data = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y_encoded)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        

        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        

        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        # Convert class weights to sample weights
        sample_weights = np.ones(len(y_train))
        for idx, label in enumerate(y_train):
            sample_weights[idx] = class_weights[label]
        
        # Convert to categorical
        y_train_cat = to_categorical(y_train)
        y_val_cat = to_categorical(y_val)
        
        fold_data.append({
            'X_train': X_train,
            'y_train': y_train_cat,
            'X_val': X_val,
            'y_val': y_val_cat,
            'sample_weights': sample_weights
        })
        
        # Print fold-specific class distribution
        print(f"\nFold {fold + 1} class distribution:")
        for class_name, count in zip(le.classes_, np.bincount(y_train)):
            print(f"{class_name}: {count}")
    
    return fold_data, le



# Main training loop
print("Starting training with enhanced balancing and regularization...")
fold_data, le = prepare_data_for_training()

# Define global input/output dims for HyperModel
input_dim = fold_data[0]['X_train'].shape[1]
num_classes = fold_data[0]['y_train'].shape[1]

# Initialize tuner with increased search space
tuner = kt.Hyperband(
    build_model,
    objective=kt.Objective('val_auc', direction='max'),
    max_epochs=50,
    factor=3,
    directory='keras_tuner_dir',
    project_name='cbb_fcnn_balanced'
)

# Train with cross-validation
histories = []
for fold, data in enumerate(fold_data):
    print(f'\nFold {fold + 1}/5')
    
    # Search for best hyperparameters
    tuner.search(
        data['X_train'], data['y_train'],
        epochs=50,
        validation_data=(data['X_val'], data['y_val']),
        callbacks=get_callbacks(fold),
        sample_weight=data['sample_weights'],
        verbose=1
    )
    
    # Get best hyperparameters and retrain
    best_hps = tuner.get_best_hyperparameters(1)[0]
    model = build_model(best_hps)
    
    # Train with best hyperparameters
    history = model.fit(
        data['X_train'], data['y_train'],
        epochs=100,
        validation_data=(data['X_val'], data['y_val']),
        callbacks=get_callbacks(fold),
        sample_weight=data['sample_weights'],
        verbose=1
    )
    
    histories.append(history.history)

def plot_training_results(histories, le):
    metrics = ['accuracy', 'auc', 'precision', 'recall']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        for i, history in enumerate(histories):
            plt.plot(history[metric], label=f'Train Fold {i+1}')
            plt.plot(history[f'val_{metric}'], label=f'Val Fold {i+1}', linestyle='--')
        plt.title(f'{metric.upper()} Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()


# Plot results
plot_training_results(histories, le)
