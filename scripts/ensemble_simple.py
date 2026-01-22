#!/usr/bin/env python3
"""
Phase 2: Simple Ensemble of Hall of Fame Models
Strategy: Averaging (Bagging)
Target: Reduce variance and lower MEE below 22.30
"""

import numpy as np
import pandas as pd
import csv
import ast
# from sklearn.model_selection import KFold  # Removed
from sklearn.preprocessing import StandardScaler
import time
import os
import json

from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee
from src.cv_utils import k_fold_split

# ============================================================================
# CONFIGURATION
# ============================================================================
HALL_OF_FAME_PATH = 'experiments/hall_of_fame.csv'
DATA_PATH = 'data/ML-CUP25-TR.csv'
TOP_N_MODELS = 10  # Ensemble the top 10 configurations
N_SPLITS = 5
RANDOM_STATE = 42
MAX_EPOCHS = 2000  # Slightly lower than single run to save time, ensemble usually converges faster
PATIENCE = 80

def load_top_configs(n=10):
    """Load top N configurations from hall of fame CSV."""
    df = pd.read_csv(HALL_OF_FAME_PATH)
    configs = []
    print(f"Loading top {n} models from Hall of Fame...")
    
    for i in range(min(n, len(df))):
        # Parse the string representation of the dict
        # The CSV has a 'params' column which is a string repr of a dict
        param_str = df.iloc[i]['params']
        
        # specific fix for numpy types in the string that 'eval' might choke on
        # The file content showed: 'learning_rate': np.float64(0.011...)
        # We need to clean this up for ast.literal_eval or json.loads
        clean_str = param_str.replace("np.float64(", "").replace(")", "")
        
        try:
            config = ast.literal_eval(clean_str)
            configs.append(config)
            print(f"  Model {i+1}: [MEE {df.iloc[i]['mee']:.4f}] Layers={config['hidden_layers']} Act={config['hidden_activation']} Drop={config['dropout_rate']}")
        except Exception as e:
            print(f"  Error parsing config {i+1}: {e}")
            
    return configs

def run_ensemble_cv(X, y, configs):
    """Run 5-fold CV using an ensemble of the provided configs."""
    # kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_mees = []
    
    print(f"\nStarting {N_SPLITS}-Fold Ensemble Cross-Validation...")
    print(f"Ensemble Size: {len(configs)} models")
    print("-" * 60)

    total_start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(k_fold_split(X, n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)):
        fold_start_time = time.time()
        
        # 1. Data Split & Scaling
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)
        
        X_train_scaled = x_scaler.transform(X_train)
        X_val_scaled = x_scaler.transform(X_val)
        y_train_scaled = y_scaler.transform(y_train)
        # y_val is kept original for final error calc, but we need scaled version for training monitoring if needed
        y_val_scaled = y_scaler.transform(y_val)

        # 2. Train Individual Models & Collect Predictions
        fold_predictions = np.zeros_like(y_val)
        
        for model_idx, config in enumerate(configs):
            # Init model
            nn = NeuralNetworkV2(
                layer_sizes=[X_train.shape[1]] + config['hidden_layers'] + [y_train.shape[1]],
                hidden_activation=config['hidden_activation'],
                weight_init=config['weight_init'],
                use_batch_norm=config['use_batch_norm'],
                dropout_rate=config['dropout_rate'],
                random_state=RANDOM_STATE + fold_idx + model_idx # Diversify seeds slightly
            )
            
            # Train
            # Using silent training to keep output clean
            nn.train(
                X_train_scaled.T, y_train_scaled.T,
                X_val_scaled.T, y_val_scaled.T,
                epochs=MAX_EPOCHS,
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate'],
                l2_lambda=config['l2_lambda'],
                momentum=config['momentum'],
                optimizer=config['optimizer'],
                patience=PATIENCE,
                verbose=False
            )
            
            # Predict
            # NNv2 prediction output shape is (features, samples) -> (4, N)
            # We need (N, 4)
            pred_scaled = nn.predict(X_val_scaled.T).T
            
            # Inverse scale to get real units
            pred_real = y_scaler.inverse_transform(pred_scaled)
            
            # Add to ensemble sum
            fold_predictions += pred_real
            
        # 3. Average Predictions
        ensemble_pred = fold_predictions / len(configs)
        
        # 4. Calculate Ensemble Error
        fold_mee = mee(y_val, ensemble_pred)
        fold_mees.append(fold_mee)
        
        elapsed = time.time() - fold_start_time
        print(f"Fold {fold_idx+1}/{N_SPLITS} | Ensemble MEE: {fold_mee:.4f} | Time: {elapsed:.1f}s")

    # Final Stats
    mean_mee = np.mean(fold_mees)
    std_mee = np.std(fold_mees)
    total_time = time.time() - total_start_time
    
    print("-" * 60)
    print(f"ENSEMBLE RESULTS:")
    print(f"Mean MEE: {mean_mee:.4f} (+/- {std_mee:.4f})")
    print(f"Total Time: {total_time:.1f}s")
    print("-" * 60)
    
    return mean_mee, std_mee

if __name__ == "__main__":
    # Check for poly features in the first config to decide data loading
    # Assumption: All hall of fame models use same feature strategy (poly=2)
    # We load raw first to check params, but assuming poly2 based on previous analysis
    
    configs = load_top_configs(TOP_N_MODELS)
    
    # Check feature engineering requirements
    use_poly = configs[0].get('use_polynomial_features', True) # Default to true if missing
    degree = configs[0].get('poly_degree', 2)
    
    print(f"Feature Strategy: Polynomial={use_poly}, Degree={degree}")
    
    # Load Data
    X, y = load_cup_data(DATA_PATH, use_polynomial_features=use_poly, poly_degree=degree)
    
    # Run
    run_ensemble_cv(X, y, configs)
