#!/usr/bin/env python3
"""
Phase 3: Advanced Feature Engineering
Strategy: Polynomial Degree 3 + PCA Dimensionality Reduction
Goal: Break the 13.75 MEE barrier (Target < 10 MEE)

Pipeline per fold:
1. Standard Scaling (on Train)
2. Polynomial Features (Degree 3)
3. PCA (Reduce to N components)
4. Neural Network Training
"""

import numpy as np
import pandas as pd
# from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee
from src.cv_utils import k_fold_split

# ============================================================================
# CONFIGURATION
# ============================================================================

# Using the best architecture from Hall of Fame
# We might need to adjust dropout or regularization if features change drastically
BASE_CONFIG = {
    'hidden_layers': [128, 84, 65],
    'learning_rate': 0.01198,
    'l2_lambda': 0.0006,
    'dropout_rate': 0.131,
    'momentum': 0.984,
    'batch_size': 64,
    'hidden_activation': 'tanh',
    'optimizer': 'adam',
    'use_batch_norm': False,
    'weight_init': 'he'
}

# Feature Engineering Settings
POLY_DEGREE = 3
PCA_COMPONENTS_LIST = [40, 60, 80, 100] # Grid search for best PCA size
CV_SPLITS = 5
RANDOM_STATE = 42
MAX_EPOCHS = 2000
PATIENCE = 100

DATA_PATH = 'data/ML-CUP25-TR.csv'

# ============================================================================
# EXPERIMENT FUNCTION
# ============================================================================

def run_feature_experiment(X, y, pca_n, config):
    """
    Run 5-fold CV with Scale -> Poly3 -> PCA(n) -> NN pipeline.
    """
    fold_mees = []
    
    print(f"\nTesting PCA Components: {pca_n} (Poly Degree {POLY_DEGREE})")
    print("-" * 60)
    
    start_time = time.time()
    
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold_split(X, n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)):
        # 1. Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 2. Scale (Fit on Train)
        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_val_s = scaler_x.transform(X_val)
        
        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train)
        y_val_s = scaler_y.transform(y_val)
        
        # 3. Polynomial Features (Fit on Train)
        # Degree 3 creates ~364 features from 12 inputs
        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_s)
        X_val_poly = poly.transform(X_val_s)
        
        # 4. PCA (Fit on Train)
        pca = PCA(n_components=pca_n, random_state=RANDOM_STATE)
        X_train_pca = pca.fit_transform(X_train_poly)
        X_val_pca = pca.transform(X_val_poly)
        
        # 5. Train Neural Network
        # Input size is now pca_n
        nn = NeuralNetworkV2(
            layer_sizes=[pca_n] + config['hidden_layers'] + [y_train.shape[1]],
            hidden_activation=config['hidden_activation'],
            weight_init=config['weight_init'],
            use_batch_norm=config['use_batch_norm'],
            dropout_rate=config['dropout_rate'],
            random_state=RANDOM_STATE + fold_idx
        )
        
        nn.train(
            X_train_pca.T, y_train_s.T,
            X_val_pca.T, y_val_s.T,
            y_scaler=scaler_y, # Pass scaler for internal MEE calc
            epochs=MAX_EPOCHS,
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            l2_lambda=config['l2_lambda'],
            momentum=config['momentum'],
            optimizer=config['optimizer'],
            patience=PATIENCE,
            verbose=False
        )
        
        # 6. Evaluate
        # Predict on Val (Shape: features x samples -> samples x features)
        pred_s = nn.predict(X_val_pca.T).T
        pred = scaler_y.inverse_transform(pred_s)
        
        score = mee(y_val, pred)
        fold_mees.append(score)
        
        print(f"  Fold {fold_idx+1}: MEE = {score:.4f}")
        
    mean_mee = np.mean(fold_mees)
    std_mee = np.std(fold_mees)
    elapsed = time.time() - start_time
    
    print(f"Result: {mean_mee:.4f} (+/- {std_mee:.4f}) | Time: {elapsed:.1f}s")
    return mean_mee, std_mee

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("PHASE 3: ADVANCED FEATURE ENGINEERING")
    print("=====================================")
    print(f"Base Config: {BASE_CONFIG['hidden_layers']} {BASE_CONFIG['hidden_activation']}")
    
    # Load RAW data (no features yet)
    X, y = load_cup_data(DATA_PATH, use_polynomial_features=False)
    
    results = {}
    
    for n_comp in PCA_COMPONENTS_LIST:
        mean, std = run_feature_experiment(X, y, n_comp, BASE_CONFIG)
        results[n_comp] = {'mean': mean, 'std': std}
        
    print("\nSUMMARY OF PHASE 3 RESULTS:")
    print("---------------------------")
    print(f"{'PCA Components':<15} | {'Mean MEE':<10} | {'Std Dev':<10}")
    print("-" * 45)
    
    best_pca = None
    best_score = float('inf')
    
    for n_comp, res in results.items():
        print(f"{n_comp:<15} | {res['mean']:<10.4f} | {res['std']:<10.4f}")
        if res['mean'] < best_score:
            best_score = res['mean']
            best_pca = n_comp
            
    print("-" * 45)
    print(f"Best Configuration: Poly=3 + PCA={best_pca} => {best_score:.4f} MEE")
