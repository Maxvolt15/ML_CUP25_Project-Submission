"""
Simple and Robust Hyperparameter Search for ML-CUP 2025

This script focuses on simpler, more robust architectures to avoid overfitting.
It targets:
- Shallow networks (1-2 layers) with fewer neurons.
- Smoother activations (Tanh, ELU).
- Stronger regularization.
"""

import numpy as np
import pandas as pd
import csv
import random
# from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, loguniform
from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee
from src.cv_utils import k_fold_split
import sys

def run_simple_search(results_file='simple_search_results.csv', n_splits=5):
    print(f"{ '='*60}")
    print("Starting Simple & Robust Hyperparameter Search")
    print(f"{ '='*60}\n")
    
    # Constrained search space for robustness
    param_dist = {
        'learning_rate': loguniform(1e-4, 1e-1),
        'l2_lambda': loguniform(1e-5, 1e-1),  # Higher regularization floor
        'batch_size': [16, 32],
        'hidden_layers': [
            [20], [30], [40], [64],          # 1 Hidden Layer
            [20, 10], [30, 15], [32, 16],    # 2 Hidden Layers (small)
            [64, 32]                         # 2 Hidden Layers (medium)
        ],
        'hidden_activation': ['tanh', 'elu'], # Smoother activations
        'optimizer': ['adam', 'sgd'],
        'momentum': uniform(loc=0.8, scale=0.19), # 0.8 - 0.99
        
        # Regularization
        'use_batch_norm': [True, False],
        'dropout_rate': uniform(0.0, 0.3), # Lower max dropout for small nets
        
        # Features
        'use_polynomial_features': [True, False],
        'poly_degree': [2],
        'use_pca_features': [False],
        'pca_components': [0]
    }
    
    # File setup
    try:
        with open(results_file, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'avg_mee', 'std_mee', 'params'])
    except FileExistsError:
        pass 

    iteration = 0
    # Run for a fixed number of iterations or until interrupted
    max_iterations = 50 
    
    while iteration < max_iterations:
        iteration += 1
        
        # --- 1. Sample Hyperparameters ---
        params = {key: dist.rvs() if hasattr(dist, 'rvs') else random.choice(dist) 
                  for key, dist in param_dist.items()}

        # Conditional logic
        if not params['use_polynomial_features']:
            params['poly_degree'] = 0
        
        # He init for ELU/ReLU, Xavier for Tanh
        if params['hidden_activation'] in ['relu', 'leaky_relu', 'elu']:
            params['weight_init'] = 'he'
        else:
            params['weight_init'] = 'xavier'

        print(f"\n[{iteration}/{max_iterations}] Testing:")
        print(f"  Arch: {params['hidden_layers']}, Act: {params['hidden_activation']}")
        print(f"  Reg: L2={params['l2_lambda']:.2e}, DO={params['dropout_rate']:.2f}, BN={params['use_batch_norm']}")
        print(f"  Train: Opt={params['optimizer']}, LR={params['learning_rate']:.5f}")

        # --- 2. Load Data ---
        X, y = load_cup_data('data/ML-CUP25-TR.csv', 
                               use_polynomial_features=params['use_polynomial_features'],
                               poly_degree=params['poly_degree'])
        
        # kf = KFold(n_splits=n_splits, shuffle=True, random_state=iteration)
        fold_mees = []
        
        # --- 3. CV Loop ---
        for fold, (train_idx, val_idx) in enumerate(k_fold_split(X, n_splits=n_splits, shuffle=True, random_state=iteration)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            x_scaler = StandardScaler().fit(X_train)
            y_scaler = StandardScaler().fit(y_train)
            
            X_train_scaled = x_scaler.transform(X_train)
            y_train_scaled = y_scaler.transform(y_train)
            X_val_scaled = x_scaler.transform(X_val)
            
            layer_sizes = [X_train_scaled.shape[1]] + params['hidden_layers'] + [y_train.shape[1]]
            
            nn = NeuralNetworkV2(
                layer_sizes=layer_sizes,
                hidden_activation=params['hidden_activation'],
                weight_init=params['weight_init'],
                use_batch_norm=params['use_batch_norm'],
                dropout_rate=params['dropout_rate'],
                random_state=42 + fold
            )
            
            nn.train(
                X_train_scaled.T, y_train_scaled.T,
                X_val_scaled.T, y_val.T, y_scaler, 
                epochs=2000, # Sufficient for small nets
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                optimizer=params['optimizer'],
                momentum=params['momentum'],
                l2_lambda=params['l2_lambda'],
                patience=150, # Higher patience for convergence
                verbose=False
            )
            
            y_pred_scaled = nn.predict(X_val_scaled.T)
            y_pred = y_scaler.inverse_transform(y_pred_scaled.T)
            fold_mees.append(mee(y_val, y_pred))
            
            # Quick check to abort bad runs early (optional optimization)
            if fold == 0 and fold_mees[0] > 100:
                print("  -> Aborting run (high error on fold 1)")
                fold_mees = [999] * n_splits # Penalty
                break

        avg_mee = np.mean(fold_mees)
        std_mee = np.std(fold_mees)
        
        print(f"  -> MEE: {avg_mee:.4f} (+/- {std_mee:.4f})")
        
        # --- 4. Save ---
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iteration, f"{avg_mee:.4f}", f"{std_mee:.4f}", str(params)])

if __name__ == '__main__':
    try:
        run_simple_search()
    except KeyboardInterrupt:
        print("\n\nSearch stopped by user.")
