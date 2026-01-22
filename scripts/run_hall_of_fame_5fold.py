#!/usr/bin/env python3
"""
Phase 1: Reproduce Hall of Fame Best Model on Proper 5-Fold CV

Simplified version that matches the genetic_search.py pattern exactly.
"""

import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime
import json
import os

from src.data_loader import load_cup_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee
from src.cv_utils import k_fold_split

# ============================================================================
# CONFIGURATION FROM HALL OF FAME
# ============================================================================

BEST_CONFIG = {
    'hidden_layers': [110, 103, 95],
    'learning_rate': 0.011984920164083283,
    'l2_lambda': 0.0006003417127125739,
    'dropout_rate': 0.13106988250978824,
    'momentum': 0.9838483610904715,  # Use mid-range momentum from hall of fame
    'batch_size': 64,
    'hidden_activation': 'tanh',
    'optimizer': 'adam',
    'use_batch_norm': False,
    'weight_init': 'he',
    'use_polynomial_features': True,
    'poly_degree': 2,
}

# CV Settings
N_SPLITS = 5
RANDOM_STATE = 42
MAX_EPOCHS = 2500
PATIENCE = 100

# Output
RESULTS_FILE = 'experiments/hall_of_fame_5fold_results.csv'
MODEL_DIR = 'experiments/hall_of_fame_models'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_dir_exists(dirpath):
    """Create directory if it doesn't exist."""
    import os
    os.makedirs(dirpath, exist_ok=True)

def train_single_fold(fold_idx, X_train, y_train, X_val, y_val, X_val_scaled, y_val_scaled, 
                       config, y_scaler):
    """
    Train a single fold.
    
    Returns:
        dict with fold results
    """
    print(f"  Fold {fold_idx + 1}/{N_SPLITS}: ", end='', flush=True)
    start_time = time.time()
    
    # Initialize network
    nn = NeuralNetworkV2(
        layer_sizes=[X_train.shape[0]] + config['hidden_layers'] + [y_train.shape[0]],
        hidden_activation=config['hidden_activation'],
        weight_init=config['weight_init'],
        use_batch_norm=config['use_batch_norm'],
        dropout_rate=config['dropout_rate'],
        random_state=RANDOM_STATE + fold_idx  # Different init per fold
    )
    
    # Train
    history = nn.train(
        X_train, y_train,
        X_val, y_val,
        y_scaler=y_scaler,
        epochs=MAX_EPOCHS,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        optimizer=config['optimizer'],
        momentum=config['momentum'],
        l2_lambda=config['l2_lambda'],
        patience=PATIENCE,
        verbose=False
    )
    
    elapsed = time.time() - start_time
    
    # Get best validation MEE
    best_val_mee = min(history['val_mee']) if history['val_mee'] else float('inf')
    best_epoch = np.argmin(history['val_mee']) if history['val_mee'] else 0
    
    # Get final train Loss
    final_train_loss = history['train_loss'][-1] if history['train_loss'] else float('inf')
    
    print(f"Val MEE: {best_val_mee:.4f} (epoch {best_epoch}) | Train Loss: {final_train_loss:.4f} | {elapsed:.1f}s")
    
    return {
        'fold': fold_idx + 1,
        'best_val_mee': best_val_mee,
        'best_epoch': best_epoch,
        'final_train_loss': final_train_loss,
        'n_samples_train': X_train.shape[0],
        'n_samples_val': X_val.shape[0],
        'elapsed_seconds': elapsed,
        'model': nn
    }

def run_k_fold_cv(X, y, config, n_splits=5):
    """
    Run k-fold cross-validation on the best config.
    """
    print(f"\n{'='*70}")
    print(f"Running {n_splits}-Fold Cross-Validation")
    print(f"Configuration: {config['hidden_layers']} {config['hidden_activation']}, "
          f"dropout={config['dropout_rate']:.4f}, poly={config['poly_degree']}")
    print(f"{'='*70}")
    
    fold_results = []
    all_val_mees = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold_split(X, n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale per fold (important for proper CV!)
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train)
        
        X_train_scaled = x_scaler.transform(X_train)
        X_val_scaled = x_scaler.transform(X_val)
        y_train_scaled = y_scaler.transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        
        # Train fold
        result = train_single_fold(
            fold_idx, 
            X_train_scaled.T, y_train_scaled.T,  # NNv2 expects (features, samples)
            X_val_scaled.T, y_val_scaled.T,
            X_val_scaled, y_val_scaled,
            config, y_scaler
        )
        
        fold_results.append(result)
        all_val_mees.append(result['best_val_mee'])
    
    # Summary
    mean_mee = np.mean(all_val_mees)
    std_mee = np.std(all_val_mees)
    min_mee = np.min(all_val_mees)
    max_mee = np.max(all_val_mees)
    
    print(f"\n{'='*70}")
    print(f"Cross-Validation Results:")
    print(f"  Mean Val MEE: {mean_mee:.4f} ± {std_mee:.4f}")
    print(f"  Min:  {min_mee:.4f}  |  Max:  {max_mee:.4f}")
    print(f"{'='*70}\n")
    
    return {
        'fold_results': fold_results,
        'mean_mee': mean_mee,
        'std_mee': std_mee,
        'min_mee': min_mee,
        'max_mee': max_mee,
        'all_mees': all_val_mees
    }

def save_results(cv_results, config):
    """Save CV results to CSV."""
    ensure_dir_exists('experiments')
    
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'fold', 'best_val_mee', 'best_epoch', 
            'final_train_loss', 'n_samples_train', 'n_samples_val', 
            'elapsed_seconds', 'config'
        ])
        
        for result in cv_results['fold_results']:
            writer.writerow([
                datetime.now().isoformat(),
                result['fold'],
                result['best_val_mee'],
                result['best_epoch'],
                result['final_train_loss'],
                result['n_samples_train'],
                result['n_samples_val'],
                result['elapsed_seconds'],
                json.dumps(config)
            ])
    
    # Summary row
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'SUMMARY',
            f"Mean: {cv_results['mean_mee']:.4f}",
            cv_results['std_mee'],
            f"Min: {cv_results['min_mee']:.4f}",
            f"Max: {cv_results['max_mee']:.4f}",
            '',
            '',
            '',
            ''
        ])
    
    print(f"✅ Results saved to {RESULTS_FILE}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 1: 5-Fold CV on Best Hall of Fame Config')
    parser.add_argument('--data', default='data/ML-CUP25-TR.csv', help='Training data path')
    parser.add_argument('--n-splits', type=int, default=N_SPLITS, help='Number of folds')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE, help='Random seed')
    args = parser.parse_args()
    
    N_SPLITS = args.n_splits
    RANDOM_STATE = args.seed
    
    # Load data
    print("Loading data with polynomial features (degree 2)...")
    X, y = load_cup_data(
        args.data,
        use_polynomial_features=BEST_CONFIG['use_polynomial_features'],
        poly_degree=BEST_CONFIG['poly_degree']
    )
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Run CV
    cv_results = run_k_fold_cv(X, y, BEST_CONFIG, n_splits=N_SPLITS)
    
    # Save results
    save_results(cv_results, BEST_CONFIG)
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE: Hall of Fame Best Config on 5-Fold CV")
    print("="*70)
    print(f"Expected MEE: ~12.27 (from genetic search on 90/10 split)")
    print(f"Actual MEE:   {cv_results['mean_mee']:.4f} ± {cv_results['std_mee']:.4f}")
    print(f"Interpretation: {'✅ VALIDATED - Ready for Phase 2 (Ensemble)' if cv_results['mean_mee'] < 15 else '⚠️ HIGHER THAN EXPECTED - May need feature tuning'}")
    print("="*70)

