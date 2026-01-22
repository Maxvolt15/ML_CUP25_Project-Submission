#!/usr/bin/env python3
"""
STEP 4: Generate Test Set Predictions
Uses ensemble of top 10 hall-of-fame models to predict on blind test set.
Output: CSV file for ML-CUP 2025 submission.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
import time

from src.data_loader import load_cup_data, load_cup_test_data
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

# Configuration
HALL_OF_FAME_PATH = 'experiments/hall_of_fame.csv'
TRAIN_DATA_PATH = 'data/ML-CUP25-TR.csv'
TEST_DATA_PATH = 'data/ML-CUP25-TS.csv'
OUTPUT_FILE = 'experiments/ML-CUP25-TS-predictions.csv'

TOP_N_MODELS = 10
MAX_EPOCHS = 2500
PATIENCE = 100
RANDOM_STATE = 42

def load_top_configs(n=10):
    """Load top N configurations from hall of fame."""
    import ast
    
    df = pd.read_csv(HALL_OF_FAME_PATH)
    configs = []
    
    print(f"Loading top {n} models from Hall of Fame...")
    for i in range(min(n, len(df))):
        param_str = df.iloc[i]['params']
        clean_str = param_str.replace("np.float64(", "").replace(")", "")
        
        try:
            config = ast.literal_eval(clean_str)
            configs.append(config)
        except Exception as e:
            print(f"Error parsing config {i+1}: {e}")
    
    return configs

def train_models_on_full_data(X_train, y_train, configs):
    """
    Train ensemble models on full training data.
    This is done once, then models make predictions on test set.
    """
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    
    X_train_scaled = x_scaler.transform(X_train).T
    y_train_scaled = y_scaler.transform(y_train).T
    
    models = []
    print(f"\nTraining {len(configs)} ensemble models on full training data...")
    
    for model_idx, config in enumerate(configs):
        print(f"  Training model {model_idx + 1}/{len(configs)}...", end=" ")
        
        nn = NeuralNetworkV2(
            layer_sizes=[X_train.shape[1]] + config['hidden_layers'] + [y_train.shape[1]],
            hidden_activation=config['hidden_activation'],
            weight_init=config['weight_init'],
            use_batch_norm=config['use_batch_norm'],
            dropout_rate=config['dropout_rate'],
            random_state=RANDOM_STATE + model_idx
        )
        
        nn.train(
            X_train_scaled, y_train_scaled,
            epochs=1500,  # Reduced from MAX_EPOCHS for speed (no validation)
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            l2_lambda=config['l2_lambda'],
            momentum=config['momentum'],
            optimizer=config['optimizer'],
            patience=0,  # No early stopping (no validation set)
            verbose=False
        )
        
        models.append((nn, x_scaler, y_scaler))
        print("✓")
    
    return models

def predict_ensemble(X_test, models):
    """
    Make predictions using ensemble of trained models.
    
    Args:
        X_test: (n_samples, n_features) test data
        models: List of (nn, x_scaler, y_scaler) tuples
    
    Returns:
        y_pred: (n_samples, 4) ensemble predictions
    """
    ensemble_pred = np.zeros((X_test.shape[0], 4))
    
    print(f"\nMaking ensemble predictions on {X_test.shape[0]} test samples...")
    
    for model_idx, (nn, x_scaler, y_scaler) in enumerate(models):
        # Scale test data
        X_test_scaled = x_scaler.transform(X_test).T
        
        # Predict
        y_pred_scaled = nn.predict(X_test_scaled).T
        
        # Inverse scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        
        # Add to ensemble
        ensemble_pred += y_pred
        
        print(f"  Model {model_idx + 1}/{len(models)} done")
    
    # Average
    ensemble_pred /= len(models)
    
    return ensemble_pred

def save_predictions(y_pred, output_file):
    """
    Save predictions in ML-CUP format.
    
    Format:
    # Suranjan Kumar Ghosh, Abinash
    # Ghosh_Abinash
    # ML-CUP25
    # 22 January 2026
    1,y1,y2,y3,y4
    ...
    """
    print(f"\nSaving predictions to {output_file}...")
    
    with open(output_file, 'w', newline='') as f:
        # 1. Mandatory Comment Header
        f.write('# Suranjan Kumar Ghosh, Abinash Boruah\n')
        f.write('# RockOn\n')
        f.write('# ML-CUP25\n')
        f.write('# 22 January 2026\n')
        
        # 2. Data Rows
        writer = csv.writer(f)
        for i, pred in enumerate(y_pred):
            # ID must be integer (i+1)
            row = [int(i + 1)] + pred.tolist()
            writer.writerow(row)
    
    print(f"✓ Saved {len(y_pred)} predictions")

def main():
    print("=" * 70)
    print("ML-CUP 2025: TEST SET PREDICTION")
    print("=" * 70)
    
    # Load configurations
    configs = load_top_configs(TOP_N_MODELS)
    if not configs:
        print("ERROR: Could not load any configurations!")
        return
    
    # Load training data and train ensemble
    print(f"\nLoading training data from {TRAIN_DATA_PATH}...")
    X_train, y_train = load_cup_data(
        TRAIN_DATA_PATH,
        use_polynomial_features=True,
        poly_degree=2
    )
    print(f"✓ Loaded {X_train.shape[0]} training samples, {X_train.shape[1]} features")
    
    # Train models
    start_time = time.time()
    models = train_models_on_full_data(X_train, y_train, configs)
    train_time = time.time() - start_time
    print(f"Training complete ({train_time:.1f}s)")
    
    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    X_test = load_cup_test_data(TEST_DATA_PATH)
    print(f"✓ Loaded {X_test.shape[0]} test samples, {X_test.shape[1]} features")
    
    # Apply same polynomial features
    from src.data_loader import add_polynomial_features
    X_test = add_polynomial_features(X_test, degree=2)
    print(f"✓ After polynomial features: {X_test.shape[1]} features")
    
    # Make predictions
    start_time = time.time()
    y_pred = predict_ensemble(X_test, models)
    pred_time = time.time() - start_time
    print(f"Prediction complete ({pred_time:.1f}s)")
    
    # Save predictions
    save_predictions(y_pred, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print(f"SUCCESS: Predictions saved to {OUTPUT_FILE}")
    print(f"Total runtime: {(train_time + pred_time):.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
