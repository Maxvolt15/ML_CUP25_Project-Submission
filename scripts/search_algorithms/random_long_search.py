"""
24-Hour Random Search for ML-CUP 2025

This script runs a pure random search for 24 hours to find optimal hyperparameters.
- Saves to a SEPARATE file from genetic_search to avoid conflicts
- Loads existing best configs and tries to beat them
- Periodic saves to prevent data loss
- Time-based stopping (24 hours)
"""

import numpy as np
import pandas as pd
import csv
import random
import time
import os
import sys
from datetime import datetime, timedelta
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_loader import load_cup_data, add_polynomial_features
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

# --- Configuration ---
SEARCH_DURATION_HOURS = 24
MAX_EPOCHS = 2500
PATIENCE = 50

# Output files - SEPARATE from genetic search
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                          'experiments', 'random_search')
HALL_OF_FAME_FILE = os.path.join(OUTPUT_DIR, 'random_hall_of_fame.csv')
FULL_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'random_full_results.csv')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_random_config():
    """Generates a completely random model configuration."""
    num_layers = random.randint(1, 4)
    hidden_layers = []
    for i in range(num_layers):
        # Funnel shape: decreasing sizes
        max_neurons = 256 if i == 0 else hidden_layers[-1]
        min_neurons = 16
        hidden_layers.append(random.randint(min_neurons, max(min_neurons, max_neurons)))
    
    # Sort descending for funnel shape
    hidden_layers.sort(reverse=True)
    
    return {
        'hidden_layers': hidden_layers,
        'learning_rate': 10 ** random.uniform(-4, -1),  # 0.0001 to 0.1
        'l2_lambda': 10 ** random.uniform(-6, -2),       # 1e-6 to 0.01
        'dropout_rate': random.uniform(0.0, 0.5),
        'momentum': random.uniform(0.8, 0.99),
        'batch_size': random.choice([16, 32, 64, 128]),
        'hidden_activation': random.choice(['tanh', 'relu', 'leaky_relu', 'elu', 'swish', 'mish']),
        'optimizer': random.choice(['adam', 'sgd', 'nesterov']),
        'use_batch_norm': random.choice([True, False]),
        'weight_init': random.choice(['he', 'xavier'])
    }


def evaluate_model(params, X_train, y_train, X_val, y_val, y_scaler):
    """Trains and evaluates a model. Returns MEE."""
    try:
        # Layer setup
        layer_sizes = [X_train.shape[0]] + params['hidden_layers'] + [y_train.shape[0]]
        
        # Init Check - match init to activation
        init = params['weight_init']
        if params['hidden_activation'] in ['relu', 'leaky_relu', 'elu', 'swish', 'mish'] and init == 'xavier':
            init = 'he'
        elif params['hidden_activation'] == 'tanh' and init == 'he':
            init = 'xavier' 
        
        nn = NeuralNetworkV2(
            layer_sizes=layer_sizes,
            hidden_activation=params['hidden_activation'],
            weight_init=init,
            use_batch_norm=params['use_batch_norm'],
            dropout_rate=params['dropout_rate'],
            random_state=random.randint(1, 10000)
        )
        
        history = nn.train(
            X_train, y_train, X_val, y_val, y_scaler,
            epochs=MAX_EPOCHS,
            batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            optimizer=params['optimizer'],
            momentum=params['momentum'],
            l2_lambda=params['l2_lambda'],
            patience=PATIENCE,
            verbose=False
        )
        
        if not history['val_mee']:
            return 999.0
            
        return min(history['val_mee'])
        
    except Exception as e:
        print(f"    Error: {e}")
        return 999.0


def load_existing_results():
    """Load existing hall of fame if it exists."""
    results = []
    if os.path.exists(HALL_OF_FAME_FILE):
        try:
            df = pd.read_csv(HALL_OF_FAME_FILE)
            for _, row in df.iterrows():
                try:
                    # Handle both string repr and actual dict
                    params = row['params']
                    if isinstance(params, str):
                        # Clean up np.float64 references
                        params = params.replace('np.float64(', '').replace(')', '')
                        params = eval(params)
                    results.append((float(row['mee']), params))
                except Exception as e:
                    print(f"    Warning: Could not parse row: {e}")
        except Exception as e:
            print(f"    Warning: Could not load existing results: {e}")
    return results


def save_hall_of_fame(all_results):
    """Save top 50 results to hall of fame."""
    # Sort by MEE and keep top 50
    sorted_results = sorted(all_results, key=lambda x: x[0])[:50]
    
    with open(HALL_OF_FAME_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'mee', 'params'])
        for i, (score, params) in enumerate(sorted_results):
            writer.writerow([i+1, score, str(params)])
    
    return sorted_results[0][0] if sorted_results else float('inf')


def save_full_results(all_results, trial_count, elapsed_hours):
    """Save all results with metadata."""
    with open(FULL_RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'mee', 'params', 'trial_count', 'elapsed_hours'])
        sorted_results = sorted(all_results, key=lambda x: x[0])
        for i, (score, params) in enumerate(sorted_results):
            writer.writerow([i+1, score, str(params), trial_count, f"{elapsed_hours:.2f}"])


def run_random_search():
    """Main random search loop."""
    print("=" * 70)
    print("STARTING 24-HOUR RANDOM HYPERPARAMETER SEARCH")
    print("=" * 70)
    print(f"Duration: {SEARCH_DURATION_HOURS} hours")
    print(f"Max epochs per model: {MAX_EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"Output: {HALL_OF_FAME_FILE}")
    print("=" * 70)
    
    # Load Data
    print("\nLoading and preprocessing data...")
    X, y = load_cup_data('data/ML-CUP25-TR.csv', use_polynomial_features=True, poly_degree=2)
    
    # 90/10 Split (same as genetic search)
    X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=999)
    
    x_scaler = StandardScaler().fit(X_train_full)
    y_scaler = StandardScaler().fit(y_train)
    
    X_train = x_scaler.transform(X_train_full).T
    X_val = x_scaler.transform(X_val_full).T
    y_train = y_scaler.transform(y_train).T
    # y_val stays UNSCALED, just transposed (for MEE in original scale)
    y_val = y_val.T
    
    print(f"Training set: {X_train.shape[1]} samples, {X_train.shape[0]} features")
    print(f"Validation set: {X_val.shape[1]} samples")
    
    # Load existing results
    all_results = load_existing_results()
    if all_results:
        print(f"\nLoaded {len(all_results)} existing results. Best MEE: {min(r[0] for r in all_results):.4f}")
    else:
        print("\nNo existing results found. Starting fresh.")
    
    # Track time
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=SEARCH_DURATION_HOURS)
    
    print(f"\nStarted at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Will end at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    trial_count = len(all_results)
    best_mee = min((r[0] for r in all_results), default=float('inf'))
    last_save_time = start_time
    
    try:
        while datetime.now() < end_time:
            trial_count += 1
            trial_start = time.time()
            
            # Generate random config
            config = get_random_config()
            
            # Evaluate
            mee_score = evaluate_model(config, X_train, y_train, X_val, y_val, y_scaler)
            
            elapsed_trial = time.time() - trial_start
            elapsed_total = (datetime.now() - start_time).total_seconds() / 3600
            remaining = (end_time - datetime.now()).total_seconds() / 3600
            
            # Track result
            all_results.append((mee_score, config))
            
            # Update best
            if mee_score < best_mee:
                best_mee = mee_score
                print(f"\n🎯 NEW BEST! Trial {trial_count}: MEE={mee_score:.4f}")
                print(f"   Config: {config['hidden_layers']} {config['hidden_activation']} {config['optimizer']}")
                print(f"   LR={config['learning_rate']:.6f}, L2={config['l2_lambda']:.6f}, Dropout={config['dropout_rate']:.2f}")
            
            # Progress update every 10 trials
            if trial_count % 10 == 0:
                print(f"Trial {trial_count}: MEE={mee_score:.4f} ({elapsed_trial:.1f}s) | "
                      f"Best={best_mee:.4f} | {elapsed_total:.1f}h elapsed, {remaining:.1f}h remaining | "
                      f"{config['hidden_layers']} {config['hidden_activation']}")
            
            # Save every 5 minutes
            if (datetime.now() - last_save_time).total_seconds() > 300:
                save_hall_of_fame(all_results)
                save_full_results(all_results, trial_count, elapsed_total)
                last_save_time = datetime.now()
                print(f"    [Auto-saved {len(all_results)} results]")
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    
    # Final save
    elapsed_hours = (datetime.now() - start_time).total_seconds() / 3600
    save_hall_of_fame(all_results)
    save_full_results(all_results, trial_count, elapsed_hours)
    
    # Summary
    print("\n" + "=" * 70)
    print("RANDOM SEARCH COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed_hours:.2f} hours")
    print(f"Total trials: {trial_count}")
    print(f"Trials per hour: {trial_count / max(elapsed_hours, 0.01):.1f}")
    print(f"\nBest MEE: {best_mee:.4f}")
    
    # Show top 5
    sorted_results = sorted(all_results, key=lambda x: x[0])[:5]
    print("\nTop 5 Configurations:")
    for i, (score, config) in enumerate(sorted_results):
        print(f"  #{i+1} MEE={score:.4f}: {config['hidden_layers']} {config['hidden_activation']} {config['optimizer']}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    run_random_search()
