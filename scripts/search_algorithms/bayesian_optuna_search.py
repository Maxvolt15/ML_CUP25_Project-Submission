"""
Bayesian Hyperparameter Optimization using Optuna

This script uses Optuna for sample-efficient hyperparameter search.
Bayesian optimization builds a probabilistic model of the objective function
and uses it to select the most promising hyperparameters to evaluate.

Why Optuna/Bayesian Optimization?
---------------------------------
1. MORE SAMPLE-EFFICIENT: Random search treats all hyperparameter combinations equally.
   Bayesian optimization learns from previous trials and focuses on promising regions.
   
2. SMARTER EXPLORATION: Uses Tree-structured Parzen Estimator (TPE) to model the 
   relationship between hyperparameters and performance.
   
3. PRUNING: Can stop unpromising trials early, saving computation time.

4. TYPICAL IMPROVEMENT: 10-30% better than random search with same budget.

References:
-----------
- Bergstra et al., "Algorithms for Hyper-Parameter Optimization", NIPS 2011
- Akiba et al., "Optuna: A Next-generation Hyperparameter Optimization Framework", 2019

Usage:
------
    python -m scripts.search_algorithms.bayesian_optuna_search

Author: Suranjan (maxvolt)
Date: January 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

# from sklearn.model_selection import KFold  # Removed - using custom k_fold_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from src.neural_network_v2 import NeuralNetworkV2
from src.data_loader import load_cup_data
from src.utils import mee
from src.cv_utils import k_fold_split  # Our custom k-fold (NOT sklearn!)

# Configuration
RANDOM_STATE = 42
N_FOLDS = 5
N_TRIALS = 50  # Number of Optuna trials (reduced for faster initial search)
TIMEOUT = 3600  # 1 hour max
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "ML-CUP25-TR.csv")

np.random.seed(RANDOM_STATE)


def preprocess_data(X, y, poly_degree=2):
    """
    Apply polynomial feature expansion and standardization.
    """
    # Polynomial features
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Standardize
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_poly)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, poly


def objective(trial):
    """
    Optuna objective function.
    
    Defines the hyperparameter search space and evaluates each configuration
    using 5-fold cross-validation.
    """
    # === Hyperparameter Search Space ===
    
    # Architecture (3 hidden layers)
    hidden1 = trial.suggest_int('hidden1', 64, 200)
    hidden2 = trial.suggest_int('hidden2', 32, 150)
    hidden3 = trial.suggest_int('hidden3', 16, 100)
    
    # Ensure decreasing layer sizes (common best practice)
    if hidden2 > hidden1:
        hidden2 = hidden1
    if hidden3 > hidden2:
        hidden3 = hidden2
    
    # Learning rate (log-uniform for wide range)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True)
    
    # Regularization
    l2_lambda = trial.suggest_float('l2_lambda', 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.35)
    
    # Training config
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Optimizer choice
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'nesterov'])
    
    # Loss function - Use MSE for training (lecture recommendation)
    # MEE is used for evaluation only
    loss_fn = 'mse'  # Fixed: train on MSE, evaluate on MEE
    
    # LR scheduler
    lr_scheduler = trial.suggest_categorical('lr_scheduler', ['none', 'cosine'])
    
    # Activation
    activation = trial.suggest_categorical('activation', ['tanh', 'relu', 'leaky_relu'])
    
    # Momentum (only relevant for Nesterov)
    momentum = trial.suggest_float('momentum', 0.85, 0.99) if optimizer == 'nesterov' else 0.9
    
    # === Load and preprocess data ===
    X, y = load_cup_data(DATA_PATH)
    X_processed, y_processed, _, scaler_y, _ = preprocess_data(X, y, poly_degree=2)
    
    input_size = X_processed.shape[1]
    output_size = y.shape[1]
    layer_sizes = [input_size, hidden1, hidden2, hidden3, output_size]
    
    # === 5-Fold Cross-Validation (using our custom k_fold_split) ===
    fold_mees = []
    
    for fold, (train_idx, val_idx) in enumerate(k_fold_split(X_processed, n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)):
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y_processed[train_idx], y_processed[val_idx]
        
        # Transpose for our NN (expects features x samples)
        X_train_T = X_train.T
        X_val_T = X_val.T
        y_train_T = y_train.T
        y_val_T = y_val.T
        
        # Initialize network
        nn = NeuralNetworkV2(
            layer_sizes=layer_sizes,
            hidden_activation=activation,
            output_activation='linear',
            weight_init='he',
            dropout_rate=dropout_rate,
            random_state=RANDOM_STATE + fold
        )
        
        # Train
        nn.train(
            X_train_T, y_train_T,
            X_val=X_val_T, y_val=y_val_T,
            y_scaler=scaler_y,
            epochs=2000,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            momentum=momentum,
            l2_lambda=l2_lambda,
            lr_scheduler=lr_scheduler,
            patience=100,
            loss_fn=loss_fn,
            verbose=False
        )
        
        # Evaluate
        y_pred_scaled = nn.predict(X_val_T)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.T)
        y_val_orig = scaler_y.inverse_transform(y_val)
        
        fold_mee = mee(y_val_orig, y_pred)
        fold_mees.append(fold_mee)
        
        # Optuna pruning: report intermediate result
        trial.report(np.mean(fold_mees), fold)
        
        # Prune if this trial is unpromising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return np.mean(fold_mees)


def run_bayesian_search():
    """
    Run Bayesian hyperparameter optimization using Optuna.
    """
    print("=" * 70)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION (Optuna)")
    print("=" * 70)
    print(f"Trials: {N_TRIALS}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Sampler: TPE (Tree-structured Parzen Estimator)")
    print(f"Pruner: Median (stops unpromising trials early)")
    print("=" * 70)
    
    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    
    study = optuna.create_study(
        direction='minimize',  # Minimize MEE
        sampler=sampler,
        pruner=pruner,
        study_name='ml_cup25_nn_optimization'
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best MEE: {study.best_value:.4f}")
    
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_df = study.trials_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(PROJECT_ROOT, "experiments", f"optuna_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    # Feature importance (hyperparameter importance)
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nHyperparameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"Could not compute importance: {e}")
    
    return study


if __name__ == "__main__":
    study = run_bayesian_search()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Configuration")
    print("=" * 70)
    print(f"""
Best MEE: {study.best_value:.4f}

Optimal Hyperparameters:
------------------------
hidden_sizes: [{study.best_params['hidden1']}, {study.best_params['hidden2']}, {study.best_params['hidden3']}]
learning_rate: {study.best_params['learning_rate']:.6f}
l2_lambda: {study.best_params['l2_lambda']:.6f}
dropout_rate: {study.best_params['dropout_rate']:.4f}
batch_size: {study.best_params['batch_size']}
optimizer: {study.best_params['optimizer']}
lr_scheduler: {study.best_params['lr_scheduler']}
activation: {study.best_params['activation']}

To use this configuration:
--------------------------
nn = NeuralNetworkV2(
    layer_sizes=[90, {study.best_params['hidden1']}, {study.best_params['hidden2']}, {study.best_params['hidden3']}, 4],
    hidden_activation='{study.best_params['activation']}',
    dropout_rate={study.best_params['dropout_rate']:.4f}
)
nn.train(..., 
    learning_rate={study.best_params['learning_rate']:.6f},
    l2_lambda={study.best_params['l2_lambda']:.6f},
    optimizer='{study.best_params['optimizer']}',
    lr_scheduler='{study.best_params['lr_scheduler']}'
)
""")
