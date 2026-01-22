#!/usr/bin/env python3
"""
============================================================================
Cascade Correlation Neural Network - From Scratch Implementation
============================================================================

This is an EXPERIMENTAL implementation of Cascade Correlation (Fahlman & Lebiere, 1990).

WHAT IS CASCADE CORRELATION?
----------------------------
Unlike fixed-architecture networks, Cascade Correlation (CC) automatically
constructs its architecture by adding one hidden unit at a time.

Algorithm:
1. Start with a network with NO hidden units (direct input→output)
2. Train output weights to minimize error
3. Train a CANDIDATE unit to maximize correlation with residual error
4. Add the best candidate to the network, FREEZE its input weights
5. Retrain output weights
6. Repeat 3-5 until convergence or max units reached

WHY IS IT HIGH RISK?
--------------------
- Complex two-phase training (correlation maximization + output training)
- Debugging is difficult (hard to know if bug or bad hyperparameters)
- Non-standard architecture (professor may question understanding)
- Slower than fixed architecture for same final size
- Can overfit by adding too many units

WHY IS IT HIGH REWARD?
----------------------
- Automatically determines network size
- Often finds smaller networks than grid search
- Shows deep understanding of neural networks
- Novel approach compared to standard MLP

USAGE:
------
    python -m scripts.search_algorithms.cascade_correlation_nn

Author: Suranjan (maxvolt)
Date: January 2026
Reference: Fahlman, S.E. & Lebiere, C. (1990). The Cascade-Correlation Learning Architecture.
============================================================================
"""

import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import KFold  # Removed - using custom k_fold_split
from src.cv_utils import k_fold_split  # Our custom k-fold (NOT sklearn!)
import time


def sigmoid(z):
    """Sigmoid activation with clipping for numerical stability."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """Derivative of sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    """Tanh activation."""
    return np.tanh(z)


def tanh_derivative(z):
    """Derivative of tanh."""
    return 1 - np.tanh(z) ** 2


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def mee(y_true, y_pred):
    """Mean Euclidean Error."""
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))


class CascadeCorrelationNN:
    """
    Cascade Correlation Neural Network implementation from scratch.
    
    This network grows its architecture automatically by adding hidden units
    one at a time, where each new unit maximizes correlation with residual error.
    
    Architecture:
        - All inputs connect to all hidden units
        - Each hidden unit connects to all previous hidden units (cascade)
        - All hidden units connect to outputs
        - No hidden-to-hidden connections within same layer
    
    Parameters:
    -----------
    n_inputs : int
        Number of input features
    n_outputs : int
        Number of output targets
    max_hidden_units : int
        Maximum number of hidden units to add
    candidate_pool_size : int
        Number of candidate units to train in parallel
    activation : str
        Activation function for hidden units ('sigmoid' or 'tanh')
    output_epochs : int
        Epochs to train output weights
    candidate_epochs : int
        Epochs to train candidate units
    output_lr : float
        Learning rate for output weight training
    candidate_lr : float
        Learning rate for candidate training
    patience : int
        Early stopping patience for output training
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_inputs, n_outputs, max_hidden_units=10,
                 candidate_pool_size=8, activation='tanh',
                 output_epochs=500, candidate_epochs=200,
                 output_lr=0.01, candidate_lr=0.1,
                 patience=50, random_state=42):
        
        np.random.seed(random_state)
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.max_hidden_units = max_hidden_units
        self.candidate_pool_size = candidate_pool_size
        self.output_epochs = output_epochs
        self.candidate_epochs = candidate_epochs
        self.output_lr = output_lr
        self.candidate_lr = candidate_lr
        self.patience = patience
        self.random_state = random_state
        
        # Activation function
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        
        # Network structure
        self.hidden_units = []  # List of (input_weights, bias) tuples
        self.output_weights = None  # Shape: (n_outputs, n_inputs + n_hidden)
        self.output_bias = None  # Shape: (n_outputs,)
        
        # Initialize output layer (direct input → output initially)
        self._init_output_weights()
        
        # Training history
        self.history = {
            'n_hidden': [],
            'train_mee': [],
            'val_mee': [],
            'correlation': []
        }
    
    def _init_output_weights(self):
        """Initialize output weights with small random values."""
        n_features = self.n_inputs + len(self.hidden_units)
        self.output_weights = np.random.randn(self.n_outputs, n_features) * 0.1
        self.output_bias = np.zeros(self.n_outputs)
    
    def _compute_hidden_activations(self, X):
        """
        Compute activations of all hidden units.
        
        Each hidden unit receives:
        - All original inputs
        - Outputs from all previous hidden units (cascade structure)
        
        Returns:
        --------
        activations : list of np.ndarray
            Activation of each hidden unit, shape (n_samples,) each
        """
        n_samples = X.shape[0]
        activations = []
        
        for i, (weights, bias) in enumerate(self.hidden_units):
            # Input to this unit = original inputs + previous hidden activations
            if i == 0:
                unit_input = X
            else:
                prev_activations = np.column_stack(activations[:i])
                unit_input = np.hstack([X, prev_activations])
            
            # Compute activation
            z = np.dot(unit_input, weights) + bias
            a = self.activation(z)
            activations.append(a)
        
        return activations
    
    def _compute_network_output(self, X):
        """
        Compute network output given inputs.
        
        Returns:
        --------
        output : np.ndarray, shape (n_samples, n_outputs)
        """
        # Get hidden activations
        hidden_activations = self._compute_hidden_activations(X)
        
        # Concatenate inputs and hidden activations for output layer
        if len(hidden_activations) > 0:
            features = np.hstack([X, np.column_stack(hidden_activations)])
        else:
            features = X
        
        # Linear output (regression)
        output = np.dot(features, self.output_weights.T) + self.output_bias
        
        return output
    
    def _train_output_weights(self, X, y, X_val=None, y_val=None, verbose=False):
        """
        Train output weights using gradient descent.
        Hidden unit weights are FROZEN.
        """
        # Get hidden activations (fixed)
        hidden_activations = self._compute_hidden_activations(X)
        
        if len(hidden_activations) > 0:
            features = np.hstack([X, np.column_stack(hidden_activations)])
        else:
            features = X
        
        # Validation features
        if X_val is not None:
            hidden_activations_val = self._compute_hidden_activations(X_val)
            if len(hidden_activations_val) > 0:
                features_val = np.hstack([X_val, np.column_stack(hidden_activations_val)])
            else:
                features_val = X_val
        
        n_samples = X.shape[0]
        best_weights = self.output_weights.copy()
        best_bias = self.output_bias.copy()
        best_val_mee = float('inf')
        patience_counter = 0
        
        for epoch in range(self.output_epochs):
            # Forward pass
            pred = np.dot(features, self.output_weights.T) + self.output_bias
            
            # Compute gradients
            error = pred - y  # (n_samples, n_outputs)
            grad_w = np.dot(error.T, features) / n_samples
            grad_b = np.mean(error, axis=0)
            
            # Update
            self.output_weights -= self.output_lr * grad_w
            self.output_bias -= self.output_lr * grad_b
            
            # Validation
            if X_val is not None:
                pred_val = np.dot(features_val, self.output_weights.T) + self.output_bias
                val_mee = mee(y_val, pred_val)
                
                if val_mee < best_val_mee:
                    best_val_mee = val_mee
                    best_weights = self.output_weights.copy()
                    best_bias = self.output_bias.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"    Output training stopped at epoch {epoch}")
                    break
        
        # Restore best
        if X_val is not None:
            self.output_weights = best_weights
            self.output_bias = best_bias
        
        return best_val_mee if X_val is not None else mee(y, self._compute_network_output(X))
    
    def _train_candidate_units(self, X, y, verbose=False):
        """
        Train a pool of candidate hidden units to maximize correlation
        with the residual error.
        
        Returns:
        --------
        best_candidate : tuple (weights, bias)
            The candidate with highest correlation
        best_correlation : float
            The correlation achieved
        """
        # Compute current residual error
        pred = self._compute_network_output(X)
        residual = y - pred  # (n_samples, n_outputs)
        
        # Center residual for correlation computation
        residual_mean = np.mean(residual, axis=0)
        residual_centered = residual - residual_mean
        
        # Prepare input to candidate units
        hidden_activations = self._compute_hidden_activations(X)
        if len(hidden_activations) > 0:
            candidate_input = np.hstack([X, np.column_stack(hidden_activations)])
        else:
            candidate_input = X
        
        n_candidate_inputs = candidate_input.shape[1]
        n_samples = X.shape[0]
        
        # Initialize candidate pool
        candidates = []
        for _ in range(self.candidate_pool_size):
            weights = np.random.randn(n_candidate_inputs) * 0.5
            bias = np.random.randn() * 0.1
            candidates.append([weights, bias])
        
        # Train each candidate to maximize correlation with residual
        for epoch in range(self.candidate_epochs):
            for c_idx, (weights, bias) in enumerate(candidates):
                # Forward pass for candidate
                z = np.dot(candidate_input, weights) + bias
                a = self.activation(z)  # (n_samples,)
                
                # Center activation
                a_mean = np.mean(a)
                a_centered = a - a_mean
                
                # Compute correlation with each output dimension
                # S = sum over outputs of (sum over samples of a_centered * residual_centered)
                correlations = np.dot(a_centered, residual_centered)  # (n_outputs,)
                S = np.sum(np.abs(correlations))  # Maximize absolute correlation
                
                # Gradient of S w.r.t. candidate weights
                # dS/da = sum_o sign(corr_o) * residual_centered_o
                signs = np.sign(correlations)
                dS_da = np.dot(residual_centered, signs)  # (n_samples,)
                
                # da/dz = activation derivative
                da_dz = self.activation_derivative(z)
                
                # dz/dw = candidate_input, dz/db = 1
                dS_dz = dS_da * da_dz
                
                grad_w = np.dot(candidate_input.T, dS_dz) / n_samples
                grad_b = np.mean(dS_dz)
                
                # Gradient ASCENT (maximize correlation)
                candidates[c_idx][0] = weights + self.candidate_lr * grad_w
                candidates[c_idx][1] = bias + self.candidate_lr * grad_b
        
        # Find best candidate
        best_correlation = -float('inf')
        best_candidate = None
        
        for weights, bias in candidates:
            z = np.dot(candidate_input, weights) + bias
            a = self.activation(z)
            a_centered = a - np.mean(a)
            
            correlations = np.dot(a_centered, residual_centered)
            S = np.sum(np.abs(correlations))
            
            if S > best_correlation:
                best_correlation = S
                best_candidate = (weights.copy(), bias)
        
        return best_candidate, best_correlation
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the Cascade Correlation network.
        
        Parameters:
        -----------
        X_train : np.ndarray, shape (n_samples, n_features)
        y_train : np.ndarray, shape (n_samples, n_outputs)
        X_val : np.ndarray, optional
        y_val : np.ndarray, optional
        verbose : bool
        
        Returns:
        --------
        self
        """
        if verbose:
            print("=" * 60)
            print("CASCADE CORRELATION TRAINING")
            print("=" * 60)
            print(f"Max hidden units: {self.max_hidden_units}")
            print(f"Candidate pool size: {self.candidate_pool_size}")
            print(f"Initial architecture: {self.n_inputs} → {self.n_outputs}")
        
        # Phase 0: Train initial output weights (no hidden units)
        if verbose:
            print(f"\nPhase 0: Training direct input→output connection...")
        
        self._init_output_weights()
        val_mee = self._train_output_weights(X_train, y_train, X_val, y_val, verbose)
        
        self.history['n_hidden'].append(0)
        self.history['val_mee'].append(val_mee)
        self.history['correlation'].append(0)
        
        if verbose:
            print(f"  Initial MEE: {val_mee:.4f}")
        
        # Add hidden units one at a time
        prev_mee = val_mee
        no_improvement_count = 0
        
        for unit_idx in range(self.max_hidden_units):
            if verbose:
                print(f"\nPhase {unit_idx + 1}: Adding hidden unit {unit_idx + 1}...")
            
            # Phase A: Train candidate units
            best_candidate, correlation = self._train_candidate_units(X_train, y_train, verbose)
            
            if verbose:
                print(f"  Best candidate correlation: {correlation:.4f}")
            
            # Add best candidate to network (FREEZE its weights)
            self.hidden_units.append(best_candidate)
            
            # Phase B: Retrain output weights with new hidden unit
            self._init_output_weights()  # Reset to include new hidden unit
            val_mee = self._train_output_weights(X_train, y_train, X_val, y_val, verbose)
            
            self.history['n_hidden'].append(unit_idx + 1)
            self.history['val_mee'].append(val_mee)
            self.history['correlation'].append(correlation)
            
            if verbose:
                improvement = prev_mee - val_mee
                print(f"  MEE: {val_mee:.4f} (improvement: {improvement:.4f})")
            
            # Early stopping
            if val_mee >= prev_mee:
                no_improvement_count += 1
                if no_improvement_count >= 3:
                    if verbose:
                        print(f"\nStopping: No improvement for 3 consecutive units")
                    # Remove last unit (didn't help)
                    self.hidden_units.pop()
                    self._init_output_weights()
                    self._train_output_weights(X_train, y_train, X_val, y_val, False)
                    break
            else:
                no_improvement_count = 0
                prev_mee = val_mee
        
        if verbose:
            print(f"\n" + "=" * 60)
            print(f"TRAINING COMPLETE")
            print(f"Final architecture: {self.n_inputs} → [{len(self.hidden_units)}] → {self.n_outputs}")
            print(f"Final MEE: {val_mee:.4f}")
            print("=" * 60)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self._compute_network_output(X)
    
    def get_architecture(self):
        """Return a description of the learned architecture."""
        return {
            'n_inputs': self.n_inputs,
            'n_hidden_units': len(self.hidden_units),
            'n_outputs': self.n_outputs,
            'cascade_structure': True
        }


# ============================================================================
# BENCHMARK SCRIPT
# ============================================================================

def run_cascade_benchmark():
    """Run Cascade Correlation on ML-CUP25 and compare with fixed architecture."""
    
    print("=" * 70)
    print("CASCADE CORRELATION BENCHMARK")
    print("Comparing: Cascade Correlation vs Fixed Architecture")
    print("=" * 70)
    
    # Import data loader
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from src.data_loader import load_cup_data
    from src.neural_network_v2 import NeuralNetworkV2
    
    # Load data
    print("\nLoading data...")
    X, y = load_cup_data('data/ML-CUP25-TR.csv', use_polynomial_features=True, poly_degree=2)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # 5-fold CV (using our custom k_fold_split)
    cascade_mees = []
    fixed_mees = []
    cascade_sizes = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold_split(X, n_splits=5, shuffle=True, random_state=42)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}")
        print(f"{'='*60}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale
        scaler_x = StandardScaler()
        X_train_s = scaler_x.fit_transform(X_train)
        X_val_s = scaler_x.transform(X_val)
        
        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train)
        y_val_s = scaler_y.transform(y_val)
        
        # --- Cascade Correlation ---
        print("\n[1] CASCADE CORRELATION")
        start_time = time.time()
        
        cc = CascadeCorrelationNN(
            n_inputs=X_train_s.shape[1],
            n_outputs=y_train.shape[1],
            max_hidden_units=15,
            candidate_pool_size=8,
            activation='tanh',
            output_epochs=500,
            candidate_epochs=100,
            output_lr=0.01,
            candidate_lr=0.05,
            patience=30,
            random_state=42 + fold_idx
        )
        
        cc.fit(X_train_s, y_train_s, X_val_s, y_val_s, verbose=True)
        
        pred_cc = cc.predict(X_val_s)
        pred_cc_unscaled = scaler_y.inverse_transform(pred_cc)
        cc_mee = mee(y_val, pred_cc_unscaled)
        cc_time = time.time() - start_time
        
        cascade_mees.append(cc_mee)
        cascade_sizes.append(len(cc.hidden_units))
        
        print(f"\nCascade Result: {cc_mee:.4f} MEE ({len(cc.hidden_units)} hidden units)")
        print(f"Time: {cc_time:.1f}s")
        
        # --- Fixed Architecture (for comparison) ---
        print("\n[2] FIXED ARCHITECTURE [128, 84, 65]")
        start_time = time.time()
        
        fixed_nn = NeuralNetworkV2(
            layer_sizes=[X_train_s.shape[1], 128, 84, 65, y_train.shape[1]],
            hidden_activation='tanh',
            weight_init='he',
            dropout_rate=0.131,
            random_state=42 + fold_idx
        )
        
        fixed_nn.train(
            X_train_s.T, y_train_s.T,
            X_val_s.T, y_val_s.T,
            y_scaler=scaler_y,
            epochs=2000,
            batch_size=64,
            learning_rate=0.012,
            l2_lambda=0.0006,
            optimizer='adam',
            patience=100,
            verbose=False
        )
        
        pred_fixed = fixed_nn.predict(X_val_s.T).T
        pred_fixed_unscaled = scaler_y.inverse_transform(pred_fixed)
        fixed_mee = mee(y_val, pred_fixed_unscaled)
        fixed_time = time.time() - start_time
        
        fixed_mees.append(fixed_mee)
        
        print(f"Fixed Result: {fixed_mee:.4f} MEE")
        print(f"Time: {fixed_time:.1f}s")
    
    # Final comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Mean MEE':<15} {'Std MEE':<15}")
    print("-" * 60)
    print(f"{'Cascade Correlation':<30} {np.mean(cascade_mees):.4f}       {np.std(cascade_mees):.4f}")
    print(f"{'Fixed [128,84,65]':<30} {np.mean(fixed_mees):.4f}       {np.std(fixed_mees):.4f}")
    
    print(f"\nCascade Architecture Sizes: {cascade_sizes}")
    print(f"Average hidden units: {np.mean(cascade_sizes):.1f}")
    
    # Verdict
    cascade_mean = np.mean(cascade_mees)
    fixed_mean = np.mean(fixed_mees)
    
    if cascade_mean < fixed_mean:
        print(f"\n✅ Cascade Correlation is BETTER by {fixed_mean - cascade_mean:.4f} MEE")
    else:
        print(f"\n⚠️ Fixed architecture is BETTER by {cascade_mean - fixed_mean:.4f} MEE")
        print("   (Cascade Correlation is experimental; fixed approach is more reliable)")


if __name__ == "__main__":
    run_cascade_benchmark()
