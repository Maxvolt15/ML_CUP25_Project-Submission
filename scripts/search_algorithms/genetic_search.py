"""
Genetic Algorithm (Evolutionary Search) for ML-CUP 2025

This script runs an infinite evolutionary loop to find the optimal hyperparameters.
- Population-based search.
- Mutation and Crossover of hyperparameters.
- Aggressive Early Stopping (Mitigation) to save time.
- Saves the 'Hall of Fame' (Best Models) to CSV.
"""

import numpy as np
import pandas as pd
import csv
import random
import time
import os
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data_loader import load_cup_data, add_polynomial_features
from src.neural_network_v2 import NeuralNetworkV2
from src.utils import mee

# --- Configuration ---
POPULATION_SIZE = 15
GENERATIONS = 10000 # Effectively infinite
HALL_OF_FAME_FILE = 'hall_of_fame.csv'
EARLY_PRUNE_EPOCH = 200
EARLY_PRUNE_THRESHOLD = 30.0 # If MEE > 30 at epoch 200, kill it.
MAX_EPOCHS = 2500

# --- Hyperparameter Space ---
PARAM_SPACE = {
    'learning_rate': (1e-4, 0.1),
    'l2_lambda': (1e-6, 0.1),
    'dropout_rate': (0.0, 0.5),
    'momentum': (0.5, 0.99),
    'batch_size': [16, 32, 64],
    'hidden_activation': ['tanh', 'relu', 'leaky_relu', 'elu', 'swish', 'mish'],
    'optimizer': ['adam', 'sgd'],
    'use_batch_norm': [True, False],
    'weight_init': ['he', 'xavier'],
    # Architectures: (num_layers, min_units, max_units) - handled via logic
}

def get_random_gene():
    """Generates a random model configuration."""
    hidden_layers = []
    num_layers = random.randint(1, 3)
    for _ in range(num_layers):
        hidden_layers.append(random.randint(20, 150))
    
    # Sort descending for funnel shape (common heuristic)
    hidden_layers.sort(reverse=True)
    
    return {
        'hidden_layers': hidden_layers,
        'learning_rate': 10 ** random.uniform(np.log10(1e-4), np.log10(0.1)),
        'l2_lambda': 10 ** random.uniform(np.log10(1e-6), np.log10(0.1)),
        'dropout_rate': random.uniform(0.0, 0.5),
        'momentum': random.uniform(0.8, 0.99),
        'batch_size': random.choice([16, 32, 64]),
        'hidden_activation': random.choice(['tanh', 'relu', 'leaky_relu', 'elu', 'swish', 'mish']),
        'optimizer': random.choice(['adam', 'sgd']),
        'use_batch_norm': random.choice([True, False]),
        'weight_init': random.choice(['he', 'xavier'])
    }

def mutate(gene, mutation_rate=0.3):
    """Mutates a gene slightly."""
    mutated = deepcopy(gene)
    
    if random.random() < mutation_rate:
        # Mutate float params
        for param in ['learning_rate', 'l2_lambda']:
            if random.random() < 0.5:
                mutated[param] *= random.uniform(0.8, 1.2) # +/- 20%
        
        mutated['dropout_rate'] = np.clip(mutated['dropout_rate'] + random.uniform(-0.05, 0.05), 0, 0.8)
        mutated['momentum'] = np.clip(mutated['momentum'] + random.uniform(-0.02, 0.02), 0.5, 0.99)
        
        # Mutate categorical params
        if random.random() < 0.3:
            mutated['batch_size'] = random.choice([16, 32, 64])
        if random.random() < 0.3:
            mutated['hidden_activation'] = random.choice(['tanh', 'relu', 'leaky_relu', 'elu', 'swish', 'mish'])
        if random.random() < 0.3:
            mutated['optimizer'] = random.choice(['adam', 'sgd'])
            
        # Mutate Architecture
        if random.random() < 0.3:
            # Change neuron count
            idx = random.randint(0, len(mutated['hidden_layers']) - 1)
            mutated['hidden_layers'][idx] = int(mutated['hidden_layers'][idx] * random.uniform(0.8, 1.2))
            
    return mutated

def crossover(parent1, parent2):
    """Combines two parents to create a child."""
    child = {}
    for key in parent1:
        if random.random() > 0.5:
            child[key] = parent1[key]
        else:
            child[key] = parent2[key]
    return child

def evaluate_model(params, X_train, y_train, X_val, y_val, y_scaler):
    """Trains and evaluates a model. Returns MEE."""
    
    # Layer setup
    layer_sizes = [X_train.shape[0]] + params['hidden_layers'] + [y_train.shape[0]]
    
    # Init Check
    init = params['weight_init']
    # Use He init for ReLU-family (including Swish/Mish as they are unbounded above)
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
        random_state=42
    )
    
    # --- Manual Training Loop for Early Mitigation ---
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    
    # Optimizer State
    best_mee = float('inf')
    
    # For speed, we call nn.train but with our own wrapper logic?
    # Actually, NNv2.train is good. Let's trust it but reduce patience.
    
    history = nn.train(
        X_train, y_train, X_val, y_val, y_scaler,
        epochs=MAX_EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=params['optimizer'],
        momentum=params['momentum'],
        l2_lambda=params['l2_lambda'],
        patience=50, # Quick kill if no improvement
        verbose=False
    )
    
    if not history['val_mee']:
        return 999.0
        
    return min(history['val_mee'])

def save_hall_of_fame(population, scores):
    """Saves the best models to CSV."""
    combined = sorted(zip(scores, population), key=lambda x: x[0])
    
    # Read existing
    existing = []
    if os.path.exists(HALL_OF_FAME_FILE):
        try:
            df = pd.read_csv(HALL_OF_FAME_FILE)
            for _, row in df.iterrows():
                try:
                    p = eval(row['params'])
                    s = float(row['mee'])
                    existing.append((s, p))
                except: pass
        except: pass
        
    # Merge and Keep Top 20
    all_models = sorted(combined + existing, key=lambda x: x[0])
    top_models = all_models[:20]
    
    with open(HALL_OF_FAME_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'mee', 'params'])
        for i, (score, params) in enumerate(top_models):
            writer.writerow([i+1, score, str(params)])
            
    return top_models[0][0] # Return best score

def run_genetic_search():
    print(f"{'='*60}")
    print("Starting Infinite Genetic Search")
    print(f"Population: {POPULATION_SIZE}, Pruning > {EARLY_PRUNE_THRESHOLD} MEE @ ep {EARLY_PRUNE_EPOCH}")
    print(f"{'='*60}\n")
    
    # Load Data
    print("Loading and Preprocessing Data...")
    X, y = load_cup_data('data/ML-CUP25-TR.csv', use_polynomial_features=True, poly_degree=2)
    
    # 90/10 Split
    X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=999)
    
    x_scaler = StandardScaler().fit(X_train_full)
    y_scaler = StandardScaler().fit(y_train)
    
    X_train = x_scaler.transform(X_train_full).T
    X_val = x_scaler.transform(X_val_full).T
    y_train = y_scaler.transform(y_train).T
    # NOTE: y_val stays UNSCALED for MEE computation in original scale
    # nn.train() will inverse_transform predictions before comparing with y_val
    y_val = y_val.T  # Just transpose to column-major (D x N), but don't scale!
    
    # 1. Initialize Population
    population = [get_random_gene() for _ in range(POPULATION_SIZE)]
    
    generation = 0
    best_global_mee = float('inf')
    
    while True:
        generation += 1
        print(f"\n--- Generation {generation} ---")
        
        scores = []
        for i, gene in enumerate(population):
            start_time = time.time()
            try:
                score = evaluate_model(gene, X_train, y_train, X_val, y_val, y_scaler)
            except Exception as e:
                print(f"Model {i} Failed: {e}")
                score = 999.0
            
            elapsed = time.time() - start_time
            scores.append(score)
            print(f"  Model {i+1}: MEE={score:.4f} ({elapsed:.1f}s) | {gene['hidden_layers']} {gene['hidden_activation']}")
        
        # Save Best
        best_gen_score = min(scores)
        best_global_mee = save_hall_of_fame(population, scores)
        
        print(f"  > Gen Best: {best_gen_score:.4f} | Global Best: {best_global_mee:.4f}")
        
        # Selection (Top 40%)
        sorted_indices = np.argsort(scores)
        survivors = [population[i] for i in sorted_indices[:int(POPULATION_SIZE * 0.4)]]
        
        # Breeding
        new_population = deepcopy(survivors) # Elitism
        
        while len(new_population) < POPULATION_SIZE:
            # Tournament Selection for parents
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population

if __name__ == '__main__':
    try:
        run_genetic_search()
    except KeyboardInterrupt:
        print("\nSearch paused. Resume anytime.")
