# Hyperparameter Settings

## 1. Final Model Configuration (Ensemble)

The final model is an ensemble of 10 Neural Networks trained with the following identical hyperparameters but different random seeds.

**Architecture:**
- **Input:** 90 features (10 raw + Degree 2 Polynomial features)
- **Hidden Layers:** [128, 84, 65]
- **Output:** 4 units (Linear activation)
- **Hidden Activation:** Tanh

**Training Hyperparameters:**
- **Optimizer:** Adam
- **Learning Rate:** 0.01198
- **Momentum:** 0.984 (beta1 for Adam)
- **Batch Size:** 64
- **Max Epochs:** 2500 (Early Stopping with Patience=100)

**Regularization:**
- **L2 Lambda:** 0.0006
- **Dropout Rate:** 0.131
- **Weight Initialization:** He Normal

## 2. Selection Methodology

**Strategy:** Two-Stage Hybrid Search

**Stage 1: Coarse Random Search** (`scripts/search_algorithms/simple_search.py`)
- Initial exploration of the hyperparameter space using constrained random search.
- Focus: Small/Medium architectures, SGD optimizer, and broad regularization ranges.
- Goal: Establish baseline performance and valid hyperparameter bounds.

**Stage 2: Evolutionary Genetic Algorithm** (`scripts/search_algorithms/genetic_search.py`)
- Efficient global optimization to refine the baseline.
- **Population:** 15 models per generation.
- **Selection:** Tournament selection of top 40% performers.
- **Outcome:** The algorithm converged on the [128, 84, 65] Tanh architecture as the most robust configuration, outperforming simpler baselines.