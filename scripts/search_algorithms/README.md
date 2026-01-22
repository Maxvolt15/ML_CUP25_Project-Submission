# scripts/search_algorithms/ - My Hyperparameter Hunt

I used these scripts to explore the vast space of hyperparameters and find the configuration that eventually became my "Hall of Fame" model.

## Scripts Overview

| Script | Purpose | Type A Requirement |
|--------|---------|-------------------|
| `genetic_search.py` | Evolutionary hyperparameter search | Architecture tuning |
| `compare_momentum.py` | **NEW:** SGD vs Nesterov vs Adam | Momentum comparison ✓ |
| `cascade_correlation_nn.py` | **NEW:** Auto-growing architecture | Advanced (optional) |

## Why I wrote these

Finding the right number of neurons, learning rate, and regularization is impossible by hand. I wrote these algorithms to automate that discovery process.

## The Scripts

### 1. `compare_momentum.py` (NEW - Type A Requirement)

**I wrote this because:** The professor explicitly recommends comparing "Momentum vs Nesterov Momentum".

**What it does:** Compares three optimizers on the same architecture:

- SGD + Standard Momentum
- SGD + Nesterov Momentum (look-ahead gradient)
- Adam (adaptive learning rate)

Usage:

```bash
python -m scripts.search_algorithms.compare_momentum
```

### 2. `cascade_correlation_nn.py` (NEW - Experimental)

**I wrote this because:** Fixed architectures feel rigid. Cascade Correlation automatically determines network size.

**What it does:** Implements the Cascade Correlation algorithm (Fahlman & Lebiere, 1990):

- Starts with no hidden units
- Adds units one at a time by maximizing correlation with residual error
- Automatically finds optimal network size

**Warning:** This is experimental and high-risk. Use the fixed architecture for your main results.

Usage:

```bash
python -m scripts.search_algorithms.cascade_correlation_nn
```

### 3. `genetic_search.py` (The Heavy Lifter)

**I wrote this because:** I wanted an evolutionary approach that could "breed" better models over time.

**How it works:** It creates a population of random models, trains them, and then combines the genes (hyperparameters) of the best ones.

**Status:** This was my primary tool for Phase 2 exploration.

Usage:

```bash
python -m scripts.search_algorithms.genetic_search
```

### 4. `hyperparameter_search.py` & `_v2.py`

**I wrote these because:** Sometimes a simple random search is enough to find a starting point.

**How it works:** It randomly samples parameters from distributions I defined and evaluates them.

### 5. `intensive_hyperparameter_search.py`

**I wrote this because:** I wanted to be absolutely sure I wasn't missing a global optimum, so I set up a very thorough (and slow) search grid.

## My Takeaway

These scripts were crucial for finding the `[128, 84, 65]` architecture. Without them, I would have been guessing in the dark.

The momentum comparison script is especially important for the report - it shows I understand the difference between standard and Nesterov momentum.
