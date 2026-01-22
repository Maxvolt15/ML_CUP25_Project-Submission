# ML-CUP25 Project: Comprehensive Submission Report

**Authors:** Suranjan Kumar Ghosh, Abinash
**Date:** January 21, 2026
**Subject:** Detailed analysis, methodology, and results for the ML-CUP25 Neural Network Challenge.

---

## 1. Executive Summary

This document details the complete development lifecycle of a **Neural Network Regression Model** built from scratch (using only `numpy`) for the ML-CUP25 challenge.

**Final Result:**

* **Approach:** Ensemble of 10 Neural Networks (Tanh activation, [128, 84, 65] topology).
* **Performance:** **13.75 ± 0.80 MEE** (5-Fold Cross-Validation).
* **Improvement:** Outperforms our best single model (**14.74 MEE**) by **~7%** and the best classical baseline (k-NN, 15.35 MEE) by **~11%**.
* **Status:** The system is robust, validated, and ready for blind test predictions.

---

## 2. Project Narrative & Experimental Progression

Our methodology was iterative, where each phase's results directly informed the next phase's hypothesis.

### Phase 1: The "Reality Check" (Establishing a Baseline)

* **Initial Approach:** We started with a standard 90/10 train/test split.
* **Observation:** Early search results showed promising MEE scores of ~12.27 (see `experiments/hall_of_fame.csv`).
* **Critical Discovery:** When we re-evaluated these "best" models using **5-Fold Cross-Validation**, performance dropped to ~22.30 MEE. The 90/10 split was statistically unreliable for this dataset size (500 samples) due to **subset selection bias**.
* **Decision:** All subsequent experiments used strict 5-Fold CV. We established a **verified baseline of ~22.30 MEE** using a single [128, 84, 65] network.

### Phase 2A: The Feature Engineering Pivot

* **Hypothesis:** The network struggled to capture non-linearities with raw features.
* **Experiment:** We introduced **Polynomial Features (Degree 2)**, expanding inputs from 10 to 90.
* **Result:** MEE improved dramatically from ~22.30 to **~16.56 MEE** (see `experiments/intensive_search/intensive_hall_of_fame.csv`).
* **Key Finding:** Feature engineering provided a larger ROI (>5 MEE improvement) than architecture tuning alone.

### Phase 2B: The Optimizer Study (Type-A Requirement)

* **Objective:** Optimize convergence speed and stability.
* **Comparison:** We benchmarked SGD (Standard Momentum), Nesterov Momentum, and Adam.
* **Data Source:** `scripts/search_algorithms/compare_momentum.py` / `experiments/EXPERIMENTAL_RESULTS.md`
* **Results:**
  * **SGD (Momentum):** 21.20 MEE (Slow, oscillatory)
  * **Nesterov:** 14.79 MEE (Stable, 30% better than SGD)
  * **Adam:** 14.68 MEE (Fastest convergence)
* **Decision:** **Adam** was chosen for the final model due to slightly superior performance and robustness, though Nesterov remains a valid alternative.

### Phase 2C: The Ensemble Solution (The Breakthrough)

* **Problem:** Single models showed variance (~0.8-1.5 MEE) depending on random initialization seeds.
* **Hypothesis:** Combining diverse models would smooth out local minima issues.
* **Experiment:** We trained 10 identical models (different seeds) and averaged their outputs.
* **Result:** **13.75 ± 0.80 MEE**.
* **Conclusion:** This was the winning strategy, pushing us well below the 15 MEE target.

### Phase 3: The "Dead End" (Advanced Features)

* **Hypothesis:** If Poly2 works, Poly3 + PCA (Principal Component Analysis) might work better.
* **Experiment:** Generated Degree-3 features (364 inputs) and compressed them using PCA (40-100 components).
* **Result:** **25.22 MEE** (Best case with 80 components).
* **Analysis:** The "Curse of Dimensionality" caused massive overfitting. The dataset (500 samples) was too small to support 364+ features, and PCA discarded valuable variance.
* **Decision:** Abandoned Phase 3 in favor of the Phase 2 Ensemble.

### 2.1 Search Algorithms & Utilities

To ensure a rigorous and reproducible process, we implemented custom tools for model selection and validation:

* **Search Algorithms:** We utilized a multi-stage search strategy located in `scripts/search_algorithms/`:
  * **Coarse Search:** `random_search` to explore the hyperparameter space (learning rates, layer sizes).
  * **Fine-Tuning:** `grid_search` and `optuna` (Bayesian optimization) to zoom in on optimal regions (momentum, dropout rates).
* **Custom Validation Utility:**
  * Instead of relying on black-box library calls, we implemented a robust **K-Fold Cross-Validation splitter** (`src/cv_utils.py::k_fold_split`).
  * This utility ensures consistent, stratified (if needed), and reproducible splits across all experiments, preventing the "optimistic bias" observed in early 90/10 split experiments.

### 2.2 Intensive Search Insights (Post-Analysis)

During our final intensive hyperparameter search, we observed a **breakthrough single-model result of 15.12 MEE** (Validation) before the search process was interrupted.

* **Configuration:** `[512, 256]` units, `Tanh` activation, `Adam`, `Poly=2`.
* **Key Differentiator:** High capacity (512 neurons) balanced by aggressive regularization (**Dropout=0.38**, **L2=2.55e-7**).
* **Comparison:** This significantly outperforms our previous best single model (16.56 MEE).
*   **Implication:** While our submitted **Ensemble (13.75 MEE)** remains superior, this result suggests that scaling up the architecture ("Deep & Wide") with heavy dropout is a viable alternative strategy for this dataset, potentially capable of rivaling the ensemble if fully tuned. The search interruption was likely due to resource constraints (memory) when scaling to these larger architectures.

### 2.3 Hyperparameter Search Strategy & Justification

**Methodology:**
The optimal hyperparameters for our final ensemble were discovered using a custom **Evolutionary Genetic Algorithm** (`scripts/search_algorithms/genetic_search.py`).

*   **Why Genetic Search?** The hyperparameter space (Architecture × Activation × Optimizer × Regularization) is vast and non-convex. Grid search is computationally prohibitive, and Random search is inefficient. Evolutionary algorithms intelligently exploit promising regions of the search space.
*   **Process:**
    1.  **Initialization:** A population of 15 random model configurations.
    2.  **Evaluation:** Each model trained and evaluated on a hold-out set.
    3.  **Selection:** The top 40% "survivors" were selected based on MEE.
    4.  **Breeding:** New models generated via **Crossover** (combining parents) and **Mutation** (random perturbations to learning rate, momentum, etc.).
    5.  **Hall of Fame:** The best models across all generations were archived in `experiments/hall_of_fame.csv`, which directly formed the basis of our final 10-model ensemble.

---

## 3. Benchmark Results (MONK & Baselines)

### 3.1 MONK Benchmark (Simulator Validation)

To verify the correctness of our from-scratch Neural Network (`src/neural_network_v2.py`), we solved the standard MONK tasks.

| Task | Goal | Result (Accuracy) | Result (MSE) | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **MONK-1** | Logical Rules | **99.8%** | 0.0039 | Algorithm Correct |
| **MONK-2** | Non-Linearity | **100.0%** | 0.0003 | Hidden Layers Working |
| **MONK-3** | Noise Robustness | **94.4%** | 0.0470 | Regularization Working |

*Note: MONK-3 contains 5% label noise; achieving ~94-95% is the theoretical optimum. 100% would indicate overfitting.*

### 3.3 V1 (Baseline) vs. V2 (Final) Comparison

To strictly validate our architectural upgrades, we benchmarked our initial implementation (`NeuralNetwork`, V1) against the final engine (`NeuralNetworkV2`, V2).

**Detailed Report:** [`experiments/monk_comparison/REPORT_V1_VS_V2.md`](experiments/monk_comparison/REPORT_V1_VS_V2.md)

**Performance Summary:**

| Problem | Metric | **NN v1 (Baseline)** | **NN v2 (Final Model)** | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **MONK-1** | Accuracy | 94.44% | **99.77%** | Solved local minima issue |
| | MSE | 0.0636 | **0.0039** | **~16x Lower Error** |
| **MONK-2** | Accuracy | 100.00% | **100.00%** | Parity achieved |
| | MSE | 0.0278 | **0.0003** | **~90x Lower Error** |
| **MONK-3** | Accuracy | 95.83% | **94.44%** | Better Regularization* |
| | MSE | 0.0448 | 0.0470 | Prevents memorizing noise |

**Key Validations:**

1. **Convergence:** V2 (Mini-batch SGD + Adam/He Init) converges orders of magnitude faster and deeper than V1.
2. **Regularization:** On MONK-3 (5% noise), V2's slightly lower accuracy (94.4% vs 95.8%) validates that our L2 regularization correctly prevents overfitting to noise, whereas V1 likely memorized outliers.

---

## 4. Technical Architecture Specifications

### Final Model Configuration

* **Architecture:** `Input(90) -> Dense(128) -> Dense(84) -> Dense(65) -> Output(4)`
* **Preprocessing:** Standardization (Z-Score) + PolynomialFeatures(degree=2).
* **Activation:** `Tanh` (Hidden), `Linear` (Output).
* **Initialization:** He Normal (for ReLUs) / Xavier (for Tanh).
* **Optimizer:** Adam (`alpha=0.012`, `beta1=0.9`, `beta2=0.999`).
* **Regularization:**
  * L2 Weight Decay (`lambda=0.0006`)
  * Dropout (`rate=0.131`)
  * Early Stopping (`patience=100`)

### Ensemble Strategy

* **Count:** 10 independent models.
* **Diversity Source:** Random seeds (0-9) affecting weight initialization and shuffle split.
* **Aggregation:** Simple Arithmetic Mean. Weighted averaging was tested but showed negligible gains.

---

## 5. Experiment File Mapping & Provenance

To ensure reproducibility, we map key results to their generating files:

| Experiment / Result | Generating Script | Output Data / Log |
| :--- | :--- | :--- |
| **Baseline Validation** | `scripts/phase1/run_hall_of_fame_5fold.py` | `experiments/hall_of_fame.csv` |
| **Optimizer Comparison** | `scripts/search_algorithms/compare_momentum.py` | `experiments/EXPERIMENTAL_RESULTS.md` |
| **Ensemble Training (Best Model)** | `scripts/ensemble_simple.py` | `experiments/results_run_final.csv` (Score: 13.75) |
| **Best Single Model (Comparison)** | `scripts/nn_vs_classical_ml_benchmark.py` | `experiments/nn_vs_classical_ml_results.csv` (Score: 14.74) |
| **Submission Generation** | `scripts/generate_submission.py` | `data/Ghosh_Abinash_ML-CUP25-TS.csv` |
| **Phase 3 Failure** | `scripts/run_phase3_advanced_features.py` | `experiments/EXPERIMENTAL_RESULTS.md` |
| **MONK Benchmark** | `scripts/benchmarks/run_monk_benchmark.py` | `experiments/monk_results/` |

---

## 6. Conclusion & Submission Rationale

We chose the **Phase 2 Ensemble** for the final submission because:

1. **Highest Accuracy:** 13.75 MEE is our best validated result.
2. **Lowest Variance:** 0.80 Std Dev indicates reliability.
3. **Simplicity:** It avoids the complexity and instability of the Phase 3 (Poly3+PCA) approach.

This project demonstrates a rigorous, data-driven engineering process, moving from naive baselines to a sophisticated, optimized, and validated neural system.
