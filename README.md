# ML-CUP25 Project - Team RockOn

**Authors:** Suranjan Kumar Ghosh, Abinash Boruah
**Project Type:** A (Neural Network from Scratch)

## 📂 Submission Structure

```text
ML_CUP25_Submission/
├── config/                  # Best model configuration
├── data/                    # Dataset files & Final Predictions
│   └── RockOn_ML-CUP25-TS.csv  # FINAL PREDICTIONS (Blind Test)
├── scripts/                 # Execution scripts
│   ├── ensemble_simple.py   # Training script (Ensemble)
│   ├── generate_submission.py # Result generation script
│   └── run_monk_benchmark.py # Simulator validation
├── src/                     # Source code (NN implementation)
├── experiments/             # Logs and result files
├── hyperparameters-setting.md # Detailed parameter documentation
├── requirements.txt         # Dependencies
└── REPORT_SUMMARY.md        # Detailed analysis report
```

## 🚀 How to Run

**Prerequisites:** Python 3.8+
Install dependencies:
```bash
pip install -r requirements.txt
```

**Important:** Run all scripts from this root directory.

### 1. Train the Final Model
Trains the ensemble of 10 networks using the optimal hyperparameters.
```bash
python -m scripts.ensemble_simple
```

### 2. Generate CUP Results (Submission)
Generates the `data/RockOn_ML-CUP25-TS.csv` file formatted for submission.
```bash
python -m scripts.generate_submission
```

### 3. Validation (MONK Benchmarks)
Verifies the correctness of the simulator on MONK datasets.
```bash
python -m scripts.run_monk_benchmark
```

## 🎲 Random Seed Handling

To ensure full reproducibility:
- **Global Seed:** 42 (used for data splitting).
- **Weights Initialization:** Each model in the ensemble uses a unique deterministic seed: `seed = 42 + fold_idx + model_idx`.
- **Validation:** We use a custom `k_fold_split` utility that guarantees consistent 5-fold splits across all experiments.

## 📄 Documentation

- **REPORT_SUMMARY.md**: Contains the full narrative, validation methodology (5-Fold CV), and detailed results.
- **hyperparameters-setting.md**: Details the specific hyperparameters and the Genetic Algorithm used to find them.
