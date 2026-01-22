import numpy as np

def k_fold_split(X, n_splits=5, shuffle=True, random_state=None):
    """
    Generates K-Fold train/test indices from scratch using NumPy.
    
    Args:
        X (np.ndarray or list): The dataset to split.
        n_splits (int): Number of folds.
        shuffle (bool): Whether to shuffle the data before splitting.
        random_state (int): Seed for reproducibility.
        
    Yields:
        train_indices (np.ndarray): Indices for the training set.
        val_indices (np.ndarray): Indices for the validation set.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    # Calculate fold sizes
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        
        # Train indices are all indices except the validation ones
        # We can use a boolean mask for efficient selection
        mask = np.ones(n_samples, dtype=bool)
        # We need to map the shuffled indices back to their original positions for the mask?
        # No, 'indices' contains the actual row numbers. 
        # But to create the complement efficiently:
        
        # Alternative: concatenate the parts before and after the val chunk
        # This works on the *shuffled* indices array
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        
        yield train_indices, val_indices
        current = stop
