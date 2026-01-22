import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA

def add_polynomial_features(X, degree=2):
    """
    Adds polynomial and interaction features to a dataset.
    Includes the original features by default.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

def add_pca_features(X, n_components=5):
    """
    Adds principal components as features to a dataset.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return np.hstack([X, X_pca])

def load_cup_data(file_path, use_polynomial_features=False, poly_degree=2, use_pca_features=False, pca_components=5):
    """
    Loads the ML-CUP dataset from a .csv file.
    Optionally adds polynomial and/or PCA features.
    """
    data = pd.read_csv(file_path, comment='#', header=None)
    data = data.drop(columns=[0])
    X = data.iloc[:, 0:12].values
    y = data.iloc[:, 12:].values

    if use_polynomial_features:
        X = add_polynomial_features(X, degree=poly_degree)
    
    if use_pca_features:
        X = add_pca_features(X, n_components=pca_components)
        
    return X, y

def load_cup_test_data(file_path):
    """
    Loads the ML-CUP blind test set from a .csv file.
    """
    data = pd.read_csv(file_path, comment='#', header=None)
    X_test = data.iloc[:, 1:].values
    return X_test

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and validation sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_scalers_and_normalize(X_train, y_train):
    """
    Creates scalers and normalizes the data.
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    return X_train_scaled, y_train_scaled, x_scaler, y_scaler

def load_monk_data(problem_name):
    """
    Loads a MONK dataset, performs one-hot encoding, and returns train/test splits.

    Args:
        problem_name (str): The name of the MONK problem (e.g., 'monks-1').

    Returns:
        tuple: A tuple containing (X_train, y_train, X_test, y_test).
    """
    train_path = f'data/{problem_name}.train'
    test_path = f'data/{problem_name}.test'
    
    # Define column names for clarity
    columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'id']

    # Read train and test data
    train_df = pd.read_csv(train_path, delim_whitespace=True, header=None, names=columns)
    test_df = pd.read_csv(test_path, delim_whitespace=True, header=None, names=columns)

    # Combine for consistent one-hot encoding
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # Separate features and target
    features = full_df.drop(columns=['class', 'id'])
    targets = full_df['class']

    # One-hot encode the features
    features_one_hot = pd.get_dummies(features.astype(str))
    
    # Split back into train and test sets
    train_size = len(train_df)
    X_train = features_one_hot[:train_size].values
    y_train = targets[:train_size].values.reshape(-1, 1)
    
    X_test = features_one_hot[train_size:].values
    y_test = targets[train_size:].values.reshape(-1, 1)

    return X_train, y_train, X_test, y_test

