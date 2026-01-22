import numpy as np

# Activation Functions
def sigmoid(z):
    """Sigmoid activation function."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    """Hyperbolic tangent activation function."""
    return np.tanh(z)

def tanh_derivative(z):
    """Derivative of the hyperbolic tangent function."""
    return 1 - np.power(np.tanh(z), 2)

def linear(z):
    """Linear activation function (identity)."""
    return z

def linear_derivative(z):
    """Derivative of the linear function."""
    return np.ones_like(z)

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU function."""
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    """Derivative of Leaky ReLU function."""
    return np.where(z > 0, 1, alpha)

def elu(z, alpha=1.0):
    """ELU activation function with clipping to prevent overflow."""
    z_clipped = np.clip(z, -500, 500)  # Prevent exp overflow
    return np.where(z > 0, z, alpha * (np.exp(z_clipped) - 1))

def elu_derivative(z, alpha=1.0):
    """Derivative of ELU function with clipping to prevent overflow."""
    z_clipped = np.clip(z, -500, 500)  # Prevent exp overflow
    return np.where(z > 0, 1, alpha * np.exp(z_clipped))

def swish(z):
    """Swish activation function: x * sigmoid(x)."""
    return z * sigmoid(z)

def swish_derivative(z):
    """Derivative of Swish function."""
    s = sigmoid(z)
    return s + z * s * (1 - s)

def mish(z):
    """Mish activation function: x * tanh(ln(1 + e^x))."""
    # Softplus: ln(1 + e^x)
    # Clipping to prevent overflow in exp
    z_clipped = np.clip(z, -500, 500)
    softplus = np.log1p(np.exp(z_clipped))
    return z * np.tanh(softplus)

def mish_derivative(z):
    """Derivative of Mish function."""
    z_clipped = np.clip(z, -500, 500)
    sp = np.log1p(np.exp(z_clipped))
    tsp = np.tanh(sp)
    
    # Derivative of softplus: sigmoid(z)
    dsp = sigmoid(z)
    
    # Derivative of tanh(u) is 1 - tanh^2(u)
    dtsp = (1 - tsp**2) * dsp
    
    return tsp + z * dtsp

# Weight Initialization
def he_init(fan_in, fan_out):
    """He initialization for ReLU activations."""
    return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in)

def xavier_init(fan_in, fan_out):
    """Xavier/Glorot initialization for tanh/sigmoid."""
    return np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))

# Loss Functions & Metrics
def mse(y_true, y_pred):
    """Mean Squared Error loss function."""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_true, y_pred):
    """Derivative of the Mean Squared Error loss function."""
    return 2 * (y_pred - y_true)

def bce(y_true, y_pred):
    """Binary Cross-Entropy loss function."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_true, y_pred):
    """Derivative of the Binary Cross-Entropy loss function."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

def mee(y_true, y_pred):
    """
    Mean Euclidean Error metric.
    Note: y_true and y_pred should have shape (num_samples, num_features)
    """
    return np.mean(np.sqrt(np.sum(np.power(y_true - y_pred, 2), axis=1)))


def mee_derivative(y_true, y_pred, epsilon=1e-8):
    """
    Derivative of Mean Euclidean Error.
    
    ⚠️ WARNING: This function exists for experimental purposes only.
    
    According to NN-part2 lectures (Prof. Micheli):
    - Training should use MSE (smooth, stable gradients)
    - MEE should be used for EVALUATION only
    
    Reasons NOT to train on MEE:
    1. Square root causes unstable gradients near zero
    2. Not as smooth as MSE
    3. Noisy gradients, especially early in training
    4. Minimizing MSE almost always minimizes MEE (monotonic relationship)
    
    The correct workflow is:
    - Train with MSE (loss_fn='mse')
    - Evaluate/validate with MEE
    - Report MEE for final results
    
    Args:
        y_true: True values, shape (D, N) where D=dimensions, N=samples
        y_pred: Predicted values, shape (D, N)
        epsilon: Small constant for numerical stability
        
    Returns:
        Gradient with same shape as y_pred
    """
    diff = y_pred - y_true  # (D, N)
    euclidean_norms = np.sqrt(np.sum(diff**2, axis=0, keepdims=True) + epsilon)
    normalized_grad = diff / euclidean_norms
    # Scale to approximate MSE gradient magnitude
    scale_factor = 2.0 * np.sqrt(y_true.shape[0])
    return normalized_grad * scale_factor


# Learning Rate Schedulers
def cosine_annealing_lr(epoch, total_epochs, lr_init, lr_min=1e-6):
    """
    Cosine annealing learning rate schedule.
    
    Gradually decreases LR following a cosine curve from lr_init to lr_min.
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        lr_init: Initial learning rate
        lr_min: Minimum learning rate
        
    Returns:
        Learning rate for current epoch
    """
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))


def cosine_warm_restart_lr(epoch, T_0=100, T_mult=2, lr_init=0.01, lr_min=1e-6):
    """
    Cosine annealing with warm restarts (SGDR).
    
    LR follows cosine decay, then "restarts" to lr_init periodically.
    Each restart period is T_mult times longer than the previous.
    
    Reference: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"
    
    Args:
        epoch: Current epoch
        T_0: Initial restart period
        T_mult: Period multiplier after each restart
        lr_init: Initial/restart learning rate
        lr_min: Minimum learning rate
        
    Returns:
        Learning rate for current epoch
    """
    T_cur = epoch
    T_i = T_0
    
    # Find which restart period we're in
    while T_cur >= T_i:
        T_cur -= T_i
        T_i *= T_mult
    
    # Cosine annealing within current period
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + np.cos(np.pi * T_cur / T_i))
