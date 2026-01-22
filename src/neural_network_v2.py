import numpy as np
import copy
from src.utils import (sigmoid, linear, tanh, relu, leaky_relu, elu, swish, mish,
                       sigmoid_derivative, linear_derivative, tanh_derivative, 
                       relu_derivative, leaky_relu_derivative, elu_derivative,
                       swish_derivative, mish_derivative,
                       he_init, xavier_init, mse, mse_derivative, mee, mee_derivative,
                       cosine_annealing_lr, cosine_warm_restart_lr)


class NeuralNetworkV2:
    """
    Advanced Neural Network implementation from scratch with:
    - Mini-batch SGD
    - Multiple optimizers (SGD+Momentum, Nesterov, Adam)
    - Learning rate decay
    - Proper weight initialization (He/Xavier)
    - Gradient clipping
    - Batch Normalization (optional)
    - Dropout (optional)
    """
    
    def __init__(self, layer_sizes, hidden_activation='relu', output_activation='linear',
                 weight_init='he', use_batch_norm=False, dropout_rate=0.0, random_state=42):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            hidden_activation: 'relu', 'leaky_relu', 'elu', 'tanh', 'sigmoid', 'swish', 'mish'
            output_activation: 'linear', 'sigmoid', 'tanh'
            weight_init: 'he' or 'xavier'
            use_batch_norm: If True, add batch normalization to hidden layers.
            dropout_rate: Probability of dropping a neuron. 0 means no dropout.
            random_state: Random seed for reproducibility
        """
        np.random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        
        # Activation function mapping
        self.activation_map = {
            'relu': (relu, relu_derivative),
            'leaky_relu': (leaky_relu, leaky_relu_derivative),
            'elu': (elu, elu_derivative),
            'tanh': (tanh, tanh_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative),
            'swish': (swish, swish_derivative),
            'mish': (mish, mish_derivative),
            'linear': (linear, linear_derivative)
        }
        
        self.hidden_act_name = hidden_activation
        self.output_act_name = output_activation
        self.hidden_activation, self.hidden_activation_derivative = self.activation_map[hidden_activation]
        self.output_activation, self.output_activation_derivative = self.activation_map[output_activation]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        if self.use_batch_norm:
            self.gamma = []
            self.beta = []
            self.running_mean = []
            self.running_var = []

        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            
            if weight_init == 'he':
                w = he_init(fan_in, fan_out)
            else:
                w = xavier_init(fan_in, fan_out)
            
            self.weights.append(w)
            self.biases.append(np.zeros((fan_out, 1)))

            # Initialize BN params for hidden layers
            if self.use_batch_norm and i < len(layer_sizes) - 2:
                self.gamma.append(np.ones((fan_out, 1)))
                self.beta.append(np.zeros((fan_out, 1)))
                self.running_mean.append(np.zeros((fan_out, 1)))
                self.running_var.append(np.ones((fan_out, 1)))

        # Adam optimizer parameters
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        if self.use_batch_norm:
            self.m_gamma = [np.zeros_like(g) for g in self.gamma]
            self.v_gamma = [np.zeros_like(g) for g in self.gamma]
            self.m_beta = [np.zeros_like(b) for b in self.beta]
            self.v_beta = [np.zeros_like(b) for b in self.beta]
        self.t = 0

        # SGD momentum
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]
        if self.use_batch_norm:
            self.velocity_gamma = [np.zeros_like(g) for g in self.gamma]
            self.velocity_beta = [np.zeros_like(b) for b in self.beta]

    def forward(self, x, training=True):
        """Forward propagation."""
        self.activation_outputs = [x]
        self.z_outputs = []
        if self.use_batch_norm:
            self.bn_cache = []
        if self.dropout_rate > 0:
            self.dropout_masks = []

        activation = x
        num_hidden_layers = len(self.weights) - 1
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            
            if self.use_batch_norm and i < num_hidden_layers:
                if training:
                    batch_mean = np.mean(z, axis=1, keepdims=True)
                    batch_var = np.var(z, axis=1, keepdims=True)
                    z_norm = (z - batch_mean) / np.sqrt(batch_var + 1e-8)
                    self.running_mean[i] = 0.9 * self.running_mean[i] + 0.1 * batch_mean
                    self.running_var[i] = 0.9 * self.running_var[i] + 0.1 * batch_var
                    self.bn_cache.append((z, z_norm, batch_mean, batch_var))
                else:
                    z_norm = (z - self.running_mean[i]) / np.sqrt(self.running_var[i] + 1e-8)
                z = self.gamma[i] * z_norm + self.beta[i]
            
            self.z_outputs.append(z)
            
            if i < num_hidden_layers:
                activation = self.hidden_activation(z)
                if self.dropout_rate > 0 and training:
                    mask = (np.random.rand(*activation.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                    self.dropout_masks.append(mask)
                    activation *= mask
            else:
                activation = self.output_activation(z)
            
            self.activation_outputs.append(activation)
        
        return activation

    def backward(self, y_true, y_pred, loss_fn='mse'):
        """
        Backward propagation with gradient computation.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            loss_fn: Loss function to use ('mse' or 'mee')
        """
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]
        if self.use_batch_norm:
            nabla_gamma = [np.zeros_like(g) for g in self.gamma]
            nabla_beta = [np.zeros_like(b) for b in self.beta]

        # Choose loss derivative based on loss function
        if loss_fn == 'mee':
            loss_grad = mee_derivative(y_true, y_pred)
        else:  # default to MSE
            loss_grad = mse_derivative(y_true, y_pred)
        
        delta = loss_grad * self.output_activation_derivative(self.z_outputs[-1])
        
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, self.activation_outputs[-2].T)
        
        num_hidden_layers = self.num_layers - 2
        for l in range(2, self.num_layers):
            d_act = np.dot(self.weights[-l + 1].T, delta)
            
            if self.dropout_rate > 0 and (self.num_layers - 1 - l) < num_hidden_layers:
                d_act *= self.dropout_masks[-(l-1)]

            if self.use_batch_norm and (self.num_layers - 1 - l) < num_hidden_layers:
                bn_idx = self.num_layers - 1 - l
                z_orig, z_norm, mean, var = self.bn_cache[bn_idx]
                m = z_orig.shape[1]
                
                d_z_norm = d_act * self.gamma[bn_idx]
                
                nabla_gamma[bn_idx] = np.sum(d_act * z_norm, axis=1, keepdims=True)
                nabla_beta[bn_idx] = np.sum(d_act, axis=1, keepdims=True)

                d_var = np.sum(d_z_norm * (z_orig - mean) * -0.5 * (var + 1e-8)**-1.5, axis=1, keepdims=True)
                d_mean = np.sum(d_z_norm * -1 / np.sqrt(var + 1e-8), axis=1, keepdims=True) + d_var * np.sum(-2 * (z_orig - mean), axis=1, keepdims=True) / m
                
                d_z = d_z_norm / np.sqrt(var + 1e-8) + d_var * 2 * (z_orig - mean) / m + d_mean / m
                delta = d_z * self.hidden_activation_derivative(self.z_outputs[-l])
            else:
                delta = d_act * self.hidden_activation_derivative(self.z_outputs[-l])

            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, self.activation_outputs[-l - 1].T)
        
        if self.use_batch_norm:
            return nabla_w, nabla_b, nabla_gamma, nabla_beta
        return nabla_w, nabla_b, None, None

    def clip_gradients(self, nabla_w, nabla_b, nabla_g=None, nabla_beta=None, max_norm=1.0):
        """Gradient clipping to prevent exploding gradients."""
        total_norm = 0
        for nw, nb in zip(nabla_w, nabla_b):
            total_norm += np.sum(nw ** 2) + np.sum(nb ** 2)
        if self.use_batch_norm and nabla_g is not None:
            for ng, nbeta in zip(nabla_g, nabla_beta):
                total_norm += np.sum(ng ** 2) + np.sum(nbeta ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            nabla_w = [nw * scale for nw in nabla_w]
            nabla_b = [nb * scale for nb in nabla_b]
            if self.use_batch_norm and nabla_g is not None:
                nabla_g = [ng * scale for ng in nabla_g]
                nabla_beta = [nbeta * scale for nbeta in nabla_beta]

        if self.use_batch_norm:
            return nabla_w, nabla_b, nabla_g, nabla_beta
        return nabla_w, nabla_b, None, None

    def update_adam(self, nabla_w, nabla_b, nabla_g, nabla_beta, learning_rate, l2_lambda, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Adam optimizer update."""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update weights and biases
            nabla_w[i] += l2_lambda * self.weights[i]
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * nabla_w[i]
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * (nabla_w[i] ** 2)
            m_w_corr = self.m_weights[i] / (1 - beta1 ** self.t)
            v_w_corr = self.v_weights[i] / (1 - beta2 ** self.t)
            self.weights[i] -= learning_rate * m_w_corr / (np.sqrt(v_w_corr) + epsilon)
            
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * nabla_b[i]
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * (nabla_b[i] ** 2)
            m_b_corr = self.m_biases[i] / (1 - beta1 ** self.t)
            v_b_corr = self.v_biases[i] / (1 - beta2 ** self.t)
            self.biases[i] -= learning_rate * m_b_corr / (np.sqrt(v_b_corr) + epsilon)

            # Update gamma and beta for hidden layers
            if self.use_batch_norm and i < len(self.gamma):
                self.m_gamma[i] = beta1 * self.m_gamma[i] + (1 - beta1) * nabla_g[i]
                self.v_gamma[i] = beta2 * self.v_gamma[i] + (1 - beta2) * (nabla_g[i] ** 2)
                m_g_corr = self.m_gamma[i] / (1 - beta1 ** self.t)
                v_g_corr = self.v_gamma[i] / (1 - beta2 ** self.t)
                self.gamma[i] -= learning_rate * m_g_corr / (np.sqrt(v_g_corr) + epsilon)

                self.m_beta[i] = beta1 * self.m_beta[i] + (1 - beta1) * nabla_beta[i]
                self.v_beta[i] = beta2 * self.v_beta[i] + (1 - beta2) * (nabla_beta[i] ** 2)
                m_beta_corr = self.m_beta[i] / (1 - beta1 ** self.t)
                v_beta_corr = self.v_beta[i] / (1 - beta2 ** self.t)
                self.beta[i] -= learning_rate * m_beta_corr / (np.sqrt(v_beta_corr) + epsilon)

    def update_sgd_momentum(self, nabla_w, nabla_b, nabla_g, nabla_beta, learning_rate, momentum, l2_lambda):
        """SGD with momentum update."""
        for i in range(len(self.weights)):
            reg_grad = l2_lambda * self.weights[i]
            
            self.velocity_weights[i] = momentum * self.velocity_weights[i] - learning_rate * (nabla_w[i] + reg_grad)
            self.weights[i] += self.velocity_weights[i]
            
            self.velocity_biases[i] = momentum * self.velocity_biases[i] - learning_rate * nabla_b[i]
            self.biases[i] += self.velocity_biases[i]

            if self.use_batch_norm and i < len(self.gamma):
                self.velocity_gamma[i] = momentum * self.velocity_gamma[i] - learning_rate * nabla_g[i]
                self.gamma[i] += self.velocity_gamma[i]
                
                self.velocity_beta[i] = momentum * self.velocity_beta[i] - learning_rate * nabla_beta[i]
                self.beta[i] += self.velocity_beta[i]

    def update_nesterov(self, nabla_w, nabla_b, nabla_g, nabla_beta, learning_rate, momentum, l2_lambda):
        """
        Nesterov Accelerated Gradient (NAG) update.
        
        Nesterov momentum "looks ahead" by computing gradients at the anticipated 
        future position, leading to faster convergence than standard momentum.
        
        Update rule:
            v_t = μ * v_{t-1} - η * ∇L(θ + μ * v_{t-1})
            θ = θ + v_t
            
        In practice (simplified form used here):
            v_t = μ * v_{t-1} - η * ∇L(θ)
            θ = θ + μ * v_t - η * ∇L(θ)
            
        This is equivalent to the "modified" Nesterov form which is easier to implement.
        Reference: Sutskever et al., "On the importance of initialization and momentum in deep learning"
        """
        for i in range(len(self.weights)):
            reg_grad = l2_lambda * self.weights[i]
            grad_w = nabla_w[i] + reg_grad
            grad_b = nabla_b[i]
            
            # Nesterov update: look-ahead correction
            # v_new = momentum * v_old - lr * gradient
            # w_new = w + momentum * v_new - lr * gradient
            #       = w + momentum * (momentum * v_old - lr * gradient) - lr * gradient
            #       = w + momentum^2 * v_old - (1 + momentum) * lr * gradient
            
            v_prev_w = self.velocity_weights[i].copy()
            v_prev_b = self.velocity_biases[i].copy()
            
            self.velocity_weights[i] = momentum * self.velocity_weights[i] - learning_rate * grad_w
            self.velocity_biases[i] = momentum * self.velocity_biases[i] - learning_rate * grad_b
            
            # Nesterov correction: add extra momentum term
            self.weights[i] += -momentum * v_prev_w + (1 + momentum) * self.velocity_weights[i]
            self.biases[i] += -momentum * v_prev_b + (1 + momentum) * self.velocity_biases[i]

            if self.use_batch_norm and i < len(self.gamma):
                v_prev_g = self.velocity_gamma[i].copy()
                v_prev_beta = self.velocity_beta[i].copy()
                
                self.velocity_gamma[i] = momentum * self.velocity_gamma[i] - learning_rate * nabla_g[i]
                self.velocity_beta[i] = momentum * self.velocity_beta[i] - learning_rate * nabla_beta[i]
                
                self.gamma[i] += -momentum * v_prev_g + (1 + momentum) * self.velocity_gamma[i]
                self.beta[i] += -momentum * v_prev_beta + (1 + momentum) * self.velocity_beta[i]

    def train(self, X_train, y_train, X_val=None, y_val=None, y_scaler=None,
              epochs=1000, batch_size=32, learning_rate=0.001, 
              optimizer='adam', momentum=0.9, l2_lambda=0.0001,
              lr_decay=0.0, lr_schedule=None, lr_scheduler='none',
              patience=50, clip_grad=True, max_grad_norm=5.0, 
              loss_fn='mse', verbose=True):
        """
        Train the neural network with mini-batch gradient descent.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            y_scaler: Scaler for inverse transforming predictions
            epochs: Maximum training epochs
            batch_size: Mini-batch size (-1 for full batch)
            learning_rate: Initial learning rate
            optimizer: 'adam', 'nesterov', or 'sgd'
            momentum: Momentum coefficient for SGD/Nesterov
            l2_lambda: L2 regularization strength
            lr_decay: Linear LR decay factor
            lr_schedule: Dict with 'milestones' and 'gamma' for step decay
            lr_scheduler: 'none', 'cosine', or 'cosine_restart'
            patience: Early stopping patience
            clip_grad: Whether to clip gradients
            max_grad_norm: Maximum gradient norm
            loss_fn: 'mse' or 'mee' - loss function for training
            verbose: Print progress
        """
        num_samples = X_train.shape[1]
        if batch_size == -1:
            batch_size = num_samples
        
        best_val_mee = float('inf')
        patience_counter = 0
        best_weights = copy.deepcopy(self.weights)
        best_biases = copy.deepcopy(self.biases)
        if self.use_batch_norm:
            best_gamma = copy.deepcopy(self.gamma)
            best_beta = copy.deepcopy(self.beta)
        best_epoch = 0
        
        current_lr = learning_rate
        history = {'train_loss': [], 'val_mee': []}
        
        for epoch in range(epochs):
            self.t = 0 # Reset adam timestep per epoch
            
            # Learning rate scheduling
            if lr_scheduler == 'cosine':
                current_lr = cosine_annealing_lr(epoch, epochs, learning_rate)
            elif lr_scheduler == 'cosine_restart':
                current_lr = cosine_warm_restart_lr(epoch, T_0=100, lr_init=learning_rate)
            elif lr_schedule is not None and epoch in lr_schedule.get('milestones', []):
                current_lr *= lr_schedule['gamma']
                if verbose:
                    print(f"LR reduced to {current_lr:.6f}")
            elif lr_decay > 0:
                current_lr = learning_rate / (1 + lr_decay * epoch)
            
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                batch_samples = end_idx - start_idx
                
                y_pred = self.forward(X_batch, training=True)
                
                nabla_w, nabla_b, nabla_g, nabla_beta = self.backward(y_batch, y_pred, loss_fn=loss_fn)
                
                nabla_w = [nw / batch_samples for nw in nabla_w]
                nabla_b = [nb / batch_samples for nb in nabla_b]
                if self.use_batch_norm:
                    nabla_g = [ng / batch_samples for ng in nabla_g]
                    nabla_beta = [nbeta / batch_samples for nbeta in nabla_beta]
                
                if clip_grad:
                    nabla_w, nabla_b, nabla_g, nabla_beta = self.clip_gradients(nabla_w, nabla_b, nabla_g, nabla_beta, max_grad_norm)
                
                if optimizer == 'adam':
                    self.update_adam(nabla_w, nabla_b, nabla_g, nabla_beta, current_lr, l2_lambda)
                elif optimizer == 'nesterov':
                    self.update_nesterov(nabla_w, nabla_b, nabla_g, nabla_beta, current_lr, momentum, l2_lambda)
                else:  # 'sgd' or default
                    self.update_sgd_momentum(nabla_w, nabla_b, nabla_g, nabla_beta, current_lr, momentum, l2_lambda)
                
                epoch_loss += mse(y_batch, y_pred) * batch_samples
            
            epoch_loss /= num_samples
            history['train_loss'].append(epoch_loss)
            
            if X_val is not None and y_val is not None:
                y_pred_val_scaled = self.forward(X_val, training=False)
                
                if y_scaler is not None:
                    y_pred_val = y_scaler.inverse_transform(y_pred_val_scaled.T)
                else:
                    y_pred_val = y_pred_val_scaled.T
                
                # y_val is (outputs, samples), y_pred_val is (samples, outputs)
                # MEE expects (samples, outputs), so transpose y_val for comparison
                val_mee_score = mee(y_val.T, y_pred_val)
                history['val_mee'].append(val_mee_score)
                
                if val_mee_score < best_val_mee:
                    best_val_mee = val_mee_score
                    best_weights = copy.deepcopy(self.weights)
                    best_biases = copy.deepcopy(self.biases)
                    if self.use_batch_norm:
                        best_gamma = copy.deepcopy(self.gamma)
                        best_beta = copy.deepcopy(self.beta)
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Val MEE={val_mee_score:.4f}, Best={best_val_mee:.4f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}. Best MEE: {best_val_mee:.4f} at epoch {best_epoch}")
                    break
            
            elif verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss={epoch_loss:.4f}")
        
        if X_val is not None:
            self.weights = best_weights
            self.biases = best_biases
            if self.use_batch_norm:
                self.gamma = best_gamma
                self.beta = best_beta
            if verbose:
                print(f"Training complete. Best MEE: {best_val_mee:.4f} at epoch {best_epoch}")
        
        return history
    
    def predict(self, x):
        """Make predictions."""
        return self.forward(x, training=False)
    
    def get_params(self):
        """Get model parameters for saving."""
        params = {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'layer_sizes': self.layer_sizes,
            'hidden_activation': self.hidden_act_name,
            'output_activation': self.output_act_name,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rate': self.dropout_rate,
        }
        if self.use_batch_norm:
            params['gamma'] = [g.copy() for g in self.gamma]
            params['beta'] = [b.copy() for b in self.beta]
            params['running_mean'] = [rm.copy() for rm in self.running_mean]
            params['running_var'] = [rv.copy() for rv in self.running_var]
        return params
    
    def set_params(self, params):
        """Set model parameters from saved state."""
        self.weights = [w.copy() for w in params['weights']]
        self.biases = [b.copy() for b in params['biases']]
        self.dropout_rate = params.get('dropout_rate', 0.0)
        if params.get('use_batch_norm', False):
            self.use_batch_norm = True
            self.gamma = [g.copy() for g in params['gamma']]
            self.beta = [b.copy() for b in params['beta']]
            self.running_mean = [rm.copy() for rm in params['running_mean']]
            self.running_var = [rv.copy() for rv in params['running_var']]

