import numpy as np
import copy
from src.utils import sigmoid, linear, tanh, sigmoid_derivative, linear_derivative, tanh_derivative, mse, mse_derivative, bce, bce_derivative, mee

class NeuralNetwork:
    """
    A simple Neural Network implementation from scratch.
    """
    def __init__(self, layer_sizes, hidden_activation=tanh, output_activation=linear, 
                 loss=mse, loss_derivative=mse_derivative, random_state=42):
        """
        Initializes the neural network.
        """
        np.random.seed(random_state)
        
        self.num_layers = len(layer_sizes)
        
        self.weights = [np.random.randn(y, x) * 0.1 for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        
        self.velocity_weights = [np.zeros_like(w) for w in self.weights]
        self.velocity_biases = [np.zeros_like(b) for b in self.biases]

        self.activations = [hidden_activation] * (len(layer_sizes) - 2) + [output_activation]
        self.activation_derivatives = {
            sigmoid: sigmoid_derivative,
            tanh: tanh_derivative,
            linear: linear_derivative
        }
        self.loss = loss
        self.loss_derivative = loss_derivative

    def forward(self, x):
        """
        Performs forward propagation.
        """
        activation = x
        self.activation_outputs = [x]
        self.z_outputs = []

        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            self.z_outputs.append(z)
            
            activation_func = self.activations[i]
            activation = activation_func(z)
            self.activation_outputs.append(activation)
            
        return activation

    def backward(self, y_true, y_pred):
        """
        Performs backward propagation to compute gradients.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        z_output = self.z_outputs[-1]
        delta = self.loss_derivative(y_true, y_pred) * self.activation_derivatives[self.activations[-1]](z_output)

        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)
        nabla_w[-1] = np.dot(delta, self.activation_outputs[-2].T)

        for l in range(2, self.num_layers):
            z = self.z_outputs[-l]
            sp = self.activation_derivatives[self.activations[-l]](z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)
            nabla_w[-l] = np.dot(delta, self.activation_outputs[-l-1].T)
            
        return nabla_w, nabla_b
    
    def predict(self, x):
        """
        Make predictions on new data.
        """
        return self.forward(x)

    def train(self, X_train, y_train, X_val=None, y_val=None, y_scaler=None, 
              epochs=1000, learning_rate=0.01, momentum=0.9, l2_lambda=0.01, patience=10):
        """
        Trains the neural network. If validation data is provided, it uses early stopping.
        """
        num_samples = X_train.shape[1]
        best_val_mee = float('inf')
        patience_counter = 0
        best_weights = copy.deepcopy(self.weights)
        best_biases = copy.deepcopy(self.biases)

        for epoch in range(epochs):
            y_pred_train = self.forward(X_train)
            
            nabla_w, nabla_b = self.backward(y_train, y_pred_train)
            
            nabla_w = [nw / num_samples for nw in nabla_w]
            nabla_b = [nb / num_samples for nb in nabla_b]

            regularization_grad = [l2_lambda * w for w in self.weights]
            
            self.velocity_weights = [momentum * vw - learning_rate * (nw + rg) for vw, nw, rg in zip(self.velocity_weights, nabla_w, regularization_grad)]
            self.velocity_biases = [momentum * vb - learning_rate * nb for vb, nb in zip(self.velocity_biases, nabla_b)]

            self.weights = [w + vw for w, vw in zip(self.weights, self.velocity_weights)]
            self.biases = [b + vb for b, vb in zip(self.biases, self.velocity_biases)]

            # Early stopping check (only if validation data is provided)
            if X_val is not None and y_val is not None and y_scaler is not None:
                if epoch % 10 == 0:
                    y_pred_val_scaled = self.forward(X_val)
                    y_pred_val = y_scaler.inverse_transform(y_pred_val_scaled.T)
                    
                    val_mee = mee(y_val, y_pred_val)
                    
                    if val_mee < best_val_mee:
                        best_val_mee = val_mee
                        best_weights = copy.deepcopy(self.weights)
                        best_biases = copy.deepcopy(self.biases)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}. Best validation MEE: {best_val_mee:.4f}")
                        self.weights = best_weights
                        self.biases = best_biases
                        return

            # If no validation data, just print training loss
            elif epoch % 100 == 0:
                loss = self.loss(y_train, y_pred_train)
                print(f"Epoch {epoch}, Training Loss: {loss}")
        
        # If early stopping was used, restore best weights
        if X_val is not None:
            self.weights = best_weights
            self.biases = best_biases
