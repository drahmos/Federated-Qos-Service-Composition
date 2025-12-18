"""
Multi-Objective QoS Predictor
Implements multi-task neural network with separate heads for each QoS objective
"""

import numpy as np
import json
import logging
import copy
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiObjectiveQoSPredictor:
    """
    Multi-task neural network for QoS prediction

    Architecture:
        Input (5D) → Shared Encoder (20D) → Task-Specific Heads (5 outputs)

    Objectives:
        - latency (response_time): minimize
        - cost: minimize
        - availability: maximize
        - reliability: maximize
        - throughput: maximize
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 20, learning_rate: float = 0.001):
        """
        Initialize multi-objective predictor

        Args:
            input_dim: Input feature dimension (5 QoS features)
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for training
        """
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError(f"Dimensions must be positive: input_dim={input_dim}, hidden_dim={hidden_dim}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Task names
        self.tasks = ['latency', 'cost', 'availability', 'reliability', 'throughput']

        # Model weights
        self.weights = None

        # Learnable task weights for uncertainty weighting
        self.log_vars = {task: 0.0 for task in self.tasks}

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize network weights using He/Xavier initialization"""

        # Shared encoder: input -> hidden1 -> hidden2
        he_std_w1 = np.sqrt(2.0 / self.input_dim)
        he_std_w2 = np.sqrt(2.0 / self.hidden_dim)

        self.weights = {
            # Shared encoder layers
            'encoder_W1': np.random.randn(self.input_dim, self.hidden_dim) * he_std_w1,
            'encoder_b1': np.zeros((1, self.hidden_dim)),
            'encoder_W2': np.random.randn(self.hidden_dim, self.hidden_dim) * he_std_w2,
            'encoder_b2': np.zeros((1, self.hidden_dim)),
        }

        # Task-specific heads: hidden -> intermediate -> output
        xavier_std = np.sqrt(1.0 / self.hidden_dim)
        for task in self.tasks:
            self.weights[f'{task}_W1'] = np.random.randn(self.hidden_dim, 10) * xavier_std
            self.weights[f'{task}_b1'] = np.zeros((1, 10))
            self.weights[f'{task}_W2'] = np.random.randn(10, 1) * np.sqrt(1.0 / 10)
            self.weights[f'{task}_b2'] = np.zeros((1, 1))

        logger.info(f"Initialized multi-objective predictor with {len(self.tasks)} tasks")
        logger.info(f"  - Shared encoder: {self.input_dim}D → {self.hidden_dim}D → {self.hidden_dim}D")
        logger.info(f"  - Task heads: {self.hidden_dim}D → 10D → 1D")

    def forward(self, X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Forward pass through network

        Args:
            X: Input features (batch_size, input_dim)

        Returns:
            predictions: Dict mapping task name to predictions
            cache: Cached values for backward pass
        """
        cache = {}
        cache['X'] = X

        # Shared encoder - Layer 1
        Z1 = np.dot(X, self.weights['encoder_W1']) + self.weights['encoder_b1']
        A1 = np.maximum(0, Z1)  # ReLU
        # Dropout simulation (only during training)
        cache['encoder_Z1'] = Z1
        cache['encoder_A1'] = A1

        # Shared encoder - Layer 2
        Z2 = np.dot(A1, self.weights['encoder_W2']) + self.weights['encoder_b2']
        A2 = np.maximum(0, Z2)  # ReLU
        cache['encoder_Z2'] = Z2
        cache['encoder_A2'] = A2  # This is the shared representation

        # Task-specific heads
        predictions = {}
        for task in self.tasks:
            # Head layer 1
            Z_h1 = np.dot(A2, self.weights[f'{task}_W1']) + self.weights[f'{task}_b1']
            A_h1 = np.maximum(0, Z_h1)  # ReLU

            # Head layer 2 (output)
            Z_h2 = np.dot(A_h1, self.weights[f'{task}_W2']) + self.weights[f'{task}_b2']
            # No activation for regression output

            cache[f'{task}_Z_h1'] = Z_h1
            cache[f'{task}_A_h1'] = A_h1
            cache[f'{task}_Z_h2'] = Z_h2

            predictions[task] = Z_h2

        return predictions, cache

    def compute_multi_task_loss(self, predictions: Dict[str, np.ndarray],
                                targets: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Compute multi-task loss with uncertainty weighting

        Loss = Σ_i (1/(2*σ_i²)) * MSE_i + log(σ_i)

        Args:
            predictions: Dict of predicted values per task
            targets: Dict of target values per task

        Returns:
            total_loss: Combined loss across all tasks
            task_losses: Individual losses per task
        """
        total_loss = 0.0
        task_losses = {}

        for task in self.tasks:
            # MSE loss for this task
            mse = np.mean((predictions[task] - targets[task]) ** 2)
            task_losses[task] = mse

            # Uncertainty weighting
            precision = np.exp(-self.log_vars[task])
            weighted_loss = 0.5 * precision * mse + 0.5 * self.log_vars[task]

            total_loss += weighted_loss

        return total_loss, task_losses

    def backward(self, predictions: Dict[str, np.ndarray],
                targets: Dict[str, np.ndarray],
                cache: Dict) -> Dict[str, np.ndarray]:
        """
        Backward pass to compute gradients

        Args:
            predictions: Forward pass predictions
            targets: Ground truth targets
            cache: Cached forward pass values

        Returns:
            gradients: Dict of gradients for all weights
        """
        m = cache['X'].shape[0]
        gradients = {}

        # Accumulator for shared encoder gradients
        dA2 = np.zeros_like(cache['encoder_A2'])

        # Backprop through each task head
        for task in self.tasks:
            # Precision for uncertainty weighting
            precision = np.exp(-self.log_vars[task])

            # Output layer gradient
            dZ_h2 = (predictions[task] - targets[task]) * precision / m
            gradients[f'{task}_W2'] = np.dot(cache[f'{task}_A_h1'].T, dZ_h2)
            gradients[f'{task}_b2'] = np.sum(dZ_h2, axis=0, keepdims=True)

            # Hidden layer gradient
            dA_h1 = np.dot(dZ_h2, self.weights[f'{task}_W2'].T)
            dZ_h1 = dA_h1 * (cache[f'{task}_Z_h1'] > 0)  # ReLU derivative
            gradients[f'{task}_W1'] = np.dot(cache['encoder_A2'].T, dZ_h1)
            gradients[f'{task}_b1'] = np.sum(dZ_h1, axis=0, keepdims=True)

            # Accumulate gradient w.r.t. shared representation
            dA2 += np.dot(dZ_h1, self.weights[f'{task}_W1'].T)

        # Backprop through shared encoder - Layer 2
        dZ2 = dA2 * (cache['encoder_Z2'] > 0)  # ReLU derivative
        gradients['encoder_W2'] = np.dot(cache['encoder_A1'].T, dZ2)
        gradients['encoder_b2'] = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop through shared encoder - Layer 1
        dA1 = np.dot(dZ2, self.weights['encoder_W2'].T)
        dZ1 = dA1 * (cache['encoder_Z1'] > 0)  # ReLU derivative
        gradients['encoder_W1'] = np.dot(cache['X'].T, dZ1)
        gradients['encoder_b1'] = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient clipping
        max_norm = 5.0
        for key in gradients:
            gradients[key] = np.clip(gradients[key], -max_norm, max_norm)

        return gradients

    def update_weights(self, gradients: Dict[str, np.ndarray]):
        """Update network weights using gradients"""
        for key in gradients:
            self.weights[key] -= self.learning_rate * gradients[key]

    def update_task_weights(self, task_losses: Dict[str, float], alpha: float = 0.01):
        """
        Update learnable task weights based on task losses

        Args:
            task_losses: Current MSE loss per task
            alpha: Update rate for log-variance parameters
        """
        # Simple gradient descent on log-variance
        for task in self.tasks:
            # Gradient of uncertainty-weighted loss w.r.t. log_var
            grad_log_var = -0.5 * np.exp(-self.log_vars[task]) * task_losses[task] + 0.5
            self.log_vars[task] -= alpha * grad_log_var

            # Prevent log_var from becoming too negative (numerical stability)
            self.log_vars[task] = np.clip(self.log_vars[task], -5.0, 5.0)

    def train_step(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """
        Single training step

        Args:
            X: Input features
            targets: Target values for each task

        Returns:
            loss: Total loss
            task_losses: Individual task losses
        """
        # Forward pass
        predictions, cache = self.forward(X)

        # Compute loss
        loss, task_losses = self.compute_multi_task_loss(predictions, targets)

        # Backward pass
        gradients = self.backward(predictions, targets, cache)

        # Update weights
        self.update_weights(gradients)

        # Update task weights
        self.update_task_weights(task_losses)

        return loss, task_losses

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict all objectives for input

        Args:
            X: Input features

        Returns:
            predictions: Dict of predictions per task
        """
        predictions, _ = self.forward(X)
        return predictions

    def get_weights(self) -> Dict:
        """Get model weights for federated aggregation"""
        return {
            'weights': copy.deepcopy(self.weights),
            'log_vars': copy.deepcopy(self.log_vars)
        }

    def set_weights(self, state: Dict):
        """Set model weights from federated aggregation"""
        self.weights = copy.deepcopy(state['weights'])
        self.log_vars = copy.deepcopy(state['log_vars'])

    def save_model(self, filepath: str):
        """Save model to file"""
        state = {
            'weights': {k: v.tolist() for k, v in self.weights.items()},
            'log_vars': self.log_vars,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'tasks': self.tasks
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.weights = {k: np.array(v) for k, v in state['weights'].items()}
        self.log_vars = state['log_vars']
        self.input_dim = state['input_dim']
        self.hidden_dim = state['hidden_dim']
        self.learning_rate = state['learning_rate']
        self.tasks = state['tasks']

        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test multi-objective predictor
    logger.info("=== Testing Multi-Objective QoS Predictor ===\n")

    # Create predictor
    predictor = MultiObjectiveQoSPredictor(input_dim=5, hidden_dim=20)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)

    # Synthetic targets for each objective
    targets = {
        'latency': np.random.uniform(10, 200, (n_samples, 1)),
        'cost': np.random.uniform(0.1, 2.0, (n_samples, 1)),
        'availability': np.random.uniform(0.8, 1.0, (n_samples, 1)),
        'reliability': np.random.uniform(0.8, 1.0, (n_samples, 1)),
        'throughput': np.random.uniform(50, 300, (n_samples, 1))
    }

    # Normalize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Normalize targets to [0, 1]
    targets_norm = {}
    for task, y in targets.items():
        y_min, y_max = y.min(), y.max()
        targets_norm[task] = (y - y_min) / (y_max - y_min + 1e-8)

    # Training
    logger.info("Training multi-objective model...")
    epochs = 50
    batch_size = 32

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_norm[batch_idx]
            targets_batch = {task: y[batch_idx] for task, y in targets_norm.items()}

            loss, task_losses = predictor.train_step(X_batch, targets_batch)
            epoch_loss += loss
            n_batches += 1

        if epoch % 10 == 0:
            avg_loss = epoch_loss / n_batches
            logger.info(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

            # Show task losses
            predictions = predictor.predict(X_norm[:5])
            logger.info("  Task weights (log-variance): " +
                       ", ".join([f"{task}: {predictor.log_vars[task]:.3f}" for task in predictor.tasks]))

    # Test predictions
    logger.info("\n=== Test Predictions ===")
    test_X = X_norm[:3]
    predictions = predictor.predict(test_X)

    for i in range(3):
        logger.info(f"\nSample {i+1}:")
        for task in predictor.tasks:
            pred_val = predictions[task][i, 0]
            logger.info(f"  {task}: {pred_val:.4f}")

    logger.info("\n✓ Multi-objective predictor test completed!")
