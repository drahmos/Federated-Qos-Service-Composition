"""
Federated Learning Server for QoS-aware Web Service Composition
Aggregates model updates from multiple clients using federated averaging
"""

import numpy as np
import json
import logging
import copy
from typing import List, Dict, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FederatedQoSServer:
    """Central server for federated learning coordination"""

    def __init__(self, num_clients: int, privacy_epsilon: float = 1.0):
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")
        if privacy_epsilon <= 0:
            raise ValueError(f"privacy_epsilon must be positive, got {privacy_epsilon}")

        self.num_clients = num_clients
        self.privacy_epsilon = privacy_epsilon  # Configurable privacy budget
        self.global_model = None
        self.round_number = 0
        self.client_updates: List[Dict] = []
        self.training_history = []
        self.previous_global_model: Optional[Dict] = None  # For convergence check
        
    def initialize_global_model(self, input_dim: int = 5, output_dim: int = 1) -> None:
        """Initialize global model using proper initialization schemes"""
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(f"Dimensions must be positive: input_dim={input_dim}, output_dim={output_dim}")

        # He initialization for ReLU activation (hidden layer)
        # Variance = 2/n_in for ReLU networks
        he_std_w1 = np.sqrt(2.0 / input_dim)

        # Xavier initialization for Sigmoid activation (output layer)
        # Variance = 1/n_in for sigmoid networks
        xavier_std_w2 = np.sqrt(1.0 / 10)

        self.global_model = {
            'W1': np.random.randn(input_dim, 10) * he_std_w1,
            'b1': np.zeros((1, 10)),
            'W2': np.random.randn(10, output_dim) * xavier_std_w2,
            'b2': np.zeros((1, output_dim))
        }

        logger.info(f"Global model initialized with He/Xavier initialization")
        logger.info(f"  - Input dim: {input_dim}, Hidden: 10, Output dim: {output_dim}")
        logger.info(f"  - W1 std: {he_std_w1:.4f}, W2 std: {xavier_std_w2:.4f}")
        
    def get_global_model(self) -> Dict:
        """Return current global model"""
        return copy.deepcopy(self.global_model)
    
    def receive_client_update(self, update: Dict) -> None:
        """Receive model update from client with validation"""
        # Validate update structure
        required_keys = ['client_id', 'weights', 'num_samples']
        for key in required_keys:
            if key not in update:
                raise ValueError(f"Client update missing required key: {key}")

        if update['num_samples'] <= 0:
            raise ValueError(f"Invalid num_samples: {update['num_samples']}")

        # Basic validation: check for extreme weight values (possible poisoning)
        for key, weight in update['weights'].items():
            if np.any(np.isnan(weight)) or np.any(np.isinf(weight)):
                logger.warning(f"Client {update['client_id']} sent invalid weights (NaN/Inf), rejecting update")
                return

            # Check for extreme values that might indicate poisoning
            weight_abs_max = np.abs(weight).max()
            if weight_abs_max > 1000:
                logger.warning(f"Client {update['client_id']} sent extreme weights (max={weight_abs_max:.2f}), rejecting update")
                return

        self.client_updates.append(update)
        logger.info(f"Received update from {update['client_id']} "
                   f"with {update['num_samples']} samples")
    
    def federated_averaging(self) -> Dict:
        """
        Aggregate client models using FedAvg algorithm
        Weighted average based on number of samples
        """
        if not self.client_updates:
            raise ValueError("No client updates received")

        # Store previous model for convergence check
        if self.global_model is not None:
            self.previous_global_model = copy.deepcopy(self.global_model)

        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in self.client_updates)

        # Initialize aggregated weights
        aggregated_weights = {}
        for key in self.global_model.keys():
            aggregated_weights[key] = np.zeros_like(self.global_model[key])

        # Weighted averaging
        for update in self.client_updates:
            weight = update['num_samples'] / total_samples
            client_weights = update['weights']

            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight * client_weights[key]

        # Update global model
        self.global_model = aggregated_weights

        # Record history
        num_clients_this_round = len(self.client_updates)
        self.training_history.append({
            'round': self.round_number,
            'num_clients': num_clients_this_round,
            'total_samples': total_samples
        })

        # Clear updates for next round
        self.client_updates = []
        self.round_number += 1

        logger.info(f"\nRound {self.round_number} completed: "
                   f"Aggregated {num_clients_this_round} client updates")

        return self.get_global_model()
    
    def adaptive_federated_averaging(self, learning_rate: float = 0.1) -> Dict:
        """
        Adaptive FedAvg with momentum and learning rate
        """
        if not self.client_updates:
            raise ValueError("No client updates received")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")

        # Store previous model for convergence check
        if self.global_model is not None:
            self.previous_global_model = copy.deepcopy(self.global_model)

        total_samples = sum(update['num_samples'] for update in self.client_updates)

        # Calculate weighted average of updates (deltas from global model)
        aggregated_delta = {}
        for key in self.global_model.keys():
            aggregated_delta[key] = np.zeros_like(self.global_model[key])

        for update in self.client_updates:
            weight = update['num_samples'] / total_samples
            client_weights = update['weights']

            for key in aggregated_delta.keys():
                # Calculate delta (difference from global model)
                delta = client_weights[key] - self.global_model[key]
                aggregated_delta[key] += weight * delta

        # Update global model with learning rate
        for key in self.global_model.keys():
            self.global_model[key] += learning_rate * aggregated_delta[key]

        num_clients_this_round = len(self.client_updates)
        self.training_history.append({
            'round': self.round_number,
            'num_clients': num_clients_this_round,
            'total_samples': total_samples,
            'learning_rate': learning_rate
        })

        self.client_updates = []
        self.round_number += 1

        logger.info(f"\nRound {self.round_number} completed (adaptive): "
                   f"Learning rate = {learning_rate}")

        return self.get_global_model()
    
    def secure_aggregation(self, sensitivity: float = 2.0) -> Dict:
        """
        Secure aggregation with differential privacy
        Adds calibrated noise to preserve privacy

        Args:
            sensitivity: L2 sensitivity of the aggregation (default: 2.0)
        """
        if not self.client_updates:
            raise ValueError("No client updates received")
        if sensitivity <= 0:
            raise ValueError(f"sensitivity must be positive, got {sensitivity}")

        # Store previous model for convergence check
        if self.global_model is not None:
            self.previous_global_model = copy.deepcopy(self.global_model)

        # Standard federated averaging
        total_samples = sum(update['num_samples'] for update in self.client_updates)
        aggregated_weights = {}

        for key in self.global_model.keys():
            aggregated_weights[key] = np.zeros_like(self.global_model[key])

        for update in self.client_updates:
            weight = update['num_samples'] / total_samples
            client_weights = update['weights']

            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight * client_weights[key]

        # Add Gaussian noise for differential privacy (epsilon-DP)
        # Using configurable epsilon from initialization
        # Sensitivity adjusted by number of clients
        adjusted_sensitivity = sensitivity / np.sqrt(len(self.client_updates))
        noise_scale = adjusted_sensitivity / self.privacy_epsilon

        for key in aggregated_weights.keys():
            noise = np.random.normal(0, noise_scale, aggregated_weights[key].shape)
            aggregated_weights[key] += noise

        self.global_model = aggregated_weights

        num_clients_this_round = len(self.client_updates)
        self.training_history.append({
            'round': self.round_number,
            'num_clients': num_clients_this_round,
            'total_samples': total_samples,
            'privacy_epsilon': self.privacy_epsilon,
            'noise_scale': noise_scale
        })

        self.client_updates = []
        self.round_number += 1

        logger.info(f"\nRound {self.round_number} completed (secure): "
                   f"Added DP noise with ε={self.privacy_epsilon}, σ={noise_scale:.6f}")

        return self.get_global_model()
    
    def evaluate_convergence(self, threshold: float = 1e-4, min_rounds: int = 5) -> Dict:
        """
        Evaluate training convergence metrics based on weight changes

        Args:
            threshold: Convergence threshold for relative weight change
            min_rounds: Minimum number of rounds before checking convergence
        """
        if len(self.training_history) < 2:
            return {
                'converged': False,
                'rounds': self.round_number,
                'weight_change': None
            }

        # Calculate weight change if previous model exists
        weight_change = None
        if self.previous_global_model is not None and self.global_model is not None:
            # Compute L2 norm of weight changes
            total_change = 0.0
            total_norm = 0.0

            for key in self.global_model.keys():
                diff = self.global_model[key] - self.previous_global_model[key]
                total_change += np.sum(diff ** 2)
                total_norm += np.sum(self.global_model[key] ** 2)

            # Relative change
            if total_norm > 0:
                weight_change = np.sqrt(total_change / total_norm)
            else:
                weight_change = np.sqrt(total_change)

        # Check convergence: weight change below threshold AND minimum rounds met
        converged = False
        if self.round_number >= min_rounds and weight_change is not None:
            converged = weight_change < threshold

        return {
            'converged': converged,
            'rounds': self.round_number,
            'weight_change': float(weight_change) if weight_change is not None else None,
            'threshold': threshold,
            'total_samples_processed': sum(h['total_samples']
                                          for h in self.training_history)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save global model to file with error handling"""
        if self.global_model is None:
            raise ValueError("No model to save - global_model is None")

        try:
            # Ensure directory exists
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            model_data = {
                'weights': {k: v.tolist() for k, v in self.global_model.items()},
                'round': self.round_number,
                'history': self.training_history,
                'privacy_epsilon': self.privacy_epsilon
            }

            with open(filepath, 'w') as f:
                json.dump(model_data, f, indent=2)

            logger.info(f"Model saved to {filepath}")

        except IOError as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load global model from file with error handling"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with open(filepath, 'r') as f:
                model_data = json.load(f)

            # Validate loaded data structure
            required_keys = ['weights', 'round', 'history']
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Invalid model file: missing '{key}'")

            self.global_model = {k: np.array(v)
                               for k, v in model_data['weights'].items()}
            self.round_number = model_data['round']
            self.training_history = model_data['history']

            # Load privacy epsilon if available (backward compatibility)
            if 'privacy_epsilon' in model_data:
                self.privacy_epsilon = model_data['privacy_epsilon']

            logger.info(f"Model loaded from {filepath} (Round {self.round_number})")

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in model file {filepath}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            raise


class FederatedTrainingCoordinator:
    """Coordinates federated training across multiple clients"""
    
    def __init__(self, server: FederatedQoSServer, clients: List):
        self.server = server
        self.clients = clients
        
    def run_training_round(self, local_epochs: int = 5,
                          aggregation_method: str = 'fedavg') -> None:
        """Execute one round of federated training"""
        if local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {local_epochs}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Training Round {self.server.round_number + 1}")
        logger.info(f"{'='*60}")

        # Distribute global model to clients
        global_model = self.server.get_global_model()
        for client in self.clients:
            client.update_model(global_model)

        # Each client trains locally
        for client in self.clients:
            logger.info(f"\nTraining on {client.client_id}...")
            client.train_local(epochs=local_epochs)

            # Send update to server
            update = client.get_model_update()
            self.server.receive_client_update(update)

        # Aggregate updates
        logger.info(f"\nAggregating updates using {aggregation_method}...")
        if aggregation_method == 'fedavg':
            self.server.federated_averaging()
        elif aggregation_method == 'adaptive':
            self.server.adaptive_federated_averaging()
        elif aggregation_method == 'secure':
            self.server.secure_aggregation()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def run_federated_training(self, num_rounds: int = 10,
                              local_epochs: int = 5,
                              aggregation_method: str = 'fedavg') -> Dict:
        """Run complete federated training process"""
        if num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {num_rounds}")

        logger.info(f"\n{'='*60}")
        logger.info(f"Federated Learning Training")
        logger.info(f"Number of Rounds: {num_rounds}")
        logger.info(f"Local Epochs per Round: {local_epochs}")
        logger.info(f"Aggregation Method: {aggregation_method}")
        logger.info(f"Number of Clients: {len(self.clients)}")
        logger.info(f"{'='*60}")

        for round_num in range(num_rounds):
            self.run_training_round(local_epochs, aggregation_method)

            # Check convergence
            convergence = self.server.evaluate_convergence()
            if convergence.get('weight_change') is not None:
                logger.info(f"Weight change: {convergence['weight_change']:.6f}")

            if convergence['converged']:
                logger.info(f"\nTraining converged after {round_num + 1} rounds")
                break

        logger.info(f"\n{'='*60}")
        logger.info("Federated Training Completed")
        logger.info(f"{'='*60}")

        return self.server.get_global_model()


if __name__ == "__main__":
    # This would typically be run with actual clients
    # See demo.py for complete example
    logger.info("Federated QoS Server")
    logger.info("Use demo.py to run complete federated learning simulation")
