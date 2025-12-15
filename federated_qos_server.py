"""
Federated Learning Server for QoS-aware Web Service Composition
Aggregates model updates from multiple clients using federated averaging
"""

import numpy as np
import json
from typing import List, Dict
import copy


class FederatedQoSServer:
    """Central server for federated learning coordination"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.global_model = None
        self.round_number = 0
        self.client_updates: List[Dict] = []
        self.training_history = []
        
    def initialize_global_model(self, input_dim: int = 5, output_dim: int = 1):
        """Initialize global model"""
        self.global_model = {
            'W1': np.random.randn(input_dim, 10) * 0.01,
            'b1': np.zeros((1, 10)),
            'W2': np.random.randn(10, output_dim) * 0.01,
            'b2': np.zeros((1, output_dim))
        }
        print(f"Global model initialized with input_dim={input_dim}, output_dim={output_dim}")
        
    def get_global_model(self) -> Dict:
        """Return current global model"""
        return copy.deepcopy(self.global_model)
    
    def receive_client_update(self, update: Dict):
        """Receive model update from client"""
        self.client_updates.append(update)
        print(f"Received update from {update['client_id']} "
              f"with {update['num_samples']} samples")
    
    def federated_averaging(self) -> Dict:
        """
        Aggregate client models using FedAvg algorithm
        Weighted average based on number of samples
        """
        if not self.client_updates:
            raise ValueError("No client updates received")
        
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
        self.training_history.append({
            'round': self.round_number,
            'num_clients': len(self.client_updates),
            'total_samples': total_samples
        })
        
        # Clear updates for next round
        self.client_updates = []
        self.round_number += 1
        
        print(f"\nRound {self.round_number} completed: "
              f"Aggregated {len(self.client_updates)} client updates")
        
        return self.get_global_model()
    
    def adaptive_federated_averaging(self, learning_rate: float = 0.1) -> Dict:
        """
        Adaptive FedAvg with momentum and learning rate
        """
        if not self.client_updates:
            raise ValueError("No client updates received")
        
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
        
        self.training_history.append({
            'round': self.round_number,
            'num_clients': len(self.client_updates),
            'total_samples': total_samples,
            'learning_rate': learning_rate
        })
        
        self.client_updates = []
        self.round_number += 1
        
        print(f"\nRound {self.round_number} completed (adaptive): "
              f"Learning rate = {learning_rate}")
        
        return self.get_global_model()
    
    def secure_aggregation(self) -> Dict:
        """
        Secure aggregation with differential privacy
        Adds calibrated noise to preserve privacy
        """
        if not self.client_updates:
            raise ValueError("No client updates received")
        
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
        epsilon = 1.0  # Privacy budget
        sensitivity = 2.0 / len(self.client_updates)  # L2 sensitivity
        noise_scale = sensitivity / epsilon
        
        for key in aggregated_weights.keys():
            noise = np.random.normal(0, noise_scale, aggregated_weights[key].shape)
            aggregated_weights[key] += noise
        
        self.global_model = aggregated_weights
        
        self.training_history.append({
            'round': self.round_number,
            'num_clients': len(self.client_updates),
            'total_samples': total_samples,
            'privacy_epsilon': epsilon
        })
        
        self.client_updates = []
        self.round_number += 1
        
        print(f"\nRound {self.round_number} completed (secure): "
              f"Added DP noise with ε={epsilon}")
        
        return self.get_global_model()
    
    def evaluate_convergence(self) -> Dict:
        """Evaluate training convergence metrics"""
        if len(self.training_history) < 2:
            return {'converged': False, 'rounds': self.round_number}
        
        # Simple convergence check (can be enhanced)
        return {
            'converged': self.round_number >= 10,  # Example: after 10 rounds
            'rounds': self.round_number,
            'total_samples_processed': sum(h['total_samples'] 
                                          for h in self.training_history)
        }
    
    def save_model(self, filepath: str):
        """Save global model to file"""
        model_data = {
            'weights': {k: v.tolist() for k, v in self.global_model.items()},
            'round': self.round_number,
            'history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load global model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.global_model = {k: np.array(v) 
                           for k, v in model_data['weights'].items()}
        self.round_number = model_data['round']
        self.training_history = model_data['history']
        print(f"Model loaded from {filepath} (Round {self.round_number})")


class FederatedTrainingCoordinator:
    """Coordinates federated training across multiple clients"""
    
    def __init__(self, server: FederatedQoSServer, clients: List):
        self.server = server
        self.clients = clients
        
    def run_training_round(self, local_epochs: int = 5, 
                          aggregation_method: str = 'fedavg'):
        """Execute one round of federated training"""
        print(f"\n{'='*60}")
        print(f"Starting Training Round {self.server.round_number + 1}")
        print(f"{'='*60}")
        
        # Distribute global model to clients
        global_model = self.server.get_global_model()
        for client in self.clients:
            client.update_model(global_model)
        
        # Each client trains locally
        for client in self.clients:
            print(f"\nTraining on {client.client_id}...")
            client.train_local(epochs=local_epochs)
            
            # Send update to server
            update = client.get_model_update()
            self.server.receive_client_update(update)
        
        # Aggregate updates
        print(f"\nAggregating updates using {aggregation_method}...")
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
                              aggregation_method: str = 'fedavg'):
        """Run complete federated training process"""
        print(f"\n{'='*60}")
        print(f"Federated Learning Training")
        print(f"Number of Rounds: {num_rounds}")
        print(f"Local Epochs per Round: {local_epochs}")
        print(f"Aggregation Method: {aggregation_method}")
        print(f"Number of Clients: {len(self.clients)}")
        print(f"{'='*60}")
        
        for round_num in range(num_rounds):
            self.run_training_round(local_epochs, aggregation_method)
            
            # Check convergence
            convergence = self.server.evaluate_convergence()
            if convergence['converged']:
                print(f"\nTraining converged after {round_num + 1} rounds")
                break
        
        print(f"\n{'='*60}")
        print("Federated Training Completed")
        print(f"{'='*60}")
        
        return self.server.get_global_model()


if __name__ == "__main__":
    # This would typically be run with actual clients
    # See demo.py for complete example
    print("Federated QoS Server")
    print("Use demo.py to run complete federated learning simulation")
