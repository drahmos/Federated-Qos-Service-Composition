"""
Federated Learning Client for QoS-aware Web Service Composition
Each client trains locally on service performance data and shares model updates
"""

import numpy as np
import json
from typing import List, Dict, Tuple
import time


class QoSMetrics:
    """QoS metrics for web services"""
    def __init__(self, response_time: float, throughput: float, 
                 availability: float, reliability: float, cost: float):
        self.response_time = response_time  # milliseconds
        self.throughput = throughput        # requests/second
        self.availability = availability    # percentage (0-1)
        self.reliability = reliability      # percentage (0-1)
        self.cost = cost                   # monetary units


class WebService:
    """Represents a web service with QoS attributes"""
    def __init__(self, service_id: str, service_type: str, qos: QoSMetrics):
        self.service_id = service_id
        self.service_type = service_type
        self.qos = qos
        
    def to_vector(self) -> np.ndarray:
        """Convert service QoS to feature vector"""
        return np.array([
            self.qos.response_time,
            self.qos.throughput,
            self.qos.availability,
            self.qos.reliability,
            self.qos.cost
        ])


class FederatedQoSClient:
    """Federated learning client for QoS prediction"""
    
    def __init__(self, client_id: str, learning_rate: float = 0.01):
        self.client_id = client_id
        self.learning_rate = learning_rate
        self.model_weights = None
        self.local_services: List[WebService] = []
        self.service_history: Dict[str, List[QoSMetrics]] = {}
        
    def initialize_model(self, input_dim: int = 5, output_dim: int = 1):
        """Initialize local model weights"""
        # Simple neural network: input -> hidden -> output
        self.model_weights = {
            'W1': np.random.randn(input_dim, 10) * 0.01,
            'b1': np.zeros((1, 10)),
            'W2': np.random.randn(10, output_dim) * 0.01,
            'b2': np.zeros((1, output_dim))
        }
        
    def add_service(self, service: WebService):
        """Add a service to local repository"""
        self.local_services.append(service)
        if service.service_id not in self.service_history:
            self.service_history[service.service_id] = []
        self.service_history[service.service_id].append(service.qos)
        
    def forward_pass(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Forward propagation"""
        cache = {}
        
        # Layer 1
        Z1 = np.dot(X, self.model_weights['W1']) + self.model_weights['b1']
        A1 = np.maximum(0, Z1)  # ReLU activation
        cache['Z1'] = Z1
        cache['A1'] = A1
        
        # Layer 2
        Z2 = np.dot(A1, self.model_weights['W2']) + self.model_weights['b2']
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
        cache['Z2'] = Z2
        cache['A2'] = A2
        
        return A2, cache
    
    def backward_pass(self, X: np.ndarray, y: np.ndarray, cache: Dict) -> Dict:
        """Backward propagation"""
        m = X.shape[0]
        gradients = {}
        
        # Output layer
        dZ2 = cache['A2'] - y
        gradients['dW2'] = np.dot(cache['A1'].T, dZ2) / m
        gradients['db2'] = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer
        dA1 = np.dot(dZ2, self.model_weights['W2'].T)
        dZ1 = dA1 * (cache['Z1'] > 0)  # ReLU derivative
        gradients['dW1'] = np.dot(X.T, dZ1) / m
        gradients['db1'] = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return gradients
    
    def train_local(self, epochs: int = 10) -> Dict:
        """Train model on local data"""
        if not self.local_services:
            return self.model_weights
        
        # Prepare training data
        X = np.array([s.to_vector() for s in self.local_services])
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Create labels (quality score: composite of QoS metrics)
        y = np.array([[
            s.qos.availability * s.qos.reliability / 
            (s.qos.response_time * s.qos.cost + 1e-8)
        ] for s in self.local_services])
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)  # Normalize to [0,1]
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            predictions, cache = self.forward_pass(X)
            
            # Compute loss (MSE)
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass
            gradients = self.backward_pass(X, y, cache)
            
            # Update weights
            self.model_weights['W1'] -= self.learning_rate * gradients['dW1']
            self.model_weights['b1'] -= self.learning_rate * gradients['db1']
            self.model_weights['W2'] -= self.learning_rate * gradients['dW2']
            self.model_weights['b2'] -= self.learning_rate * gradients['db2']
            
            if epoch % 5 == 0:
                print(f"Client {self.client_id} - Epoch {epoch}, Loss: {loss:.4f}")
        
        return self.model_weights
    
    def update_model(self, global_weights: Dict):
        """Update local model with global weights"""
        self.model_weights = global_weights.copy()
        
    def predict_qos_score(self, service: WebService) -> float:
        """Predict QoS score for a service"""
        X = service.to_vector().reshape(1, -1)
        X = (X - X.mean()) / (X.std() + 1e-8)
        score, _ = self.forward_pass(X)
        return float(score[0, 0])
    
    def get_model_update(self) -> Dict:
        """Get model weights for federated aggregation"""
        return {
            'client_id': self.client_id,
            'weights': self.model_weights,
            'num_samples': len(self.local_services)
        }


class ServiceComposer:
    """Compose services based on QoS predictions"""
    
    def __init__(self, client: FederatedQoSClient):
        self.client = client
        
    def compose_services(self, service_types: List[str], 
                         available_services: Dict[str, List[WebService]],
                         qos_constraints: Dict[str, float]) -> List[WebService]:
        """
        Compose optimal service workflow
        Args:
            service_types: Ordered list of service types needed
            available_services: Dict mapping service type to available services
            qos_constraints: QoS requirements (max_response_time, min_availability, etc.)
        """
        composition = []
        
        for service_type in service_types:
            if service_type not in available_services:
                raise ValueError(f"No services available for type: {service_type}")
            
            # Filter services by constraints
            candidates = available_services[service_type]
            valid_candidates = [
                s for s in candidates
                if self._meets_constraints(s, qos_constraints)
            ]
            
            if not valid_candidates:
                print(f"Warning: No services meet constraints for {service_type}")
                valid_candidates = candidates
            
            # Select best service based on predicted QoS score
            best_service = max(valid_candidates, 
                             key=lambda s: self.client.predict_qos_score(s))
            composition.append(best_service)
        
        return composition
    
    def _meets_constraints(self, service: WebService, 
                          constraints: Dict[str, float]) -> bool:
        """Check if service meets QoS constraints"""
        if 'max_response_time' in constraints:
            if service.qos.response_time > constraints['max_response_time']:
                return False
        if 'min_availability' in constraints:
            if service.qos.availability < constraints['min_availability']:
                return False
        if 'min_reliability' in constraints:
            if service.qos.reliability < constraints['min_reliability']:
                return False
        if 'max_cost' in constraints:
            if service.qos.cost > constraints['max_cost']:
                return False
        return True
    
    def evaluate_composition(self, composition: List[WebService]) -> Dict:
        """Evaluate overall QoS of composition"""
        total_response_time = sum(s.qos.response_time for s in composition)
        avg_availability = np.mean([s.qos.availability for s in composition])
        avg_reliability = np.mean([s.qos.reliability for s in composition])
        total_cost = sum(s.qos.cost for s in composition)
        
        return {
            'total_response_time': total_response_time,
            'avg_availability': avg_availability,
            'avg_reliability': avg_reliability,
            'total_cost': total_cost,
            'num_services': len(composition)
        }


if __name__ == "__main__":
    # Example usage
    print("=== Federated QoS Client Demo ===\n")
    
    # Create client
    client = FederatedQoSClient("client_1", learning_rate=0.01)
    client.initialize_model()
    
    # Add sample services
    services_data = [
        ("s1", "authentication", QoSMetrics(50, 100, 0.99, 0.98, 0.5)),
        ("s2", "authentication", QoSMetrics(80, 80, 0.95, 0.97, 0.3)),
        ("s3", "payment", QoSMetrics(120, 50, 0.98, 0.99, 1.0)),
        ("s4", "payment", QoSMetrics(100, 60, 0.97, 0.96, 0.8)),
        ("s5", "notification", QoSMetrics(30, 200, 0.99, 0.99, 0.2)),
    ]
    
    for sid, stype, qos in services_data:
        service = WebService(sid, stype, qos)
        client.add_service(service)
    
    # Train local model
    print("Training local model...")
    client.train_local(epochs=20)
    
    # Predict QoS scores
    print("\nQoS Score Predictions:")
    for service in client.local_services:
        score = client.predict_qos_score(service)
        print(f"  {service.service_id} ({service.service_type}): {score:.4f}")
    
    # Service composition
    print("\n=== Service Composition ===")
    composer = ServiceComposer(client)
    
    available = {
        "authentication": [client.local_services[0], client.local_services[1]],
        "payment": [client.local_services[2], client.local_services[3]],
        "notification": [client.local_services[4]]
    }
    
    constraints = {
        'max_response_time': 200,
        'min_availability': 0.95,
        'max_cost': 2.0
    }
    
    composition = composer.compose_services(
        ["authentication", "payment", "notification"],
        available,
        constraints
    )
    
    print("\nSelected Composition:")
    for s in composition:
        print(f"  {s.service_type}: {s.service_id}")
    
    metrics = composer.evaluate_composition(composition)
    print("\nComposition Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
