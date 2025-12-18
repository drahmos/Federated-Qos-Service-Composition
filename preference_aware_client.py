"""
Preference-Aware Client for Federated Multi-Objective Learning
Learns user preferences from composition history and trains multi-objective models
"""

import numpy as np
import logging
import copy
from typing import List, Dict, Optional
from multi_objective_predictor import MultiObjectiveQoSPredictor
from federated_qos_client import WebService, QoSMetrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreferenceAwareClient:
    """
    Client with preference learning for multi-objective federated learning

    Features:
    - Multi-objective local model
    - Preference vector learning from history
    - Local training on QoS data
    - Model updates for federated aggregation
    """

    def __init__(self, client_id: str, initial_preferences: np.ndarray = None,
                 input_dim: int = 5, hidden_dim: int = 20, learning_rate: float = 0.001):
        """
        Initialize preference-aware client

        Args:
            client_id: Unique client identifier
            initial_preferences: Initial preference vector (length 5), or None for uniform
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.cluster_id = None  # Assigned by server

        # Multi-objective local model
        self.local_model = MultiObjectiveQoSPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate
        )

        # Preference vector (weights for each objective)
        if initial_preferences is None:
            # Default: uniform preferences
            self.preference_vector = np.ones(5) / 5
        else:
            if len(initial_preferences) != 5:
                raise ValueError(f"initial_preferences must have length 5, got {len(initial_preferences)}")
            # Normalize to sum to 1
            self.preference_vector = np.array(initial_preferences) / (np.sum(initial_preferences) + 1e-8)

        # Local data
        self.local_services: List[WebService] = []

        # Composition history for preference learning
        self.composition_history = []

        # Normalization parameters (per objective)
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        self.objective_stats = {
            'latency': {'min': None, 'max': None},
            'cost': {'min': None, 'max': None},
            'availability': {'min': None, 'max': None},
            'reliability': {'min': None, 'max': None},
            'throughput': {'min': None, 'max': None}
        }

        logger.info(f"Initialized {client_id} with preferences: "
                   f"{self._format_preference(self.preference_vector)}")

    def add_service(self, service: WebService) -> None:
        """
        Add service to local repository

        Args:
            service: WebService instance
        """
        self.local_services.append(service)

    def train_local(self, epochs: int = 10, batch_size: int = 32) -> None:
        """
        Train local multi-objective model

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if not self.local_services:
            logger.warning(f"{self.client_id}: No services to train on")
            return

        # Prepare features
        X = np.array([s.to_vector() for s in self.local_services])

        # Normalize features
        if self.feature_mean is None or self.feature_std is None:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std

        # Prepare targets (one for each objective)
        targets_raw = {
            'latency': np.array([[s.qos.response_time] for s in self.local_services]),
            'cost': np.array([[s.qos.cost] for s in self.local_services]),
            'availability': np.array([[s.qos.availability] for s in self.local_services]),
            'reliability': np.array([[s.qos.reliability] for s in self.local_services]),
            'throughput': np.array([[s.qos.throughput] for s in self.local_services])
        }

        # Normalize targets to [0, 1]
        targets_norm = {}
        for task, y in targets_raw.items():
            if self.objective_stats[task]['min'] is None:
                self.objective_stats[task]['min'] = float(y.min())
                self.objective_stats[task]['max'] = float(y.max())

            y_min = self.objective_stats[task]['min']
            y_max = self.objective_stats[task]['max']

            if y_max - y_min > 1e-8:
                targets_norm[task] = (y - y_min) / (y_max - y_min)
            else:
                # All values are identical
                targets_norm[task] = np.full_like(y, 0.5)

        n_samples = len(X_norm)

        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i+batch_size, n_samples)]

                X_batch = X_norm[batch_indices]
                targets_batch = {task: y[batch_indices] for task, y in targets_norm.items()}

                # Training step
                loss, task_losses = self.local_model.train_step(X_batch, targets_batch)

                epoch_loss += loss
                n_batches += 1

            if epoch % 5 == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                logger.info(f"{self.client_id} - Epoch {epoch}, Loss: {avg_loss:.4f}")

    def predict_objectives(self, service: WebService) -> Dict[str, float]:
        """
        Predict all QoS objectives for a service

        Args:
            service: WebService instance

        Returns:
            predictions: Dict mapping objective name to predicted value
        """
        x = service.to_vector()

        # Normalize features
        if self.feature_mean is not None and self.feature_std is not None:
            x_norm = (x - self.feature_mean) / self.feature_std
        else:
            logger.warning(f"{self.client_id}: No normalization parameters, using raw features")
            x_norm = x

        x_batch = x_norm.reshape(1, -1)

        # Predict
        predictions_norm = self.local_model.predict(x_batch)

        # Denormalize predictions
        predictions = {}
        for task in self.local_model.tasks:
            pred_norm = predictions_norm[task][0, 0]

            # Denormalize if stats available
            if self.objective_stats[task]['min'] is not None:
                y_min = self.objective_stats[task]['min']
                y_max = self.objective_stats[task]['max']
                predictions[task] = pred_norm * (y_max - y_min) + y_min
            else:
                predictions[task] = pred_norm

        return predictions

    def learn_preferences_from_history(self) -> None:
        """
        Learn user preferences from composition history using variance-based approach

        Updates self.preference_vector based on selected services in history
        """
        if len(self.composition_history) < 5:
            logger.info(f"{self.client_id}: Insufficient history for preference learning "
                       f"({len(self.composition_history)} < 5)")
            return

        # Extract objective values from selected services
        selected_objectives = []

        for composition in self.composition_history:
            for service in composition.get('services', []):
                obj_values = [
                    service.qos.response_time,
                    service.qos.cost,
                    service.qos.availability,
                    service.qos.reliability,
                    service.qos.throughput
                ]
                selected_objectives.append(obj_values)

        if not selected_objectives:
            return

        selected_objectives = np.array(selected_objectives)

        # Inverse Reinforcement Learning approach:
        # User prefers objectives with low variance (consistent selection)
        objective_variances = np.var(selected_objectives, axis=0)

        # Invert and normalize to get preference weights
        # Higher variance -> lower preference weight
        preference_weights = 1.0 / (objective_variances + 1e-8)
        new_preferences = preference_weights / np.sum(preference_weights)

        # Smooth update (exponential moving average)
        alpha = 0.3  # Learning rate for preference update
        self.preference_vector = (1 - alpha) * self.preference_vector + alpha * new_preferences

        logger.info(f"{self.client_id} updated preferences from {len(self.composition_history)} compositions:")
        logger.info(f"  {self._format_preference(self.preference_vector)}")

    def add_composition_to_history(self, services: List[WebService], metadata: Dict = None) -> None:
        """
        Add a service composition to history

        Args:
            services: List of selected services
            metadata: Optional metadata about the composition
        """
        composition = {
            'services': services,
            'metadata': metadata or {},
            'timestamp': len(self.composition_history)
        }
        self.composition_history.append(composition)

    def get_model_update(self) -> Dict:
        """
        Get model state for federated aggregation

        Returns:
            update: Dict containing model state, num_samples, preference, and cluster_id
        """
        return {
            'client_id': self.client_id,
            'model_state': self.local_model.get_weights(),
            'num_samples': len(self.local_services),
            'preference_vector': self.preference_vector.copy(),
            'cluster_id': self.cluster_id
        }

    def update_model(self, global_state: Dict) -> None:
        """
        Update local model with global weights from server

        Args:
            global_state: Model state from server
        """
        self.local_model.set_weights(global_state)
        logger.info(f"{self.client_id}: Updated model from global state")

    def _format_preference(self, preference: np.ndarray) -> str:
        """Format preference vector for display"""
        tasks = ['latency', 'cost', 'avail', 'reliab', 'throughput']
        return ', '.join([f"{task}: {preference[i]:.2f}" for i, task in enumerate(tasks)])


if __name__ == "__main__":
    # Test preference-aware client
    logger.info("=== Testing Preference-Aware Client ===\n")

    # Create client with cost-sensitive preferences
    preferences = np.array([0.1, 0.5, 0.2, 0.1, 0.1])  # Cost-focused
    client = PreferenceAwareClient("client_test", initial_preferences=preferences)

    # Add sample services
    logger.info("Adding sample services...")
    services_data = [
        ("s1", "auth", QoSMetrics(50, 100, 0.99, 0.98, 0.5)),
        ("s2", "auth", QoSMetrics(80, 80, 0.95, 0.97, 0.3)),
        ("s3", "payment", QoSMetrics(120, 50, 0.98, 0.99, 1.0)),
        ("s4", "payment", QoSMetrics(100, 60, 0.97, 0.96, 0.8)),
        ("s5", "notify", QoSMetrics(30, 200, 0.99, 0.99, 0.2)),
    ]

    for sid, stype, qos in services_data:
        service = WebService(sid, stype, qos)
        client.add_service(service)

    # Train local model
    logger.info("\nTraining local model...")
    client.train_local(epochs=20, batch_size=3)

    # Test predictions
    logger.info("\n=== Multi-Objective Predictions ===")
    for service in client.local_services[:3]:
        predictions = client.predict_objectives(service)
        logger.info(f"\n{service.service_id} ({service.service_type}):")
        logger.info(f"  Actual: RT={service.qos.response_time}, Cost={service.qos.cost:.2f}, "
                   f"Avail={service.qos.availability:.2f}")
        logger.info(f"  Predicted: RT={predictions['latency']:.1f}, Cost={predictions['cost']:.2f}, "
                   f"Avail={predictions['availability']:.2f}")

    # Test preference learning
    logger.info("\n=== Preference Learning ===")
    logger.info("Adding compositions to history...")

    # Simulate composition history (user selects low-cost services)
    for _ in range(10):
        selected_services = [client.local_services[1], client.local_services[3]]  # Low-cost options
        client.add_composition_to_history(selected_services)

    client.learn_preferences_from_history()

    # Get model update
    logger.info("\n=== Model Update ===")
    update = client.get_model_update()
    logger.info(f"Update contains: {update.keys()}")
    logger.info(f"  Samples: {update['num_samples']}")
    logger.info(f"  Preference: {client._format_preference(update['preference_vector'])}")

    logger.info("\n✓ Preference-aware client test completed!")
