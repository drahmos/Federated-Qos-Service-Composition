"""
Federated Multi-Objective Server
Manages multiple global models (one per preference cluster) for federated learning
"""

import numpy as np
import json
import logging
import copy
from typing import List, Dict, Optional
from multi_objective_predictor import MultiObjectiveQoSPredictor
from preference_clustering import PreferenceClusteringModule

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FederatedMultiObjectiveServer:
    """
    Central server for federated multi-objective learning

    Manages:
    - Multiple global models (one per preference cluster)
    - Preference-based client clustering
    - Per-cluster aggregation
    - Pareto optimality validation
    """

    def __init__(self, num_clients: int, n_preference_clusters: int = 3,
                 input_dim: int = 5, hidden_dim: int = 20):
        """
        Initialize federated multi-objective server

        Args:
            num_clients: Expected number of clients
            n_preference_clusters: Number of preference-based clusters
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
        """
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")
        if n_preference_clusters <= 0:
            raise ValueError(f"n_preference_clusters must be positive, got {n_preference_clusters}")

        self.num_clients = num_clients
        self.n_clusters = n_preference_clusters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Multiple global models (one per preference cluster)
        self.global_models: Dict[int, MultiObjectiveQoSPredictor] = {}
        for i in range(n_preference_clusters):
            self.global_models[i] = MultiObjectiveQoSPredictor(
                input_dim=input_dim,
                hidden_dim=hidden_dim
            )

        # Preference clustering
        self.preference_clusterer = PreferenceClusteringModule(
            n_clusters=n_preference_clusters,
            distance_metric='cosine'
        )

        # Training state
        self.round_number = 0
        self.training_history = []
        self.client_updates: Dict[int, List[Dict]] = {i: [] for i in range(n_preference_clusters)}

        logger.info(f"Initialized federated multi-objective server")
        logger.info(f"  - {n_preference_clusters} preference clusters")
        logger.info(f"  - {num_clients} expected clients")
        logger.info(f"  - Model: {input_dim}D → {hidden_dim}D → 5 objectives")

    def initialize_global_models(self):
        """Initialize all global models with same random weights"""
        # Get base weights from first model
        base_state = self.global_models[0].get_weights()

        # Copy to all other models
        for cluster_id in range(1, self.n_clusters):
            self.global_models[cluster_id].set_weights(base_state)

        logger.info(f"Initialized {self.n_clusters} global models with shared weights")

    def update_client_clusters(self, client_preferences: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Re-cluster clients based on current preferences

        Args:
            client_preferences: Dict mapping client_id -> preference vector (length 5)

        Returns:
            assignments: Dict mapping client_id -> cluster_id
        """
        if not client_preferences:
            raise ValueError("client_preferences cannot be empty")

        # Perform clustering
        assignments = self.preference_clusterer.cluster_clients(client_preferences)

        # Compute silhouette score
        silhouette = self.preference_clusterer.compute_silhouette_score(client_preferences)

        logger.info(f"\n=== Preference Clustering (Round {self.round_number}) ===")
        logger.info(f"Silhouette Score: {silhouette:.4f}")

        # Log cluster details
        for cluster_id in range(self.n_clusters):
            members = self.preference_clusterer.get_cluster_members(cluster_id)
            logger.info(f"\nCluster {cluster_id}: {len(members)} clients")

            if self.preference_clusterer.cluster_centroids is not None:
                centroid = self.preference_clusterer.cluster_centroids[cluster_id]
                tasks = ['latency', 'cost', 'avail', 'reliab', 'throughput']
                centroid_str = ', '.join([f"{task}: {centroid[i]:.2f}" for i, task in enumerate(tasks)])
                logger.info(f"  Centroid: [{centroid_str}]")

        return assignments

    def receive_client_update(self, update: Dict) -> None:
        """
        Receive model update from client

        Args:
            update: Dict containing:
                - 'client_id': Client identifier
                - 'model_state': Model weights and log_vars
                - 'num_samples': Number of training samples
                - 'preference_vector': Client's preference vector
                - 'cluster_id': Assigned cluster ID
        """
        # Validate update structure
        required_keys = ['client_id', 'model_state', 'num_samples', 'cluster_id']
        for key in required_keys:
            if key not in update:
                raise ValueError(f"Client update missing required key: {key}")

        if update['num_samples'] <= 0:
            raise ValueError(f"Invalid num_samples: {update['num_samples']}")

        cluster_id = update['cluster_id']
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        # Store update for this cluster
        self.client_updates[cluster_id].append(update)

        logger.info(f"Received update from {update['client_id']} "
                   f"(Cluster {cluster_id}, {update['num_samples']} samples)")

    def aggregate_cluster(self, cluster_id: int) -> None:
        """
        Aggregate updates from clients in a specific cluster using FedAvg

        Args:
            cluster_id: Which cluster to aggregate
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        updates = self.client_updates[cluster_id]

        if not updates:
            logger.warning(f"No updates for cluster {cluster_id}, skipping aggregation")
            return

        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in updates)

        # Get current global model state
        global_state = self.global_models[cluster_id].get_weights()

        # Initialize aggregated state
        aggregated_weights = {}
        for key in global_state['weights'].keys():
            aggregated_weights[key] = np.zeros_like(global_state['weights'][key])

        aggregated_log_vars = {task: 0.0 for task in self.global_models[cluster_id].tasks}

        # Weighted averaging
        for update in updates:
            weight = update['num_samples'] / total_samples
            client_state = update['model_state']

            # Aggregate model weights
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight * client_state['weights'][key]

            # Aggregate log_vars
            for task in aggregated_log_vars.keys():
                aggregated_log_vars[task] += weight * client_state['log_vars'][task]

        # Update global model
        self.global_models[cluster_id].set_weights({
            'weights': aggregated_weights,
            'log_vars': aggregated_log_vars
        })

        logger.info(f"Cluster {cluster_id}: Aggregated {len(updates)} updates "
                   f"({total_samples} total samples)")

        # Clear updates for next round
        self.client_updates[cluster_id] = []

    def aggregate_all_clusters(self) -> None:
        """Aggregate updates for all clusters"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Aggregating Round {self.round_number + 1}")
        logger.info(f"{'='*60}")

        for cluster_id in range(self.n_clusters):
            self.aggregate_cluster(cluster_id)

        # Record history
        self.training_history.append({
            'round': self.round_number,
            'clusters': {
                cluster_id: len(self.client_updates[cluster_id])
                for cluster_id in range(self.n_clusters)
            }
        })

        self.round_number += 1

    def get_global_model(self, cluster_id: int) -> Dict:
        """
        Get global model for a specific cluster

        Args:
            cluster_id: Cluster identifier

        Returns:
            model_state: Deep copy of model state
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"Invalid cluster_id: {cluster_id}")

        return self.global_models[cluster_id].get_weights()

    def validate_pareto_optimality(self, test_services: List) -> Dict:
        """
        Validate that global models produce Pareto-optimal predictions

        Args:
            test_services: List of WebService objects for testing

        Returns:
            metrics: Dict containing Pareto quality metrics per cluster
        """
        if not test_services:
            logger.warning("No test services provided for validation")
            return {}

        results = {}

        logger.info(f"\n=== Pareto Optimality Validation ===")

        for cluster_id, model in self.global_models.items():
            # Prepare test data
            X_test = np.array([s.to_vector() for s in test_services])

            # Normalize features (using simple standardization)
            X_mean = X_test.mean(axis=0)
            X_std = X_test.std(axis=0) + 1e-8
            X_norm = (X_test - X_mean) / X_std

            # Predict all objectives
            predictions = model.predict(X_norm)

            # Convert to objective matrix (n_services x n_objectives)
            # For Pareto analysis, convert to minimization problem
            obj_matrix = np.column_stack([
                predictions['latency'].flatten(),      # minimize
                predictions['cost'].flatten(),         # minimize
                -predictions['availability'].flatten(), # maximize -> minimize
                -predictions['reliability'].flatten(),  # maximize -> minimize
                -predictions['throughput'].flatten()    # maximize -> minimize
            ])

            # Compute Pareto frontier
            pareto_mask = self._is_pareto_efficient(obj_matrix)
            n_pareto = np.sum(pareto_mask)

            # Compute hypervolume (simplified)
            hypervolume = self._compute_hypervolume_simple(obj_matrix[pareto_mask])

            results[cluster_id] = {
                'n_pareto_optimal': n_pareto,
                'pareto_ratio': n_pareto / len(test_services),
                'hypervolume': hypervolume,
                'n_services': len(test_services)
            }

            logger.info(f"Cluster {cluster_id}: "
                       f"{n_pareto}/{len(test_services)} Pareto-optimal "
                       f"({results[cluster_id]['pareto_ratio']:.1%}), "
                       f"HV={hypervolume:.4f}")

        return results

    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto-efficient points (minimization)

        Args:
            costs: (n_points, n_objectives) array

        Returns:
            is_efficient: (n_points,) boolean array
        """
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                # Point i is dominated if another point is better in all objectives
                # For minimization: point j dominates i if costs[j] <= costs[i] for all objectives
                # and costs[j] < costs[i] for at least one objective
                dominated_by = np.all(costs[:n_points] <= costs[i], axis=1) & \
                              np.any(costs[:n_points] < costs[i], axis=1)
                is_efficient[i] = not np.any(dominated_by)

        return is_efficient

    @staticmethod
    def _compute_hypervolume_simple(pareto_front: np.ndarray, reference_point: np.ndarray = None) -> float:
        """
        Compute simplified hypervolume indicator

        Args:
            pareto_front: (n_pareto, n_objectives) array
            reference_point: Reference point (default: worst values * 1.1)

        Returns:
            hypervolume: Approximated hypervolume value
        """
        if len(pareto_front) == 0:
            return 0.0

        if reference_point is None:
            # Use worst values as reference
            reference_point = np.max(pareto_front, axis=0) * 1.1

        # Simple approximation: sum of dominated volumes
        volumes = np.prod(np.maximum(0, reference_point - pareto_front), axis=1)
        return float(np.sum(volumes))

    def save_models(self, directory: str):
        """
        Save all global models to directory

        Args:
            directory: Directory path to save models
        """
        import os
        os.makedirs(directory, exist_ok=True)

        for cluster_id, model in self.global_models.items():
            filepath = os.path.join(directory, f"global_model_cluster_{cluster_id}.json")
            model.save_model(filepath)

        # Save server state
        server_state = {
            'round_number': self.round_number,
            'n_clusters': self.n_clusters,
            'training_history': self.training_history,
            'cluster_stats': self.preference_clusterer.get_cluster_statistics()
        }

        server_filepath = os.path.join(directory, "server_state.json")
        with open(server_filepath, 'w') as f:
            json.dump(server_state, f, indent=2)

        logger.info(f"Saved {self.n_clusters} global models to {directory}")

    def load_models(self, directory: str):
        """
        Load all global models from directory

        Args:
            directory: Directory path containing saved models
        """
        import os

        for cluster_id in range(self.n_clusters):
            filepath = os.path.join(directory, f"global_model_cluster_{cluster_id}.json")
            if os.path.exists(filepath):
                self.global_models[cluster_id].load_model(filepath)
            else:
                logger.warning(f"Model file not found: {filepath}")

        # Load server state
        server_filepath = os.path.join(directory, "server_state.json")
        if os.path.exists(server_filepath):
            with open(server_filepath, 'r') as f:
                server_state = json.load(f)
            self.round_number = server_state.get('round_number', 0)
            self.training_history = server_state.get('training_history', [])

        logger.info(f"Loaded {self.n_clusters} global models from {directory}")


class FederatedMultiObjectiveCoordinator:
    """Coordinates federated multi-objective training across clients"""

    def __init__(self, server: FederatedMultiObjectiveServer, clients: List):
        """
        Initialize coordinator

        Args:
            server: FederatedMultiObjectiveServer instance
            clients: List of PreferenceAwareClient instances
        """
        self.server = server
        self.clients = clients

    def run_training_round(self, local_epochs: int = 10) -> None:
        """
        Execute one round of federated multi-objective training

        Args:
            local_epochs: Number of local training epochs per client
        """
        if local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {local_epochs}")

        logger.info(f"\n{'='*70}")
        logger.info(f"Training Round {self.server.round_number + 1}")
        logger.info(f"{'='*70}")

        # Phase 1: Collect client preferences and perform clustering
        client_preferences = {
            client.client_id: client.preference_vector
            for client in self.clients
        }

        cluster_assignments = self.server.update_client_clusters(client_preferences)

        # Phase 2: Distribute global models to clients
        for client in self.clients:
            cluster_id = cluster_assignments[client.client_id]
            global_model_state = self.server.get_global_model(cluster_id)
            client.update_model(global_model_state)
            client.cluster_id = cluster_id

        # Phase 3: Local training
        logger.info(f"\n--- Local Training ---")
        for client in self.clients:
            logger.info(f"\nTraining {client.client_id} (Cluster {client.cluster_id})...")
            client.train_local(epochs=local_epochs)

            # Send update to server
            update = client.get_model_update()
            self.server.receive_client_update(update)

        # Phase 4: Aggregate updates
        logger.info(f"\n--- Aggregation ---")
        self.server.aggregate_all_clusters()

    def run_federated_training(self, num_rounds: int = 10, local_epochs: int = 10,
                               test_services: List = None) -> None:
        """
        Run complete federated multi-objective training process

        Args:
            num_rounds: Number of federated rounds
            local_epochs: Local epochs per round
            test_services: Optional test services for validation
        """
        if num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive, got {num_rounds}")

        logger.info(f"\n{'='*70}")
        logger.info(f"Federated Multi-Objective Learning")
        logger.info(f"{'='*70}")
        logger.info(f"Rounds: {num_rounds}")
        logger.info(f"Local Epochs: {local_epochs}")
        logger.info(f"Clients: {len(self.clients)}")
        logger.info(f"Preference Clusters: {self.server.n_clusters}")

        for round_num in range(num_rounds):
            self.run_training_round(local_epochs)

            # Validate Pareto optimality if test services provided
            if test_services and round_num % 5 == 0:
                self.server.validate_pareto_optimality(test_services)

        logger.info(f"\n{'='*70}")
        logger.info("Federated Multi-Objective Training Completed")
        logger.info(f"{'='*70}")


if __name__ == "__main__":
    logger.info("=== Federated Multi-Objective Server ===")
    logger.info("This module requires PreferenceAwareClient for full testing")
    logger.info("See federated_mo_demo.py for complete example")
