"""
Preference Clustering Module
Groups clients by QoS preference vectors using k-means clustering
"""

import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreferenceClusteringModule:
    """
    Clusters clients by preference vectors using k-means with cosine distance

    Preference vector format: [latency_weight, cost_weight, availability_weight,
                               reliability_weight, throughput_weight]
    where weights sum to 1.0
    """

    def __init__(self, n_clusters: int = 3, max_iterations: int = 100, distance_metric: str = 'cosine'):
        """
        Initialize preference clustering module

        Args:
            n_clusters: Number of preference clusters
            max_iterations: Maximum iterations for k-means
            distance_metric: Distance metric ('cosine' or 'euclidean')
        """
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if distance_metric not in ['cosine', 'euclidean']:
            raise ValueError(f"distance_metric must be 'cosine' or 'euclidean', got {distance_metric}")

        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric

        # Cluster state
        self.cluster_centroids = None
        self.client_assignments = {}
        self.cluster_sizes = {}

    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute distance between two preference vectors

        Args:
            vec1: First preference vector
            vec2: Second preference vector

        Returns:
            distance: Distance value
        """
        if self.distance_metric == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm_product < 1e-10:
                return 1.0
            cosine_sim = dot_product / norm_product
            return 1.0 - cosine_sim
        else:  # euclidean
            return np.linalg.norm(vec1 - vec2)

    def _normalize_preferences(self, preferences: np.ndarray) -> np.ndarray:
        """
        Normalize preference vectors

        Args:
            preferences: Array of preference vectors (n_clients, n_objectives)

        Returns:
            normalized: Normalized preference vectors
        """
        # Ensure preferences sum to 1
        row_sums = preferences.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1.0  # Avoid division by zero
        normalized = preferences / row_sums

        # For cosine distance, normalize to unit length
        if self.distance_metric == 'cosine':
            norms = np.linalg.norm(normalized, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            normalized = normalized / norms

        return normalized

    def cluster_clients(self, client_preferences: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Cluster clients by preference vectors using k-means

        Args:
            client_preferences: Dict mapping client_id -> preference vector (length 5)

        Returns:
            cluster_assignments: Dict mapping client_id -> cluster_id
        """
        if not client_preferences:
            raise ValueError("client_preferences cannot be empty")

        # Convert to matrix
        client_ids = list(client_preferences.keys())
        n_clients = len(client_ids)

        if n_clients < self.n_clusters:
            logger.warning(f"Fewer clients ({n_clients}) than clusters ({self.n_clusters}), reducing clusters")
            self.n_clusters = n_clients

        X = np.array([client_preferences[cid] for cid in client_ids])

        # Normalize preference vectors
        X_normalized = self._normalize_preferences(X)

        # K-means clustering
        cluster_labels = self._kmeans(X_normalized)

        # Store results
        self.client_assignments = {
            client_ids[i]: int(cluster_labels[i])
            for i in range(len(client_ids))
        }

        # Compute cluster sizes
        self.cluster_sizes = {}
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            self.cluster_sizes[cluster_id] = len(members)

        # Log clustering results
        logger.info(f"\n=== Preference Clustering Results ===")
        logger.info(f"Clustered {n_clients} clients into {self.n_clusters} groups")
        for cluster_id in range(self.n_clusters):
            members = self.get_cluster_members(cluster_id)
            centroid = self.cluster_centroids[cluster_id]
            logger.info(f"\nCluster {cluster_id}: {len(members)} clients")
            logger.info(f"  Members: {members[:5]}{'...' if len(members) > 5 else ''}")
            logger.info(f"  Centroid: {self._format_preference(centroid)}")

        return self.client_assignments

    def _kmeans(self, X: np.ndarray) -> np.ndarray:
        """
        K-means clustering algorithm

        Args:
            X: Normalized data matrix (n_samples, n_features)

        Returns:
            labels: Cluster assignments (n_samples,)
        """
        n_samples = X.shape[0]

        # Initialize centroids using k-means++
        centroids = self._kmeans_plus_plus_init(X)

        labels = np.zeros(n_samples, dtype=int)
        prev_labels = np.ones(n_samples, dtype=int) * -1

        # K-means iterations
        for iteration in range(self.max_iterations):
            # Assign points to nearest centroid
            for i in range(n_samples):
                distances = [self._compute_distance(X[i], centroid) for centroid in centroids]
                labels[i] = np.argmin(distances)

            # Check convergence
            if np.array_equal(labels, prev_labels):
                logger.info(f"K-means converged after {iteration + 1} iterations")
                break

            prev_labels = labels.copy()

            # Update centroids
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    centroids[k] = cluster_points.mean(axis=0)
                    # Re-normalize
                    if self.distance_metric == 'cosine':
                        norm = np.linalg.norm(centroids[k])
                        if norm > 1e-10:
                            centroids[k] /= norm

        self.cluster_centroids = centroids
        return labels

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """
        K-means++ initialization for better initial centroids

        Args:
            X: Data matrix

        Returns:
            centroids: Initial centroids
        """
        n_samples = X.shape[0]
        centroids = []

        # Choose first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids.append(X[first_idx].copy())

        # Choose remaining centroids
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.array([
                min([self._compute_distance(x, c) for c in centroids])
                for x in X
            ])

            # Choose next centroid with probability proportional to distance squared
            distances_sq = distances ** 2
            probabilities = distances_sq / (distances_sq.sum() + 1e-10)

            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx].copy())

        return np.array(centroids)

    def get_cluster_members(self, cluster_id: int) -> List[str]:
        """
        Get all client IDs in a specific cluster

        Args:
            cluster_id: Cluster identifier

        Returns:
            members: List of client IDs in this cluster
        """
        return [cid for cid, cid_cluster in self.client_assignments.items()
                if cid_cluster == cluster_id]

    def assign_new_client(self, preference_vector: np.ndarray) -> int:
        """
        Assign a new client to the nearest cluster

        Args:
            preference_vector: Client's preference vector

        Returns:
            cluster_id: Assigned cluster ID
        """
        if self.cluster_centroids is None:
            raise ValueError("Must call cluster_clients() before assign_new_client()")

        # Normalize preference
        preference_normalized = self._normalize_preferences(preference_vector.reshape(1, -1))[0]

        # Find nearest centroid
        distances = [self._compute_distance(preference_normalized, centroid)
                    for centroid in self.cluster_centroids]

        cluster_id = int(np.argmin(distances))

        logger.info(f"New client assigned to cluster {cluster_id}")
        logger.info(f"  Preference: {self._format_preference(preference_vector)}")

        return cluster_id

    def get_cluster_statistics(self) -> Dict:
        """
        Get statistics about clusters

        Returns:
            stats: Dict containing cluster statistics
        """
        if not self.client_assignments:
            return {}

        stats = {
            'n_clusters': self.n_clusters,
            'n_clients': len(self.client_assignments),
            'cluster_sizes': self.cluster_sizes,
            'centroids': {}
        }

        if self.cluster_centroids is not None:
            for cluster_id in range(self.n_clusters):
                stats['centroids'][cluster_id] = {
                    'vector': self.cluster_centroids[cluster_id].tolist(),
                    'formatted': self._format_preference(self.cluster_centroids[cluster_id])
                }

        return stats

    def _format_preference(self, preference: np.ndarray) -> str:
        """Format preference vector for display"""
        tasks = ['latency', 'cost', 'availability', 'reliability', 'throughput']
        return ', '.join([f"{task}: {preference[i]:.3f}" for i, task in enumerate(tasks)])

    def compute_silhouette_score(self, client_preferences: Dict[str, np.ndarray]) -> float:
        """
        Compute silhouette score to measure cluster quality

        Args:
            client_preferences: Dict mapping client_id -> preference vector

        Returns:
            score: Silhouette score (higher is better, range [-1, 1])
        """
        if len(self.client_assignments) < 2:
            return 0.0

        client_ids = list(client_preferences.keys())
        X = np.array([client_preferences[cid] for cid in client_ids])
        X_normalized = self._normalize_preferences(X)

        silhouette_vals = []

        for i, client_id in enumerate(client_ids):
            cluster_id = self.client_assignments[client_id]

            # Compute a(i): mean distance to points in same cluster
            same_cluster_points = [
                X_normalized[j] for j, cid in enumerate(client_ids)
                if self.client_assignments[cid] == cluster_id and j != i
            ]

            if len(same_cluster_points) == 0:
                a_i = 0
            else:
                distances = [self._compute_distance(X_normalized[i], p) for p in same_cluster_points]
                a_i = np.mean(distances)

            # Compute b(i): min mean distance to points in other clusters
            b_i = float('inf')
            for other_cluster in range(self.n_clusters):
                if other_cluster == cluster_id:
                    continue

                other_cluster_points = [
                    X_normalized[j] for j, cid in enumerate(client_ids)
                    if self.client_assignments[cid] == other_cluster
                ]

                if len(other_cluster_points) > 0:
                    distances = [self._compute_distance(X_normalized[i], p) for p in other_cluster_points]
                    mean_dist = np.mean(distances)
                    b_i = min(b_i, mean_dist)

            # Silhouette coefficient for this point
            if b_i == float('inf'):
                s_i = 0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)

            silhouette_vals.append(s_i)

        return np.mean(silhouette_vals)


if __name__ == "__main__":
    # Test preference clustering
    logger.info("=== Testing Preference Clustering Module ===\n")

    # Create module
    clustering = PreferenceClusteringModule(n_clusters=3, distance_metric='cosine')

    # Generate synthetic client preferences
    np.random.seed(42)
    n_clients = 20

    # Create 3 archetypical preference profiles
    archetypes = {
        'cost_sensitive': np.array([0.1, 0.5, 0.2, 0.1, 0.1]),
        'performance_focused': np.array([0.5, 0.1, 0.1, 0.2, 0.1]),
        'reliability_focused': np.array([0.1, 0.1, 0.3, 0.4, 0.1])
    }

    # Generate clients with preferences close to archetypes
    client_preferences = {}
    for i in range(n_clients):
        archetype_name = np.random.choice(list(archetypes.keys()))
        base_pref = archetypes[archetype_name].copy()

        # Add some noise
        noise = np.random.randn(5) * 0.05
        pref = base_pref + noise
        pref = np.abs(pref)  # Ensure non-negative
        pref = pref / pref.sum()  # Normalize

        client_preferences[f"client_{i}"] = pref

    # Cluster clients
    logger.info("Clustering clients by preferences...")
    assignments = clustering.cluster_clients(client_preferences)

    # Compute silhouette score
    silhouette = clustering.compute_silhouette_score(client_preferences)
    logger.info(f"\nSilhouette Score: {silhouette:.4f}")

    # Get statistics
    stats = clustering.get_cluster_statistics()
    logger.info(f"\n=== Cluster Statistics ===")
    logger.info(f"Total clients: {stats['n_clients']}")
    logger.info(f"Cluster sizes: {stats['cluster_sizes']}")

    # Test assigning new client
    logger.info(f"\n=== Testing New Client Assignment ===")
    new_preference = np.array([0.45, 0.15, 0.1, 0.2, 0.1])  # Performance-focused
    assigned_cluster = clustering.assign_new_client(new_preference)
    logger.info(f"Assigned to cluster: {assigned_cluster}")

    logger.info("\n✓ Preference clustering test completed!")
