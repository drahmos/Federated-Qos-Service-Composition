"""
Pareto Service Composer
Composes optimal service workflows using Pareto optimization and user preferences
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from preference_aware_client import PreferenceAwareClient
from federated_qos_client import WebService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParetoServiceComposer:
    """
    Service composition using Pareto optimization

    Features:
    - Multi-objective prediction for all candidates
    - Pareto frontier computation
    - Preference-based scalarization for selection
    - Constraint-based filtering
    """

    def __init__(self, client: PreferenceAwareClient):
        """
        Initialize Pareto service composer

        Args:
            client: PreferenceAwareClient with trained multi-objective model
        """
        self.client = client

    def predict_all_objectives(self, service: WebService) -> Dict[str, float]:
        """
        Predict all QoS objectives for a service

        Args:
            service: WebService instance

        Returns:
            predictions: Dict mapping objective name to predicted value
        """
        return self.client.predict_objectives(service)

    def compute_pareto_frontier(self, services: List[WebService]) -> List[WebService]:
        """
        Compute Pareto frontier from list of services

        Args:
            services: List of candidate services

        Returns:
            pareto_services: List of Pareto-optimal services
        """
        if not services:
            return []

        # Predict objectives for all services
        objective_matrix = []
        for service in services:
            preds = self.predict_all_objectives(service)

            # Convert to minimization problem (negate maximize objectives)
            obj_vector = [
                preds['latency'],       # minimize
                preds['cost'],          # minimize
                -preds['availability'],  # maximize -> minimize (negate)
                -preds['reliability'],   # maximize -> minimize (negate)
                -preds['throughput']     # maximize -> minimize (negate)
            ]
            objective_matrix.append(obj_vector)

        objective_matrix = np.array(objective_matrix)

        # Find Pareto-efficient points
        pareto_mask = self._is_pareto_efficient(objective_matrix)

        pareto_services = [services[i] for i in range(len(services)) if pareto_mask[i]]

        logger.info(f"Pareto frontier: {len(pareto_services)}/{len(services)} services "
                   f"({len(pareto_services)/len(services):.1%})")

        return pareto_services

    @staticmethod
    def _is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
        """
        Find Pareto-efficient points (minimization problem)

        Args:
            costs: (n_points, n_objectives) array

        Returns:
            is_efficient: (n_points,) boolean array
        """
        n_points = costs.shape[0]
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                # Check if point i is dominated by any other point
                # Point j dominates i if:
                # - All objectives of j are <= objectives of i
                # - At least one objective of j is < objective of i
                for j in range(n_points):
                    if i != j:
                        # j dominates i?
                        all_better_or_equal = np.all(costs[j] <= costs[i])
                        at_least_one_better = np.any(costs[j] < costs[i])

                        if all_better_or_equal and at_least_one_better:
                            is_efficient[i] = False
                            break

        return is_efficient

    def scalarize_objectives(self, service: WebService,
                            preference_weights: np.ndarray = None) -> float:
        """
        Scalarize multi-objective predictions using weighted sum

        Args:
            service: Service to evaluate
            preference_weights: User preference vector (length 5), or None to use client's

        Returns:
            scalar_score: Weighted combination of objectives (higher is better)
        """
        preds = self.predict_all_objectives(service)

        # Use client's preferences if not provided
        if preference_weights is None:
            preference_weights = self.client.preference_vector

        # Normalize preferences
        weights = preference_weights / (np.sum(preference_weights) + 1e-8)

        # Compute weighted score
        # Note: For minimize objectives, we use negative weight
        # For maximize objectives, we use positive weight
        score = (
            -weights[0] * preds['latency'] +        # minimize latency
            -weights[1] * preds['cost'] +           # minimize cost
            weights[2] * preds['availability'] +    # maximize availability
            weights[3] * preds['reliability'] +     # maximize reliability
            weights[4] * preds['throughput']        # maximize throughput
        )

        return score

    def compose_services(self,
                        service_types: List[str],
                        available_services: Dict[str, List[WebService]],
                        user_preferences: np.ndarray = None,
                        qos_constraints: Dict[str, float] = None) -> List[WebService]:
        """
        Compose optimal service workflow using Pareto optimization

        Args:
            service_types: Ordered list of service types needed
            available_services: Dict mapping service type -> list of services
            user_preferences: User preference weights (length 5), or None for client's
            qos_constraints: Optional hard constraints (e.g., {'max_latency': 100})

        Returns:
            composition: List of selected services (one per type)
        """
        if not service_types:
            raise ValueError("service_types cannot be empty")

        # Use client's preferences if not provided
        if user_preferences is None:
            user_preferences = self.client.preference_vector

        composition = []

        for service_type in service_types:
            if service_type not in available_services:
                raise ValueError(f"No services available for type: {service_type}")

            candidates = available_services[service_type]

            if not candidates:
                raise ValueError(f"No candidate services for type: {service_type}")

            # Apply hard constraints if specified
            if qos_constraints:
                filtered_candidates = [
                    s for s in candidates
                    if self._meets_constraints(s, qos_constraints)
                ]

                if filtered_candidates:
                    candidates = filtered_candidates
                else:
                    logger.warning(f"No services meet constraints for {service_type}, "
                                 f"using all candidates")

            # Compute Pareto frontier
            pareto_candidates = self.compute_pareto_frontier(candidates)

            if not pareto_candidates:
                # Fallback: use all candidates if Pareto frontier is empty
                logger.warning(f"Empty Pareto frontier for {service_type}, using all candidates")
                pareto_candidates = candidates

            # Select best from Pareto frontier using scalarization
            best_service = max(pareto_candidates,
                             key=lambda s: self.scalarize_objectives(s, user_preferences))

            composition.append(best_service)

            logger.info(f"Selected {best_service.service_id} for {service_type}")

        return composition

    def _meets_constraints(self, service: WebService,
                          constraints: Dict[str, float]) -> bool:
        """
        Check if service meets hard QoS constraints

        Args:
            service: Service to check
            constraints: Dict of constraint name -> threshold value

        Returns:
            meets_constraints: True if all constraints are satisfied
        """
        preds = self.predict_all_objectives(service)

        # Check each constraint
        if 'max_latency' in constraints:
            if preds['latency'] > constraints['max_latency']:
                return False

        if 'max_cost' in constraints:
            if preds['cost'] > constraints['max_cost']:
                return False

        if 'min_availability' in constraints:
            if preds['availability'] < constraints['min_availability']:
                return False

        if 'min_reliability' in constraints:
            if preds['reliability'] < constraints['min_reliability']:
                return False

        if 'min_throughput' in constraints:
            if preds['throughput'] < constraints['min_throughput']:
                return False

        return True

    def evaluate_composition(self, composition: List[WebService]) -> Dict:
        """
        Evaluate overall QoS of composition using predictions

        Args:
            composition: List of selected services

        Returns:
            evaluation: Dict containing aggregate QoS metrics
        """
        if not composition:
            return {}

        # Predict objectives for each service
        predictions = [self.predict_all_objectives(s) for s in composition]

        # Aggregate metrics
        # For workflow: latencies add up, costs add up, availability/reliability multiply (approximation)
        total_latency = sum(p['latency'] for p in predictions)
        total_cost = sum(p['cost'] for p in predictions)

        # For series composition: availability = product of availabilities
        # For simplicity, we use average here
        avg_availability = np.mean([p['availability'] for p in predictions])
        avg_reliability = np.mean([p['reliability'] for p in predictions])
        avg_throughput = np.mean([p['throughput'] for p in predictions])

        evaluation = {
            'total_latency': total_latency,
            'total_cost': total_cost,
            'avg_availability': avg_availability,
            'avg_reliability': avg_reliability,
            'avg_throughput': avg_throughput,
            'num_services': len(composition)
        }

        return evaluation

    def compute_composition_score(self, composition: List[WebService],
                                 user_preferences: np.ndarray = None) -> float:
        """
        Compute overall score for a composition

        Args:
            composition: List of services
            user_preferences: Preference weights, or None for client's

        Returns:
            score: Weighted composition score
        """
        if user_preferences is None:
            user_preferences = self.client.preference_vector

        evaluation = self.evaluate_composition(composition)

        if not evaluation:
            return 0.0

        # Normalize preferences
        weights = user_preferences / (np.sum(user_preferences) + 1e-8)

        # Compute weighted score
        score = (
            -weights[0] * evaluation['total_latency'] +
            -weights[1] * evaluation['total_cost'] +
            weights[2] * evaluation['avg_availability'] +
            weights[3] * evaluation['avg_reliability'] +
            weights[4] * evaluation['avg_throughput']
        )

        return score

    def compare_compositions(self, composition1: List[WebService],
                           composition2: List[WebService],
                           user_preferences: np.ndarray = None) -> Dict:
        """
        Compare two service compositions

        Args:
            composition1: First composition
            composition2: Second composition
            user_preferences: Preference weights

        Returns:
            comparison: Dict containing comparison results
        """
        eval1 = self.evaluate_composition(composition1)
        eval2 = self.evaluate_composition(composition2)

        score1 = self.compute_composition_score(composition1, user_preferences)
        score2 = self.compute_composition_score(composition2, user_preferences)

        comparison = {
            'composition1': eval1,
            'composition2': eval2,
            'score1': score1,
            'score2': score2,
            'winner': 'composition1' if score1 > score2 else 'composition2',
            'score_difference': abs(score1 - score2)
        }

        return comparison


if __name__ == "__main__":
    # Test Pareto service composer
    logger.info("=== Testing Pareto Service Composer ===\n")

    from federated_qos_client import QoSMetrics

    # Create client with balanced preferences
    preferences = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    client = PreferenceAwareClient("client_composer_test", initial_preferences=preferences)

    # Add training services
    logger.info("Adding and training on sample services...")
    training_services = [
        ("s1", "auth", QoSMetrics(50, 100, 0.99, 0.98, 0.5)),
        ("s2", "auth", QoSMetrics(80, 80, 0.95, 0.97, 0.3)),
        ("s3", "payment", QoSMetrics(120, 50, 0.98, 0.99, 1.0)),
        ("s4", "payment", QoSMetrics(100, 60, 0.97, 0.96, 0.8)),
        ("s5", "notify", QoSMetrics(30, 200, 0.99, 0.99, 0.2)),
        ("s6", "notify", QoSMetrics(40, 150, 0.97, 0.98, 0.4)),
    ]

    for sid, stype, qos in training_services:
        service = WebService(sid, stype, qos)
        client.add_service(service)

    client.train_local(epochs=30, batch_size=3)

    # Create composer
    composer = ParetoServiceComposer(client)

    # Create candidate services for composition
    logger.info("\n=== Service Composition ===")

    available = {
        "auth": [
            WebService("auth_1", "auth", QoSMetrics(45, 110, 0.99, 0.98, 0.6)),
            WebService("auth_2", "auth", QoSMetrics(85, 75, 0.94, 0.96, 0.4)),
            WebService("auth_3", "auth", QoSMetrics(60, 90, 0.97, 0.97, 0.5)),
        ],
        "payment": [
            WebService("pay_1", "payment", QoSMetrics(115, 55, 0.98, 0.99, 0.9)),
            WebService("pay_2", "payment", QoSMetrics(95, 65, 0.96, 0.95, 0.7)),
            WebService("pay_3", "payment", QoSMetrics(130, 45, 0.99, 0.98, 1.1)),
        ],
        "notify": [
            WebService("not_1", "notify", QoSMetrics(35, 180, 0.98, 0.99, 0.3)),
            WebService("not_2", "notify", QoSMetrics(45, 140, 0.96, 0.97, 0.5)),
        ]
    }

    # Compose services with constraints
    constraints = {
        'max_latency': 300,
        'max_cost': 400,
        'min_availability': 0.90
    }

    composition = composer.compose_services(
        service_types=["auth", "payment", "notify"],
        available_services=available,
        qos_constraints=constraints
    )

    logger.info("\n=== Selected Composition ===")
    for service in composition:
        preds = composer.predict_all_objectives(service)
        logger.info(f"{service.service_id} ({service.service_type}):")
        logger.info(f"  Predicted: latency={preds['latency']:.1f}, cost={preds['cost']:.2f}, "
                   f"avail={preds['availability']:.3f}")

    # Evaluate composition
    logger.info("\n=== Composition Evaluation ===")
    evaluation = composer.evaluate_composition(composition)
    for key, value in evaluation.items():
        logger.info(f"  {key}: {value:.2f}")

    score = composer.compute_composition_score(composition)
    logger.info(f"\nComposition Score: {score:.4f}")

    logger.info("\n✓ Pareto service composer test completed!")
