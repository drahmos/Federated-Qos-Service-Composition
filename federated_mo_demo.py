"""
Comprehensive Demo: Federated Multi-Objective QoS Learning with Pareto Optimization

This demo showcases the complete system:
1. Multiple clients with different preferences
2. Preference-based clustering
3. Multi-objective federated learning
4. Pareto-optimal service composition
"""

import numpy as np
import logging
from typing import List, Dict

from federated_qos_client import WebService, QoSMetrics
from multi_objective_predictor import MultiObjectiveQoSPredictor
from preference_clustering import PreferenceClusteringModule
from federated_mo_server import FederatedMultiObjectiveServer, FederatedMultiObjectiveCoordinator
from preference_aware_client import PreferenceAwareClient
from pareto_composer import ParetoServiceComposer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_services(n_services: int, service_type: str, seed: int = None) -> List[WebService]:
    """
    Generate synthetic web services with realistic QoS correlations

    Args:
        n_services: Number of services to generate
        service_type: Type of service
        seed: Random seed

    Returns:
        services: List of WebService instances
    """
    if seed is not None:
        np.random.seed(seed)

    services = []

    for i in range(n_services):
        # Generate correlated QoS metrics
        # Higher cost -> better availability/reliability (positive correlation)
        # Higher latency -> lower cost (negative correlation)

        base_latency = np.random.uniform(20, 200)
        base_cost = np.random.uniform(0.1, 2.0)

        # Correlations
        latency = base_latency + np.random.normal(0, 10)
        cost = max(0.1, base_cost - 0.005 * (latency - 100) + np.random.normal(0, 0.2))
        availability = min(0.999, 0.85 + 0.05 * (cost / 2.0) + np.random.normal(0, 0.02))
        reliability = min(0.999, 0.85 + 0.05 * (cost / 2.0) + np.random.normal(0, 0.02))
        throughput = max(10, 150 - 0.3 * latency + np.random.normal(0, 20))

        # Clamp values
        latency = max(10, latency)
        cost = max(0.1, cost)
        availability = max(0.7, min(0.999, availability))
        reliability = max(0.7, min(0.999, reliability))
        throughput = max(10, throughput)

        qos = QoSMetrics(latency, throughput, availability, reliability, cost)
        service = WebService(f"{service_type}_{i}", service_type, qos)
        services.append(service)

    return services


def create_clients_with_diverse_preferences(n_clients: int) -> List[PreferenceAwareClient]:
    """
    Create clients with diverse preference profiles

    Args:
        n_clients: Number of clients to create

    Returns:
        clients: List of PreferenceAwareClient instances
    """
    # Define archetypical preference profiles
    archetypes = {
        'cost_sensitive': np.array([0.1, 0.5, 0.2, 0.1, 0.1]),
        'performance_focused': np.array([0.5, 0.1, 0.1, 0.2, 0.1]),
        'reliability_focused': np.array([0.1, 0.1, 0.3, 0.4, 0.1]),
        'balanced': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        'throughput_optimized': np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    }

    archetype_names = list(archetypes.keys())
    clients = []

    for i in range(n_clients):
        # Assign archetype cyclically
        archetype_name = archetype_names[i % len(archetype_names)]
        base_pref = archetypes[archetype_name].copy()

        # Add some noise to create variation
        noise = np.random.randn(5) * 0.05
        pref = base_pref + noise
        pref = np.abs(pref)  # Ensure non-negative
        pref = pref / pref.sum()  # Normalize

        client = PreferenceAwareClient(
            client_id=f"client_{i}",
            initial_preferences=pref,
            input_dim=5,
            hidden_dim=20,
            learning_rate=0.001
        )

        clients.append(client)

    return clients


def distribute_services_to_clients(clients: List[PreferenceAwareClient],
                                   all_services: Dict[str, List[WebService]],
                                   services_per_client: int = 20) -> None:
    """
    Distribute services to clients (simulating local data)

    Args:
        clients: List of clients
        all_services: Dict of service_type -> list of services
        services_per_client: Number of services per client
    """
    all_service_list = []
    for service_list in all_services.values():
        all_service_list.extend(service_list)

    for client in clients:
        # Randomly sample services
        selected_indices = np.random.choice(
            len(all_service_list),
            size=min(services_per_client, len(all_service_list)),
            replace=False
        )

        for idx in selected_indices:
            client.add_service(all_service_list[idx])

        logger.info(f"{client.client_id}: Assigned {len(client.local_services)} services")


def main():
    """Main demo function"""

    logger.info("=" * 80)
    logger.info("FEDERATED MULTI-OBJECTIVE QOS LEARNING WITH PARETO OPTIMIZATION")
    logger.info("=" * 80)

    # ========== Configuration ==========
    NUM_CLIENTS = 10
    NUM_PREFERENCE_CLUSTERS = 3
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 8
    SERVICES_PER_TYPE = 30
    SERVICES_PER_CLIENT = 20

    np.random.seed(42)

    # ========== Phase 1: Setup ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: SETUP")
    logger.info("=" * 80)

    # Generate synthetic services
    logger.info("\nGenerating synthetic services...")
    service_types = ['authentication', 'payment', 'database', 'notification', 'analytics']

    all_services = {}
    for i, stype in enumerate(service_types):
        all_services[stype] = generate_synthetic_services(
            n_services=SERVICES_PER_TYPE,
            service_type=stype,
            seed=42 + i
        )
        logger.info(f"  {stype}: {len(all_services[stype])} services")

    # Create server
    logger.info(f"\nInitializing federated server with {NUM_PREFERENCE_CLUSTERS} preference clusters...")
    server = FederatedMultiObjectiveServer(
        num_clients=NUM_CLIENTS,
        n_preference_clusters=NUM_PREFERENCE_CLUSTERS,
        input_dim=5,
        hidden_dim=20
    )
    server.initialize_global_models()

    # Create clients with diverse preferences
    logger.info(f"\nCreating {NUM_CLIENTS} clients with diverse preferences...")
    clients = create_clients_with_diverse_preferences(NUM_CLIENTS)

    # Display client preferences
    logger.info("\nClient Preference Profiles:")
    for client in clients:
        tasks = ['latency', 'cost', 'avail', 'reliab', 'throughput']
        pref_str = ', '.join([f"{task}={client.preference_vector[i]:.2f}"
                             for i, task in enumerate(tasks)])
        logger.info(f"  {client.client_id}: [{pref_str}]")

    # Distribute services to clients
    logger.info(f"\nDistributing services to clients ({SERVICES_PER_CLIENT} per client)...")
    distribute_services_to_clients(clients, all_services, SERVICES_PER_CLIENT)

    # ========== Phase 2: Federated Training ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: FEDERATED MULTI-OBJECTIVE TRAINING")
    logger.info("=" * 80)

    # Create coordinator
    coordinator = FederatedMultiObjectiveCoordinator(server, clients)

    # Prepare test services for validation
    test_services = []
    for stype in service_types[:2]:  # Use subset for testing
        test_services.extend(all_services[stype][:5])

    # Run federated training
    coordinator.run_federated_training(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        test_services=test_services
    )

    # ========== Phase 3: Final Validation ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: FINAL VALIDATION")
    logger.info("=" * 80)

    # Validate Pareto optimality
    logger.info("\nValidating Pareto optimality of global models...")
    pareto_metrics = server.validate_pareto_optimality(test_services)

    # Display cluster statistics
    logger.info("\nFinal Cluster Statistics:")
    cluster_stats = server.preference_clusterer.get_cluster_statistics()
    for cluster_id in range(NUM_PREFERENCE_CLUSTERS):
        logger.info(f"\nCluster {cluster_id}:")
        logger.info(f"  Members: {cluster_stats['cluster_sizes'].get(cluster_id, 0)} clients")
        if cluster_id in cluster_stats['centroids']:
            logger.info(f"  Centroid: {cluster_stats['centroids'][cluster_id]['formatted']}")
        if cluster_id in pareto_metrics:
            logger.info(f"  Pareto ratio: {pareto_metrics[cluster_id]['pareto_ratio']:.1%}")
            logger.info(f"  Hypervolume: {pareto_metrics[cluster_id]['hypervolume']:.4f}")

    # ========== Phase 4: Service Composition ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 4: PARETO-OPTIMAL SERVICE COMPOSITION")
    logger.info("=" * 80)

    # Select a client for composition demo
    demo_client = clients[0]
    logger.info(f"\nUsing {demo_client.client_id} for composition demo")
    logger.info(f"Preferences: {demo_client._format_preference(demo_client.preference_vector)}")

    # Create composer
    composer = ParetoServiceComposer(demo_client)

    # Prepare candidate services for composition
    composition_candidates = {
        stype: all_services[stype][:10]  # Use first 10 of each type
        for stype in service_types[:3]  # Use first 3 types
    }

    logger.info(f"\nComposing workflow: {list(composition_candidates.keys())}")

    # Compose with constraints
    constraints = {
        'max_latency': 150,
        'max_cost': 1.5,
        'min_availability': 0.85
    }

    logger.info(f"Constraints: {constraints}")

    composition = composer.compose_services(
        service_types=list(composition_candidates.keys()),
        available_services=composition_candidates,
        qos_constraints=constraints
    )

    # Display selected composition
    logger.info("\n=== Selected Composition ===")
    for service in composition:
        preds = composer.predict_all_objectives(service)
        logger.info(f"\n{service.service_id} ({service.service_type}):")
        logger.info(f"  Actual QoS:")
        logger.info(f"    latency={service.qos.response_time:.1f}, "
                   f"cost={service.qos.cost:.2f}, "
                   f"avail={service.qos.availability:.3f}")
        logger.info(f"  Predicted QoS:")
        logger.info(f"    latency={preds['latency']:.1f}, "
                   f"cost={preds['cost']:.2f}, "
                   f"avail={preds['availability']:.3f}")

    # Evaluate composition
    evaluation = composer.evaluate_composition(composition)
    logger.info("\n=== Composition Evaluation ===")
    logger.info(f"Total latency: {evaluation['total_latency']:.2f} ms")
    logger.info(f"Total cost: {evaluation['total_cost']:.2f}")
    logger.info(f"Avg availability: {evaluation['avg_availability']:.3f}")
    logger.info(f"Avg reliability: {evaluation['avg_reliability']:.3f}")
    logger.info(f"Avg throughput: {evaluation['avg_throughput']:.2f}")

    score = composer.compute_composition_score(composition)
    logger.info(f"\nComposition Score: {score:.4f}")

    # ========== Phase 5: Preference Learning Demo ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 5: PREFERENCE LEARNING FROM HISTORY")
    logger.info("=" * 80)

    logger.info(f"\nSimulating composition history for {demo_client.client_id}...")

    # Add compositions to history
    for _ in range(10):
        # Simulate user selections (prefer certain services)
        simulated_composition = [
            composition_candidates[stype][np.random.randint(0, 5)]
            for stype in list(composition_candidates.keys())
        ]
        demo_client.add_composition_to_history(simulated_composition)

    logger.info(f"Old preferences: {demo_client._format_preference(demo_client.preference_vector)}")

    # Learn preferences from history
    demo_client.learn_preferences_from_history()

    logger.info(f"New preferences: {demo_client._format_preference(demo_client.preference_vector)}")

    # ========== Summary ==========
    logger.info("\n" + "=" * 80)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 80)

    logger.info(f"""
Successfully demonstrated:
✓ Multi-objective neural network predictor with {len(demo_client.local_model.tasks)} objectives
✓ Preference-based client clustering ({NUM_PREFERENCE_CLUSTERS} clusters)
✓ Federated multi-objective learning ({NUM_ROUNDS} rounds, {NUM_CLIENTS} clients)
✓ Pareto-optimal service composition
✓ Preference learning from composition history

Key Results:
- Trained {NUM_CLIENTS} clients on {len(test_services)} test services
- Average Pareto ratio: {np.mean([m['pareto_ratio'] for m in pareto_metrics.values()]):.1%}
- Composed workflow with {len(composition)} services meeting all constraints
- Client preferences successfully learned from {len(demo_client.composition_history)} compositions
    """)

    logger.info("=" * 80)
    logger.info("DEMO COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
