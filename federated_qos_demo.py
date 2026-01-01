"""
Complete Demo of Federated Learning for QoS-aware Web Service Composition
Simulates multiple clients training collaboratively with a central server
"""

import numpy as np
import logging
from federated_qos_client import (
    FederatedQoSClient, WebService, QoSMetrics, ServiceComposer
)
from federated_qos_server import (
    FederatedQoSServer, FederatedTrainingCoordinator
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_synthetic_services(num_services: int, service_types: list) -> list:
    """Generate synthetic web services with varying QoS"""
    services = []
    
    for i in range(num_services):
        service_type = np.random.choice(service_types)
        
        # Generate QoS metrics with some variation
        qos = QoSMetrics(
            response_time=np.random.uniform(20, 200),  # ms
            throughput=np.random.uniform(50, 200),     # req/s
            availability=np.random.uniform(0.90, 0.99),
            reliability=np.random.uniform(0.90, 0.99),
            cost=np.random.uniform(0.1, 2.0)           # cost units
        )
        
        service = WebService(
            service_id=f"{service_type}_s{i}",
            service_type=service_type,
            qos=qos
        )
        services.append(service)
    
    return services


def distribute_services_to_clients(services: list, num_clients: int) -> dict:
    """Distribute services across clients (simulating different service providers)"""
    client_services = {f"client_{i}": [] for i in range(num_clients)}
    
    # Randomly assign services to clients
    for service in services:
        client_id = f"client_{np.random.randint(0, num_clients)}"
        client_services[client_id].append(service)
    
    return client_services


def demonstrate_federated_learning():
    """Main demonstration of federated QoS learning"""
    
    logger.info("="*80)
    logger.info("FEDERATED LEARNING FOR QoS-AWARE WEB SERVICE COMPOSITION")
    logger.info("="*80)

    # Configuration
    NUM_CLIENTS = 5
    NUM_SERVICES = 100
    SERVICE_TYPES = ["authentication", "payment", "database", "notification", "analytics"]
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 8

    logger.info(f"\nConfiguration:")
    logger.info(f"  Number of Clients: {NUM_CLIENTS}")
    logger.info(f"  Total Services: {NUM_SERVICES}")
    logger.info(f"  Service Types: {SERVICE_TYPES}")
    logger.info(f"  Training Rounds: {NUM_ROUNDS}")
    logger.info(f"  Local Epochs per Round: {LOCAL_EPOCHS}")
    
    # Generate synthetic services
    logger.info(f"\n{'-'*80}")
    logger.info("Step 1: Generating Synthetic Services")
    logger.info(f"{'-'*80}")
    all_services = generate_synthetic_services(NUM_SERVICES, SERVICE_TYPES)
    logger.info(f"Generated {len(all_services)} services")
    
    # Distribute services to clients
    logger.info(f"\n{'-'*80}")
    logger.info("Step 2: Distributing Services to Clients")
    logger.info(f"{'-'*80}")
    client_service_map = distribute_services_to_clients(all_services, NUM_CLIENTS)
    for client_id, services in client_service_map.items():
        logger.info(f"  {client_id}: {len(services)} services")
    
    # Initialize server
    logger.info(f"\n{'-'*80}")
    logger.info("Step 3: Initializing Federated Server")
    logger.info(f"{'-'*80}")
    server = FederatedQoSServer(num_clients=NUM_CLIENTS)
    server.initialize_global_model(input_dim=5, output_dim=1)
    
    # Initialize clients
    logger.info(f"\n{'-'*80}")
    logger.info("Step 4: Initializing Clients")
    logger.info(f"{'-'*80}")
    clients = []
    for i in range(NUM_CLIENTS):
        client_id = f"client_{i}"
        client = FederatedQoSClient(client_id, learning_rate=0.02)
        client.initialize_model()
        
        # Add services to client
        for service in client_service_map[client_id]:
            client.add_service(service)
        
        clients.append(client)
        logger.info(f"  Initialized {client_id} with {len(client.local_services)} services")
    
    # Run federated training
    logger.info(f"\n{'-'*80}")
    logger.info("Step 5: Running Federated Training")
    logger.info(f"{'-'*80}")
    coordinator = FederatedTrainingCoordinator(server, clients)
    
    # Train with standard FedAvg
    final_model = coordinator.run_federated_training(
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        aggregation_method='fedavg'
    )
    
    # Save model
    server.save_model('qos_federated_model.json')
    
    # Demonstrate service composition
    logger.info(f"\n{'-'*80}")
    logger.info("Step 6: Service Composition with Trained Model")
    logger.info(f"{'-'*80}")
    
    # Update all clients with final global model
    for client in clients:
        client.update_model(final_model)
    
    # Use first client for demonstration
    demo_client = clients[0]
    composer = ServiceComposer(demo_client)
    
    # Prepare available services by type
    available_services = {stype: [] for stype in SERVICE_TYPES}
    for service in all_services[:30]:  # Use subset for demo
        available_services[service.service_type].append(service)
    
    # Define composition requirements
    workflow = ["authentication", "database", "analytics", "notification"]
    constraints = {
        'max_response_time': 500,
        'min_availability': 0.92,
        'min_reliability': 0.92,
        'max_cost': 5.0
    }
    
    logger.info(f"\nWorkflow: {' -> '.join(workflow)}")
    logger.info(f"Constraints: {constraints}")
    
    # Compose services
    try:
        composition = composer.compose_services(workflow, available_services, constraints)
        
        logger.info(f"\nOptimal Service Composition:")
        for i, service in enumerate(composition, 1):
            score = demo_client.predict_qos_score(service)
            logger.info(f"  {i}. {service.service_id}")
            logger.info(f"     Type: {service.service_type}")
            logger.info(f"     QoS Score: {score:.4f}")
            logger.info(f"     Response Time: {service.qos.response_time:.2f} ms")
            logger.info(f"     Availability: {service.qos.availability:.2%}")
            logger.info(f"     Cost: ${service.qos.cost:.2f}")
        
        # Evaluate overall composition
        metrics = composer.evaluate_composition(composition)
        logger.info(f"\nComposition Quality Metrics:")
        logger.info(f"  Total Response Time: {metrics['total_response_time']:.2f} ms")
        logger.info(f"  Average Availability: {metrics['avg_availability']:.2%}")
        logger.info(f"  Average Reliability: {metrics['avg_reliability']:.2%}")
        logger.info(f"  Total Cost: ${metrics['total_cost']:.2f}")
        
    except Exception as e:
        logger.info(f"Error in composition: {e}")
    
    # Compare with different aggregation methods
    logger.info(f"\n{'-'*80}")
    logger.info("Step 7: Comparing Aggregation Methods")
    logger.info(f"{'-'*80}")
    
    # Reset and try adaptive aggregation
    server2 = FederatedQoSServer(num_clients=NUM_CLIENTS)
    server2.initialize_global_model()
    
    clients2 = []
    for i in range(NUM_CLIENTS):
        client_id = f"client_{i}"
        client = FederatedQoSClient(client_id, learning_rate=0.02)
        client.initialize_model()
        for service in client_service_map[client_id]:
            client.add_service(service)
        clients2.append(client)
    
    coordinator2 = FederatedTrainingCoordinator(server2, clients2)
    logger.info("\nTraining with Adaptive FedAvg:")
    coordinator2.run_federated_training(
        num_rounds=10,
        local_epochs=LOCAL_EPOCHS,
        aggregation_method='adaptive'
    )
    
    # Reset and try secure aggregation
    server3 = FederatedQoSServer(num_clients=NUM_CLIENTS)
    server3.initialize_global_model()
    
    clients3 = []
    for i in range(NUM_CLIENTS):
        client_id = f"client_{i}"
        client = FederatedQoSClient(client_id, learning_rate=0.02)
        client.initialize_model()
        for service in client_service_map[client_id]:
            client.add_service(service)
        clients3.append(client)
    
    coordinator3 = FederatedTrainingCoordinator(server3, clients3)
    logger.info("\nTraining with Secure Aggregation (Differential Privacy):")
    coordinator3.run_federated_training(
        num_rounds=10,
        local_epochs=LOCAL_EPOCHS,
        aggregation_method='secure'
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("DEMONSTRATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info("\nKey Features Demonstrated:")
    logger.info("  ✓ Federated learning across multiple clients")
    logger.info("  ✓ Privacy-preserving model aggregation")
    logger.info("  ✓ QoS-aware service composition")
    logger.info("  ✓ Multiple aggregation strategies (FedAvg, Adaptive, Secure)")
    logger.info("  ✓ Constraint-based service selection")
    logger.info("  ✓ Model persistence and loading")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_federated_learning()
