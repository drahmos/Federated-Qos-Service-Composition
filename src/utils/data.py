"""
Data generation and loading utilities for QoS-aware service composition
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class Service:
    """Represents a web service"""
    id: int
    name: str
    domain: str
    functionality: str
    input_schema: List[str]
    output_schema: List[str]
    qos_values: np.ndarray  # QoS attributes
    qos_history: List[np.ndarray]  # Historical QoS values
    
    def __repr__(self):
        return f"Service({self.id}, {self.name}, domain={self.domain})"


@dataclass
class Workflow:
    """Represents a service workflow (composition template)"""
    id: int
    name: str
    domain: str
    nodes: List[Dict]  # Service nodes with positions
    edges: List[Tuple[int, int]]  # Dependencies
    requirements: np.ndarray  # QoS requirements


class QoSAttribute:
    """QoS attribute definitions"""
    RESPONSE_TIME = 0
    THROUGHPUT = 1
    AVAILABILITY = 2
    RELIABILITY = 3
    COST = 4
    
    @staticmethod
    def get_name(idx: int) -> str:
        names = ["response_time", "throughput", "availability", "reliability", "cost"]
        return names[idx]
    
    @staticmethod
    def get_optimization_direction(idx: int) -> str:
        """Returns 'min' or 'max' for optimization direction"""
        if idx in [QoSAttribute.RESPONSE_TIME, QoSAttribute.COST]:
            return "min"
        else:
            return "max"


class DataGenerator:
    """Generates synthetic QoS data and service information"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_service(
        self,
        service_id: int,
        domain: str,
        functionality: str,
        qos_base: Optional[np.ndarray] = None
    ) -> Service:
        """Generate a service with QoS values"""
        
        # Default QoS base values if not provided
        if qos_base is None:
            qos_base = np.array([
                np.random.uniform(100, 2000),    # response_time (ms)
                np.random.uniform(10, 1000),      # throughput (req/s)
                np.random.uniform(0.90, 0.999),   # availability
                np.random.uniform(0.85, 0.995),   # reliability
                np.random.uniform(0.01, 1.0)     # cost ($)
            ])
        
        # Add domain-specific variation
        domain_factor = self._get_domain_factor(domain)
        qos_values = qos_base * domain_factor
        
        # Generate QoS history (100 historical points)
        qos_history = []
        for _ in range(100):
            noise = np.random.normal(0, 0.05, size=qos_values.shape)
            historical_qos = qos_values * (1 + noise)
            historical_qos = np.clip(historical_qos, 0, None)
            qos_history.append(historical_qos)
        
        return Service(
            id=service_id,
            name=f"{domain}_{functionality}_{service_id}",
            domain=domain,
            functionality=functionality,
            input_schema=[f"param_{i}" for i in range(np.random.randint(1, 5))],
            output_schema=[f"result_{i}" for i in range(np.random.randint(1, 3))],
            qos_values=qos_values,
            qos_history=qos_history
        )
    
    def _get_domain_factor(self, domain: str) -> np.ndarray:
        """Domain-specific QoS variation factors"""
        factors = {
            "healthcare": np.array([1.2, 0.8, 1.1, 1.2, 1.3]),
            "fintech": np.array([0.9, 1.2, 1.15, 1.15, 1.5]),
            "ecommerce": np.array([1.0, 1.0, 0.95, 0.95, 0.8]),
            "iot": np.array([1.5, 0.6, 0.9, 0.9, 0.5]),
            "travel": np.array([1.1, 0.9, 0.98, 0.98, 1.1]),
            "education": np.array([1.0, 0.85, 0.92, 0.92, 0.7])
        }
        return factors.get(domain, np.ones(5))
    
    def generate_domain_services(
        self,
        domain: str,
        num_services: int,
        functionalities: Optional[List[str]] = None
    ) -> List[Service]:
        """Generate all services for a domain"""
        
        if functionalities is None:
            functionalities = [
                "authentication", "data_processing", "payment",
                "notification", "analytics", "storage", "search",
                "recommendation", "reporting", "api_gateway"
            ]
        
        services = []
        services_per_func = num_services // len(functionalities)
        
        service_id = 0
        for func in functionalities:
            for _ in range(services_per_func):
                qos_base = self._get_functionality_qos_base(func)
                service = self.generate_service(service_id, domain, func, qos_base)
                services.append(service)
                service_id += 1
        
        # Add remaining services
        while len(services) < num_services:
            func = np.random.choice(functionalities)
            qos_base = self._get_functionality_qos_base(func)
            service = self.generate_service(service_id, domain, func, qos_base)
            services.append(service)
            service_id += 1
        
        return services
    
    def _get_functionality_qos_base(self, functionality: str) -> np.ndarray:
        """Base QoS values for different functionalities"""
        base_values = {
            "authentication": np.array([200, 500, 0.99, 0.98, 0.1]),
            "data_processing": np.array([800, 200, 0.95, 0.94, 0.5]),
            "payment": np.array([500, 100, 0.999, 0.999, 0.2]),
            "notification": np.array([100, 1000, 0.98, 0.97, 0.05]),
            "analytics": np.array([1500, 50, 0.92, 0.91, 1.0]),
            "storage": np.array([300, 800, 0.995, 0.99, 0.15]),
            "search": np.array([400, 400, 0.97, 0.96, 0.08]),
            "recommendation": np.array([600, 300, 0.94, 0.93, 0.3]),
            "reporting": np.array([1000, 100, 0.93, 0.92, 0.4]),
            "api_gateway": np.array([150, 800, 0.998, 0.995, 0.12])
        }
        return base_values.get(functionality, np.array([500, 300, 0.95, 0.94, 0.5]))
    
    def generate_workflow(
        self,
        workflow_id: int,
        domain: str,
        services: List[Service],
        num_nodes: int
    ) -> Workflow:
        """Generate a workflow (composition template)"""
        
        # Select services for the workflow
        selected_indices = np.random.choice(len(services), min(num_nodes, len(services)), replace=False)
        selected_services = [services[i] for i in selected_indices]
        
        # Create nodes
        nodes = []
        for i, service in enumerate(selected_services):
            nodes.append({
                "id": i,
                "service_id": service.id,
                "service_name": service.name,
                "position": i
            })
        
        # Create edges (sequential + some parallel branches)
        edges = []
        for i in range(len(nodes) - 1):
            if np.random.random() > 0.3:  # 70% sequential connection
                edges.append((i, i + 1))
        
        # Add some parallel branches
        if len(nodes) > 3:
            for _ in range(np.random.randint(1, 3)):
                src = np.random.randint(0, len(nodes) - 2)
                dst = src + np.random.randint(2, min(4, len(nodes) - src))
                if (src, dst) not in edges:
                    edges.append((src, dst))
        
        # Generate QoS requirements
        requirements = self._generate_requirements(selected_services)
        
        return Workflow(
            id=workflow_id,
            name=f"{domain}_workflow_{workflow_id}",
            domain=domain,
            nodes=nodes,
            edges=edges,
            requirements=requirements
        )
    
    def _generate_requirements(self, services: List[Service]) -> np.ndarray:
        """Generate QoS requirements based on selected services"""
        
        # Compute aggregated QoS
        agg_qos = np.array([s.qos_values for s in services])
        
        # Requirements based on mean service QoS (achievable targets)
        requirements = agg_qos.mean(axis=0)
        
        # Make requirements slightly challenging but achievable
        # These adjustments make it possible to meet requirements with good service selection
        requirements[0] *= 0.95  # response_time: 5% better (achievable)
        requirements[1] *= 1.05  # throughput: 5% higher (achievable)
        requirements[2] = max(0.90, min(0.99, agg_qos.mean(axis=0)[2]))  # availability: realistic target
        requirements[3] = max(0.88, min(0.98, agg_qos.mean(axis=0)[3]))  # reliability: realistic target
        requirements[4] *= 1.05  # cost: 5% higher budget (easier to satisfy)
        
        return requirements
    
    def generate_composition_dataset(
        self,
        domains_config: Dict[str, int],
        num_workflows_per_domain: int = 30,
        seed: int = 42
    ) -> Tuple[Dict[str, List[Service]], Dict[str, List[Workflow]]]:
        """Generate complete dataset of services and workflows"""
        
        np.random.seed(seed)
        
        # Generate services for each domain
        all_services = {}
        all_workflows = {}
        
        for domain, num_services in domains_config.items():
            print(f"Generating {num_services} services for domain: {domain}")
            services = self.generate_domain_services(domain, num_services)
            all_services[domain] = services
            
            # Generate workflows
            print(f"Generating {num_workflows_per_domain} workflows for domain: {domain}")
            workflows = []
            for i in range(num_workflows_per_domain):
                num_nodes = np.random.randint(3, 8)
                workflow = self.generate_workflow(i, domain, services, num_nodes)
                workflows.append(workflow)
            all_workflows[domain] = workflows
        
        return all_services, all_workflows
    
    def save_dataset(
        self,
        services: Dict[str, List[Service]],
        workflows: Dict[str, List[Workflow]],
        filepath: str
    ):
        """Save dataset to file"""
        data = {
            "services": {
                domain: [
                    {
                        "id": s.id,
                        "name": s.name,
                        "domain": s.domain,
                        "functionality": s.functionality,
                        "input_schema": s.input_schema,
                        "output_schema": s.output_schema,
                        "qos_values": s.qos_values.tolist(),
                        "qos_history": [h.tolist() for h in s.qos_history]
                    }
                    for s in services_list
                ]
                for domain, services_list in services.items()
            },
            "workflows": {
                domain: [
                    {
                        "id": w.id,
                        "name": w.name,
                        "domain": w.domain,
                        "nodes": w.nodes,
                        "edges": w.edges,
                        "requirements": w.requirements.tolist()
                    }
                    for w in workflow_list
                ]
                for domain, workflow_list in workflows.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_dataset(self, filepath: str) -> Tuple[Dict[str, List[Service]], Dict[str, List[Workflow]]]:
        """Load dataset from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        services = {}
        for domain, services_data in data["services"].items():
            services[domain] = [
                Service(
                    id=s["id"],
                    name=s["name"],
                    domain=s["domain"],
                    functionality=s["functionality"],
                    input_schema=s["input_schema"],
                    output_schema=s["output_schema"],
                    qos_values=np.array(s["qos_values"]),
                    qos_history=[np.array(h) for h in s["qos_history"]]
                )
                for s in services_data
            ]
        
        workflows = {}
        for domain, workflows_data in data["workflows"].items():
            workflows[domain] = [
                Workflow(
                    id=w["id"],
                    name=w["name"],
                    domain=w["domain"],
                    nodes=w["nodes"],
                    edges=[tuple(e) for e in w["edges"]],
                    requirements=np.array(w["requirements"])
                )
                for w in workflows_data
            ]
        
        return services, workflows


class CompositionExecutor:
    """Executes service compositions and computes actual QoS"""
    
    @staticmethod
    def compute_composite_qos(
        composition: Dict,
        services: Dict[int, Service],
        edges: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Compute composite QoS for a given composition"""
        
        # Get services in composition
        composition_services = [
            services[node["service_id"]]
            for node in composition["nodes"]
        ]
        
        # Aggregate QoS based on workflow structure
        if len(edges) == len(composition_services) - 1:
            # Sequential composition
            composite_qos = np.zeros(5)
            for service in composition_services:
                composite_qos[0] += service.qos_values[0]  # response_time (sum)
                composite_qos[1] = min(composite_qos[1], service.qos_values[1]) if composite_qos[1] > 0 else service.qos_values[1]  # throughput (min)
                composite_qos[2] *= service.qos_values[2]  # availability (product)
                composite_qos[3] *= service.qos_values[3]  # reliability (product)
                composite_qos[4] += service.qos_values[4]  # cost (sum)
        else:
            # Mixed composition (parallel + sequential)
            # Simplified: use weighted average
            qos_matrix = np.array([s.qos_values for s in composition_services])
            composite_qos = qos_matrix.mean(axis=0)
        
        return composite_qos
    
    @staticmethod
    def compute_actual_qos(
        predicted_qos: np.ndarray,
        noise_level: float = 0.1
    ) -> np.ndarray:
        """Simulate actual QoS with noise"""
        noise = np.random.normal(0, noise_level, size=predicted_qos.shape)
        actual = predicted_qos * (1 + noise)
        return np.clip(actual, 0, None)
    
    @staticmethod
    def check_requirements(
        actual_qos: np.ndarray,
        requirements: np.ndarray
    ) -> bool:
        """Check if composition meets requirements"""
        for i, (actual, req) in enumerate(zip(actual_qos, requirements)):
            direction = QoSAttribute.get_optimization_direction(i)
            if direction == "min":
                if actual > req * 1.1:  # Allow 10% tolerance
                    return False
            else:
                if actual < req * 0.9:  # Allow 10% tolerance
                    return False
        return True


if __name__ == "__main__":
    # Test data generation
    generator = DataGenerator(seed=42)
    
    # Generate sample dataset
    domains_config = {
        "healthcare": 20,
        "fintech": 25,
        "ecommerce": 30
    }
    
    services, workflows = generator.generate_composition_dataset(
        domains_config,
        num_workflows_per_domain=10
    )
    
    print("\n=== Sample Services ===")
    for domain, services_list in services.items():
        print(f"\n{domain}: {len(services_list)} services")
        print(f"  First service: {services_list[0]}")
        print(f"  QoS: {services_list[0].qos_values}")
    
    print("\n=== Sample Workflows ===")
    for domain, workflow_list in workflows.items():
        print(f"\n{domain}: {len(workflow_list)} workflows")
        print(f"  First workflow: {workflow_list[0].name}")
        print(f"  Nodes: {len(workflow_list[0].nodes)}")
        print(f"  Requirements: {workflow_list[0].requirements}")
    
    # Save dataset
    generator.save_dataset(services, workflows, "data/dataset.json")
    print("\nDataset saved to data/dataset.json")
