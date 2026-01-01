"""
Baseline methods for comparison with meta-learning approach
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
import random


class BaselineComposer(ABC):
    """Abstract base class for composition baselines"""
    
    @abstractmethod
    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Select services for composition"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get method name"""
        pass


class RandomComposer(BaselineComposer):
    """Random service selection as baseline"""
    
    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Randomly select services for each workflow node"""
        num_nodes = len(workflow.nodes)
        selected = []
        
        for i in range(num_nodes):
            idx = random.randint(0, len(available_services) - 1)
            selected.append(available_services[idx].id)
        
        return selected
    
    def get_name(self) -> str:
        return "Random"


class GreedyComposer(BaselineComposer):
    """Greedy selection based on individual service QoS"""
    
    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Greedy selection of best services per position"""
        selected = []
        
        # Score each service
        service_scores = []
        for service in available_services:
            score = self._compute_service_score(service, requirements)
            service_scores.append((score, service))
        
        # Sort by score
        service_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select top services
        num_nodes = min(len(workflow.nodes), len(available_services))
        for i in range(num_nodes):
            selected.append(service_scores[i][1].id)
        
        return selected
    
    def _compute_service_score(self, service, requirements: np.ndarray) -> float:
        """Compute score for a single service"""
        qos = service.qos_values
        score = 0.0
        
        # For minimization attributes (response_time, cost)
        if qos[0] <= requirements[0]:
            score += 1.0
        if qos[4] <= requirements[4]:
            score += 1.0
        
        # For maximization attributes
        if qos[1] >= requirements[1]:
            score += 1.0
        if qos[2] >= requirements[2]:
            score += 1.0
        if qos[3] >= requirements[3]:
            score += 1.0
        
        return score / 5.0
    
    def get_name(self) -> str:
        return "Greedy"


class GeneticAlgorithmComposer(BaselineComposer):
    """Genetic Algorithm for service composition optimization"""
    
    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Run GA to find optimal composition"""
        num_nodes = len(workflow.nodes)
        
        # Initialize population
        population = self._initialize_population(
            population_size=self.population_size,
            num_nodes=num_nodes,
            available_services=available_services
        )
        
        # Evolve
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = [
                self._evaluate_fitness(individual, available_services, workflow, requirements)
                for individual in population
            ]
            
            # Selection
            selected = self._selection(population, fitness)
            
            # Crossover
            offspring = self._crossover(selected)
            
            # Mutation
            offspring = self._mutate(offspring, available_services)
            
            # Elitism: keep best individual
            best_idx = np.argmax(fitness)
            offspring[0] = population[best_idx]
            
            population = offspring
        
        # Return best solution
        fitness = [
            self._evaluate_fitness(individual, available_services, workflow, requirements)
            for individual in population
        ]
        best_idx = np.argmax(fitness)
        
        return population[best_idx]
    
    def _initialize_population(
        self,
        population_size: int,
        num_nodes: int,
        available_services: List
    ) -> List[List[int]]:
        """Initialize random population"""
        population = []
        num_services = len(available_services)
        
        for _ in range(population_size):
            individual = [
                random.randint(0, num_services - 1) for _ in range(num_nodes)
            ]
            population.append(individual)
        
        return population
    
    def _evaluate_fitness(
        self,
        individual: List[int],
        available_services: List,
        workflow,
        requirements: np.ndarray
    ) -> float:
        """Evaluate fitness of an individual"""
        # Get selected services
        selected_services = [available_services[idx] for idx in individual]
        
        # Compute composite QoS (simplified sequential)
        composite_qos = np.array([
            sum(s.qos_values[i] for s in selected_services)
            for i in range(5)
        ])
        
        # Compute fitness
        fitness = 0.0
        
        # Penalty for violating requirements
        for i in range(5):
            if i in [0, 4]:  # Minimization
                if composite_qos[i] > requirements[i]:
                    penalty = (composite_qos[i] - requirements[i]) / requirements[i]
                    fitness -= penalty
                else:
                    fitness += 0.2
            else:  # Maximization
                if composite_qos[i] < requirements[i]:
                    penalty = (requirements[i] - composite_qos[i]) / requirements[i]
                    fitness -= penalty
                else:
                    fitness += 0.2
        
        return fitness
    
    def _selection(
        self,
        population: List[List[int]],
        fitness: List[float]
    ) -> List[List[int]]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Select tournament participants
            participants = random.sample(
                list(zip(population, fitness)),
                tournament_size
            )
            
            # Select winner
            winner = max(participants, key=lambda x: x[1])[0]
            selected.append(winner)
        
        return selected
    
    def _crossover(
        self,
        parents: List[List[int]]
    ) -> List[List[int]]:
        """Uniform crossover"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            if random.random() < self.crossover_rate:
                child1, child2 = self._uniform_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])
        
        return offspring
    
    def _uniform_crossover(
        self,
        parent1: List[int],
        parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Uniform crossover operation"""
        child1 = []
        child2 = []
        
        for g1, g2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(g1)
                child2.append(g2)
            else:
                child1.append(g2)
                child2.append(g1)
        
        return child1, child2
    
    def _mutate(
        self,
        population: List[List[int]],
        available_services: List
    ) -> List[List[int]]:
        """Mutation operation"""
        num_services = len(available_services)
        
        for individual in population:
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
                    individual[i] = random.randint(0, num_services - 1)
        
        return population
    
    def get_name(self) -> str:
        return "Genetic Algorithm"


class TransferLearningComposer(BaselineComposer):
    """Transfer learning: fine-tune on new domain without meta-learning"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        training_steps: int = 100
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim)
        )
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.is_trained = False
    
    def pre_train(
        self,
        source_domain_data: List[Tuple[torch.Tensor, int, float]]
    ):
        """Pre-train on source domain"""
        print(f"Pre-training on source domain with {len(source_domain_data)} samples...")
        
        for step in range(self.training_steps):
            # Sample batch
            batch = random.sample(source_domain_data, min(32, len(source_domain_data)))
            states = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch])
            rewards = torch.tensor([b[2] for b in batch])
            
            # Forward pass
            logits = self.policy(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(actions)), actions]
            
            # Policy loss
            loss = -(selected_log_probs * rewards).mean()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.is_trained = True
        print("Pre-training complete!")
    
    def fine_tune(
        self,
        target_domain_data: List[Tuple[torch.Tensor, int, float]]
    ):
        """Fine-tune on new domain"""
        print(f"Fine-tuning on target domain with {len(target_domain_data)} samples...")
        
        for step in range(min(self.training_steps // 2, len(target_domain_data))):
            # Sample batch
            batch = random.sample(target_domain_data, min(32, len(target_domain_data)))
            states = torch.stack([b[0] for b in batch])
            actions = torch.tensor([b[1] for b in batch])
            rewards = torch.tensor([b[2] for b in batch])
            
            # Forward pass
            logits = self.policy(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(len(actions)), actions]
            
            # Policy loss
            loss = -(selected_log_probs * rewards).mean()
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        print("Fine-tuning complete!")
    
    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Select services using learned policy"""
        if not self.is_trained:
            return self._random_compose(workflow, available_services)
        
        selected = []
        
        for node in workflow.nodes:
            # Create state representation
            state = self._create_state(workflow, node, requirements, len(available_services))
            
            # Get action
            with torch.no_grad():
                logits = self.policy(state)
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.argmax(probs).item())
            
            # Select service
            action = min(action, len(available_services) - 1)
            selected.append(available_services[action].id)
        
        return selected
    
    def _create_state(
        self,
        workflow,
        node,
        requirements: np.ndarray,
        num_services: int
    ) -> torch.Tensor:
        """Create state representation (12-dim to match MAML)"""
        state_features = []

        pos_feature = np.zeros(1)
        pos_feature[0] = node["position"] / 10.0
        state_features.append(pos_feature)

        state_features.append(requirements / 1000.0)

        state_features.append(np.array([num_services / 100.0]))

        state = np.concatenate(state_features)
        return torch.FloatTensor(state)
    
    def _random_compose(
        self,
        workflow,
        available_services: List
    ) -> List[int]:
        """Fallback to random composition"""
        num_nodes = len(workflow.nodes)
        return [
            random.choice(available_services).id
            for _ in range(num_nodes)
        ]
    
    def get_name(self) -> str:
        return "Transfer Learning"


class MultiTaskLearningComposer(BaselineComposer):
    """Multi-task learning: train on multiple domains jointly"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_domains: int = 3,
        learning_rate: float = 0.001
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_domains = num_domains
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Domain-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(256, action_dim) for _ in range(num_domains)
        ])
        
        self.optimizer = optim.Adam(
            list(self.shared.parameters()) + list(self.heads.parameters()),
            lr=learning_rate
        )
    
    def train(
        self,
        multi_domain_data: Dict[int, List[Tuple[torch.Tensor, int, float]]]
    ):
        """Train on multiple domains"""
        print(f"Training MTL on {len(multi_domain_data)} domains...")
        
        for step in range(100):
            domain_losses = []
            
            for domain_id, domain_data in multi_domain_data.items():
                # Sample batch
                batch = random.sample(domain_data, min(32, len(domain_data)))
                states = torch.stack([b[0] for b in batch])
                actions = torch.tensor([b[1] for b in batch])
                rewards = torch.tensor([b[2] for b in batch])
                
                # Forward pass through shared layers
                shared_features = self.shared(states)
                
                # Forward through domain-specific head
                logits = self.heads[domain_id](shared_features)
                log_probs = torch.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs[torch.arange(len(actions)), actions]
                
                # Domain-specific loss
                loss = -(selected_log_probs * rewards).mean()
                domain_losses.append(loss)
            
            # Average loss across domains and update
            total_loss = torch.stack(domain_losses).mean()
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        print("MTL training complete!")
    
    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray,
        domain_id: int = 0
    ) -> List[int]:
        """Select services using MTL model"""
        selected = []
        
        for node in workflow.nodes:
            # Create state representation
            state = self._create_state(workflow, node, requirements, len(available_services))
            
            # Get action from domain-specific head
            with torch.no_grad():
                shared_features = self.shared(state)
                logits = self.heads[domain_id](shared_features)
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.argmax(probs).item())
            
            # Select service
            action = min(action, len(available_services) - 1)
            selected.append(available_services[action].id)
        
        return selected
    
    def _create_state(
        self,
        workflow,
        node,
        requirements: np.ndarray,
        num_services: int
    ) -> torch.Tensor:
        """Create state representation (12-dim to match MAML)"""
        state_features = []

        pos_feature = np.zeros(1)
        pos_feature[0] = node["position"] / 10.0
        state_features.append(pos_feature)

        state_features.append(requirements / 1000.0)

        state_features.append(np.array([num_services / 100.0]))

        state = np.concatenate(state_features)
        return torch.FloatTensor(state)
    
    def get_name(self) -> str:
        return "Multi-Task Learning"


if __name__ == "__main__":
    # Test baseline methods
    print("Testing baseline methods...")
    
    from src.utils.data import DataGenerator, Workflow, Service
    from src.utils.data import CompositionExecutor
    
    # Generate test data
    generator = DataGenerator(seed=42)
    services = generator.generate_domain_services("test", 20)
    
    # Create a simple workflow
    workflow = Workflow(
        id=0,
        name="test_workflow",
        domain="test",
        nodes=[{"id": i, "service_id": services[i].id, "position": i} for i in range(5)],
        edges=[(i, i+1) for i in range(4)],
        requirements=np.array([500.0, 200.0, 0.97, 0.96, 1.5])
    )
    
    # Test Random Composer
    random_composer = RandomComposer()
    composition = random_composer.compose(workflow, services, workflow.requirements)
    print(f"Random composer selected: {composition}")
    
    # Test Greedy Composer
    greedy_composer = GreedyComposer()
    composition = greedy_composer.compose(workflow, services, workflow.requirements)
    print(f"Greedy composer selected: {composition}")
    
    # Test Genetic Algorithm
    ga_composer = GeneticAlgorithmComposer(
        population_size=20,
        generations=50
    )
    composition = ga_composer.compose(workflow, services, workflow.requirements)
    print(f"GA composer selected: {composition}")
    
    # Test Transfer Learning
    tl_composer = TransferLearningComposer(
        state_dim=16,
        action_dim=20
    )
    
    # Create dummy training data
    source_data = [
        (torch.randn(16), np.random.randint(0, 20), np.random.random())
        for _ in range(100)
    ]
    
    tl_composer.pre_train(source_data)
    composition = tl_composer.compose(workflow, services, workflow.requirements)
    print(f"TL composer selected: {composition}")
    
    # Test Multi-Task Learning
    mtl_composer = MultiTaskLearningComposer(
        state_dim=16,
        action_dim=20,
        num_domains=3
    )
    
    multi_domain_data = {
        0: [(torch.randn(16), np.random.randint(0, 20), np.random.random()) for _ in range(100)],
        1: [(torch.randn(16), np.random.randint(0, 20), np.random.random()) for _ in range(100)],
        2: [(torch.randn(16), np.random.randint(0, 20), np.random.random()) for _ in range(100)]
    }
    
    mtl_composer.train(multi_domain_data)
    composition = mtl_composer.compose(workflow, services, workflow.requirements, domain_id=0)
    print(f"MTL composer selected: {composition}")
    
    print("\nAll baseline methods tested successfully!")
