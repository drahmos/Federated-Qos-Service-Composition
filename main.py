"""
Main evaluation script for Meta-QoS Composition Framework
"""
import sys
import os
import time
import torch
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.config import config, TrainingConfig
from utils.data import (
    DataGenerator, Service, Workflow, CompositionExecutor, QoSAttribute
)
from utils.metrics import (
    MetricsCalculator, ResultsAggregator, CompositionResult
)
from utils.visualization import ResultsVisualizer

from core.meta_learning.maml_engine import MAMLComposer
from core.meta_learning.baselines import (
    RandomComposer, GreedyComposer, GeneticAlgorithmComposer,
    TransferLearningComposer, MultiTaskLearningComposer
)


class Evaluator:
    """Main evaluation class for comparing all methods"""
    
    def __init__(self, cfg: TrainingConfig):
        self.config = cfg
        self.data_generator = DataGenerator(seed=cfg.seed)
        self.composition_executor = CompositionExecutor()
        self.visualizer = ResultsVisualizer(output_dir=cfg.plot_dir)
        self.results_aggregator = ResultsAggregator()
        
        # Store results
        self.all_results = {}
        self.training_curves = {}
        self.adaptation_times = {}
        self.cold_start_performance = {}
    
    def generate_datasets(self):
        """Generate training and test datasets"""
        print("=" * 60)
        print("Generating Datasets")
        print("=" * 60)
        
        # Generate datasets for each domain
        domains_config = {}
        for domain in self.config.train_domains + self.config.val_domains + self.config.test_domains:
            num_services = np.random.randint(15, 30)
            domains_config[domain] = num_services
        
        self.all_services, self.all_workflows = \
            self.data_generator.generate_composition_dataset(
                domains_config,
                num_workflows_per_domain=25,
                seed=self.config.seed
            )
        
        print(f"\nGenerated datasets:")
        for domain, services in self.all_services.items():
            print(f"  {domain}: {len(services)} services, "
                  f"{len(self.all_workflows[domain])} workflows")
    
    def prepare_training_data(
        self,
        domain: str,
        num_samples: int = 1000
    ) -> List[torch.Tensor]:
        """Prepare training data for RL/meta-learning methods"""
        training_data = []
        
        workflows = self.all_workflows[domain]
        services = self.all_services[domain]
        
        for _ in range(num_samples):
            # Sample random workflow
            workflow = np.random.choice(workflows)
            
            # Randomly select services
            selected_indices = np.random.choice(
                len(services),
                size=min(len(workflow.nodes), len(services)),
                replace=False
            )
            selected_services = [services[i] for i in selected_indices]
            
            # Compute composite QoS
            composite_qos = self.composition_executor.compute_composite_qos(
                {"nodes": workflow.nodes},
                {s.id: s for s in services},
                workflow.edges
            )
            
            # Compute reward based on requirements
            reward = self._compute_reward(
                composite_qos,
                workflow.requirements
            )
            
            # Create state representation
            state = self._create_state(workflow, composite_qos, len(services))
            
            # Create action (service selection)
            action = selected_indices[0]
            
            training_data.append((state, action, reward))
        
        return training_data
    
    def _create_state(
        self,
        workflow: Workflow,
        qos: np.ndarray,
        num_services: int
    ) -> torch.Tensor:
        """Create state tensor (7-dim: 1 pos + 5 req + 1 num_services)"""
        state_features = []

        pos_feature = np.array([0.5])  # Use fixed position for training
        state_features.append(pos_feature)

        state_features.append(workflow.requirements / 1000.0)

        state_features.append(np.array([num_services / 100.0]))

        state = np.concatenate(state_features)
        return torch.FloatTensor(state)
    
    def _compute_reward(
        self,
        qos: np.ndarray,
        requirements: np.ndarray
    ) -> float:
        """Compute reward based on QoS satisfaction"""
        reward = 0.0
        
        for i in range(5):
            direction = QoSAttribute.get_optimization_direction(i)
            if direction == "min":
                if qos[i] <= requirements[i]:
                    reward += 1.0
                else:
                    reward -= (qos[i] - requirements[i]) / requirements[i]
            else:
                if qos[i] >= requirements[i]:
                    reward += 1.0
                else:
                    reward -= (requirements[i] - qos[i]) / requirements[i]
        
        return max(reward, -1.0)
    
    def evaluate_method(
        self,
        method_name: str,
        composer,
        test_domain: str,
        num_compositions: int = 50
    ) -> List[CompositionResult]:
        """Evaluate a composition method"""
        print(f"\nEvaluating {method_name} on {test_domain}...")
        results = []
        start_time = time.time()
        
        workflows = self.all_workflows[test_domain]
        services = self.all_services[test_domain]
        
        for i in range(min(num_compositions, len(workflows))):
            workflow = workflows[i]
            
            try:
                # Select services
                selected_ids = composer.compose(
                    workflow,
                    services,
                    workflow.requirements
                )
                
                # Get selected services
                selected_services = []
                for sid in selected_ids:
                    found = False
                    for s in services:
                        if s.id == sid:
                            selected_services.append(s)
                            found = True
                            break
                    if not found and len(services) > 0:
                        selected_services.append(services[0])
                
                # Compute composite QoS
                predicted_qos = self.composition_executor.compute_composite_qos(
                    {"nodes": workflow.nodes},
                    {s.id: s for s in services},
                    workflow.edges
                )
                
                # Simulate actual QoS with noise
                actual_qos = self.composition_executor.compute_actual_qos(
                    predicted_qos,
                    noise_level=0.1
                )
                
                # Check requirements
                satisfied = self.composition_executor.check_requirements(
                    actual_qos,
                    workflow.requirements
                )
                
                # Compute cost
                cost = sum(s.qos_values[4] for s in selected_services)
                
                results.append(CompositionResult(
                    composition_id=f"{test_domain}_{i}",
                    domain=test_domain,
                    selected_services=selected_ids,
                    predicted_qos=predicted_qos,
                    actual_qos=actual_qos,
                    requirements=workflow.requirements,
                    cost=cost,
                    execution_time=time.time() - start_time,
                    satisfied=satisfied
                ))
            except Exception as e:
                print(f"Error in composition {i}: {e}")
        
        elapsed = time.time() - start_time
        print(f"Completed {len(results)}/{num_compositions} compositions in {elapsed:.2f}s")
        
        return results
    
    def train_meta_learning(self):
        """Train meta-learning model"""
        print("\n" + "=" * 60)
        print("Training Meta-Learning Model")
        print("=" * 60)
        
        # Prepare training data for each source domain
        domains_data = {}
        for domain in self.config.train_domains:
            domains_data[domain] = self.prepare_training_data(domain, num_samples=200)
        
        # Initialize MAML
        state_dim = 7  # State dimension (1 pos + 5 req + 1 num_services)
        action_dim = 30  # Max services
        
        self.maml_composer = MAMLComposer(
            state_dim=state_dim,
            action_dim=action_dim,
            inner_lr=self.config.meta_learning.inner_lr,
            meta_lr=self.config.meta_learning.meta_lr,
            num_inner_steps=self.config.meta_learning.num_inner_steps
        )
        
        # Convert to format expected by MAML
        maml_domains_data = {}
        for domain_name, domain_list in domains_data.items():
            states = [d[0] for d in domain_list]
            actions = [d[1] for d in domain_list]
            rewards = [d[2] for d in domain_list]
            
            maml_domains_data[domain_name] = {
                "states": states,
                "actions": actions,
                "rewards": rewards
            }
        
        # Meta-train
        self.maml_composer.meta_train(
            maml_domains_data,
            num_iterations=self.config.meta_learning.num_meta_iterations,
            batch_size=self.config.meta_learning.batch_size,
            support_size=self.config.meta_learning.support_set_size,
            query_size=self.config.meta_learning.query_set_size,
            log_interval=100
        )
        
        self.training_curves["Meta-Learning"] = self.maml_composer.meta_losses
        
        # Save checkpoint
        self.maml_composer.save_checkpoint(
            os.path.join(self.config.checkpoint_dir, "meta_policy.pt")
        )
    
    def train_baselines(self):
        """Train baseline methods that need training"""
        print("\n" + "=" * 60)
        print("Training Baseline Methods")
        print("=" * 60)
        
        # Prepare training data
        source_data = self.prepare_training_data(
            self.config.train_domains[0],
            num_samples=500
        )
        
        multi_domain_data = {}
        for i, domain in enumerate(self.config.train_domains[:3]):
            multi_domain_data[i] = self.prepare_training_data(domain, num_samples=200)
        
        # Transfer Learning
        print("\nTraining Transfer Learning...")
        state_dim = 7  # State dimension (1 pos + 5 req + 1 num_services)
        action_dim = 30
        
        self.tl_composer = TransferLearningComposer(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=0.001,
            training_steps=50
        )
        self.tl_composer.pre_train(source_data)
        
        # Multi-Task Learning
        print("\nTraining Multi-Task Learning...")
        self.mtl_composer = MultiTaskLearningComposer(
            state_dim=state_dim,
            action_dim=action_dim,
            num_domains=len(multi_domain_data),
            learning_rate=0.001
        )
        self.mtl_composer.train(multi_domain_data)
    
    def run_full_evaluation(self):
        """Run complete evaluation"""
        print("\n" + "=" * 60)
        print("Running Full Evaluation")
        print("=" * 60)
        
        # Train methods
        self.train_meta_learning()
        self.train_baselines()
        
        # Initialize baseline composers
        random_composer = RandomComposer()
        greedy_composer = GreedyComposer()
        ga_composer = GeneticAlgorithmComposer(
            population_size=20,
            generations=30
        )
        
        # Evaluate on test domain
        test_domain = self.config.test_domains[0]
        print(f"\n\nEvaluating on test domain: {test_domain}")
        print("=" * 60)
        
        # Evaluate each method
        methods = {
            "Meta-Learning (Proposed)": self.maml_composer,
            "Transfer Learning": self.tl_composer,
            "Multi-Task Learning": self.mtl_composer,
            "Genetic Algorithm": ga_composer,
            "Greedy": greedy_composer,
            "Random": random_composer
        }
        
        results_by_method = {}
        adaptation_times = {}
        
        for method_name, composer in methods.items():
            start_time = time.time()
            results = self.evaluate_method(
                method_name,
                composer,
                test_domain,
                num_compositions=50
            )
            elapsed = time.time() - start_time
            
            results_by_method[method_name] = results
            adaptation_times[method_name] = elapsed
            self.results_aggregator.add_results(method_name, results)
            
            # Calculate metrics
            calculator = MetricsCalculator()
            rsr = calculator.requirement_satisfaction_rate(results)
            avg_dev = calculator.average_qos_deviation(results)
            
            print(f"{method_name}: RSR = {rsr:.4f}, QoS Dev = {avg_dev:.4f}")
        
        # Generate visualizations
        self.generate_all_visualizations(results_by_method, adaptation_times)
    
    def generate_all_visualizations(
        self,
        results_by_method: Dict[str, List[CompositionResult]],
        adaptation_times: Dict[str, float]
    ):
        """Generate all visualization plots"""
        print("\n" + "=" * 60)
        print("Generating Visualizations")
        print("=" * 60)
        
        calculator = MetricsCalculator()
        
        # 1. Requirement Satisfaction Rate
        rsr_by_method = {
            method: calculator.requirement_satisfaction_rate(results)
            for method, results in results_by_method.items()
        }
        self.visualizer.plot_requirement_satisfaction_rate(rsr_by_method)
        
        # 2. QoS Deviation
        qos_dev_by_method = {
            method: calculator.average_qos_deviation(results)
            for method, results in results_by_method.items()
        }
        self.visualizer.plot_qos_deviation(qos_dev_by_method)
        
        # 3. Adaptation Time
        self.visualizer.plot_adaptation_time(adaptation_times)
        
        # 4. Individual QoS Satisfaction
        qos_individual_by_method = {
            method: calculator.individual_requirement_satisfaction(results)
            for method, results in results_by_method.items()
        }
        self.visualizer.plot_individual_qos_satisfaction(qos_individual_by_method)
        
        # 5. Training Curves
        if self.training_curves:
            self.visualizer.plot_adaptation_curves(self.training_curves)
        
        # 6. Comparison Table
        summary = self.results_aggregator.get_summary()
        self.visualizer.plot_comparison_table(summary)
        
        # 7. Radar Chart
        radar_results = {}
        radar_metrics = ["RSR", "Cost Efficiency", "Throughput", "Availability", "Reliability"]
        for method, results in results_by_method.items():
            rsr = calculator.requirement_satisfaction_rate(results)
            cost_inv = 1.0 / (calculator.average_composite_cost(results) + 0.1)
            
            individual_qos = calculator.individual_requirement_satisfaction(results)
            throughput_sat = individual_qos[1]
            avail_sat = individual_qos[2]
            reliab_sat = individual_qos[3]
            
            # Normalize to [0, 1]
            radar_results[method] = [
                rsr, min(cost_inv * 0.5, 1.0), throughput_sat, avail_sat, reliab_sat
            ]
        
        self.visualizer.plot_radar_comparison(radar_results, radar_metrics)
        
        print("\nAll visualizations generated!")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "=" * 80)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 80)
        print(self.results_aggregator.get_comparison_table())
        print("=" * 80)


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("Meta-QoS Composition Framework - Full Evaluation")
    print("=" * 80)
    
    # Create configuration
    cfg = TrainingConfig()
    
    # Update configuration for extended evaluation
    cfg.meta_learning.num_meta_iterations = 2000
    cfg.train_domains = ["healthcare", "fintech", "ecommerce"]
    cfg.test_domains = ["education"]
    
    # Create evaluator
    evaluator = Evaluator(cfg)
    
    # Generate datasets
    evaluator.generate_datasets()
    
    # Run full evaluation
    evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
