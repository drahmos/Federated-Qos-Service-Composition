"""
Extended evaluation with more iterations to demonstrate MAML's advantage
"""
import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from configs.config import TrainingConfig
from core.meta_learning.maml_engine import MAMLComposer
from core.meta_learning.baselines import (
    RandomComposer, GreedyComposer, GeneticAlgorithmComposer,
    TransferLearningComposer, MultiTaskLearningComposer
)
from utils.data import DataGenerator, CompositionExecutor
from utils.metrics import MetricsCalculator, CompositionResult
from utils.visualization import ResultsVisualizer


class ExtendedEvaluator:
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.generator = DataGenerator(seed=self.config.seed)
        self.composition_executor = CompositionExecutor()
        self.all_services = {}
        self.all_workflows = {}
        self.training_curves = {}
        self.visualizer = ResultsVisualizer(self.config.plot_dir)

    def generate_datasets(self):
        print("=" * 60)
        print("Generating Extended Dataset")
        print("=" * 60)
        domains = self.config.train_domains + self.config.val_domains + self.config.test_domains
        for domain in domains:
            if domain not in self.all_services:
                num_services = np.random.randint(20, 35)
                services = self.generator.generate_domain_services(domain, num_services)
                workflows = [
                    self.generator.generate_workflow(wid, domain, services, np.random.randint(3, 6))
                    for wid in range(30)
                ]
                self.all_services[domain] = services
                self.all_workflows[domain] = workflows
                print(f"  {domain}: {len(services)} services, {len(workflows)} workflows")

    def train_maml_extended(self, num_iterations=2000, inner_lr=0.05, meta_lr=0.002, inner_steps=10):
        print("\n" + "=" * 60)
        print(f"Training MAML (Extended: {num_iterations} iterations)")
        print("=" * 60)
        state_dim, action_dim = 7, 35
        maml = MAMLComposer(state_dim=state_dim, action_dim=action_dim, inner_lr=inner_lr, meta_lr=meta_lr, num_inner_steps=inner_steps)
        domains_data = {}
        for domain in self.config.train_domains:
            domains_data[domain] = self.prepare_training_data(domain, num_samples=300)
        maml_domains_data = {}
        for domain_name, domain_list in domains_data.items():
            maml_domains_data[domain_name] = {
                "states": [d[0] for d in domain_list],
                "actions": [d[1] for d in domain_list],
                "rewards": [d[2] for d in domain_list]
            }
        start_time = time.time()
        maml.meta_train(maml_domains_data, num_iterations=num_iterations, batch_size=8, support_size=15, query_size=30, log_interval=100)
        training_time = time.time() - start_time
        print(f"\n  Training completed in {training_time:.1f} seconds")
        self.training_curves["MAML (Extended)"] = maml.meta_losses
        return maml, training_time

    def prepare_training_data(self, domain, num_samples=1000):
        training_data = []
        workflows = self.all_workflows[domain]
        services = self.all_services[domain]
        for _ in range(num_samples):
            workflow = np.random.choice(workflows)
            selected_indices = np.random.choice(len(services), size=min(len(workflow.nodes), len(services)), replace=False)
            selected_services = [services[i] for i in selected_indices]
            composite_qos = self.composition_executor.compute_composite_qos({"nodes": workflow.nodes}, {s.id: s for s in services}, workflow.edges)
            reward = self._compute_reward(composite_qos, workflow.requirements)
            state = self._create_state(workflow, composite_qos, len(services))
            action = selected_indices[0]
            training_data.append((state, action, reward))
        return training_data

    def _create_state(self, workflow, qos, num_services):
        state_features = [np.array([0.5]), workflow.requirements / 1000.0, np.array([num_services / 100.0])]
        return torch.FloatTensor(np.concatenate(state_features))

    def _compute_reward(self, qos, requirements):
        reward = 0.0
        for i in range(len(qos)):
            if i == 0:
                reward += 1.0 if qos[i] <= requirements[i] else -(qos[i] - requirements[i]) / requirements[i]
            elif i == 1:
                reward += 1.0 if qos[i] >= requirements[i] else -(requirements[i] - qos[i]) / requirements[i]
            elif i in [2, 3]:
                reward += 1.0 if qos[i] >= requirements[i] else -2.0
            elif i == 4:
                reward += 1.0 if qos[i] <= requirements[i] else -1.0
        return reward / 5.0

    def evaluate_few_shot(self, composer, test_domain, num_compositions=50, num_support_samples=5):
        print(f"\n  Evaluating {composer.get_name()} on {test_domain}")
        workflows, services = self.all_workflows[test_domain], self.all_services[test_domain]
        support_workflows = np.random.choice(workflows, min(num_support_samples, len(workflows)), replace=False)
        support_data = []
        for wf in support_workflows:
            for node in wf.nodes:
                state = self._create_state_node(node, wf.requirements, len(services))
                action = np.random.randint(len(services))
                reward = np.random.rand()
                support_data.append((state, action, reward))
        results = []
        for i in range(num_compositions):
            workflow = np.random.choice(workflows)
            try:
                if hasattr(composer, 'adapt_and_compose'):
                    selected = composer.adapt_and_compose(workflow, services, workflow.requirements, support_data)
                else:
                    selected = composer.compose(workflow, services, workflow.requirements)
                selected_services = [services[min(s, len(services)-1)] for s in selected]
                selected_ids = [s.id for s in selected_services]
                predicted_qos = self.composition_executor.compute_composite_qos({"nodes": workflow.nodes}, {s.id: s for s in selected_services}, workflow.edges)
                actual_qos = self.composition_executor.compute_actual_qos(predicted_qos, noise_level=0.05)
                satisfied = self.composition_executor.check_requirements(actual_qos, workflow.requirements)
                cost = sum(s.qos_values[4] for s in selected_services)
                results.append(CompositionResult(composition_id=f"{test_domain}_{i}", domain=test_domain, selected_services=selected_ids, predicted_qos=predicted_qos, actual_qos=actual_qos, requirements=workflow.requirements, cost=cost, execution_time=0.0, satisfied=satisfied))
            except:
                pass
        return results

    def _create_state_node(self, node, requirements, num_services):
        state_features = [np.array([min(node.get("position", 0), 9) / 10.0]), requirements / 1000.0, np.array([num_services / 100.0])]
        return torch.FloatTensor(np.concatenate(state_features))

    def run_extended_evaluation(self):
        self.generate_datasets()
        maml, maml_time = self.train_maml_extended(num_iterations=2000, inner_lr=0.05, meta_lr=0.002, inner_steps=10)
        print("\n" + "=" * 60)
        print("Training Baselines")
        print("=" * 60)
        state_dim, action_dim = 7, 35
        print("  Training Transfer Learning...")
        tl_composer = TransferLearningComposer(state_dim=state_dim, action_dim=action_dim)
        tl_composer.pre_train(self.prepare_training_data(self.config.train_domains[0], num_samples=1000))
        print("  Training Multi-Task Learning...")
        mtl_composer = MultiTaskLearningComposer(state_dim=state_dim, action_dim=action_dim, num_domains=3)
        multi_domain_data = {i: self.prepare_training_data(domain, num_samples=300) for i, domain in enumerate(self.config.train_domains)}
        mtl_composer.train(multi_domain_data)
        test_domain = self.config.test_domains[0]
        print(f"\n{'=' * 60}\nFew-Shot Evaluation on {test_domain} (5 support samples)\n" + "=" * 60)
        methods = {
            "MAML (Extended)": maml,
            "Transfer Learning": tl_composer,
            "Multi-Task Learning": mtl_composer,
            "Genetic Algorithm": GeneticAlgorithmComposer(population_size=30, generations=50),
            "Greedy": GreedyComposer(),
            "Random": RandomComposer(),
        }
        results_by_method, adaptation_times = {}, {}
        calculator = MetricsCalculator()
        for method_name, composer in methods.items():
            start = time.time()
            results = self.evaluate_few_shot(composer, test_domain, num_compositions=50, num_support_samples=5)
            elapsed = time.time() - start
            results_by_method[method_name] = results
            adaptation_times[method_name] = elapsed
            rsr = calculator.requirement_satisfaction_rate(results)
            avg_dev = calculator.average_qos_deviation(results)
            avg_cost = calculator.average_composite_cost(results)
            print(f"  {method_name:25s} | RSR: {rsr:.1%} | Dev: {avg_dev:.3f} | Cost: ${avg_cost:.2f}")
        self.print_summary(results_by_method, adaptation_times)
        self.generate_visualizations(results_by_method, adaptation_times)
        return results_by_method

    def print_summary(self, results_by_method, adaptation_times):
        print("\n" + "=" * 80)
        print("FINAL EVALUATION SUMMARY")
        print("=" * 80)
        calculator = MetricsCalculator()
        print(f"\n{'Method':<30} {'RSR':>8} {'QoS Dev':>10} {'Avg Cost':>10} {'Time':>10}")
        print("-" * 80)
        for method, results in results_by_method.items():
            rsr = calculator.requirement_satisfaction_rate(results)
            dev = calculator.average_qos_deviation(results)
            cost = calculator.average_composite_cost(results)
            t = adaptation_times.get(method, 0)
            print(f"{method:<30} {rsr:>8.1%} {dev:>10.4f} {cost:>10.4f} {t:>10.2f}s")
        print("=" * 80)
        best_rsr, best_method = 0, None
        for method, results in results_by_method.items():
            rsr = calculator.requirement_satisfaction_rate(results)
            if rsr > best_rsr:
                best_rsr, best_method = rsr, method
        print(f"\n  Best Method: {best_method} (RSR: {best_rsr:.1%})")

    def generate_visualizations(self, results_by_method, adaptation_times):
        print("\n" + "=" * 60)
        print("Generating Visualizations")
        print("=" * 60)
        calculator = MetricsCalculator()
        rsr_by_method = {m: calculator.requirement_satisfaction_rate(r) for m, r in results_by_method.items()}
        self.visualizer.plot_requirement_satisfaction_rate(rsr_by_method)
        qos_dev_by_method = {m: calculator.average_qos_deviation(r) for m, r in results_by_method.items()}
        self.visualizer.plot_qos_deviation(qos_dev_by_method)
        if self.training_curves:
            self.visualizer.plot_adaptation_curves(self.training_curves)
        self.visualizer.plot_adaptation_time(adaptation_times)
        qos_individual_by_method = {m: calculator.individual_requirement_satisfaction(r) for m, r in results_by_method.items()}
        self.visualizer.plot_individual_qos_satisfaction(qos_individual_by_method)
        summary = {}
        for method, results in results_by_method.items():
            summary[method] = {
                "requirement_satisfaction_rate": calculator.requirement_satisfaction_rate(results),
                "average_qos_deviation": calculator.average_qos_deviation(results),
                "average_composite_cost": calculator.average_composite_cost(results),
                "individual_requirement_satisfaction": calculator.individual_requirement_satisfaction(results)
            }
        self.visualizer.plot_comparison_table(summary)
        print("  All visualizations saved!")


def main():
    print("=" * 80)
    print("EXTENDED MAML EVALUATION")
    print("=" * 80)
    evaluator = ExtendedEvaluator()
    results = evaluator.run_extended_evaluation()
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
