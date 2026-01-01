"""
Unit tests for Meta-QoS Composition Framework
"""
import pytest
import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.meta_learning.maml_engine import MAMLComposer, CompositionPolicy, ReptileComposer
from core.meta_learning.baselines import (
    RandomComposer, GreedyComposer, GeneticAlgorithmComposer,
    TransferLearningComposer, MultiTaskLearningComposer
)
from utils.metrics import QoSMetrics, MetricsCalculator, CompositionResult
from utils.visualization import ResultsVisualizer


class TestQoSMetrics:
    """Tests for QoS metrics and requirements"""

    def test_metric_names(self):
        """Test that all QoS metric names are defined"""
        assert QoSMetrics.NAMES == ["response_time", "throughput", "availability", "reliability", "cost"]
        assert len(QoSMetrics.NAMES) == 5

    def test_optimization_directions(self):
        """Test optimization directions (MIN/MAX)"""
        # Response time and cost should be MIN (lower is better)
        assert QoSMetrics.get_direction(0) == QoSMetrics.OptimizationDirection.MIN  # response_time
        assert QoSMetrics.get_direction(4) == QoSMetrics.OptimizationDirection.MIN  # cost

        # Throughput, availability, reliability should be MAX
        assert QoSMetrics.get_direction(1) == QoSMetrics.OptimizationDirection.MAX  # throughput
        assert QoSMetrics.get_direction(2) == QoSMetrics.OptimizationDirection.MAX  # availability
        assert QoSMetrics.get_direction(3) == QoSMetrics.OptimizationDirection.MAX  # reliability

    def test_is_better(self):
        """Test comparison logic"""
        # Lower is better for response time
        assert QoSMetrics.is_better(100, 200, 0) is True
        assert QoSMetrics.is_better(200, 100, 0) is False

        # Higher is better for availability
        assert QoSMetrics.is_better(0.95, 0.90, 2) is True
        assert QoSMetrics.is_better(0.90, 0.95, 2) is False


class TestMetricsCalculator:
    """Tests for metrics calculation"""

    def test_requirement_satisfaction_rate_empty(self):
        """Test RSR calculation with empty results"""
        assert MetricsCalculator.requirement_satisfaction_rate([]) == 0.0

    def test_requirement_satisfaction_rate(self):
        """Test RSR with sample results"""
        results = [
            CompositionResult(
                composition_id="c1",
                domain="test",
                selected_services=[1, 2],
                predicted_qos=np.array([100, 50, 0.95, 0.90, 0.5]),
                actual_qos=np.array([150, 60, 0.94, 0.89, 0.6]),
                requirements=np.array([200, 100, 0.90, 0.85, 1.0]),
                cost=0.6,
                execution_time=0.1,
                satisfied=True
            ),
            CompositionResult(
                composition_id="c2",
                domain="test",
                selected_services=[3],
                predicted_qos=np.array([300, 30, 0.85, 0.80, 0.8]),
                actual_qos=np.array([350, 35, 0.84, 0.79, 0.9]),
                requirements=np.array([200, 100, 0.90, 0.85, 1.0]),
                cost=0.9,
                execution_time=0.1,
                satisfied=False
            )
        ]
        rsr = MetricsCalculator.requirement_satisfaction_rate(results)
        assert rsr == pytest.approx(0.5, rel=1e-3)

    def test_qos_deviation(self):
        """Test QoS deviation calculation"""
        actual = np.array([500, 200, 0.95, 0.90, 1.0])
        requirements = np.array([600, 250, 0.90, 0.85, 1.5])

        deviation = MetricsCalculator.qos_deviation(actual, requirements)
        # Should have small deviation since all requirements met or close
        assert deviation >= 0

    def test_composite_utility(self):
        """Test composite utility calculation"""
        qos = np.array([500, 200, 0.95, 0.90, 1.0])
        utility = MetricsCalculator.composite_utility(qos)
        assert 0.0 <= utility <= 1.0

    def test_normalize_qos(self):
        """Test QoS normalization"""
        qos = np.array([500, 200, 0.95, 0.90, 2.0])
        normalized = MetricsCalculator._normalize_qos(qos)

        assert len(normalized) == 5
        assert all(0.0 <= v <= 1.0 for v in normalized)


class TestCompositionPolicy:
    """Tests for the neural network policy"""

    def test_policy_forward(self):
        """Test policy network forward pass"""
        state_dim = 7
        action_dim = 30
        hidden_dim = 128

        policy = CompositionPolicy(state_dim, action_dim, hidden_dim)

        # Test with single state
        state = torch.randn(state_dim)
        logits = policy(state)

        assert logits.shape == (action_dim,)

        # Test with batch
        batch = torch.randn(8, state_dim)
        batch_logits = policy(batch)
        assert batch_logits.shape == (8, action_dim)

    def test_policy_action_selection(self):
        """Test action selection from policy"""
        state_dim = 7
        action_dim = 30

        policy = CompositionPolicy(state_dim, action_dim)

        state = torch.randn(state_dim)
        logits = policy(state)
        probs = torch.softmax(logits, dim=-1)

        # Action probabilities should sum to 1
        assert torch.sum(probs).item() == pytest.approx(1.0, rel=1e-5)

        # Greedy action should be valid
        greedy_action = torch.argmax(probs).item()
        assert 0 <= greedy_action < action_dim


class TestMAMLComposer:
    """Tests for MAML-based composer"""

    def test_maml_initialization(self):
        """Test MAML composer initialization"""
        composer = MAMLComposer(
            state_dim=7,
            action_dim=30,
            inner_lr=0.01,
            meta_lr=0.001,
            num_inner_steps=5
        )

        assert composer.state_dim == 7
        assert composer.action_dim == 30
        assert composer.inner_lr == 0.01
        assert composer.meta_lr == 0.001

    def test_inner_loop_adapt(self):
        """Test inner loop adaptation"""
        composer = MAMLComposer(state_dim=7, action_dim=30)

        # Create fake support data
        states = [torch.randn(7) for _ in range(10)]
        actions = [torch.randint(0, 30, (1,)).item() for _ in range(10)]
        rewards = [np.random.rand() for _ in range(10)]

        adapted_policy = composer.adapt_to_domain(states, actions, rewards)

        # Adapted policy should be a new policy instance
        assert adapted_policy is not composer.meta_policy

    def test_compose_method(self):
        """Test service composition"""
        composer = MAMLComposer(state_dim=7, action_dim=30)

        # Create mock workflow and services
        workflow = type('Workflow', (), {
            'nodes': [
                {"position": 0, "id": "node_0"},
                {"position": 1, "id": "node_1"},
                {"position": 2, "id": "node_2"}
            ]
        })()

        available_services = [type('Service', (), {'id': i})() for i in range(20)]
        requirements = np.array([500, 100, 0.90, 0.85, 1.0])

        selected = composer.compose(workflow, available_services, requirements)

        assert len(selected) == 3
        assert all(isinstance(s, int) for s in selected)

    def test_adapt_and_compose(self):
        """Test adaptation + composition"""
        composer = MAMLComposer(state_dim=7, action_dim=30)

        workflow = type('Workflow', (), {
            'nodes': [{"position": i, "id": f"node_{i}"} for i in range(3)]
        })()

        services = [type('Service', (), {'id': i})() for i in range(20)]
        requirements = np.array([500, 100, 0.90, 0.85, 1.0])

        # Support data
        support_data = [
            (torch.randn(7), 5, 0.8),
            (torch.randn(7), 10, 0.7),
            (torch.randn(7), 8, 0.9)
        ]

        selected = composer.adapt_and_compose(
            workflow, services, requirements, support_data
        )

        assert len(selected) == 3

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        composer = MAMLComposer(state_dim=7, action_dim=30)

        # Train a bit
        for _ in range(10):
            states = [torch.randn(7) for _ in range(5)]
            actions = [torch.randint(0, 30, (1,)).item() for _ in range(5)]
            rewards = [np.random.rand() for _ in range(5)]
            composer.adapt_to_domain(states, actions, rewards)

        # Save checkpoint
        checkpoint_path = "/tmp/maml_test_checkpoint.pt"
        composer.save_checkpoint(checkpoint_path)

        # Load into new composer
        new_composer = MAMLComposer(state_dim=7, action_dim=30)
        new_composer.load_checkpoint(checkpoint_path)

        # Should have same state
        assert len(new_composer.meta_losses) > 0

        # Cleanup
        os.remove(checkpoint_path)


class TestBaselineComposers:
    """Tests for baseline composition methods"""

    def test_random_composer(self):
        """Test random composition"""
        composer = RandomComposer(state_dim=7, action_dim=30)

        workflow = type('Workflow', (), {
            'nodes': [{"position": i} for i in range(3)]
        })()

        services = [type('Service', (), {'id': i})() for i in range(20)]

        selected = composer.compose(workflow, services, np.zeros(5))

        assert len(selected) == 3
        assert all(0 <= s < 20 for s in selected)

    def test_greedy_composer(self):
        """Test greedy composition"""
        composer = GreedyComposer(state_dim=7, action_dim=30)

        workflow = type('Workflow', (), {
            'nodes': [{"position": i} for i in range(2)]
        })()

        services = [type('Service', (), {'id': i, 'qos': np.array([100 * i, 50 * i, 0.9, 0.8, 0.5])})() for i in range(10)]
        requirements = np.array([500, 100, 0.85, 0.75, 1.0])

        selected = composer.compose(workflow, services, requirements)

        assert len(selected) == 2

    def test_genetic_algorithm_composer(self):
        """Test GA composition"""
        composer = GeneticAlgorithmComposer(state_dim=7, action_dim=30)

        workflow = type('Workflow', (), {
            'nodes': [{"position": i} for i in range(3)]
        })()

        services = [type('Service', (), {'id': i})() for i in range(20)]

        selected = composer.compose(workflow, services, np.zeros(5))

        assert len(selected) == 3

    def test_transfer_learning_composer(self):
        """Test transfer learning composer"""
        composer = TransferLearningComposer(state_dim=7, action_dim=30)

        # Pre-train
        source_data = [
            (torch.randn(7), 5, 0.8) for _ in range(100)
        ]
        composer.pre_train(source_data)

        # Compose
        workflow = type('Workflow', (), {
            'nodes': [{"position": i} for i in range(2)]
        })()

        services = [type('Service', (), {'id': i})() for i in range(20)]
        selected = composer.compose(workflow, services, np.zeros(5))

        assert len(selected) == 2

    def test_multi_task_composer(self):
        """Test multi-task learning composer"""
        composer = MultiTaskLearningComposer(state_dim=7, action_dim=30)

        # Train on multiple domains
        multi_domain_data = {
            "domain1": [(torch.randn(7), 5, 0.8) for _ in range(50)],
            "domain2": [(torch.randn(7), 10, 0.7) for _ in range(50)],
            "domain3": [(torch.randn(7), 8, 0.9) for _ in range(50)]
        }
        composer.train(multi_domain_data)

        # Compose
        workflow = type('Workflow', (), {
            'nodes': [{"position": i} for i in range(2)]
        })()

        services = [type('Service', (), {'id': i})() for i in range(20)]
        selected = composer.compose(workflow, services, np.zeros(5))

        assert len(selected) == 2


class TestReptileComposer:
    """Tests for Reptile (first-order MAML) composer"""

    def test_reptile_initialization(self):
        """Test Reptile composer initialization"""
        composer = ReptileComposer(state_dim=7, action_dim=30)

        assert composer.inner_lr == 0.01
        assert composer.meta_lr == 0.001
        assert composer.num_inner_steps == 5

    def test_reptile_compose(self):
        """Test Reptile composition"""
        composer = ReptileComposer(state_dim=7, action_dim=30)

        workflow = type('Workflow', (), {
            'nodes': [{"position": i} for i in range(2)]
        })()

        services = [type('Service', (), {'id': i})() for i in range(20)]

        selected = composer.compose(workflow, services, np.zeros(5))

        assert len(selected) == 2


class TestResultsVisualizer:
    """Tests for visualization utilities"""

    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ResultsVisualizer(output_dir=tmpdir)
            assert visualizer.output_dir == tmpdir

    def test_plot_rsr(self):
        """Test RSR plot generation"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ResultsVisualizer(output_dir=tmpdir)

            results = {
                "MAML": 0.85,
                "Transfer Learning": 0.70,
                "Random": 0.30
            }

            visualizer.plot_requirement_satisfaction_rate(results)

            expected_path = os.path.join(tmpdir, "rsr_comparison.png")
            assert os.path.exists(expected_path)

    def test_plot_qos_deviation(self):
        """Test QoS deviation plot"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer = ResultsVisualizer(output_dir=tmpdir)

            results = {
                "MAML": 0.10,
                "GA": 0.20,
                "Greedy": 0.30
            }

            visualizer.plot_qos_deviation(results)

            expected_path = os.path.join(tmpdir, "qos_deviation_comparison.png")
            assert os.path.exists(expected_path)


class TestIntegration:
    """Integration tests for end-to-end functionality"""

    def test_full_composition_pipeline(self):
        """Test complete composition pipeline"""
        # Create composer
        maml_composer = MAMLComposer(state_dim=7, action_dim=30)
        baseline_composers = [
            RandomComposer(state_dim=7, action_dim=30),
            GreedyComposer(state_dim=7, action_dim=30),
            GeneticAlgorithmComposer(state_dim=7, action_dim=30)
        ]

        # Create test workflow
        workflow = type('Workflow', (), {
            'nodes': [
                {"position": 0, "id": "node_0"},
                {"position": 1, "id": "node_1"},
                {"position": 2, "id": "node_2"}
            ]
        })()

        services = [
            type('Service', (), {
                'id': i,
                'qos': np.array([100 + i*10, 50 + i*5, 0.9 + i*0.01, 0.8 + i*0.02, 0.5 - i*0.01])
            })()
            for i in range(30)
        ]

        requirements = np.array([500, 100, 0.85, 0.80, 0.8])

        # Test all composers
        results = {}
        for composer in [maml_composer] + baseline_composers:
            selected = composer.compose(workflow, services, requirements)
            results[composer.get_name()] = len(selected)

        assert len(results) == 4
        assert all(len(s) == 3 for s in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
