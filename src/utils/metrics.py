"""
Evaluation metrics for QoS-aware service composition
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OptimizationDirection(Enum):
    MIN = "min"
    MAX = "max"


class QoSMetrics:
    """QoS attribute definitions and optimization directions"""
    
    RESPONSE_TIME = 0
    THROUGHPUT = 1
    AVAILABILITY = 2
    RELIABILITY = 3
    COST = 4
    
    NAMES = ["response_time", "throughput", "availability", "reliability", "cost"]
    DIRECTIONS = [
        OptimizationDirection.MIN,  # response_time
        OptimizationDirection.MAX,  # throughput
        OptimizationDirection.MAX,  # availability
        OptimizationDirection.MAX,  # reliability
        OptimizationDirection.MIN   # cost
    ]
    
    @classmethod
    def get_name(cls, idx: int) -> str:
        return cls.NAMES[idx]
    
    @classmethod
    def get_direction(cls, idx: int) -> OptimizationDirection:
        return cls.DIRECTIONS[idx]
    
    @classmethod
    def is_better(cls, value1: float, value2: float, idx: int) -> bool:
        """Check if value1 is better than value2 for attribute idx"""
        direction = cls.get_direction(idx)
        if direction == OptimizationDirection.MIN:
            return value1 < value2
        else:
            return value1 > value2


@dataclass
class CompositionResult:
    """Result of a composition attempt"""
    composition_id: str
    domain: str
    selected_services: List[int]
    predicted_qos: np.ndarray
    actual_qos: np.ndarray
    requirements: np.ndarray
    cost: float
    execution_time: float
    satisfied: bool


class MetricsCalculator:
    """Calculates various evaluation metrics"""
    
    @staticmethod
    def requirement_satisfaction_rate(
        results: List[CompositionResult]
    ) -> float:
        """Percentage of compositions meeting all requirements"""
        if len(results) == 0:
            return 0.0
        satisfied_count = sum(1 for r in results if r.satisfied)
        return satisfied_count / len(results)
    
    @staticmethod
    def qos_deviation(
        actual_qos: np.ndarray,
        requirements: np.ndarray,
        normalize: bool = True
    ) -> float:
        """Calculate normalized deviation from requirements"""
        deviations = []
        for i in range(len(actual_qos)):
            direction = QoSMetrics.get_direction(i)
            if direction == OptimizationDirection.MIN:
                if actual_qos[i] > requirements[i]:
                    dev = (actual_qos[i] - requirements[i]) / requirements[i] if normalize else actual_qos[i] - requirements[i]
                    deviations.append(dev)
            else:
                if actual_qos[i] < requirements[i]:
                    dev = (requirements[i] - actual_qos[i]) / requirements[i] if normalize else requirements[i] - actual_qos[i]
                    deviations.append(dev)
        
        return float(np.mean(deviations)) if deviations else 0.0
    
    @staticmethod
    def average_qos_deviation(
        results: List[CompositionResult]
    ) -> float:
        """Average QoS deviation across all results"""
        if len(results) == 0:
            return 0.0
        deviations = [
            MetricsCalculator.qos_deviation(r.actual_qos, r.requirements)
            for r in results
        ]
        return float(np.mean(deviations))
    
    @staticmethod
    def composite_utility(
        actual_qos: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Calculate weighted utility of composite service"""
        if weights is None:
            weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])  # Default weights
        
        # Normalize QoS values to [0, 1]
        normalized_qos = MetricsCalculator._normalize_qos(actual_qos)
        
        return float(np.sum(weights * normalized_qos))
    
    @staticmethod
    def _normalize_qos(qos: np.ndarray) -> np.ndarray:
        """Normalize QoS values to [0, 1] range"""
        # Normalize each attribute independently
        normalized = np.zeros_like(qos)
        
        # Response time: normalize by 2000ms max
        normalized[0] = np.clip(qos[0] / 2000, 0, 1)
        
        # Throughput: normalize by 1000 req/s max
        normalized[1] = np.clip(qos[1] / 1000, 0, 1)
        
        # Availability: already in [0, 1]
        normalized[2] = qos[2]
        
        # Reliability: already in [0, 1]
        normalized[3] = qos[3]
        
        # Cost: normalize by $10 max
        normalized[4] = np.clip(qos[4] / 10, 0, 1)
        
        # For minimization attributes, invert (lower is better)
        for i in [0, 4]:  # response_time and cost
            normalized[i] = 1.0 - normalized[i]
        
        return normalized
    
    @staticmethod
    def average_composite_cost(
        results: List[CompositionResult]
    ) -> float:
        """Average cost of compositions"""
        if len(results) == 0:
            return 0.0
        return float(np.mean([r.cost for r in results]))
    
    @staticmethod
    def individual_requirement_satisfaction(
        results: List[CompositionResult]
    ) -> Dict[int, float]:
        """Satisfaction rate for each individual QoS requirement"""
        if len(results) == 0:
            return {i: 0.0 for i in range(5)}
        
        satisfaction_counts = np.zeros(5)
        
        for r in results:
            for i in range(5):
                direction = QoSMetrics.get_direction(i)
                if direction == OptimizationDirection.MIN:
                    if r.actual_qos[i] <= r.requirements[i] * 1.1:  # 10% tolerance
                        satisfaction_counts[i] += 1
                else:
                    if r.actual_qos[i] >= r.requirements[i] * 0.9:  # 10% tolerance
                        satisfaction_counts[i] += 1
        
        return {i: satisfaction_counts[i] / len(results) for i in range(5)}


class AdaptationMetrics:
    """Metrics for evaluating adaptation efficiency"""
    
    @staticmethod
    def cold_start_performance(
        results_by_samples: Dict[int, List[CompositionResult]],
        reference_performance: float
    ) -> Dict[int, float]:
        """Calculate cold-start performance at different sample counts"""
        performance = {}
        for n_samples, results in results_by_samples.items():
            if len(results) == 0:
                performance[n_samples] = 0.0
            else:
                rsr = MetricsCalculator.requirement_satisfaction_rate(results)
                performance[n_samples] = rsr / reference_performance
        return performance
    
    @staticmethod
    def samples_to_threshold(
        results_by_samples: Dict[int, List[CompositionResult]],
        threshold: float = 0.9
    ) -> int:
        """Number of samples needed to reach threshold performance"""
        for n_samples, results in sorted(results_by_samples.items()):
            rsr = MetricsCalculator.requirement_satisfaction_rate(results)
            if rsr >= threshold:
                return n_samples
        return int(float('inf'))
    
    @staticmethod
    def convergence_rate(
        losses: List[float]
    ) -> float:
        """Calculate convergence rate (percentage of loss reduction)"""
        if len(losses) < 2:
            return 0.0
        initial_loss = losses[0]
        final_loss = losses[-1]
        if initial_loss == 0:
            return 100.0
        return ((initial_loss - final_loss) / initial_loss) * 100


class RobustnessMetrics:
    """Metrics for evaluating robustness and concept drift handling"""
    
    @staticmethod
    def drift_detection_latency(
        actual_drift_time: float,
        detection_time: float
    ) -> float:
        """Time to detect concept drift"""
        return detection_time - actual_drift_time
    
    @staticmethod
    def recovery_time(
        drift_detected_at: float,
        performance_recovered_at: float
    ) -> float:
        """Time to recover performance after drift"""
        return performance_recovered_at - drift_detected_at
    
    @staticmethod
    def uncertainty_calibration(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        actual_values: np.ndarray
    ) -> float:
        """Brier score for uncertainty calibration"""
        # Expected calibration error
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(
                    np.abs(predictions[in_bin] - actual_values[in_bin]) < 0.1
                )
                avg_confidence_in_bin = np.mean(uncertainties[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece


class ComparativeMetrics:
    """Metrics for comparing different approaches"""
    
    @staticmethod
    def improvement_percentage(
        baseline_value: float,
        proposed_value: float,
        higher_is_better: bool = True
    ) -> float:
        """Calculate percentage improvement"""
        if baseline_value == 0:
            return 0.0 if proposed_value == 0 else 100.0
        
        if higher_is_better:
            improvement = ((proposed_value - baseline_value) / baseline_value) * 100
        else:
            improvement = ((baseline_value - proposed_value) / baseline_value) * 100
        
        return improvement
    
    @staticmethod
    def statistical_significance(
        values1: List[float],
        values2: List[float]
    ) -> Tuple[float, float]:
        """Perform t-test and return p-value and significance threshold"""
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(values1, values2)
        return p_value, 0.05


class ResultsAggregator:
    """Aggregates results from multiple runs"""
    
    def __init__(self):
        self.results: Dict[str, List[CompositionResult]] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
    
    def add_results(self, method_name: str, results: List[CompositionResult]):
        """Add results for a method"""
        if method_name not in self.results:
            self.results[method_name] = []
        self.results[method_name].extend(results)
    
    def add_metrics(self, method_name: str, metrics: Dict):
        """Add metrics for a method"""
        if method_name not in self.metrics_history:
            self.metrics_history[method_name] = []
        self.metrics_history[method_name].append(metrics)
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary metrics for all methods"""
        summary = {}
        
        for method_name, results in self.results.items():
            calculator = MetricsCalculator()
            
            summary[method_name] = {
                "num_compositions": len(results),
                "requirement_satisfaction_rate": calculator.requirement_satisfaction_rate(results),
                "average_qos_deviation": calculator.average_qos_deviation(results),
                "average_cost": calculator.average_composite_cost(results),
                "individual_satisfaction": calculator.individual_requirement_satisfaction(results)
            }
        
        return summary
    
    def get_comparison_table(self) -> str:
        """Generate a formatted comparison table"""
        summary = self.get_summary()
        
        # Header
        lines = []
        lines.append("=" * 120)
        lines.append(f"{'Method':<25} {'RSR':<10} {'QoS Dev':<10} {'Avg Cost':<10} {'Resp':<8} {'Thru':<8} {'Avail':<8} {'Rel':<8}")
        lines.append("=" * 120)
        
        # Rows
        for method_name, metrics in summary.items():
            lines.append(
                f"{method_name:<25} "
                f"{metrics['requirement_satisfaction_rate']:<10.4f} "
                f"{metrics['average_qos_deviation']:<10.4f} "
                f"{metrics['average_cost']:<10.4f} "
                f"{metrics['individual_satisfaction'][0]:<8.4f} "
                f"{metrics['individual_satisfaction'][1]:<8.4f} "
                f"{metrics['individual_satisfaction'][2]:<8.4f} "
                f"{metrics['individual_satisfaction'][3]:<8.4f}"
            )
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing QoS Metrics...")
    
    # Create sample results
    results = [
        CompositionResult(
            composition_id="c1",
            domain="healthcare",
            selected_services=[1, 2, 3],
            predicted_qos=np.array([500, 200, 0.98, 0.97, 1.0]),
            actual_qos=np.array([480, 210, 0.985, 0.975, 0.95]),
            requirements=np.array([600, 150, 0.97, 0.96, 1.5]),
            cost=0.95,
            execution_time=0.5,
            satisfied=True
        ),
        CompositionResult(
            composition_id="c2",
            domain="fintech",
            selected_services=[4, 5],
            predicted_qos=np.array([800, 150, 0.95, 0.94, 2.0]),
            actual_qos=np.array([850, 140, 0.94, 0.93, 2.1]),
            requirements=np.array([700, 180, 0.97, 0.96, 1.8]),
            cost=2.1,
            execution_time=0.6,
            satisfied=False
        )
    ]
    
    calculator = MetricsCalculator()
    
    rsr = calculator.requirement_satisfaction_rate(results)
    print(f"Requirement Satisfaction Rate: {rsr:.4f}")
    
    avg_dev = calculator.average_qos_deviation(results)
    print(f"Average QoS Deviation: {avg_dev:.4f}")
    
    avg_cost = calculator.average_composite_cost(results)
    print(f"Average Cost: ${avg_cost:.2f}")
    
    individual = calculator.individual_requirement_satisfaction(results)
    print("\nIndividual Requirement Satisfaction:")
    for i, rate in individual.items():
        print(f"  {QoSMetrics.get_name(i)}: {rate:.4f}")
    
    aggregator = ResultsAggregator()
    aggregator.add_results("Proposed", results)
    aggregator.add_results("Baseline", results[:1])
    
    print("\n" + aggregator.get_comparison_table())
