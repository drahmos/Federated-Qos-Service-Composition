"""
Visualization utilities for results and metrics
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ResultsVisualizer:
    """Visualize experiment results"""
    
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_requirement_satisfaction_rate(
        self,
        results_by_method: Dict[str, float],
        title: str = "Requirement Satisfaction Rate Comparison",
        save_path: Optional[str] = None
    ):
        """Bar chart of requirement satisfaction rates"""
        methods = list(results_by_method.keys())
        values = list(results_by_method.values())
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        bars = ax.bar(methods, values, color='steelblue', alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Requirement Satisfaction Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/rsr_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_qos_deviation(
        self,
        results_by_method: Dict[str, float],
        title: str = "QoS Deviation Comparison (Lower is Better)",
        save_path: Optional[str] = None
    ):
        """Bar chart of QoS deviations"""
        methods = list(results_by_method.keys())
        values = list(results_by_method.values())
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        colors = ['red' if v > 0.2 else 'green' for v in values]
        bars = ax.bar(methods, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average QoS Deviation', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/qos_deviation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_cold_start_performance(
        self,
        results_by_samples: Dict[str, Dict[int, float]],
        reference_performance: float,
        title: str = "Cold-Start Performance by Number of Samples",
        save_path: Optional[str] = None
    ):
        """Line plot showing cold-start performance over sample counts"""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        for method, performance_dict in results_by_samples.items():
            sample_counts = sorted(performance_dict.keys())
            performances = [performance_dict[sc] for sc in sample_counts]
            
            ax.plot(sample_counts, performances, 
                   marker='o', linewidth=2, markersize=8, label=method)
        
        # Add reference line
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, 
                  label='Reference (100%)')
        
        ax.set_xlabel('Number of Training Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance (relative to reference)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/cold_start_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_adaptation_curves(
        self,
        training_losses: Dict[str, List[float]],
        title: str = "Training Loss Curves",
        save_path: Optional[str] = None
    ):
        """Plot training loss curves for different methods"""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        for method, losses in training_losses.items():
            ax.plot(losses, linewidth=2, alpha=0.8, label=method)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_individual_qos_satisfaction(
        self,
        results_by_method: Dict[str, Dict[int, float]],
        title: str = "Individual QoS Requirement Satisfaction",
        save_path: Optional[str] = None
    ):
        """Grouped bar chart of individual QoS satisfaction rates"""
        qos_names = ["Response Time", "Throughput", "Availability", 
                    "Reliability", "Cost"]
        methods = list(results_by_method.keys())
        x = np.arange(len(qos_names))
        width = 0.8 / len(methods)
        
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        
        for i, method in enumerate(methods):
            satisfaction = list(results_by_method[method].values())
            offset = (i - len(methods)/2 + 0.5) * width
            bars = ax.bar(x + offset, satisfaction, width, 
                         label=method, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, satisfaction):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('QoS Attribute', fontsize=12, fontweight='bold')
        ax.set_ylabel('Satisfaction Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(qos_names)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/individual_qos_satisfaction.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_adaptation_time(
        self,
        adaptation_times: Dict[str, float],
        title: str = "Adaptation Time Comparison (Lower is Better)",
        save_path: Optional[str] = None
    ):
        """Bar chart of adaptation times"""
        methods = list(adaptation_times.keys())
        times = list(adaptation_times.values())
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        colors = ['green' if t < 5 else 'orange' if t < 10 else 'red' 
                 for t in times]
        bars = ax.bar(methods, times, color=colors, alpha=0.7)
        
        # Add time labels
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time_val:.1f}s',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Adaptation Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/adaptation_time.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_radar_comparison(
        self,
        results_by_method: Dict[str, List[float]],
        metrics_names: List[str],
        title: str = "Multi-Metric Comparison",
        save_path: Optional[str] = None
    ):
        """Radar chart for comparing multiple metrics"""
        angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
        angles = angles + angles[:1]  # Complete the circle
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        colors = plt.get_cmap('Set3')(np.linspace(0, 1, len(results_by_method)))
        
        for (method, values), color in zip(results_by_method.items(), colors):
            values = values + values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/radar_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_comparison_table(
        self,
        summary: Dict[str, Dict],
        title: str = "Performance Comparison Summary",
        save_path: Optional[str] = None
    ):
        """Create a formatted table visualization"""
        methods = list(summary.keys())
        metrics = list(summary[methods[0]].keys())
        
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data
        table_data = [["Method"] + metrics]
        for method in methods:
            row = [method]
            for metric in metrics:
                value = summary[method][metric]
                if isinstance(value, dict):
                    row.append(f"{np.mean(list(value.values())):.3f}")
                else:
                    row.append(f"{value:.3f}")
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.2] + [0.15] * len(metrics))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[i])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/comparison_table.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)
    
    def plot_convergence_analysis(
        self,
        losses_by_method: Dict[str, List[float]],
        window_size: int = 100,
        title: str = "Convergence Analysis",
        save_path: Optional[str] = None
    ):
        """Plot smoothed convergence curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Raw losses
        for method, losses in losses_by_method.items():
            ax1.plot(list(losses), alpha=0.5, linewidth=1, label=method)
        
        ax1.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=10, fontweight='bold')
        ax1.set_title('Training Loss (Raw)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Smoothed losses
        for method, losses in losses_by_method.items():
            # Compute moving average
            smoothed = np.convolve(list(losses), np.ones(window_size)/window_size,
                               mode='valid')
            ax2.plot(smoothed, linewidth=2, label=method)
        
        ax2.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Loss (smoothed)', fontsize=10, fontweight='bold')
        ax2.set_title(f'Training Loss (Smoothed, window={window_size})',
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.set_yscale('log')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.output_dir}/convergence_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close(fig)


if __name__ == "__main__":
    # Test visualization
    print("Testing Results Visualizer...")
    
    visualizer = ResultsVisualizer()
    
    # Test requirement satisfaction rate plot
    rsr_results = {
        "Proposed (Meta-Learning)": 0.92,
        "Transfer Learning": 0.78,
        "Multi-Task Learning": 0.81,
        "Genetic Algorithm": 0.65,
        "Greedy": 0.55,
        "Random": 0.35
    }
    visualizer.plot_requirement_satisfaction_rate(rsr_results)
    
    # Test QoS deviation plot
    qos_dev_results = {
        "Proposed (Meta-Learning)": 0.08,
        "Transfer Learning": 0.15,
        "Multi-Task Learning": 0.12,
        "Genetic Algorithm": 0.20,
        "Greedy": 0.28,
        "Random": 0.42
    }
    visualizer.plot_qos_deviation(qos_dev_results)
    
    # Test cold-start performance
    cold_start_results = {
        "Proposed (Meta-Learning)": {1: 0.65, 3: 0.78, 5: 0.85, 10: 0.92, 20: 0.94},
        "Transfer Learning": {1: 0.35, 3: 0.45, 5: 0.55, 10: 0.72, 20: 0.82},
        "Multi-Task Learning": {1: 0.40, 3: 0.50, 5: 0.60, 10: 0.75, 20: 0.85}
    }
    visualizer.plot_cold_start_performance(cold_start_results, reference_performance=0.95)
    
    # Test adaptation time
    adaptation_times = {
        "Proposed (Meta-Learning)": 180.0,
        "Transfer Learning": 1920.0,
        "Multi-Task Learning": 0.0,
        "Genetic Algorithm": 60.0,
        "Greedy": 0.0,
        "Random": 0.0
    }
    visualizer.plot_adaptation_time(adaptation_times)
    
    # Test individual QoS satisfaction
    qos_individual = {
        "Proposed (Meta-Learning)": {0: 0.95, 1: 0.92, 2: 0.90, 3: 0.93, 4: 0.88},
        "Transfer Learning": {0: 0.85, 1: 0.80, 2: 0.82, 3: 0.80, 4: 0.75},
        "Greedy": {0: 0.70, 1: 0.60, 2: 0.65, 3: 0.58, 4: 0.50}
    }
    visualizer.plot_individual_qos_satisfaction(qos_individual)
    
    # Test radar comparison
    radar_results = {
        "Proposed (Meta-Learning)": [0.92, 0.85, 0.90, 0.88, 0.95],
        "Transfer Learning": [0.78, 0.65, 0.72, 0.70, 0.82],
        "Greedy": [0.55, 0.45, 0.50, 0.48, 0.60]
    }
    radar_metrics = ["RSR", "Utility", "Availability", "Reliability", "Cost Efficiency"]
    visualizer.plot_radar_comparison(radar_results, radar_metrics)
    
    # Test convergence analysis
    convergence_losses = {
        "Proposed (Meta-Learning)": list(np.exp(-np.linspace(0, 5, 1000)) + 0.01),
        "Transfer Learning": list(np.exp(-np.linspace(0, 3, 1000)) + 0.02),
        "Random": list(np.ones(1000) * 0.8)
    }
    visualizer.plot_convergence_analysis(convergence_losses)
    
    print("\nAll visualizations generated successfully!")
