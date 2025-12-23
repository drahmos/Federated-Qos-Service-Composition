"""
Configuration management for Meta-QoS Composition Framework
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class MetaLearningConfig:
    """Meta-learning configuration"""
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    num_inner_steps: int = 5
    num_meta_iterations: int = 5000
    support_set_size: int = 10
    query_set_size: int = 20
    batch_size: int = 8


@dataclass
class RLConfig:
    """Reinforcement learning configuration"""
    algorithm: str = "PPO"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4


@dataclass
class UncertaintyConfig:
    """Uncertainty estimation configuration"""
    dropout_rate: float = 0.2
    mc_samples: int = 100
    uncertainty_threshold: float = 0.5
    beta: float = 2.0  # UCB exploration parameter


@dataclass
class BanditConfig:
    """Contextual bandit configuration"""
    alpha: float = 1.0
    context_dim: int = 256


@dataclass
class KnowledgeGraphConfig:
    """Knowledge graph configuration"""
    embedding_dim: int = 256
    num_gnn_layers: int = 3
    num_heads: int = 8
    hidden_dim: int = 512


@dataclass
class DomainConfig:
    """Domain-specific configuration"""
    name: str
    num_services: int
    num_workflows: int
    num_qos_attrs: int = 5


@dataclass
class TrainingConfig:
    """Overall training configuration"""
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    bandits: BanditConfig = field(default_factory=BanditConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    seed: int = 42
    log_interval: int = 100
    checkpoint_interval: int = 500
    
    # Domains for training
    train_domains: List[str] = field(default_factory=lambda: ["healthcare", "fintech", "ecommerce"])
    val_domains: List[str] = field(default_factory=lambda: ["iot", "travel"])
    test_domains: List[str] = field(default_factory=lambda: ["education"])
    
    # Output directories
    checkpoint_dir: str = "results/checkpoints"
    log_dir: str = "results/logs"
    plot_dir: str = "results/plots"
    
    def __post_init__(self):
        """Create output directories"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)


# Global configuration instance
config = TrainingConfig()
