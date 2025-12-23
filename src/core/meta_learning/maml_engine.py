"""
Meta-learning engine for cross-domain QoS adaptation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import copy


class CompositionPolicy(nn.Module):
    """Base policy network for service composition"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Build policy network
        layers = []
        input_dim = state_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax)"""
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> int:
        """Sample an action from the policy"""
        probs = self.get_action_probs(state)
        
        if deterministic:
            return int(torch.argmax(probs).item())
        else:
            return int(torch.multinomial(probs, 1).item())


class MAMLComposer:
    """
    Model-Agnostic Meta-Learning for QoS-aware service composition
    Implements gradient-based meta-learning for rapid cross-domain adaptation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        hidden_dim: int = 256
    ):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize meta-policy
        self.meta_policy = CompositionPolicy(
            state_dim, action_dim, hidden_dim
        )
        
        # Domain-specific policies (adapted versions)
        self.domain_policies: Dict[str, nn.Module] = {}
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.meta_policy.parameters(),
            lr=meta_lr
        )
        
        # Training history
        self.meta_losses: List[float] = []
        self.inner_losses: List[List[float]] = []
    
    def inner_loop_adapt(
        self,
        support_states: List[torch.Tensor],
        support_actions: List[torch.Tensor],
        support_rewards: List[float],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Inner loop: adapt policy to specific domain/task using support set
        
        Args:
            support_states: List of state tensors
            support_actions: List of action indices
            support_rewards: List of reward values
            num_steps: Number of gradient steps (default: self.num_inner_steps)
        
        Returns:
            Adapted policy copy
        """
        num_steps = num_steps or self.num_inner_steps
        
        # Create a copy of meta-policy for adaptation
        adapted_policy = copy.deepcopy(self.meta_policy)
        adapted_policy.train()
        
        # Convert to tensors
        states_tensor = torch.stack(support_states)
        actions_tensor = torch.tensor(support_actions, dtype=torch.long)
        rewards_tensor = torch.tensor(support_rewards, dtype=torch.float32)
        
        # Inner loop gradient steps
        for step in range(num_steps):
            # Forward pass
            logits = adapted_policy(states_tensor)
            
            # Compute loss (policy gradient)
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[
                torch.arange(len(actions_tensor)), actions_tensor
            ]
            
            # Policy loss: negative expected reward
            loss = -(selected_log_probs * rewards_tensor).mean()
            
            # Gradient step
            adapted_policy.zero_grad()
            loss.backward()
            
            # Manually update parameters
            with torch.no_grad():
                for param in adapted_policy.parameters():
                    if param.grad is not None:
                        param.data -= self.inner_lr * param.grad
        
        return adapted_policy
    
    def compute_query_loss(
        self,
        policy: nn.Module,
        query_states: List[torch.Tensor],
        query_actions: List[torch.Tensor],
        query_rewards: List[float]
    ) -> torch.Tensor:
        """
        Compute loss on query set for meta-optimization
        
        Args:
            policy: The adapted policy
            query_states: List of state tensors
            query_actions: List of action indices
            query_rewards: List of reward values
        
        Returns:
            Query loss tensor
        """
        states_tensor = torch.stack(query_states)
        actions_tensor = torch.tensor(query_actions, dtype=torch.long)
        rewards_tensor = torch.tensor(query_rewards, dtype=torch.float32)
        
        # Forward pass
        logits = policy(states_tensor)
        
        # Compute policy loss
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[
            torch.arange(len(actions_tensor)), actions_tensor
        ]
        
        loss = -(selected_log_probs * rewards_tensor).mean()
        
        return loss
    
    def outer_loop_update(
        self,
        batch_domains: List[Dict[str, List]]
    ) -> float:
        """
        Outer loop: meta-optimize across batch of domains
        
        Args:
            batch_domains: List of domain tasks, each containing:
                - support_states, support_actions, support_rewards
                - query_states, query_actions, query_rewards
        
        Returns:
            Meta loss value
        """
        domain_losses = []
        
        # Process each domain in the batch
        for domain_data in batch_domains:
            # Inner loop adaptation
            adapted_policy = self.inner_loop_adapt(
                domain_data["support_states"],
                domain_data["support_actions"],
                domain_data["support_rewards"]
            )
            
            # Compute query loss
            query_loss = self.compute_query_loss(
                adapted_policy,
                domain_data["query_states"],
                domain_data["query_actions"],
                domain_data["query_rewards"]
            )
            
            domain_losses.append(query_loss)
        
        # Average loss across domains
        meta_loss = torch.stack(domain_losses).mean()
        
        # Meta-gradient computation
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.meta_policy.parameters(),
            max_norm=1.0
        )
        
        # Meta-update
        self.meta_optimizer.step()
        
        self.meta_losses.append(float(meta_loss.item()))
        
        return float(meta_loss.item())
    
    def meta_train(
        self,
        domains_data: Dict[str, Dict],
        num_iterations: int = 5000,
        batch_size: int = 8,
        support_size: int = 10,
        query_size: int = 20,
        log_interval: int = 100
    ):
        """
        Meta-training loop across all domains
        
        Args:
            domains_data: Dictionary mapping domain names to their data
            num_iterations: Number of meta-iterations
            batch_size: Number of domains per meta-batch
            support_size: Number of samples in support set
            query_size: Number of samples in query set
            log_interval: Logging interval
        """
        domain_names = list(domains_data.keys())
        
        print(f"Starting meta-training on {len(domain_names)} domains")
        print(f"Meta-iterations: {num_iterations}, Batch size: {batch_size}")
        print(f"Support size: {support_size}, Query size: {query_size}")
        
        for iteration in range(num_iterations):
            # Sample batch of domains
            batch_domain_names = np.random.choice(
                domain_names, 
                size=min(batch_size, len(domain_names)),
                replace=False
            )
            
            batch_domains = []
            for domain_name in batch_domain_names:
                domain_data = domains_data[domain_name]
                
                # Sample support and query sets
                indices = np.random.permutation(len(domain_data["states"]))
                support_indices = indices[:support_size]
                query_indices = indices[support_size:support_size + query_size]
                
                domain_task = {
                    "support_states": [
                        domain_data["states"][i] for i in support_indices
                    ],
                    "support_actions": [
                        domain_data["actions"][i] for i in support_indices
                    ],
                    "support_rewards": [
                        domain_data["rewards"][i] for i in support_indices
                    ],
                    "query_states": [
                        domain_data["states"][i] for i in query_indices
                    ],
                    "query_actions": [
                        domain_data["actions"][i] for i in query_indices
                    ],
                    "query_rewards": [
                        domain_data["rewards"][i] for i in query_indices
                    ]
                }
                batch_domains.append(domain_task)
            
            # Outer loop update
            meta_loss = self.outer_loop_update(batch_domains)
            
            # Logging
            if iteration % log_interval == 0:
                avg_loss = np.mean(self.meta_losses[-log_interval:])
                print(
                    f"Iteration {iteration}: Meta Loss = {meta_loss:.4f}, "
                    f"Avg (last {log_interval}) = {avg_loss:.4f}"
                )
        
        print("Meta-training complete!")
    
    def adapt_to_domain(
        self,
        support_states: List[torch.Tensor],
        support_actions: List[torch.Tensor],
        support_rewards: List[float],
        num_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Adapt meta-policy to a new domain using few-shot learning
        
        Args:
            support_states: Support set states
            support_actions: Support set actions
            support_rewards: Support set rewards
            num_steps: Number of adaptation steps
        
        Returns:
            Adapted policy for the new domain
        """
        print(f"Adapting to new domain with {len(support_states)} samples...")
        
        adapted_policy = self.inner_loop_adapt(
            support_states,
            support_actions,
            support_rewards,
            num_steps=num_steps
        )
        
        print("Adaptation complete!")
        return adapted_policy
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'meta_policy_state_dict': self.meta_policy.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'meta_losses': self.meta_losses,
            'config': {
                'inner_lr': self.inner_lr,
                'meta_lr': self.meta_lr,
                'num_inner_steps': self.num_inner_steps,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        self.meta_policy.load_state_dict(checkpoint['meta_policy_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_losses = checkpoint['meta_losses']
        print(f"Checkpoint loaded from {filepath}")


class ReptileComposer:
    """
    Reptile (First-Order MAML) for faster meta-training
    Simplified version that doesn't compute second-order derivatives
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        num_inner_steps: int = 5,
        hidden_dim: int = 256
    ):
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        self.meta_policy = CompositionPolicy(state_dim, action_dim, hidden_dim)
        self.meta_optimizer = torch.optim.Adam(
            self.meta_policy.parameters(),
            lr=meta_lr
        )
        self.meta_losses: List[float] = []
    
    def adapt_step(self, policy: nn.Module, loss: torch.Tensor):
        """Single adaptation step"""
        policy.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in policy.parameters():
                if param.grad is not None:
                    param.data -= self.inner_lr * param.grad
    
    def meta_train_reptile(
        self,
        domains_data: Dict[str, Dict],
        num_iterations: int = 5000,
        batch_size: int = 8,
        support_size: int = 10
    ):
        """
        Reptile meta-training (first-order)
        """
        domain_names = list(domains_data.keys())
        
        print(f"Reptile meta-training on {len(domain_names)} domains")
        
        for iteration in range(num_iterations):
            # Sample batch of domains
            batch_domain_names = np.random.choice(
                domain_names,
                size=min(batch_size, len(domain_names)),
                replace=False
            )
            
            # Initialize gradient accumulator
            theta_grad: List[torch.Tensor] = []
            for param in self.meta_policy.parameters():
                theta_grad.append(torch.zeros_like(param.data))
            
            for domain_name in batch_domain_names:
                domain_data = domains_data[domain_name]
                
                # Create copy for adaptation
                adapted_policy = copy.deepcopy(self.meta_policy)
                
                # Inner loop adaptation
                indices = np.random.choice(
                    len(domain_data["states"]),
                    size=min(support_size, len(domain_data["states"])),
                    replace=False
                )
                
                for _ in range(self.num_inner_steps):
                    idx = np.random.choice(indices)
                    state = domain_data["states"][idx:idx+1]
                    action = torch.tensor([domain_data["actions"][idx]])
                    reward = torch.tensor([domain_data["rewards"][idx]])
                    
                    logits = adapted_policy(state)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected_log_probs = log_probs[0, action]
                    loss = -(selected_log_probs * reward).mean()
                    
                    self.adapt_step(adapted_policy, loss)
                
                # Accumulate gradient toward adapted parameters
                for i, (meta_param, adapted_param) in enumerate(
                    zip(self.meta_policy.parameters(),
                        adapted_policy.parameters())
                ):
                    theta_grad[i] = theta_grad[i] + (adapted_param.data - meta_param.data)
            
            # Meta-update (Reptile)
            self.meta_optimizer.zero_grad()
            for i, param in enumerate(self.meta_policy.parameters()):
                param.grad = -theta_grad[i] / len(batch_domain_names)
            
            self.meta_optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Reptile Iteration {iteration} complete")
        
        print("Reptile meta-training complete!")


if __name__ == "__main__":
    # Test MAML implementation
    print("Testing MAML Composer...")
    
    state_dim = 64
    action_dim = 20
    
    maml = MAMLComposer(
        state_dim=state_dim,
        action_dim=action_dim,
        inner_lr=0.01,
        meta_lr=0.001
    )
    
    print(f"Meta-policy initialized with {sum(p.numel() for p in maml.meta_policy.parameters())} parameters")
    
    # Create dummy domains data
    domains_data = {
        "domain1": {
            "states": [torch.randn(state_dim) for _ in range(100)],
            "actions": [np.random.randint(0, action_dim) for _ in range(100)],
            "rewards": [np.random.random() for _ in range(100)]
        },
        "domain2": {
            "states": [torch.randn(state_dim) for _ in range(100)],
            "actions": [np.random.randint(0, action_dim) for _ in range(100)],
            "rewards": [np.random.random() for _ in range(100)]
        },
        "domain3": {
            "states": [torch.randn(state_dim) for _ in range(100)],
            "actions": [np.random.randint(0, action_dim) for _ in range(100)],
            "rewards": [np.random.random() for _ in range(100)]
        }
    }
    
    # Test single meta-iteration
    batch_domains = [
        {
            "support_states": domains_data["domain1"]["states"][:10],
            "support_actions": domains_data["domain1"]["actions"][:10],
            "support_rewards": domains_data["domain1"]["rewards"][:10],
            "query_states": domains_data["domain1"]["states"][10:30],
            "query_actions": domains_data["domain1"]["actions"][10:30],
            "query_rewards": domains_data["domain1"]["rewards"][10:30]
        }
    ]
    
    loss = maml.outer_loop_update(batch_domains)
    print(f"First meta-iteration loss: {loss:.4f}")
    
    print("\nTesting Reptile Composer...")
    reptile = ReptileComposer(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Test Reptile
    reptile.meta_train_reptile(
        domains_data,
        num_iterations=10,
        batch_size=2,
        support_size=5
    )
    
    print("Tests complete!")
