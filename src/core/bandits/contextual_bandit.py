"""
Contextual Bandit implementation for exploration in service composition
This module provides contextual bandit algorithms for service selection
with exploration-exploitation trade-off.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn


class ContextualBanditPolicy(nn.Module):
    """Neural network for contextual bandit policy"""

    def __init__(self, context_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class ContextualBandit:
    """
    Contextual Bandit for service selection with uncertainty-aware exploration

    Uses Thompson Sampling or UCB for exploration in the context of
    service composition.
    """

    def __init__(
        self,
        context_dim: int,
        num_services: int,
        hidden_dim: int = 128,
        algorithm: str = "thompson",
        alpha: float = 1.0,
        device: str = "cpu"
    ):
        self.context_dim = context_dim
        self.num_services = num_services
        self.algorithm = algorithm
        self.alpha = alpha
        self.device = device

        # Neural network policy
        self.policy = ContextualBanditPolicy(context_dim, num_services, hidden_dim)
        self.policy.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=0.001)

        # Beta parameters for Thompson Sampling (Beta-Bernoulli bandit)
        self.beta_params = {
            'alpha': torch.ones(num_services) * alpha,
            'beta': torch.ones(num_services) * alpha
        }

        # Reward history
        self.rewards = []
        self.contexts = []
        self.actions = []

        # Q-value estimates
        self.Q = np.zeros(num_services)
        self.N = np.zeros(num_services)

    def select_action(
        self,
        context: np.ndarray,
        epsilon: float = 0.1
    ) -> Tuple[int, np.ndarray]:
        """
        Select service using contextual bandit policy

        Args:
            context: State/context features
            epsilon: Exploration rate

        Returns:
            action: Selected service index
            probs: Action probabilities
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy(context_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Exploration
        if np.random.random() < epsilon:
            action = np.random.randint(self.num_services)
        else:
            # Exploitation with algorithm choice
            if self.algorithm == "thompson":
                action = self._thompson_sample()
            elif self.algorithm == "ucb":
                action = self._ucb_select()
            else:
                action = int(np.argmax(probs))

        return action, probs

    def _thompson_sample(self) -> int:
        """Thompson Sampling for exploration"""
        samples = np.random.beta(
            self.beta_params['alpha'].cpu().numpy(),
            self.beta_params['beta'].cpu().numpy()
        )
        return int(np.argmax(samples))

    def _ucb_select(self) -> int:
        """Upper Confidence Bound selection"""
        if np.sum(self.N) == 0:
            return np.random.randint(self.num_services)

        ucb_values = self.Q + self.alpha * np.sqrt(
            np.log(np.sum(self.N) + 1) / (self.N + 1)
        )
        return int(np.argmax(ucb_values))

    def update(
        self,
        context: np.ndarray,
        action: int,
        reward: float
    ):
        """Update bandit with new observation"""
        # Store transition
        self.contexts.append(context)
        self.actions.append(action)
        self.rewards.append(reward)

        # Update beta parameters (for Thompson Sampling)
        if reward > 0:
            self.beta_params['alpha'][action] += reward
        else:
            self.beta_params['beta'][action] += (1 - reward)

        # Update Q-value (incremental mean)
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

        # Neural network update (simplified)
        self._update_policy(context, action, reward)

    def _update_policy(
        self,
        context: np.ndarray,
        action: int,
        reward: float
    ):
        """Update neural network policy"""
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)

        self.optimizer.zero_grad()

        logits = self.policy(context_tensor)
        log_probs = torch.log_softmax(logits, dim=-1)

        loss = -log_probs[0, action] * reward_tensor

        loss.backward()
        self.optimizer.step()

    def get_stats(self) -> Dict:
        """Get bandit statistics"""
        return {
            'total_pulls': len(self.rewards),
            'mean_reward': np.mean(self.rewards) if self.rewards else 0,
            'action_counts': self.N.tolist(),
            'Q_values': self.Q.tolist(),
            'algorithm': self.algorithm
        }

    def reset(self):
        """Reset bandit to initial state"""
        self.beta_params = {
            'alpha': torch.ones(self.num_services) * self.alpha,
            'beta': torch.ones(self.num_services) * self.alpha
        }
        self.Q = np.zeros(self.num_services)
        self.N = np.zeros(self.num_services)
        self.rewards = []
        self.contexts = []
        self.actions = []


class BanditComposer:
    """
    Service composer using contextual bandits for exploration
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        algorithm: str = "thompson",
        alpha: float = 1.0
    ):
        self.bandit = ContextualBandit(
            context_dim=state_dim,
            num_services=action_dim,
            algorithm=algorithm,
            alpha=alpha
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Compose services using bandit policy"""
        selected = []

        for node in workflow.nodes:
            state = self._create_state(node, requirements, len(available_services))
            action, _ = self.bandit.select_action(state, epsilon=0.1)
            action = min(action, len(available_services) - 1)
            selected.append(available_services[action].id)

        return selected

    def _create_state(
        self,
        node,
        requirements: np.ndarray,
        num_services: int
    ) -> np.ndarray:
        """Create state representation"""
        state = np.zeros(self.state_dim)
        state[0] = min(node.get("position", 0), 9) / 10.0
        state[1:6] = requirements / 1000.0
        state[6] = num_services / 100.0
        return state

    def update(self, context: np.ndarray, action: int, reward: float):
        """Update bandit with reward"""
        self.bandit.update(context, action, reward)

    def get_name(self) -> str:
        return f"Contextual Bandit ({self.bandit.algorithm})"


if __name__ == "__main__":
    # Test contextual bandit
    bandit = ContextualBandit(context_dim=7, num_services=30)

    print("Testing Contextual Bandit...")

    # Simulate interactions
    for t in range(100):
        context = np.random.randn(7)
        action, probs = bandit.select_action(context, epsilon=0.2)
        reward = np.random.random() * (action / 30)  # Higher actions get higher rewards
        bandit.update(context, action, reward)

        if (t + 1) % 20 == 0:
            stats = bandit.get_stats()
            print(f"Step {t+1}: Mean reward = {stats['mean_reward']:.3f}")

    print("\nBandit statistics:")
    print(bandit.get_stats())
