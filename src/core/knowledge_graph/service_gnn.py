"""
Knowledge Graph embedding for service relationships
Uses Graph Neural Networks to encode service dependencies and QoS relationships
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ServiceNode:
    """Service node in knowledge graph"""
    id: int
    name: str
    category: str
    qos_features: np.ndarray
    neighbors: List[int] = None


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer"""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.W = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            adj: Adjacency matrix [N, N]
        """
        N = x.size(0)
        num_heads = self.num_heads

        # Transform
        x = self.W(x)  # [N, out_dim * num_heads]
        x = x.view(N, num_heads, self.out_dim)  # [N, num_heads, out_dim]

        # Compute attention
        x_expanded = x.unsqueeze(0).expand(N, N, num_heads, self.out_dim)
        x_j_expanded = x_expanded.permute(1, 0, 2, 3)

        attention_input = torch.cat([x_expanded, x_j_expanded], dim=-1)
        attention_scores = self.attn(attention_input).squeeze(-1)  # [N, N, num_heads]
        attention_scores = attention_scores / np.sqrt(self.out_dim)

        # Mask attention
        mask = (1 - adj) * -1e9
        attention_scores = attention_scores + mask.unsqueeze(-1)

        # Softmax
        attention = F.softmax(attention_scores, dim=1)  # [N, N, num_heads]

        # Aggregate
        out = torch.matmul(attention, x_j_expanded)  # [N, num_heads, out_dim]
        out = out.view(N, num_heads * self.out_dim)  # [N, num_heads * out_dim]

        return out


class ServiceGNN(nn.Module):
    """Graph Neural Network for service embedding"""

    def __init__(
        self,
        num_services: int,
        input_dim: int = 5,  # QoS features
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_services = num_services
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initial feature embedding
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim // num_heads, num_heads)
            )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features: Node features [N, input_dim]
            adj: Adjacency matrix [N, N]
        """
        # Initial embedding
        x = self.feature_embedding(features)
        x = F.relu(x)
        x = self.dropout(x)

        # GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)

        # Output
        embeddings = self.output_proj(x)

        return embeddings

    def get_service_embedding(self, service_id: int) -> torch.Tensor:
        """Get embedding for a specific service"""
        with torch.no_grad():
            # Create dummy features for single service
            features = torch.zeros(1, self.input_dim)
            adj = torch.ones(1, 1)

            embedding = self.forward(features, adj)
            return embedding[0]


class ServiceKnowledgeGraph:
    """
    Knowledge graph for service relationships and QoS prediction
    """

    def __init__(
        self,
        num_services: int,
        qos_dim: int = 5,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        self.num_services = num_services
        self.qos_dim = qos_dim
        self.device = device

        # Initialize service features (QoS values)
        self.service_features = torch.randn(num_services, qos_dim)

        # Create default adjacency (fully connected - services can be composed)
        self.adjacency = torch.ones(num_services, num_services) - torch.eye(num_services)
        self.adjacency = self.adjacency.to(device)

        # Initialize GNN
        self.gnn = ServiceGNN(
            num_services=num_services,
            input_dim=qos_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            device=device
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=0.001)

    def update_service(
        self,
        service_id: int,
        qos_values: np.ndarray,
        neighbors: List[int] = None
    ):
        """Update service features in the graph"""
        with torch.no_grad():
            self.service_features[service_id] = torch.FloatTensor(qos_values)

            # Update adjacency if neighbors provided
            if neighbors is not None:
                self.adjacency[service_id] = 0
                self.adjacency[service_id, neighbors] = 1
                self.adjacency[neighbors, service_id]  # 1

    def compute_embeddings(self) -> torch.Tensor:
        """Compute embeddings for all services"""
        return self.gnn(self.service_features.to(self.device), self.adjacency)

    def get_service_similarity(self, s1: int, s2: int) -> float:
        """Compute similarity between two services"""
        embeddings = self.compute_embeddings()
        emb1 = embeddings[s1]
        emb2 = embeddings[s2]

        # Cosine similarity
        similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))

        return similarity.item()

    def predict_qos(
        self,
        service_id: int,
        context_services: List[int] = None
    ) -> np.ndarray:
        """Predict QoS for a service based on graph structure"""
        with torch.no_grad():
            embeddings = self.compute_embeddings()

            if context_services is None:
                # Use learned embedding to predict
                emb = embeddings[service_id]
                # Simple decoder to QoS
                qos = torch.sigmoid(emb[:self.qos_dim]) * 1000
                return qos.cpu().numpy()

            # Aggregate from context services
            context_emb = embeddings[context_services].mean(dim=0)
            emb = (embeddings[service_id] + context_emb) / 2

            qos = torch.sigmoid(emb[:self.qos_dim]) * 1000
            return qos.cpu().numpy()

    def find_similar_services(
        self,
        service_id: int,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most similar services to given service"""
        embeddings = self.compute_embeddings()
        target = embeddings[service_id]

        # Compute similarities
        similarities = F.cosine_similarity(
            target.unsqueeze(0),
            embeddings
        )

        # Get top-k (excluding self)
        _, indices = torch.topk(similarities, top_k + 1)
        results = []

        for idx in indices:
            if idx.item() != service_id:
                results.append((idx.item(), similarities[idx].item()))

        return results[:top_k]

    def train_step(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Train GNN on QoS prediction task

        Args:
            features: Service features [N, qos_dim]
            adj: Adjacency matrix [N, N]
            targets: Target QoS values [N, qos_dim]
        """
        self.optimizer.zero_grad()

        embeddings = self.gnn(features.to(self.device), adj.to(self.device))

        # Predict QoS from embeddings
        predictions = torch.sigmoid(embeddings[:, :self.qos_dim]) * 1000

        # Loss
        loss = F.mse_loss(predictions, targets.to(self.device))

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()


class GNNComposer:
    """
    Service composer using knowledge graph embeddings
    """

    def __init__(
        self,
        num_services: int,
        state_dim: int = 7,
        action_dim: int = 30,
        embedding_dim: int = 64
    ):
        self.knowledge_graph = ServiceKnowledgeGraph(
            num_services=num_services,
            qos_dim=5,
            embedding_dim=embedding_dim
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def compose(
        self,
        workflow,
        available_services: List,
        requirements: np.ndarray
    ) -> List[int]:
        """Compose services using GNN-based policy"""
        # Compute embeddings
        embeddings = self.knowledge_graph.compute_embeddings()

        selected = []

        for node in workflow.nodes:
            # Create state
            state = self._create_state(node, requirements, len(available_services))

            # Get service embeddings for available services
            service_emb = embeddings[:len(available_services)]

            # Combine state with average embedding
            combined = torch.cat([
                torch.FloatTensor(state),
                service_emb.mean(dim=0)
            ])

            # Get action probabilities
            with torch.no_grad():
                logits = self.policy(combined.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                action = int(torch.argmax(probs).item())

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

    def get_name(self) -> str:
        return "Knowledge Graph (GNN)"


if __name__ == "__main__":
    print("Testing Knowledge Graph...")

    # Create knowledge graph
    kg = ServiceKnowledgeGraph(num_services=50)

    # Update some services
    for i in range(50):
        qos = np.random.rand(5) * 1000
        kg.update_service(i, qos)

    # Compute embeddings
    embeddings = kg.compute_embeddings()
    print(f"Embeddings shape: {embeddings.shape}")

    # Find similar services
    similar = kg.find_similar_services(0, top_k=5)
    print(f"Similar to service 0: {similar}")

    # Train step
    features = torch.randn(50, 5)
    adj = torch.bernoulli(torch.ones(50, 50) * 0.3)
    adj = adj - torch.diag(torch.diag(adj))  # Remove self-loops
    targets = torch.randn(50, 5) * 1000

    loss = kg.train_step(features, adj, targets)
    print(f"Training loss: {loss:.4f}")

    # Test GNN composer
    print("\nTesting GNN Composer...")
    composer = GNNComposer(num_services=50)

    workflow = type('Workflow', (), {
        'nodes': [{"position": i} for i in range(3)]
    })()

    services = [type('Service', (), {'id': i})() for i in range(30)]
    requirements = np.array([500, 100, 0.90, 0.85, 1.0])

    selected = composer.compose(workflow, services, requirements)
    print(f"Selected services: {selected}")
