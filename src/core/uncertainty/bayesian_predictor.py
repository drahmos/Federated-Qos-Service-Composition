"""
Uncertainty estimation module for QoS predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class BayesianQoSPredictor(nn.Module):
    """
    Bayesian Neural Network with Monte Carlo Dropout for uncertainty estimation
    Provides epistemic uncertainty estimates for QoS predictions
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Output layer (predict QoS values)
        layers.append(nn.Linear(hidden_dim, 5))  # 5 QoS attributes
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            QoS predictions of shape (batch_size, 5)
        """
        return self.network(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict QoS values with uncertainty estimates using MC Dropout
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            training: Whether to keep dropout enabled
        
        Returns:
            mean_predictions: Mean QoS predictions
            uncertainty: Standard deviation of predictions
        """
        self.train(training)  # Enable dropout if training=True
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch_size, 5)
        
        # Compute statistics
        mean_predictions = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_predictions, uncertainty


class UncertaintyAwareExplorer:
    """
    Exploration strategy that uses uncertainty estimates to guide exploration
    Implements UCB (Upper Confidence Bound) style exploration
    """
    
    def __init__(
        self,
        predictor: BayesianQoSPredictor,
        uncertainty_threshold: float = 0.5,
        exploration_beta: float = 2.0
    ):
        self.predictor = predictor
        self.uncertainty_threshold = uncertainty_threshold
        self.exploration_beta = exploration_beta
    
    def compute_ucb_score(
        self,
        mean_qos: torch.Tensor,
        uncertainty: torch.Tensor,
        requirements: torch.Tensor,
        qos_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Upper Confidence Bound score for each service candidate
        
        Args:
            mean_qos: Predicted mean QoS values
            uncertainty: Uncertainty estimates
            requirements: QoS requirements
            qos_weights: Weights for each QoS attribute
        
        Returns:
            UCB scores for each candidate
        """
        # Compute satisfaction score (how well predictions meet requirements)
        satisfaction = self._compute_satisfaction_score(mean_qos, requirements)
        
        # UCB = satisfaction + beta * uncertainty
        ucb_scores = satisfaction + self.exploration_beta * uncertainty.mean(dim=-1)
        
        return ucb_scores
    
    def _compute_satisfaction_score(
        self,
        predicted_qos: torch.Tensor,
        requirements: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute how well predictions satisfy requirements
        
        Args:
            predicted_qos: Shape (batch_size, 5)
            requirements: Shape (5,)
        
        Returns:
            Satisfaction scores, shape (batch_size,)
        """
        # For minimization attributes (response_time, cost)
        min_attrs = [0, 4]
        # For maximization attributes (throughput, availability, reliability)
        max_attrs = [1, 2, 3]
        
        satisfaction = torch.zeros(predicted_qos.size(0))
        
        for i in min_attrs:
            # Lower is better - normalize and invert
            normalized = torch.clamp(predicted_qos[:, i] / (requirements[i] * 1.5), 0, 1)
            satisfaction += (1.0 - normalized)
        
        for i in max_attrs:
            # Higher is better
            normalized = torch.clamp(predicted_qos[:, i] / requirements[i], 0, 2)
            satisfaction += torch.clamp(normalized, 0, 1)
        
        return satisfaction / 5.0
    
    def select_service(
        self,
        candidate_services: List[torch.Tensor],
        requirements: torch.Tensor,
        exploration_rate: float = 0.1
    ) -> int:
        """
        Select a service using uncertainty-guided exploration
        
        Args:
            candidate_services: List of service features
            requirements: QoS requirements
            exploration_rate: Base exploration probability
        
        Returns:
            Index of selected service
        """
        if not candidate_services:
            return -1
        
        # Stack candidates
        candidates_tensor = torch.stack(candidate_services)
        
        # Predict with uncertainty
        mean_qos, uncertainty = self.predictor.predict_with_uncertainty(
            candidates_tensor,
            num_samples=50
        )
        
        # Compute overall uncertainty for each candidate
        overall_uncertainty = uncertainty.mean(dim=1)
        
        # Adjust exploration rate based on uncertainty
        max_uncertainty = overall_uncertainty.max()
        adjusted_exploration = exploration_rate + (max_uncertainty * 0.5)
        adjusted_exploration = min(adjusted_exploration, 0.5)
        
        # Explore or exploit
        if np.random.random() < adjusted_exploration:
            # Explore: select based on UCB
            ucb_scores = self.compute_ucb_score(
                mean_qos,
                uncertainty,
                requirements,
                torch.ones(5)  # Equal weights
            )
            return int(torch.argmax(ucb_scores).item())
        else:
            # Exploit: select best predicted performance
            satisfaction = self._compute_satisfaction_score(mean_qos, requirements)
            return int(torch.argmax(satisfaction).item())


class EnsembleQoSPredictor:
    """
    Ensemble of multiple predictors for robust uncertainty estimation
    Combines predictions from multiple models
    """
    
    def __init__(
        self,
        input_dim: int,
        num_models: int = 5,
        hidden_dim: int = 256
    ):
        self.num_models = num_models
        self.models = nn.ModuleList([
            BayesianQoSPredictor(input_dim, hidden_dim)
            for _ in range(num_models)
        ])
    
    def predict(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict using ensemble
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty
        
        Returns:
            mean_predictions: Mean across ensemble
            uncertainty: Standard deviation across ensemble (if requested)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch_size, 5)
        
        mean_predictions = predictions.mean(dim=0)
        
        if return_uncertainty:
            uncertainty = predictions.std(dim=0)
            return mean_predictions, uncertainty
        
        return mean_predictions, None
    
    def fit_ensemble(
        self,
        train_data: List[Tuple[torch.Tensor, torch.Tensor]],
        epochs: int = 100
    ):
        """
        Train each model in the ensemble
        
        Args:
            train_data: List of (input, target) pairs
            epochs: Training epochs
        """
        # Create slightly different training sets for each model
        for i, model in enumerate(self.models):
            # Bootstrap sample
            indices = np.random.choice(
                len(train_data),
                size=len(train_data),
                replace=True
            )
            bootstrap_data = [train_data[idx] for idx in indices]
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                
                for x, y in bootstrap_data:
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    print(f"Model {i}, Epoch {epoch + 1}: Loss = {epoch_loss / len(bootstrap_data):.4f}")


class CalibrationChecker:
    """
    Checks if uncertainty estimates are well-calibrated
    """
    
    @staticmethod
    def compute_calibration_error(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        actual_values: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            predictions: Predicted QoS values
            uncertainties: Uncertainty estimates
            actual_values: Actual QoS values
            num_bins: Number of calibration bins
        
        Returns:
            ECE value
        """
        errors = np.abs(predictions - actual_values)
        normalized_errors = (errors < 0.1).astype(float)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        ece = 0.0
        
        for i in range(num_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (uncertainties >= lower) & (uncertainties < upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy = np.mean(normalized_errors[in_bin])
                avg_confidence = np.mean(uncertainties[in_bin])
                ece += np.abs(accuracy - avg_confidence) * prop_in_bin
        
        return ece
    
    @staticmethod
    def plot_reliability_diagram(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        actual_values: np.ndarray
    ):
        """Plot reliability diagram for visualization"""
        import matplotlib.pyplot as plt
        
        errors = np.abs(predictions - actual_values)
        normalized_errors = (errors < 0.1).astype(float)
        
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        
        bin_accs = []
        bin_confs = []
        
        for i in range(num_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (uncertainties >= lower) & (uncertainties < upper)
            
            if np.sum(in_bin) > 0:
                bin_accs.append(np.mean(normalized_errors[in_bin]))
                bin_confs.append(np.mean(uncertainties[in_bin]))
            else:
                bin_accs.append(0)
                bin_confs.append(0.5 * (lower + upper))
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.plot(bin_confs, bin_accs, 'o-', label='Model')
        plt.xlabel('Confidence (Uncertainty)')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()


if __name__ == "__main__":
    # Test uncertainty estimation
    print("Testing Bayesian QoS Predictor...")
    
    input_dim = 64
    predictor = BayesianQoSPredictor(input_dim, hidden_dim=128)
    
    # Test predictions
    batch_size = 10
    x = torch.randn(batch_size, input_dim)
    
    # Single prediction
    pred = predictor(x)
    print(f"Single prediction shape: {pred.shape}")
    
    # Prediction with uncertainty
    mean, uncertainty = predictor.predict_with_uncertainty(x, num_samples=50)
    print(f"Mean prediction shape: {mean.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Average uncertainty: {uncertainty.mean():.4f}")
    
    # Test exploration
    explorer = UncertaintyAwareExplorer(predictor)
    
    candidates = [torch.randn(input_dim) for _ in range(5)]
    requirements = torch.tensor([500.0, 200.0, 0.97, 0.96, 1.5])
    
    selected = explorer.select_service(candidates, requirements)
    print(f"Selected service index: {selected}")
    
    # Test ensemble
    print("\nTesting Ensemble QoS Predictor...")
    ensemble = EnsembleQoSPredictor(input_dim, num_models=3)
    mean_pred, ens_uncertainty = ensemble.predict(x, return_uncertainty=True)
    print(f"Ensemble prediction shape: {mean_pred.shape}")
    if ens_uncertainty is not None:
        print(f"Ensemble uncertainty shape: {ens_uncertainty.shape}")
    
    # Test calibration
    print("\nTesting Calibration Checker...")
    pred_np = np.random.rand(100, 5)
    unc_np = np.random.rand(100, 5)
    actual_np = np.random.rand(100, 5)
    
    ece = CalibrationChecker.compute_calibration_error(
        pred_np.flatten(), unc_np.flatten(), actual_np.flatten()
    )
    print(f"Expected Calibration Error: {ece:.4f}")
    
    print("\nAll tests passed!")
