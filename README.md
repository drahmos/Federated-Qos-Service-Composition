# Federated Learning for QoS-aware Web Service Composition

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-1.21+-orange.svg)](https://numpy.org/)

A comprehensive implementation of **federated learning** applied to **Quality-of-Service (QoS) aware web service composition**. This system enables multiple service providers to collaboratively train a machine learning model for predicting service quality without sharing raw service data, preserving privacy while improving prediction accuracy.

## 🌟 Features

- **🔐 Privacy-Preserving Learning**: No raw service data shared between clients
- **🤖 Neural Network QoS Prediction**: Predicts service quality from multiple metrics
- **🔄 Multiple Aggregation Methods**: 
  - Standard FedAvg (Federated Averaging)
  - Adaptive FedAvg with learning rate
  - Secure Aggregation with Differential Privacy
- **🎯 Intelligent Service Composition**: Constraint-based optimal service selection
- **📊 5 QoS Metrics**: Response time, throughput, availability, reliability, cost
- **⚡ Efficient**: Lightweight model, fast training, minimal network overhead

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/federated-qos-service-composition.git
cd federated-qos-service-composition

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

Run the complete demonstration:

```bash
python federated_qos_demo.py
```

This will:
1. Generate 100 synthetic web services
2. Distribute them across 5 clients
3. Train a federated model over 15 rounds
4. Demonstrate service composition with QoS constraints
5. Compare different aggregation methods

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                  Federated QoS Server                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Global Model Aggregation                  │  │
│  │  • FedAvg  • Adaptive  • Secure (DP)             │  │
│  └───────────────────────────────────────────────────┘  │
└───────────┬─────────────────────────────┬───────────────┘
            │                             │
    ┌───────▼────────┐           ┌───────▼────────┐
    │   Client 1     │    ...    │   Client N     │
    │ ┌────────────┐ │           │ ┌────────────┐ │
    │ │Local Model │ │           │ │Local Model │ │
    │ │  Training  │ │           │ │  Training  │ │
    │ └────────────┘ │           │ └────────────┘ │
    │ ┌────────────┐ │           │ ┌────────────┐ │
    │ │  Service   │ │           │ │  Service   │ │
    │ │ Repository │ │           │ │ Repository │ │
    │ └────────────┘ │           │ └────────────┘ │
    └────────────────┘           └────────────────┘
```

### Neural Network Architecture

```
Input Layer (5 QoS features)
    ↓
Hidden Layer (10 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
    ↓
Quality Score (0-1)
```

## 💡 Usage Examples

### Basic Client Usage

```python
from federated_qos_client import FederatedQoSClient, WebService, QoSMetrics

# Create a client
client = FederatedQoSClient("client_1", learning_rate=0.01)
client.initialize_model()

# Add a service
service = WebService(
    service_id="auth_service_1",
    service_type="authentication",
    qos=QoSMetrics(
        response_time=50.0,      # ms
        throughput=100.0,        # req/s
        availability=0.99,       # 99%
        reliability=0.98,        # 98%
        cost=0.5                 # cost units
    )
)
client.add_service(service)

# Train locally
client.train_local(epochs=10)

# Predict quality score
score = client.predict_qos_score(service)
print(f"QoS Score: {score:.4f}")
```

### Server Coordination

```python
from federated_qos_server import FederatedQoSServer, FederatedTrainingCoordinator

# Initialize server
server = FederatedQoSServer(num_clients=5)
server.initialize_global_model()

# Create coordinator
coordinator = FederatedTrainingCoordinator(server, clients)

# Run federated training
final_model = coordinator.run_federated_training(
    num_rounds=10,
    local_epochs=5,
    aggregation_method='fedavg'  # or 'adaptive' or 'secure'
)

# Save the trained model
server.save_model('qos_model.json')
```

### Service Composition

```python
from federated_qos_client import ServiceComposer

# Create composer
composer = ServiceComposer(client)

# Define workflow
workflow = ["authentication", "payment", "notification"]

# Define QoS constraints
constraints = {
    'max_response_time': 200,   # ms
    'min_availability': 0.95,   # 95%
    'max_cost': 2.0
}

# Compose optimal services
composition = composer.compose_services(
    workflow,
    available_services,
    constraints
)

# Evaluate composition
metrics = composer.evaluate_composition(composition)
print(f"Total Response Time: {metrics['total_response_time']} ms")
print(f"Total Cost: ${metrics['total_cost']}")
```

## 📚 API Reference

### QoSMetrics
- `response_time`: Service latency (milliseconds)
- `throughput`: Processing rate (requests/second)
- `availability`: Uptime percentage (0-1)
- `reliability`: Success rate (0-1)
- `cost`: Monetary cost per invocation

### FederatedQoSClient
- `initialize_model()`: Initialize neural network
- `add_service(service)`: Add service to local repository
- `train_local(epochs)`: Train on local data
- `predict_qos_score(service)`: Predict quality score
- `get_model_update()`: Get weights for aggregation

### FederatedQoSServer
- `initialize_global_model()`: Initialize global model
- `receive_client_update(update)`: Receive client update
- `federated_averaging()`: Standard FedAvg aggregation
- `adaptive_federated_averaging()`: Adaptive aggregation
- `secure_aggregation()`: DP-enabled aggregation
- `save_model(filepath)`: Save model to disk

## 🔬 How It Works

### Federated Learning Process

1. **Initialization**: Server creates global model, clients initialize local copies
2. **Distribution**: Global model sent to all clients
3. **Local Training**: Each client trains on private data
4. **Aggregation**: Server combines client updates using weighted averaging
5. **Update**: New global model distributed to clients
6. **Repeat**: Process continues for multiple rounds

### Aggregation Methods

**Standard FedAvg**:
```
global_weights = Σ(weight_i × local_weights_i)
where weight_i = num_samples_i / total_samples
```

**Adaptive FedAvg**:
```
delta = Σ(weight_i × (local_weights_i - global_weights))
global_weights += learning_rate × delta
```

**Secure Aggregation**:
```
aggregated_weights += Gaussian_Noise(0, σ²)
where σ = sensitivity / privacy_budget
```

## 📊 Results

The system demonstrates:
- ✅ **Privacy preservation**: No raw data exchange
- ✅ **Improved accuracy**: Collaborative learning from multiple sources
- ✅ **Efficient composition**: Fast QoS-aware service selection
- ✅ **Scalability**: Supports arbitrary number of clients
- ✅ **Flexibility**: Multiple aggregation strategies

## 🛠️ Project Structure

```
federated-qos-service-composition/
├── federated_qos_client.py      # Client implementation
├── federated_qos_server.py      # Server implementation
├── federated_qos_demo.py        # Complete demonstration
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
└── .gitignore                    # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{federated_qos_2024,
  title={Federated Learning for QoS-aware Web Service Composition},
  year={2024},
  url={https://github.com/YOUR_USERNAME/federated-qos-service-composition}
}
```

## 🔗 References

- McMahan et al. (2017) - Communication-Efficient Learning of Deep Networks from Decentralized Data
- QoS-aware Service Composition in Service-Oriented Architecture
- Differential Privacy in Machine Learning

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ using Federated Learning**
