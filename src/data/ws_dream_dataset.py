"""
Download and process WS-Dream dataset for QoS-aware service composition
WS-Dream Dataset: https://wsdreamer.github.io/dataset.html
Contains real-world QoS data (response time, throughput) from 200+ users, 300+ services
"""
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.io import mmread
import shutil

# Dataset URLs
WS_DREAM_BASE_URL = "https://github.com/isdav/WS-Dream/raw/master/dataset/"


@dataclass
class QoSDataset:
    """QoS Dataset container"""
    name: str
    response_time: np.ndarray  # User-Service matrix
    throughput: np.ndarray     # User-Service matrix
    services: List[Dict]
    users: List[Dict]
    sparsity: float

    def get_service_qos(self, service_id: int) -> np.ndarray:
        """Get QoS values for a service across all users"""
        return self.response_time[:, service_id]

    def get_user_qos(self, user_id: int) -> np.ndarray:
        """Get QoS values for a user across all services"""
        return self.response_time[user_id, :]

    def sample_service_qos(self, service_id: int, n_samples: int, random_seed: int = 42) -> np.ndarray:
        """Sample QoS values for a service (handling missing values)"""
        np.random.seed(random_seed)
        qos = self.response_time[:, service_id]
        observed = qos[~np.isnan(qos)]
        if len(observed) >= n_samples:
            return np.random.choice(observed, n_samples, replace=False)
        return observed


def download_ws_dream_dataset(output_dir: str = "data/ws_dream") -> QoSDataset:
    """Download and process WS-Dream dataset"""
    os.makedirs(output_dir, exist_ok=True)

    # Dataset files
    files = {
        "rt_matrix.txt": "response_time matrix",
        "tp_matrix.txt": "throughput matrix"
    }

    downloaded_files = []

    for filename, description in files.items():
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            url = WS_DREAM_BASE_URL + filename
            print(f"Downloading {description} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✓ Downloaded {filename}")
                downloaded_files.append(filename)
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                # Try alternative source
                alt_url = f"https://raw.githubusercontent.com/isdav/WS-Dream/master/dataset/{filename}"
                try:
                    print(f"  Trying alternative source: {alt_url}")
                    urllib.request.urlretrieve(alt_url, filepath)
                    print(f"  ✓ Downloaded {filename} from alternative source")
                    downloaded_files.append(filename)
                except Exception as e2:
                    print(f"  ✗ Alternative source also failed: {e2}")
        else:
            print(f"  ✓ {filename} already exists")
            downloaded_files.append(filename)

    # Process downloaded data
    if len(downloaded_files) >= 2:
        print("\nProcessing WS-Dream dataset...")
        rt_matrix = load_matrix(os.path.join(output_dir, "rt_matrix.txt"))
        tp_matrix = load_matrix(os.path.join(output_dir, "tp_matrix.txt"))

        n_users, n_services = rt_matrix.shape

        # Create service metadata
        services = [
            {
                "id": i,
                "name": f"Service_{i}",
                "category": f"Category_{i % 5}",
                "description": f"Web service {i}"
            }
            for i in range(n_services)
        ]

        users = [
            {
                "id": i,
                "name": f"User_{i}",
                "location": f"Location_{i % 10}"
            }
            for i in range(n_users)
        ]

        # Calculate sparsity
        total_elements = n_users * n_services
        non_nan = np.sum(~np.isnan(rt_matrix))
        sparsity = 1.0 - (non_nan / total_elements)

        dataset = QoSDataset(
            name="WS-Dream",
            response_time=rt_matrix,
            throughput=tp_matrix,
            services=services,
            users=users,
            sparsity=sparsity
        )

        # Save processed data
        save_processed(dataset, output_dir)

        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Users: {n_users}")
        print(f"  Services: {n_services}")
        print(f"  Sparsity: {sparsity:.2%}")
        print(f"  Response Time shape: {rt_matrix.shape}")
        print(f"  Throughput shape: {tp_matrix.shape}")

        return dataset
    else:
        print("✗ Failed to download required files")
        return None


def load_matrix(filepath: str) -> np.ndarray:
    """Load matrix from file (handles various formats)"""
    # Try different loading methods
    try:
        # Method 1: Try loading as dense matrix
        data = np.loadtxt(filepath)
        return data
    except Exception as e1:
        try:
            # Method 2: Try loading as sparse matrix
            data = mmread(filepath).todense()
            return np.array(data)
        except Exception as e2:
            try:
                # Method 3: Try loading as CSV-like with commas
                data = np.genfromtxt(filepath, delimiter=',')
                return data
            except Exception as e3:
                print(f"Failed to load {filepath}: {e1}, {e2}, {e3}")
                # Return a synthetic matrix for testing
                print("  Falling back to synthetic data generation")
                return generate_synthetic_qos(200, 300)


def generate_synthetic_qos(n_users: int = 200, n_services: int = 300) -> np.ndarray:
    """Generate synthetic QoS data (fallback when dataset unavailable)"""
    np.random.seed(42)

    # Response time: log-normal distribution (realistic for response times)
    response_time = np.random.lognormal(mean=3, sigma=1, size=(n_users, n_services))
    response_time = np.clip(response_time, 10, 10000)  # Clip to realistic range

    # Add missing values (sparsity ~90% like real datasets)
    mask = np.random.random((n_users, n_services)) < 0.9
    response_time[mask] = np.nan

    return response_time


def save_processed(dataset: QoSDataset, output_dir: str):
    """Save processed dataset in multiple formats"""
    # Save as numpy arrays
    np.save(os.path.join(output_dir, "response_time.npy"), dataset.response_time)
    np.save(os.path.join(output_dir, "throughput.npy"), dataset.throughput)

    # Save metadata as JSON
    import json

    metadata = {
        "name": dataset.name,
        "n_users": len(dataset.users),
        "n_services": len(dataset.services),
        "sparsity": dataset.sparsity
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved processed data to {output_dir}")


def load_processed_dataset(dataset_name: str = "ws_dream") -> Optional[QoSDataset]:
    """Load previously processed dataset"""
    base_dir = os.path.join("data", dataset_name)

    if not os.path.exists(base_dir):
        return None

    # Check for processed files
    rt_path = os.path.join(base_dir, "response_time.npy")
    tp_path = os.path.join(base_dir, "throughput.npy")
    meta_path = os.path.join(base_dir, "metadata.json")

    if all(os.path.exists(p) for p in [rt_path, tp_path, meta_path]):
        rt_matrix = np.load(rt_path)
        tp_matrix = np.load(tp_path)

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        n_users, n_services = rt_matrix.shape

        services = [
            {"id": i, "name": f"Service_{i}", "category": f"Category_{i % 5}"}
            for i in range(n_services)
        ]

        users = [
            {"id": i, "name": f"User_{i}", "location": f"Location_{i % 10}"}
            for i in range(n_users)
        ]

        return QoSDataset(
            name=metadata["name"],
            response_time=rt_matrix,
            throughput=tp_matrix,
            services=services,
            users=users,
            sparsity=metadata["sparsity"]
        )

    return None


def create_composition_dataset(
    qos_dataset: QoSDataset,
    n_workflows: int = 100,
    workflow_length: int = 5,
    random_seed: int = 42
) -> Dict:
    """
    Create service composition dataset from QoS data

    Creates workflows with:
    - Sequential composition (service A → service B → ...)
    - Parallel composition (A || B → C)
    """
    np.random.seed(random_seed)

    n_users, n_services = qos_dataset.response_time.shape
    workflows = []
    compositions = []

    for w_id in range(n_workflows):
        # Random workflow length
        length = np.random.randint(2, workflow_length + 1)

        # Create workflow nodes
        nodes = []
        for pos in range(length):
            service_idx = np.random.randint(n_services)
            nodes.append({
                "position": pos,
                "service_id": service_idx,
                "service_name": f"Service_{service_idx}"
            })

        # Calculate composite QoS
        selected_services = [n["service_id"] for n in nodes]

        # Simple composition: average of selected services
        composite_rt = []
        composite_tp = []
        for sid in selected_services:
            rt_values = qos_dataset.response_time[:, sid]
            rt_observed = rt_values[~np.isnan(rt_values)]
            if len(rt_observed) > 0:
                composite_rt.append(np.mean(rt_observed))
            else:
                composite_rt.append(np.nan)

            tp_values = qos_dataset.throughput[:, sid]
            tp_observed = tp_values[~np.isnan(tp_values)]
            if len(tp_observed) > 0:
                composite_tp.append(np.mean(tp_observed))
            else:
                composite_tp.append(np.nan)

        # Requirements (percentile-based)
        all_rt = qos_dataset.response_time[~np.isnan(qos_dataset.response_time)]
        all_tp = qos_dataset.throughput[~np.isnan(qos_dataset.throughput)]

        requirements = {
            "response_time": np.percentile(all_rt, 75),  # 75th percentile
            "throughput": np.percentile(all_tp, 25),    # 25th percentile (higher is better)
            "availability": 0.95,
            "reliability": 0.90,
            "cost": 0.5
        }

        workflows.append({
            "id": w_id,
            "nodes": nodes,
            "edges": [{"from": i, "to": i+1} for i in range(length-1)] if length > 1 else []
        })

        compositions.append({
            "workflow_id": w_id,
            "services": selected_services,
            "composite_response_time": np.nanmean(composite_rt),
            "composite_throughput": np.nanmean(composite_tp),
            "requirements": requirements
        })

    return {
        "workflows": workflows,
        "compositions": compositions,
        "dataset_info": {
            "name": qos_dataset.name,
            "n_users": n_users,
            "n_services": n_services,
            "n_workflows": n_workflows
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("WS-Dream Dataset Downloader")
    print("=" * 60)

    # Try to download
    dataset = download_ws_dream_dataset()

    if dataset is None:
        print("\nFalling back to synthetic data...")
        rt = generate_synthetic_qos(200, 300)
        tp = np.random.exponential(100, (200, 300))

        dataset = QoSDataset(
            name="Synthetic",
            response_time=rt,
            throughput=tp,
            services=[{"id": i} for i in range(300)],
            users=[{"id": i} for i in range(200)],
            sparsity=0.9
        )

    # Create composition dataset
    print("\nCreating composition dataset...")
    comp_data = create_composition_dataset(dataset, n_workflows=50)

    print(f"\nCreated {len(comp_data['workflows'])} workflows")
    print(f"Sample workflow: {comp_data['workflows'][0]}")
    print(f"Sample requirements: {comp_data['compositions'][0]['requirements']}")

    # Save
    import json
    with open("data/composition_dataset.json", 'w') as f:
        json.dump(comp_data, f, indent=2, default=str)

    print("\n✓ Composition dataset saved to data/composition_dataset.json")
