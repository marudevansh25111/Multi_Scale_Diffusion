# Multi-Scale Diffusion Graph Clustering

A graph clustering implementation using multi-scale diffusion techniques with performance monitoring.

## Installation

```bash
pip install torch torch-geometric scikit-learn psutil numpy tqdm
```

## Usage

Run the experiments on different datasets:

```bash
# Cora dataset
python run_cora_simple_monitor.py --dataset Cora --epochs 100 --lr 0.005

# CiteSeer dataset
python run_citeseer_simple_monitor.py --dataset Cora --epochs 100 --lr 0.005

# PubMed dataset
python run_pubmed_simple_monitor.py --dataset Cora --epochs 100 --lr 0.005
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset name | Cora |
| `--hidden_dims` | Hidden dimensions | 256 |
| `--walk_length` | Random walk length | 3 |
| `--walks_per_node` | Walks per node | 10 |
| `--batch_size` | Batch size | 2708 |
| `--lr` | Learning rate | 0.01 |
| `--epochs` | Training epochs | 200 |
| `--log_steps` | Logging frequency | 1 |

## Files

- `simple_monitor.py` - Performance monitoring system
- `run_cora_simple_monitor.py` - Cora dataset runner
- `run_citeseer_simple_monitor.py` - CiteSeer dataset runner  
- `run_pubmed_simple_monitor.py` - PubMed dataset runner
- `small_model_monitored.py` - Lightweight model implementation
- `clustering_metric.py` - Evaluation metrics

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- scikit-learn
- psutil
- numpy
- tqdm
