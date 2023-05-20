"""
Utility classes for data handling.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"

import torch
from torch_geometric.data.dataset import Dataset

from datasets.arrow_dataset import Dataset

def pyg_to_arrow(pyg_dataset: Dataset):
    """
    Converts a PyG dataset to a Hugging Face dataset.
    
    Adds a column "num_nodes" to the Hugging Face dataset, which contains the number of nodes for each graph.
    Also renames the column "x" to "node_feat" to comply with huggingface standards.

    Args
    ----
    pyg_dataset: torch_geometric.data.dataset.Dataset
        PyG dataset to convert."""

    # Prepare data for PyArrow Table
    data_for_arrow = {}

    # Iterate over all keys in the first data object in the PyG dataset
    keys = pyg_dataset[0].keys
    for key in keys:
        data_for_arrow[key] = []
    data_for_arrow["num_nodes"] = []

    for graph in pyg_dataset:
        for key in keys:
            if key == "x":
                data_for_arrow["num_nodes"].append(len(graph[key]))
            feature = graph[key]
            if isinstance(feature, torch.Tensor):
                data_for_arrow[key].append(feature.tolist())
            else:
                data_for_arrow[key].append(feature)

    # Convert Dict to Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data_for_arrow)
    hf_dataset = hf_dataset.rename_column("x", "node_feat")

    print(f"Resulting Dataset: {hf_dataset}")

    return hf_dataset