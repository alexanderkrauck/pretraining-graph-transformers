#%%
import torch_geometric as pyg
import numpy as np
from ogb.lsc import PCQM4Mv2Dataset
root_dir = "data"

def summarize_data(data, print_classes = True):
    print(f"n classes: {data.num_classes}")
    data.print_summary()

    if print_classes:
        true_array = np.zeros(data.num_classes)
        num_array = np.zeros(data.num_classes)


        for i in data:
            y = i.y.numpy()
            nans = np.isnan(y)
            true_array = true_array + np.nan_to_num(y)
            num_array = num_array + np.invert(nans)


        for i in range(data.num_classes):
            print(f"{i:>3}: {int(true_array[0,i])} of {int(num_array[0,i])}. {true_array[0,i]/num_array[0,i]:>4f} are true. "\
                f"{num_array[0,i]/len(data):.4f} of the data have this label.")
# %%
data = pyg.datasets.MoleculeNet(root_dir, "Tox21")
summarize_data(data)
# %% The x1 object is 1 sample of the dataset.
# x1.x is of the shape [num_nodes, num_node_features]
# x1.edge_index is of the shape [2, num_edges]
x1 = data[0]
x1

# %%
#%%
from datasets import load_dataset

# There is only one split on the hub
dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="data/huggingface")

dataset = dataset.shuffle(seed=0)
train_dataset = dataset["train"]

train_dataset[0]

#it seems that pyg and huggingface datasets very similar for graphs in terms of structure.
#huggingface uses dictonaries/lists, pyg uses tensors and a Data class object.
#%%

import pyarrow as pa
from datasets.arrow_dataset import Dataset
import torch

def pyg_to_arrow(pyg_dataset):
    # Prepare data for PyArrow Table
    data_for_arrow = {}

    # Iterate over all keys in the first data object in the PyG dataset
    keys = pyg_dataset[0].keys
    for key in keys:
        data_for_arrow[key] = []

    for graph in pyg_dataset:
        for key in keys:
            feature = graph[key]
            if isinstance(feature, torch.Tensor):
                data_for_arrow[key].append(feature.tolist())
            else:
                data_for_arrow[key].append(feature)

    # Convert Arrow Table to Hugging Face Dataset
    print(data_for_arrow.keys())
    print(len(data_for_arrow["edge_attr"]))
    print(len(data_for_arrow["x"]))

    hf_dataset = Dataset.from_dict(data_for_arrow)

    return hf_dataset
# %%
arrow_ds = pyg_to_arrow(data)
# %%
