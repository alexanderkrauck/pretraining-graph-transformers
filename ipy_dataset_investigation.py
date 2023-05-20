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
import utils.data as data_utils

arrow_ds = data_utils.pyg_to_arrow(data, to_disk_location="data/tox21/processed/arrow")
#%%
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
arrow_ds_processed = arrow_ds.map(preprocess_item, batched=False)

#%%
for k in arrow_ds_processed.features.keys():
    print(k, arrow_ds_processed[0][k])
# %%
train_dataset
# %%
for k in train_dataset.features.keys():
    print(k)
    print(arrow_ds[0][k],"\n",train_dataset[0][k])

# %%
data = pyg.datasets.QM9(root_dir+"/qm9")

# %%
arrow_ds = data_utils.pyg_to_arrow(data)
# %%

print(arrow_ds[20000]["edge_index"])
arrow_ds
# %%
from datasets import