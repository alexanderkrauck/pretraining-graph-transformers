#%%
import os
os.chdir("..")
#%%
from utils import data as data_utils
import torch_geometric as pyg
from utils import ZincWithRDKit
from datasets import DatasetDict, load_from_disk, IterableDataset
from os.path import join
import subprocess
from utils import graphormer_data_collator_improved_3d as graphormer_collator_utils_3d
from tqdm import tqdm
import numpy as np
import time
import yaml

from utils.modeling_graphormer_improved_3d import (
    Graphormer3DForGraphClassification,
    Graphormer3DConfig
)

import torch
import matplotlib.pyplot as plt
#%%
config_file_path = "configs/dummy_config_3d.yml"
#%%
dataset_path =  "data/ZINC/processed/arrow"
#dataset_path =  "data/ZINC/processed/arrow_processed"
dataset = DatasetDict.load_from_disk(
    dataset_path, keep_in_memory=False
)
example_batch = [dataset["train"][i] for i in range(64)]

#%%
with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)
#%%
model_config = Graphormer3DConfig(**config["model_args"])
model = Graphormer3DForGraphClassification(model_config).to("cuda")

# %%
collator = graphormer_collator_utils_3d.Graphormer3DDataCollator(model_config, on_the_fly_processing=True, collator_mode="classification")
# %%
collated_batch = collator(example_batch)
collated_batch = {k: v.to("cuda") for k, v in collated_batch.items() if isinstance(v, torch.Tensor)}
#%%
out = model(**collated_batch)
# %%
len(out)
# %%
