#%%
import os
os.chdir("..")

from utils import graphormer_data_collator_improved as graphormer_collator_utils
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk

np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)


#%%

#dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="data/huggingface")
dataset = DatasetDict.load_from_disk("data/tox21_original/processed/arrow")
dataset_train = dataset["train"]
#dataset_train = load_from_disk("data/ZINC/processed/arrow")
graphormer_collator_utils.preprocess_item(dataset_train[0])
#%%
dataset = dataset.map(graphormer_collator_utils.preprocess_item, batched=False)
# %%
edge_attr = np.array([1,2,3])
edge_attr = edge_attr[:, None]
edge_attr
# %%
# %%
