#%%
import os
os.chdir("..")

from utils import graphormer_data_collator_improved as graphormer_collator_utils
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import GraphormerForGraphClassification, GraphormerConfig

np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)


#%%
dataset = DatasetDict.load_from_disk("data/tox21_original/processed/arrow_processed")
dataset_train = dataset["train"]
# %%
collator = graphormer_collator_utils.GraphormerDataCollator()
cnfg = GraphormerConfig(
    num_classes = 12,
    embedding_dim = 128,
    num_attention_heads = 8,
    num_hidden_layers = 8
)
model = GraphormerForGraphClassification(cnfg).cuda()
# %%
example_batch = [dataset["train"][i] for i in range(8)]
collated_batch = collator(example_batch)
# %%
collated_batch = {k: v.to("cuda") for k, v in collated_batch.items()}
# %%
output = model(**collated_batch)

# %%
