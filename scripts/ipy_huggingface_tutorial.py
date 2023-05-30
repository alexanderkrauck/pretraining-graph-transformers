#from https://huggingface.co/blog/graphml-classification
#%%
import os
os.chdir("..")

from datasets import load_dataset

# There is only one split on the hub
dataset = load_dataset("OGB/ogbg-molhiv", cache_dir="data/huggingface")

dataset = dataset.shuffle(seed=0)

#%%
dataset["train"][0]
# %%
import networkx as nx
import matplotlib.pyplot as plt

# We want to plot the first train graph
graph = dataset["train"][0]

edges = graph["edge_index"]
num_edges = len(edges[0])
num_nodes = graph["num_nodes"]

# Conversion to networkx format
G = nx.Graph()
G.add_nodes_from(range(num_nodes))
G.add_edges_from([(edges[0][i], edges[1][i]) for i in range(num_edges)])

# Plot
nx.draw(G)
# %%
from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
dataset_processed = dataset.map(preprocess_item, batched=False)
#dataset_processed = dataset
#%%
dataset_processed["train"][0].keys()

# %%
from transformers import GraphormerForGraphClassification, GraphormerConfig

model = GraphormerForGraphClassification.from_pretrained(
    "clefourrier/pcqm4mv2_graphormer_base",
    num_classes=2, # num_classes for the downstream task 
    ignore_mismatched_sizes=True,
).cuda()

cnfg = GraphormerConfig(
    num_classes = 2,
    embedding_dim = 128,
    num_attention_heads = 8,
    num_hidden_layers = 8
)
model = GraphormerForGraphClassification(cnfg).cuda()
# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "graph-classification",
    logging_dir="graph-classification",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    auto_find_batch_size=False, # batch size can be changed automatically to prevent OOMs
    gradient_accumulation_steps=10,
    dataloader_num_workers=4, #1, 
    num_train_epochs=20,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    push_to_hub=False,
)
# %%
def preproc_weak(item, keep_features=False):
    if "labels" not in item:
        item["labels"] = item["y"]
    return item

test_dataset = dataset["train"].map(preproc_weak, batched=False)

#%%
from torch.utils.data import DataLoader
collator = GraphormerDataCollator(on_the_fly_processing=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
)
# %%
train_results = trainer.train()
# %%
paramsum = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(paramsum)
# %%
processed = collator([dataset["train"][i] for i in range(20)])
# %%
for key in processed.keys():
    print(key, type(processed[key]), processed[key].shape)
# %%
[dataset["train"][i]["num_nodes"] for i in range(20)]
# %%
[len(dataset["train"][i]["edge_attr"]) for i in range(20)]

# %%
