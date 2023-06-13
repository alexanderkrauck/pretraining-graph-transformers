# TODO: consider adding the download of the data to this script
# TODO: consider adding the huggingface data preprocessing to this script
# %%
import os

os.chdir("..")

from utils import data as data_utils
import torch_geometric as pyg
from utils import ZincWithRDKit
from datasets import DatasetDict, load_from_disk, IterableDataset
from os.path import join
import subprocess
from utils import graphormer_data_collator_improved as graphormer_collator_utils
from tqdm import tqdm
import numpy as np
import time


data_dir = "./data"
# %%
# load pcqm4mv2 data
subprocess.run(
    [
        "wget",
        "-P",
        join(data_dir, "pcqm4mv2/raw"),
        "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz",
    ]
)
subprocess.run(
    [
        "tar",
        "-xf",
        join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf.tar.gz"),
        "-C",
        join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf"),
    ]
)

# %%
# load tox21 original data
subprocess.run(
    [
        "wget",
        "-P",
        join(data_dir, "tox21_original/raw"),
        "http://bioinf.jku.at/research/DeepTox/tox21_compoundData.csv",
    ]
)
subprocess.run(
    [
        "wget",
        "-P",
        join(data_dir, "tox21_original/raw"),
        "http://bioinf.jku.at/research/DeepTox/tox21.sdf.gz",
    ]
)
subprocess.run(["gunzip", join(data_dir, "tox21_original/raw/tox21.sdf.gz")])


# %%
# load tox21 data
subprocess.run(
    [
        "wget",
        "-P",
        join(data_dir, "tox21/raw"),
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    ]
)
subprocess.run(["gunzip", join(data_dir, "tox21/raw/tox21.csv.gz")])

# %%
# load pcba data
subprocess.run(
    [
        "wget",
        "-P",
        join(data_dir, "pcba/raw"),
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
    ]
)
subprocess.run(["gunzip", join(data_dir, "pcba/raw/pcba.csv.gz")])

# %%
# load qm9 data
subprocess.run(
    [
        "wget",
        "-P",
        join(data_dir, "qm9/raw"),
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
    ]
)
subprocess.run(
    ["unzip", join(data_dir, "qm9/raw/qm9.zip"), "-d", join(data_dir, "qm9/raw")]
)
subprocess.run(["rm", join(data_dir, "qm9/raw/qm9.zip")])

# %%
# load zinc data
zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="train")
zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="val")
zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="test")
del zinc

# Here start the data processing to arrow format

# %%
ds = data_utils.sdf_to_arrow(
    join(data_dir, "pcqm4mv2/raw/pcqm4m-v2-train.sdf"),
    to_disk_location=join(data_dir, "pcqm4mv2/processed/arrow"),
)

# %%
ds_train = data_utils.sdf_to_arrow(
    join(data_dir, "tox21_original/raw/tox21.sdf"),
    to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
    target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
    split_column=4,
    take_split="training",
)
ds_val = data_utils.sdf_to_arrow(
    join(data_dir, "tox21_original/raw/tox21.sdf"),
    to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
    target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
    split_column=4,
    take_split="validation",
)
ds_test = data_utils.sdf_to_arrow(
    join(data_dir, "tox21_original/raw/tox21.sdf"),
    to_disk_location=join(data_dir, "tox21_original/processed/arrow"),
    target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    csv_with_metainfo=join(data_dir, "tox21_original/raw/tox21_compoundData.csv"),
    split_column=4,
    take_split="test",
)

dataset_dict = DatasetDict(
    {
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test,
    }
)
dataset_dict.save_to_disk(join(data_dir, "tox21_original/processed/arrow"))
# %%
ds = data_utils.csv_to_arrow(
    join(data_dir, "tox21/raw/tox21.csv"),
    to_disk_location=join(data_dir, "tox21/processed/arrow"),
    include_conformer=True,
    id_column=12,
)


# %%
ds = data_utils.sdf_to_arrow(
    join(data_dir, "qm9/raw/gdb9.sdf"),
    to_disk_location=join(data_dir, "qm9/processed/arrow"),
    csv_with_metainfo=join(data_dir, "qm9/raw/gdb9.sdf.csv"),
    target_columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
)

# %%
zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="train")
mol_list = zinc.to_rdkit_molecule_list()
ds_train = data_utils.rdkit_to_arrow(
    mol_list,
    target_list=zinc.y.numpy().tolist(),
)

zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="val")
mol_list = zinc.to_rdkit_molecule_list()
ds_val = data_utils.rdkit_to_arrow(
    mol_list,
    target_list=zinc.y.numpy().tolist(),
)

zinc = ZincWithRDKit(join(data_dir, "ZINC"), subset=True, split="test")
mol_list = zinc.to_rdkit_molecule_list()
ds_test = data_utils.rdkit_to_arrow(
    mol_list,
    target_list=zinc.y.numpy().tolist(),
)

dataset_dict = DatasetDict(
    {
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test,
    }
)

dataset_dict.save_to_disk(join(data_dir, "ZINC/processed/arrow"))

# %%
ds = data_utils.csv_to_arrow(
    join(data_dir, "pcba/raw/pcba.csv"),
    include_conformer=False,  # NOTE: for now because its very slow
    to_disk_location=join(data_dir, "pcba/processed/arrow"),
    smiles_column=-1,
    id_column=-2,
    target_columns=list(range(0, 128)),
)

# Here start the mapping of the data to input format to the model

# %%
data_utils.map_arrow_dataset_from_disk(
    join(data_dir, "pcqm4mv2/processed"), is_dataset_dict=False
)
# %%
data_utils.map_arrow_dataset_from_disk(
    join(data_dir, "tox21_original/processed"), is_dataset_dict=True
)
# %%
data_utils.map_arrow_dataset_from_disk(
    join(data_dir, "tox21/processed"), is_dataset_dict=False
)
# %%
data_utils.map_arrow_dataset_from_disk(
    join(data_dir, "qm9/processed"), is_dataset_dict=False
)
# %%
data_utils.map_arrow_dataset_from_disk(
    join(data_dir, "ZINC/processed"), is_dataset_dict=True
)
# %%
data_utils.map_arrow_dataset_from_disk(
    join(data_dir, "pcba/processed"), is_dataset_dict=False
)


# %% Test out for samples with 0 edges
dataset = DatasetDict.load_from_disk("data/tox21_original/processed/arrow")
regular_sample = dataset["train"].filter(lambda example: example["id"] == 0)[0]
zero_edge_sample = dataset["train"].filter(lambda example: example["id"] == 10206)[0]
# %%
proc1 = graphormer_collator_utils.preprocess_item(zero_edge_sample)
proc2 = graphormer_collator_utils.preprocess_item(regular_sample)
# %%
collator = graphormer_collator_utils.GraphormerDataCollator()
collator([proc1, proc2])

# %% Test for batching, dataset and collator with focus on execution time
collator = graphormer_collator_utils.GraphormerDataCollator()
#dataset_path =  "/home/alexander/temp/ZINC/processed/arrow"
dataset_path =  "data/pcqm4mv2/processed/arrow"
#dataset_path =  "data/ZINC/processed/arrow_processed"
dataset = load_from_disk(
    dataset_path, keep_in_memory=False
)
if isinstance(dataset, DatasetDict):
    dataset = dataset["train"]


dataset.cleanup_cache_files()
dataset_size = len(dataset)
tot_batches = 1000
batch_size = 256

#dataset = dataset.shuffle()

#%% random index data
_start = time.time()
n = batch_size * tot_batches
for i in np.random.default_rng(43).integers(0, len(dataset), size=n):
    _ = dataset[int(i)]
print((time.time() - _start)/tot_batches)
#%% only loading
batches = []
for e in tqdm(range(tot_batches)):
    data_batch = [dataset[(i + e * batch_size) % dataset_size] for i in range(batch_size)]
    batches.append(data_batch)
#%% only loading
samples = []
for i in tqdm(range(100 * batch_size)):
    samples.append(dataset[i + 1000000])
#%% only loading and preparing (for not processed arrow) no collater
batches = []
for e in tqdm(range(tot_batches)):
    data_batch = [graphormer_collator_utils.preprocess_item(dataset[(i + e) % dataset_size]) for i in range(64)]
    batches.append(data_batch)
#%% only collater, no loading. I think this time i can divide by number of cpus.
for data_batch in tqdm(batches):
    collated_batch = collator(data_batch)
    {k: v.clone().detach().to("cuda") for k, v in collated_batch.items()}

# %% loading + collater
for e in tqdm(range(10)):
    data_batch = [dataset[(i + e * batch_size) % dataset_size] for i in range(batch_size)]

    collator(data_batch)
# %%
ds2 = dataset.with_format("numpy")
# %% other dataset testing
class PreloadedDataset:
    def __init__(self, dataset):
        #dataset.set_format(type='numpy', columns=list(dataset.column_names))
        self.rows = []
        for i in tqdm(dataset):
            self.rows.append(i)
        del dataset

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)

#%% usage
dataset = PreloadedDataset(dataset= dataset)
# %%
batches = []
dataset_size = len(dataset)
for e in tqdm(range(157)):
    data_batch = [dataset[(i + e * 64) % dataset_size] for i in range(64)]
    batches.append(data_batch)
# %%
collator = graphormer_collator_utils.GraphormerDataCollator(on_the_fly_processing=False)
for batch in tqdm(batches):

    collator(batch)
# %%
