# %%
from utils import data as data_utils
import torch_geometric as pyg
from utils import ZincWithRDKit
from datasets import DatasetDict

# %%
ds = data_utils.sdf_to_arrow(
    "data/pcqm4mv2/raw/pcqm4m-v2-train.sdf",
    to_disk_location="data/pcqm4mv2/processed/arrow",
)

# %%
ds_train = data_utils.sdf_to_arrow(
    "data/tox21_original/raw/tox21_original.sdf",
    to_disk_location="data/tox21_original/processed/arrow",
    target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    csv_with_metainfo="data/tox21_original/raw/infofile.csv",
    split_column=4,
    take_split="training",
)
ds_val = data_utils.sdf_to_arrow(
    "data/tox21_original/raw/tox21_original.sdf",
    to_disk_location="data/tox21_original/processed/arrow",
    target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    csv_with_metainfo="data/tox21_original/raw/infofile.csv",
    split_column=4,
    take_split="validation",
)
ds_test = data_utils.sdf_to_arrow(
    "data/tox21_original/raw/tox21_original.sdf",
    to_disk_location="data/tox21_original/processed/arrow",
    target_columns=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    csv_with_metainfo="data/tox21_original/raw/infofile.csv",
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
dataset_dict.save_to_disk("data/tox21_original/processed/arrow")
# %%
ds = data_utils.csv_to_arrow(
    "data/tox21/raw/tox21.csv",
    to_disk_location="data/tox21/processed/arrow",
    include_conformer=True,
    id_column=12,
)


# %%
ds = data_utils.sdf_to_arrow(
    "data/qm9/raw/gdb9.sdf",
    to_disk_location="data/qm9/processed/arrow",
    csv_with_metainfo="data/qm9/raw/gdb9.sdf.csv",
    target_columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
)

# %%
zinc = ZincWithRDKit("data/ZINC", subset=True, split="train")
mol_list = zinc.to_rdkit_molecule_list()
ds_train = data_utils.rdkit_to_arrow(
    mol_list,
    target_list=zinc.y.numpy().tolist(),
)

zinc = ZincWithRDKit("data/ZINC", subset=True, split="val")
mol_list = zinc.to_rdkit_molecule_list()
ds_val = data_utils.rdkit_to_arrow(
    mol_list,
    target_list=zinc.y.numpy().tolist(),
)

zinc = ZincWithRDKit("data/ZINC", subset=True, split="test")
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

dataset_dict.save_to_disk("data/ZINC/processed/arrow")

# %%
ds = data_utils.csv_to_arrow(
    "data/pcba/raw/pcba.csv",
    to_disk_location="data/pcba/processed/arrow",
    smiles_column=-1,
    id_column=-2,
    target_columns=list(range(0, 128)),
)
# %%
