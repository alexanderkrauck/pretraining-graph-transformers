#%%
from utils import data as data_utils
import torch_geometric as pyg
from utils import ZincWithRDKit
#%%
data_utils.sdf_to_arrow("data/pcqm4mv2/raw/pcqm4m-v2-train.sdf", to_disk_location="data/pcqm4mv2/processed/temp")

#%%
data_utils.sdf_to_arrow("data/tox21_original/raw/tox21_original.sdf", to_disk_location="data/tox21_original/processed/temp")

#%%
zinc = ZincWithRDKit("data/ZINC", subset=True)
mol_list = zinc.to_rdkit_molecule_list()
data_utils.rdkit_to_arrow(mol_list, "data/ZINC/processed/arrow")
