# %%
from rdkit import Chem

from utils import data as data_utils
import torch_geometric as pyg
import tqdm

data = pyg.datasets.ZINC("data/ZINC", subset=True)

flag = 0
for idx, d in enumerate(data):
    if 26 in d.x:
        if flag < 1:
            flag += 1
        else:
            print(d.x)

            break
    #rdkit_graph = data_utils.pyg_to_rdkit(d)
#%%
for idx, (send, rec) in enumerate(d.edge_index.T):
    if send == 26:
        print(d.edge_attr[idx])

# %%
print(idx)

mol = data_utils.pyg_to_rdkit(d)
# %%
mol
# %%
#mol = Chem.AddHs(mol)
for idx, atom in enumerate(mol.GetAtoms()):
    print(idx, atom.GetSymbol(), atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetDegree())
    for bond in atom.GetBonds():
        print(bond.GetBondType())
# %%
mol.GetAtoms()[26].SetNumExplicitHs(0)
# %%
mol.UpdatePropertyCache(strict=False)
Chem.GetSSSR(mol)
Chem.SanitizeMol(mol)

# %%
mols = []
for d in tqdm.tqdm(data):
    mol = data_utils.pyg_to_rdkit(d)
    mols.append(mol)

# %%
tqdm(data)
# %%
from utils import ZincWithRDKit

zinc = ZincWithRDKit("data/ZINC", subset=True)
mol_list = zinc.to_rdkit_molecule_list()


# %%
from utils import data as data_utils
data_utils.rdkit_to_arrow(mol_list, "data/ZINC/processed/arrow")
# %%
