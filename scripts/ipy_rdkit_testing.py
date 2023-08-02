# %%
import os
os.chdir("..")

from rdkit import Chem

from utils import data as data_utils
import torch_geometric as pyg
import tqdm
import numpy as np
np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)

#%%
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
# %% test bounds method
import os
os.chdir("..")

from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem
import numpy as np
import matplotlib.pyplot as plt
import rdkit.RDLogger as RDLogger
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*")


np.set_printoptions(threshold=np.inf, edgeitems=30, linewidth=100000)

import utils.preprocessing as preprocessing_utils

from transformers.utils import is_cython_available, requires_backends


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from transformers.models.graphormer import algos_graphormer  # noqa E402

# Load molecule

#mol = Chem.MolFromSmiles('CCOc1ccc2nc(S(N)(=O)=O)sc2c1')
mol = Chem.MolFromSmiles('O=C1NC(=O)NC(=O)C1')

#need removeHs=False, sanitize=False to get the same number of nodes as the ogb dataset

# Generate distance bounds

bounds_matrix = GetMoleculeBoundsMatrix(mol)
# %%
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)


# %%
i,j,k = 1,2,3
actual_angle = rdMolTransforms.GetAngleRad(mol.GetConformer(0), j, i, k)
actual_angle
# %%
coords = np.array([list(mol.GetConformer(0).GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])

# %%
distij = np.linalg.norm(coords[i] - coords[j])
distik = np.linalg.norm(coords[i] - coords[k])
distjk = np.linalg.norm(coords[j] - coords[k])

# %%
np.arccos((distjk**2 - distij**2 - distik**2) / (-2 * distik * distij))



# %%
adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
#%%
triplets, cosines = preprocessing_utils.generate_angle_bounds(mol, bounds_matrix, shortest_path_result, n_hops=3)

#%%
def generate_from_csv(
    csv_file,
    smiles_column = None,
    has_header = True,
    include_conformer = True,
    id_column = None,
    target_columns = None,
):

    mols = []
    with open(csv_file, "r") as file:
        if has_header:
            header = file.readline().strip().split(",")
            if smiles_column is None:
                assert (
                    "smiles" in header
                ), "If the csv file has a header, it must contain a smiles column."
                smiles_column = header.index("smiles")
        else:
            assert (
                smiles_column is not None
            ), "If the csv file does not have a header, you must specify the smiles column."

        mol_index = 0
        for line in file:
            line = line.strip()
            if line == "":
                continue
            split_line = line.split(",")
            smiles = split_line[smiles_column]

            mol = Chem.MolFromSmiles(smiles)
            mols.append(mol)
    return mols
#%%
#suppl = Chem.SDMolSupplier(f'data/pcqm4mv2/raw/pcqm4m-v2-train.sdf', removeHs=True, sanitize=True)
suppl = generate_from_csv(f'data/tox21/raw/tox21.csv', has_header=True)

times = -1
cosines_list = []
for idx, mol in tqdm(enumerate(suppl)):
    if mol is None or times == idx:
        break
    try: 
        bounds_matrix = GetMoleculeBoundsMatrix(mol)
    except RuntimeError:
        print("gag")
        continue
    # try:
    #     AllChem.EmbedMolecule(mol)
    #     AllChem.MMFFOptimizeMolecule(mol)
    # except ValueError:
    #     try:
    #         mol = Chem.AddHs(mol)
    #         AllChem.EmbedMolecule(mol)
    #         AllChem.MMFFOptimizeMolecule(mol)
    #         mol = Chem.RemoveHs(mol)
    #     except ValueError:
    #         continue

    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)

    _, cosines = preprocessing_utils.generate_angle_bounds(mol, bounds_matrix, shortest_path_result, n_hops=2)

    cosines_list.append(cosines)

cat_cos = np.concatenate(cosines_list)
flat_cat_cos = cat_cos.flatten()


#%%

# Define the number of bins
n = 50

# Define the minimum bin width
min_bin_width = 0.01

# Use percentile to generate bin edges
bins = np.percentile(flat_cat_cos, np.linspace(0, 100, n+1))

# Create a list to store the final bin edges
final_bins = [bins[0]]

# Iterate through the bin edges
for i in tqdm(range(1, len(bins))):
    # If the width of the bin is below the threshold,
    # replace the last bin edge with the current one
    if bins[i] - final_bins[-1] < min_bin_width:
        final_bins[-1] = bins[i]
    # Otherwise, add a new bin edge
    else:
        final_bins.append(bins[i])

# Convert the list to a numpy array
almost_final_bins = np.array(final_bins)
final_bins = almost_final_bins[1:]

print(f"[{','.join([str(i) for i in final_bins])}]")
print(len(final_bins))
#%%
indices = np.digitize(flat_cat_cos, final_bins)
plt.hist(indices, bins=len(final_bins))
plt.show()
# %%


plt.hist(flat_cat_cos, bins=almost_final_bins, edgecolor='black', density=True)
#plt.legend(["min","max","actual"])
plt.xlim(-1.5,1.5)
plt.ylabel('Probability Density')
plt.xlabel('Cosine of Angle')
plt.title('Distribution of Cosines of Angles in Tox21 Dataset')

plt.show()
#%%

# %% test DFT (Density Functional Theory) calculation time
from pyscf import gto, scf

# Define your molecule in SMILES format
suppl = Chem.SDMolSupplier("data/tox21_original/raw/tox21.sdf", removeHs=False, sanitize=True)

# Convert SMILES to 3D coordinates
# This can be done with RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

for mol in suppl:
    try:
        if len(mol.GetAtoms()) > 100:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
            break
    except AttributeError:
        continue
    except ValueError:
        continue


# Now we need to convert the optimized molecule to a format PySCF can use
# This will be a string with each atom and its coordinates
atom_coords = ""
for atom in mol.GetAtoms():
    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
    atom_coords += "{0} {1} {2} {3};".format(atom.GetSymbol(), pos.x, pos.y, pos.z)


#%% 
# Define molecule for PySCF
mol_pyscf = gto.M(atom=atom_coords, basis='sto3g')

# Perform DFT calculation
mf = scf.RHF(mol_pyscf)
mf.kernel()

# Print out the total energy
print('Total Energy: ', mf.e_tot)

# %%
