"""
Utility functions for data handling.

This file contains modified code based on Fey Matthias and Lenssen Jan E. implementation.

Original Source: https://github.com/pyg-team/pytorch_geometric
Original Code Title: Took some parts from torch_geometric/utils/smiles.py

The modifications made in this file include:
- x_map and e_map.
- Parts of the process_molecule function.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"




import torch
from torch_geometric.data.dataset import Dataset

from datasets import Dataset, Value, Features, Sequence
from rdkit import Chem
from rdkit.Chem import AllChem


from tqdm import tqdm

x_map = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}

#TODO: check correctness of this mapping
zinc_mapping = {
    0: 6,   # 'C': 0 -> Carbon atomic number is 6
    1: 8,   # 'O': 1 -> Oxygen atomic number is 8
    2: 7,   # 'N': 2 -> Nitrogen atomic number is 7
    3: 9,   # 'F': 3 -> Fluorine atomic number is 9
    4: 6,   # 'C H1': 4 -> Carbon atomic number is 6
    5: 16,  # 'S': 5 -> Sulfur atomic number is 16
    6: 17,  # 'Cl': 6 -> Chlorine atomic number is 17
    7: 8,   # 'O -': 7 -> Oxygen atomic number is 8
    8: 7,   # 'N H1 +': 8 -> Nitrogen atomic number is 7
    9: 35,  # 'Br': 9 -> Bromine atomic number is 35
    10: 7,  # 'N H3 +': 10 -> Nitrogen atomic number is 7
    11: 7,  # 'N H2 +': 11 -> Nitrogen atomic number is 7
    12: 7,  # 'N +': 12 -> Nitrogen atomic number is 7
    13: 7,  # 'N -': 13 -> Nitrogen atomic number is 7
    14: 16, # 'S -': 14 -> Sulfur atomic number is 16
    15: 53, # 'I': 15 -> Iodine atomic number is 53
    16: 15, # 'P': 16 -> Phosphorus atomic number is 15
    17: 8,  # 'O H1 +': 17 -> Oxygen atomic number is 8
    18: 7,  # 'N H1 -': 18 -> Nitrogen atomic number is 7
    19: 8,  # 'O +': 19 -> Oxygen atomic number is 8
    20: 16, # 'S +': 20 -> Sulfur atomic number is 16
    21: 15, # 'P H1': 21 -> Phosphorus atomic number is 15
    22: 15, # 'P H2': 22 -> Phosphorus atomic number is 15
    23: 6,  # 'C H2 -': 23 -> Carbon atomic number is 6
    24: 15, # 'P +': 24 -> Phosphorus atomic number is 15
    25: 16, # 'S H1 +': 25 -> Sulfur atomic number is 16
    26: 6,  # 'C H1 -': 26 -> Carbon atomic number is 6
    27: 15  # 'P H1 +': 27 -> Phosphorus atomic number is 15
}

#TODO: unify the features. node_feat and edge_attr should be the same as when loading sdf files.
def pyg_to_arrow(pyg_dataset: Dataset, to_disk_location: str = None):
    """
    Converts a PyG dataset to a Hugging Face dataset.

    Adds a column "num_nodes" to the Hugging Face dataset, which contains the number of nodes for each graph.
    Also renames the column "x" to "node_feat" to comply with huggingface standards.

    Args
    ----
    pyg_dataset: torch_geometric.data.dataset.Dataset
        PyG dataset to convert.
    to_disk_location: str
        Path to save the dataset to (the directory)."""

    # Prepare data for PyArrow Table
    data_for_arrow = {}

    # Iterate over all keys in the first data object in the PyG dataset
    keys = pyg_dataset[0].keys
    for key in keys:
        data_for_arrow[key] = []
    data_for_arrow["num_nodes"] = []

    for graph in tqdm(pyg_dataset):
        for key in keys:
            if key == "x":
                data_for_arrow["num_nodes"].append(len(graph[key]))
            feature = graph[key]
            if isinstance(feature, torch.Tensor):
                data_for_arrow[key].append(feature.tolist())
            else:
                data_for_arrow[key].append(feature)

    # Convert Dict to Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data_for_arrow)
    hf_dataset = hf_dataset.rename_column("x", "node_feat")

    print(f"Resulting Dataset: {hf_dataset}")

    if to_disk_location is not None:
        hf_dataset.save_to_disk(to_disk_location)
        print(f"Saved dataset to {to_disk_location}.")

    return hf_dataset

#TODO: consider moving this whole thing into a separate file where we create a new dataset class that inherits from PyG's ZINC dataset class.
def pyg_to_rdkit(data):

    # Create an RDKit molecule from the graph
    mol = Chem.RWMol()

    # Add atoms to the molecule
    for features in data.x:
        atomic_num = zinc_mapping[int(features[0])]  # Assuming the atomic number is in the first element of x
        atom = Chem.Atom(atomic_num)
        mol.AddAtom(atom)

    # Add bonds to the molecule
    for idx, edge in enumerate(data.edge_index.T):
        i, j = edge[0].item(), edge[1].item()
        if i < j: #because the edge_index is undirected, we only need to add the bond once.
            continue
        bond_type = Chem.rdchem.BondType.values[data.edge_attr[idx].item()]  # Assuming the bond type is in the first element of edge_attr
        mol.AddBond(i, j, order=bond_type)

    # Convert the RDKit molecule to a non-editable molecule
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSSSR(mol)
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.AtomValenceException:
        print("Sanitization failed. Skipping molecule.")
        return None

    AllChem.EmbedMolecule(mol)

    # Optimize the generated conformer
    AllChem.MMFFOptimizeMolecule(mol)

    return mol

#TODO: consider adding a way to add target features to the dataset. like for tox21.
def sdf_to_arrow(
    sdf_file: str, to_disk_location: str = None, cache_dir: str = "./data/huggingface"
):
    """
    Converts a sdf file to a Hugging Face dataset.

    Args
    ----
    sdf_file: str
        Path to sdf file to convert.
    to_disk_location: str
        Path to save the dataset to (the directory).
    cache_dir: str
        Path to cache directory for Hugging Face dataset."""

    features = Features(
        {
            "edge_attr": Sequence(
                feature=Sequence(
                    feature=Value(dtype="int64", id=None), length=-1, id=None
                ),
                length=-1,
                id=None,
            ),
            "node_feat": Sequence(
                feature=Sequence(
                    feature=Value(dtype="int64", id=None), length=-1, id=None
                ),
                length=-1,
                id=None,
            ),
            "edge_index": Sequence(
                feature=Sequence(
                    feature=Value(dtype="int64", id=None), length=-1, id=None
                ),
                length=-1,
                id=None,
            ),
            "pos": Sequence(
                feature=Sequence(
                    feature=Value(dtype="float32", id=None), length=-1, id=None
                ),
                length=-1,
                id=None,
            ),
            "smiles": Value("string"),
            "num_nodes": Value("int64"),
        }
    )

    dataset = Dataset.from_generator(
        generate_from_sdf,
        gen_kwargs={"sdf_file": sdf_file},
        features=features,
        cache_dir=cache_dir,
    )
    dataset.save_to_disk(to_disk_location)


def generate_from_sdf(sdf_file):
    """
    Generate processed molecule representations from an SDF file.

    Args
    ----
    sdf_file: str
        Path to the SDF file to generate molecules from.

    Yields:
        Processed molecule dictionary.
    """

    suppl = Chem.SDMolSupplier(sdf_file, removeHs=True, sanitize=True)
    for mol in suppl:
        if mol is not None:
            yield process_molecule(mol)



def process_molecule(mol: Chem.Mol):
    """
    Process an RDKit molecule and extract features for graph representation.

    Args
    ----
    mol: Chem.Mol
        The RDKit molecule object to process.

    Returns
    -------
    dict: A dictionary containing the extracted features for graph representation.
        The dictionary contains the following keys:
        - 'edge_attr': A list of edge attributes.
        - 'node_feat': A list of node features.
        - 'smiles': The SMILES representation of the molecule.
        - 'num_nodes': The number of nodes (atoms) in the molecule.
        - 'edge_index': A list of edge indices.
        - 'pos': A list of 3D coordinates for each atom.
    """

    return_dict = {
        "edge_attr": [],
        "node_feat": [],
        "smiles": Chem.MolToSmiles(mol),
        "num_nodes": 0,
    }

    for atom in mol.GetAtoms():
        x = []
        x.append(x_map["atomic_num"].index(atom.GetAtomicNum())) #TODO: this index call is not necessary I think
        x.append(x_map["chirality"].index(str(atom.GetChiralTag())))
        x.append(x_map["degree"].index(atom.GetTotalDegree()))
        x.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
        x.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
        x.append(x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()))
        x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        x.append(x_map["is_in_ring"].index(atom.IsInRing()))
        return_dict["node_feat"].append(x)

    edge_indices_send, edge_indices_rec = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map["bond_type"].index(str(bond.GetBondType())))
        e.append(e_map["stereo"].index(str(bond.GetStereo())))
        e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices_send += [i, j]
        edge_indices_rec += [j, i]
        return_dict["edge_attr"].extend([e, e])

    return_dict["pos"] = mol.GetConformer().GetPositions().tolist()

    return_dict["num_nodes"] = len(return_dict["node_feat"])
    return_dict["edge_index"] = [edge_indices_send, edge_indices_rec]
    return return_dict
