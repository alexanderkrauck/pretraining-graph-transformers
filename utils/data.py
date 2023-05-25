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
import rdkit.RDLogger as RDLogger

import pandas as pd

from tqdm import tqdm

from typing import Iterable, Optional

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


# TODO: unify the features. node_feat and edge_attr should be the same as when loading sdf files.
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


def load_smiles_dataset(dataset_location: str, smiles_column: Optional[int] = None):
    """
    Loads a dataset from a csv file.

    Args
    ----
    dataset_location: str
        Path to the csv file.
    smiles_column: int
        Column index of the SMILES strings."""

    dataset = pd.read_csv(dataset_location)
    if smiles_column is None:
        if "smiles" in dataset.columns:
            smiles_list = dataset["smiles"].tolist()
        else:
            raise ValueError("No smiles column found. Please specify the column index.")
    else:
        smiles_list = dataset.iloc[:, smiles_column].tolist()
    mol_list = []
    print("Converting SMILES to rdkit molecules.")
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)

    return mol_list


# TODO: consider adding a way to add target features to the dataset. like for tox21.
def rdkit_to_arrow(
    rdkit_iterable: Iterable[Chem.Mol],
    to_disk_location: str = None,
    cache_dir: str = "./data/huggingface",
):
    """
    Converts a sdf file to a Hugging Face dataset.

    Args
    ----
    rdkit_iterable: Iterable[Chem.Mol]
        Iterable of molecules from rdkit.
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
        generate_from_rdkit,
        gen_kwargs={"rdkit_iterable": rdkit_iterable},
        features=features,
        cache_dir=cache_dir,
    )
    dataset.save_to_disk(to_disk_location)

    return dataset


def csv_to_arrow(
    csv_file: str,
    to_disk_location: str = None,
    cache_dir: str = "./data/huggingface",
    include_conformer: bool = True,
    **kwargs,
):
    """
    Converts a csv file to a Hugging Face dataset.

    Args
    ----
    csv_file: str
        Path to the csv file.
    to_disk_location: str
        Path to save the dataset to (the directory).
    cache_dir: str
        Path to cache directory for Hugging Face dataset.
    include_conformer: bool
        Whether to include the conformer in the dataset.
    **kwargs:
        Additional keyword arguments for customizing the CSV reading process.
        Example: smiles_column=1, has_header=True.
    """

    features_dict = {
        "edge_attr": Sequence(
            feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None),
            length=-1,
            id=None,
        ),
        "node_feat": Sequence(
            feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None),
            length=-1,
            id=None,
        ),
        "edge_index": Sequence(
            feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None),
            length=-1,
            id=None,
        ),
        "smiles": Value("string"),
        "num_nodes": Value("int64"),
    }

    if include_conformer:
        features_dict["pos"] = Sequence(
            feature=Sequence(
                feature=Value(dtype="float32", id=None), length=-1, id=None
            ),
            length=-1,
            id=None,
        )

    features = Features(features_dict)

    dataset = Dataset.from_generator(
        generate_from_csv,
        gen_kwargs={
            "csv_file": csv_file,
            "include_conformer": include_conformer,
            **kwargs,
        },
        features=features,
        cache_dir=cache_dir,
    )
    dataset.save_to_disk(to_disk_location)

    return dataset


def generate_from_csv(
    csv_file,
    smiles_column: Optional[int] = None,
    has_header: Optional[bool] = True,
    include_conformer: Optional[bool] = True,
):
    """
    Generates a Hugging Face dataset from a csv file.

    Args
    ----
    csv_file: str
        Path to the csv file.
    smiles_column: int
        Column index of the SMILES strings.
    has_header: bool
        Whether the csv file has a header or not.
    """

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

        line_count = 0
        RDLogger.DisableLog("rdApp.warning")
        for line in file:
            line_count += 1

            line = line.strip()
            if line == "":
                continue
            smiles = line.split(",")[smiles_column]

            mol = Chem.MolFromSmiles(smiles)

            if include_conformer:
                try:
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(
                        mol
                    )  # Optimize the generated conformer
                    mol.GetConformer()
                except ValueError:
                    print(
                        f"Could not generate conformer for molecule {line_count} with SMILES {smiles}."
                    )
                    continue

            if mol is not None:
                yield process_molecule(mol, include_conformer=include_conformer)

        RDLogger.EnableLog("rdApp.warning")


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

    return dataset


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


def generate_from_rdkit(rdkit_iterable: Iterable[Chem.Mol]):
    """
    Generate processed molecule representations from a list of RDKit molecules.

    Args
    ----
    rdkit_iterable: Iterable[Chem.Mol]
        A iterable of RDKit molecules to generate representations from.

    Yields:
        Processed molecule dictionary.
    """

    for mol in rdkit_iterable:
        if mol is not None:
            yield process_molecule(mol)


def process_molecule(mol: Chem.Mol, include_conformer: bool = True):
    """
    Process an RDKit molecule and extract features for graph representation.

    Args
    ----
    mol: Chem.Mol
        The RDKit molecule object to process.
    include_conformer: bool
        Whether to include the 3D conformer of the molecule.

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
        x.append(
            x_map["atomic_num"].index(atom.GetAtomicNum())
        )  # TODO: this index call is not necessary I think
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

    if include_conformer:
        return_dict["pos"] = mol.GetConformer().GetPositions().tolist()

    return_dict["num_nodes"] = len(return_dict["node_feat"])
    return_dict["edge_index"] = [edge_indices_send, edge_indices_rec]
    return return_dict
