"""
Utility functions for data preprocessing.

Parts are from the Source: https://github.com/pyg-team/pytorch_geometric
With the Title: Took some parts from torch_geometric/utils/smiles.py

The modified results from this source are:
- x_map and e_map.
- Parts of the process_molecule function.

Modified Code Copyright (c) 2023 PyG Team <team@pyg.org>
Modified Code Copyright (c) 2023 Alexander Krauck

This code is distributed under the MIT license. See LICENSE_PYTORCH_GEOMETRIC.txt and LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"


from os.path import join

from datasets import Dataset, Value, Features, Sequence, load_from_disk, DatasetDict
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.RDLogger as RDLogger
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem import rdMolTransforms
import numpy as np


import pandas as pd

from typing import Optional, List

from utils import graphormer_data_collator_improved as graphormer_collator_utils

from transformers.utils import is_cython_available, requires_backends


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from transformers.models.graphormer import algos_graphormer  # noqa E402

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


def sdf_to_arrow(
    sdf_file: str,
    to_disk_location: Optional[str] = None,
    cache_dir: str = "./data/huggingface",
    id_mol_property: Optional[str] = "_Name",
    **kwargs,
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
        Path to cache directory for Hugging Face dataset.
    id_mol_property: Optional[str]
        Name of the molecule property to use as the id.
    **kwargs:
        Additional keyword arguments for customizing the SDF reading process.
        Example: target_columns=[2, 3, 4] csv_with_metainfo='targets.csv', split_column=1, take_split='training'
    """

    features = get_arrow_features(include_name=(id_mol_property is not None))

    dataset = Dataset.from_generator(
        generate_from_sdf,
        gen_kwargs={"sdf_file": sdf_file, "id_mol_property": id_mol_property, **kwargs},
        # features=features,
        cache_dir=cache_dir,
    )
    if to_disk_location is not None:
        dataset.save_to_disk(to_disk_location)

    return dataset


def rdkit_to_arrow(
    rdkit_mol_list: List[Chem.Mol],
    target_list: Optional[List] = None,
    to_disk_location: Optional[str] = None,
    cache_dir: str = "./data/huggingface",
):
    """
    Converts a list of rdkit molecules to an arrow dataset.

    Args
    ----
    rdkit_mol_list: List[Chem.Mol]
        Iterable of molecules from rdkit.
    target_list: Optional[List]
        List of targets for each molecule.
    to_disk_location: str
        Path to save the dataset to (the directory).
    cache_dir: str
        Path to cache directory for Hugging Face dataset."""

    features = get_arrow_features()

    dataset = Dataset.from_generator(
        generate_from_rdkit,
        gen_kwargs={"rdkit_mol_list": rdkit_mol_list, "target_list": target_list},
        # features=features,
        cache_dir=cache_dir,
    )

    if to_disk_location is not None:
        dataset.save_to_disk(to_disk_location)

    return dataset


def csv_to_arrow(
    csv_file: str,
    to_disk_location: Optional[str] = None,
    cache_dir: str = "./data/huggingface",
    include_conformer: bool = True,
    id_column: Optional[int] = None,
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
    id_column: Optional[int]
        The column index of the id column.
    **kwargs:
        Additional keyword arguments for customizing the CSV reading process.
        Example: smiles_column=1, has_header=True, target_columns=[2, 3, 4]
    """

    features = get_arrow_features(
        include_3d_positions=include_conformer, include_name=(id_column is not None)
    )

    dataset = Dataset.from_generator(
        generate_from_csv,
        gen_kwargs={
            "csv_file": csv_file,
            "include_conformer": include_conformer,
            "id_column": id_column,
            **kwargs,
        },
        # features=features,
        cache_dir=cache_dir,
    )

    if to_disk_location is not None:
        dataset.save_to_disk(to_disk_location)

    return dataset


def generate_from_sdf(
    sdf_file: str,
    id_mol_property: Optional[str] = "_Name",
    csv_with_metainfo: Optional[str] = None,
    target_columns: Optional[List[int]] = None,
    split_column: Optional[int] = None,
    take_split: Optional[str] = None,
):
    """
    Generate processed molecule representations from an SDF file.

    Args
    ----
    sdf_file: str
        Path to the SDF file to generate molecules from.
    id_mol_property: Optional[str]
        Name of the molecule property to use as the molecule name if desired.
    csv_with_metainfo: Optional[str]
        Path to the CSV file containing the targets for each molecule.
    target_columns: Optional[List[int]]
        List of column indices to use as targets.
    split_column: Optional[int]
        Column index of the csv_with_metainfo file that contains the split information. ("train", "val", "test")
    take_split: Optional[str]
        Which split to take. ("training", "validation", "test")

    Yields:
        Processed molecule dictionary.
    """
    # TODO: low priority. but this is currently heavily based on the tox21_orignal dataset and not generalized
    assert (
        take_split is None or split_column is not None
    ), "Must provide split column if take_split is provided."
    assert (
        split_column is None or csv_with_metainfo is not None
    ), "Must provide csv_with_metainfo if split_column is provided."

    if csv_with_metainfo is not None:
        assert (
            target_columns is not None
        ), "Must provide target columns if csv is provided."

        metainfo = pd.read_csv(csv_with_metainfo)
        targets = metainfo.iloc[:, target_columns].to_numpy().tolist()

        if split_column is not None:
            splits = metainfo.iloc[:, split_column].to_numpy().tolist()

    suppl = Chem.SDMolSupplier(sdf_file, removeHs=True, sanitize=True)
    mol_index = 0
    for mol in suppl:
        if mol is not None:
            if (
                split_column is not None
                and take_split is not None
                and splits[mol_index] != take_split
            ):
                mol_index += 1
                continue

            mol_dict = process_molecule(mol)
            if id_mol_property is not None:
                mol_dict["name"] = mol.GetProp(id_mol_property)
            mol_dict["id"] = mol_index
            if csv_with_metainfo is not None:
                mol_dict["target"] = targets[mol_index]

            yield mol_dict

        mol_index += 1


def generate_from_rdkit(
    rdkit_mol_list: List[Chem.Mol], target_list: Optional[List] = None
):
    """
    Generate processed molecule representations from a list of RDKit molecules.

    Args
    ----
    rdkit_mol_list: List[Chem.Mol]
        A iterable of RDKit molecules to generate representations from.

    Yields:
        Processed molecule dictionary.
    """

    for mol_index, mol in enumerate(rdkit_mol_list):
        if mol is not None:
            mol_dict = process_molecule(mol)
            mol_dict["id"] = mol_index
            if target_list is not None:
                mol_dict["target"] = target_list[mol_index]
            yield mol_dict


def generate_from_csv(
    csv_file,
    smiles_column: Optional[int] = None,
    has_header: Optional[bool] = True,
    include_conformer: Optional[bool] = True,
    id_column: Optional[int] = None,
    target_columns: Optional[List[int]] = None,
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
    id_column: int
        index of the column containing the unique id for each molecule if there is one.
    target_columns: List[int]
        List of column indices containing the target values.
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

        mol_index = 0
        RDLogger.DisableLog("rdApp.warning")
        for line in file:
            line = line.strip()
            if line == "":
                continue
            split_line = line.split(",")
            smiles = split_line[smiles_column]

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
                        f"Could not generate conformer for molecule index {mol_index} with SMILES {smiles}."
                    )
                    continue

            if mol is not None:
                mol_dict = process_molecule(mol, include_conformer=include_conformer)
                mol_dict["id"] = mol_index
                if id_column is not None:
                    mol_dict["name"] = split_line[id_column]
                if target_columns is not None:
                    mol_dict["target"] = [
                        float("nan")
                        if split_line[target_column] == ""
                        else float(split_line[target_column])
                        for target_column in target_columns
                    ]
                yield mol_dict
        mol_index += 1

        RDLogger.EnableLog("rdApp.warning")


def get_arrow_features(
    include_3d_positions: bool = True, include_name: str = False
) -> Features:
    """
    Returns the Hugging Face features for the processed molecule representation.
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
        "id": Value("int64"),
    }

    if include_3d_positions:
        features_dict["pos"] = Sequence(
            feature=Sequence(
                feature=Value(dtype="float32", id=None), length=-1, id=None
            ),
            length=-1,
            id=None,
        )

    if include_name:
        features_dict["name"] = Value("string")

    return Features(features_dict)


def process_molecule(
    mol: Chem.Mol,
    include_conformer: bool = True,
    generate_bounds: bool = False,
    n_bound_hops: int = 2,
):
    """
    Process an RDKit molecule and extract features for graph representation.

    Args
    ----
    mol: Chem.Mol
        The RDKit molecule object to process.
    include_conformer: bool
        Whether to include the 3D conformer of the molecule.
    generate_bounds: bool
        Whether to generate the bounds for the molecule.
    n_bound_hops: int
        The number of hops to generate bounds for.

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
        x.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
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

    if generate_bounds:
        adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
        shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
        return_dict["shortest_path_result"] = shortest_path_result
        return_dict["path"] = path
        return_dict["distance_bounds"] = GetMoleculeBoundsMatrix(mol)
        triplet, cosine_bounds = generate_angle_bounds(
            mol, return_dict["distance_bounds"], n_bound_hops, do_conformer=False
        )
        return_dict["triplet"] = triplet
        return_dict["cosine_bounds"] = cosine_bounds

    return return_dict


def generate_angle_bounds(
    mol, bounds, shortest_path_distance, n_hops=5, do_conformer=False
):
    """
    Generate angle bounds for a molecule.

    Parameters:
    mol: RDKit molecule object
    bounds: Distance bounds matrix of shape (num_atoms, num_atoms)
    shortest_path_distance: Shortest path distance matrix of shape (num_atoms, num_atoms)
    n_hops: Number of hops to generate bounds for

    Returns:
    angle_bounds: List of angle bounds for each triplet of atoms
    """

    # Initialize a list to store the indices of the triplets
    triplets = []
    actual_angles = []

    # Loop over the atoms in the molecule
    for atom_i in mol.GetAtoms():
        i = atom_i.GetIdx()

        atoms_within_n_hops = np.where(
            (shortest_path_distance[i] <= n_hops) & (shortest_path_distance[i] > 0)
        )[0]
        # Loop over the neighbors of the atom
        for j in atoms_within_n_hops:
            # Loop over the other neighbors of the atom
            for k in atoms_within_n_hops:
                # Skip the case where the neighbor is the same as the previous neighbor, or where the neighbor index is less than or equal to the previous neighbor index
                if k <= j:
                    continue

                j, k = int(j), int(k)
                # Add the triplet to the list
                triplets.append([i, j, k])

                if mol.GetNumConformers() != 0 and do_conformer:
                    actual_angle = rdMolTransforms.GetAngleRad(
                        mol.GetConformer(0), j, i, k
                    )
                    actual_angles.append(actual_angle)

    # Convert the list of triplets to a numpy array
    if len(triplets) == 0:
        if do_conformer:
            return np.zeros((0, 3)), np.zeros((0, 3))
        else:
            return np.zeros((0, 3)), np.zeros((0, 2))

    triplets = np.array(triplets)
    if mol.GetNumConformers() != 0:
        actual_angles = np.array(actual_angles)

    # Extract the distance bounds for the sides of the triangle
    a_min_values = np.minimum(
        bounds[triplets[:, 0], triplets[:, 1]], bounds[triplets[:, 1], triplets[:, 0]]
    )
    a_max_values = np.maximum(
        bounds[triplets[:, 0], triplets[:, 1]], bounds[triplets[:, 1], triplets[:, 0]]
    )
    b_min_values = np.minimum(
        bounds[triplets[:, 0], triplets[:, 2]], bounds[triplets[:, 2], triplets[:, 0]]
    )
    b_max_values = np.maximum(
        bounds[triplets[:, 0], triplets[:, 2]], bounds[triplets[:, 2], triplets[:, 0]]
    )
    c_min_values = np.minimum(
        bounds[triplets[:, 1], triplets[:, 2]], bounds[triplets[:, 2], triplets[:, 1]]
    )
    c_max_values = np.maximum(
        bounds[triplets[:, 1], triplets[:, 2]], bounds[triplets[:, 2], triplets[:, 1]]
    )

    # Calculate the minimum and maximum possible cosines for each vertex of the triangle
    # Note that this can contain values outside of [-1, 1] because of how the bounds matrix is calculated
    gamma_min_values = (a_min_values**2 + b_min_values**2 - c_max_values**2) / (
        2 * a_min_values * b_min_values
    )
    gamma_max_values = (a_max_values**2 + b_max_values**2 - c_min_values**2) / (
        2 * a_max_values * b_max_values
    )

    to_stack = [gamma_min_values[:, None], gamma_max_values[:, None]]
    if mol.GetNumConformers() != 0 and do_conformer:
        to_stack.append(np.cos(np.array(actual_angles)[:, None]))
    # Combine the indices and angle bounds into a single array
    angle_bounds = np.hstack(to_stack)

    return triplets, angle_bounds


def map_arrow_dataset_from_disk(dataset_location: str, is_dataset_dict: bool = False):
    """"""

    source_dataset_location = join(dataset_location, "arrow")
    destination_dataset_location = join(dataset_location, "arrow_processed")

    if is_dataset_dict:
        dataset = DatasetDict.load_from_disk(source_dataset_location)
    else:
        dataset = load_from_disk(source_dataset_location)

    dataset = dataset.map(
        graphormer_collator_utils.preprocess_item,
        batched=False,
        load_from_cache_file=False,
    )

    dataset.save_to_disk(destination_dataset_location)
