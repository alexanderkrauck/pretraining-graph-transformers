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


from os.path import join

import torch
from torch_geometric.data.dataset import Dataset as PyGDataset
from torch.utils.data import Dataset as TorchDataset
import wandb

from datasets import Dataset, Value, Features, Sequence, load_from_disk, DatasetDict
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.RDLogger as RDLogger

import pandas as pd

from tqdm import tqdm

from typing import Iterable, Optional, List, Union

from utils import graphormer_data_collator_improved as graphormer_collator_utils
import random

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
    return return_dict


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


def prepare_dataset_for_training(
    pretraining: bool,
    seed: int,
    dataset_name: str,
    data_dir: str,
    memory_mode: str,
    train_split: Optional[float] = None,
    **kwargs,
):
    """
    Prepare the dataset for training.

    Args
    ----
        pretraining (bool): Whether to use the pretraining dataset or the finetuning dataset.
        seed (int): The random seed to use for splitting the dataset.
        dataset_name (str): Name of the dataset.
        data_dir (str): Path to the data directory.
        memory_mode (str): Whether to load the dataset in memory or not. Can be one of ['full', 'half', 'cache'].
        train_split (float): The percentage of the dataset to use for training. Only used for finetuning.
    """

    path_extension = "_processed" if memory_mode == "full" else ""

    if not pretraining:
        if dataset_name == "tox21_original":
            dataset = DatasetDict.load_from_disk(
                join(data_dir, "tox21_original/processed/arrow" + path_extension),
                keep_in_memory=True,
            )

        if dataset_name == "tox21":
            dataset = load_from_disk(
                join(data_dir, "tox21/processed/arrow" + path_extension),
                keep_in_memory=True,
            )

        if dataset_name == "ZINC":
            dataset = DatasetDict.load_from_disk(
                join(data_dir, "ZINC/processed/arrow" + path_extension),
                keep_in_memory=True,
            )

        if dataset_name == "qm9":
            dataset = load_from_disk(
                join(data_dir, "qm9/processed/arrow" + path_extension)
            )

    else:
        if dataset_name == "pcqm4mv2":
            dataset = load_from_disk(
                join(data_dir, "pcqm4mv2/processed/arrow" + path_extension),
                keep_in_memory=False,
            )
        if dataset_name == "pcba":
            dataset = load_from_disk(
                join(data_dir, "pcba/processed/arrow" + path_extension)
            )
        if dataset_name == "qm9":
            dataset = load_from_disk(
                join(data_dir, "qm9/processed/arrow" + path_extension)
            )

    if dataset is None:
        raise ValueError(f"Invalid dataset name for pretraining = {pretraining}.")

    if memory_mode in ["full", "half"]:
        dataset = to_preloaded_dataset(
            dataset, format_numpy=True if memory_mode == "full" else False
        )

    if not isinstance(dataset, DatasetDict):
        split_dataset(dataset, train_split, seed)

    return dataset


class PreloadedDataset(TorchDataset):
    """
    A preloaded dataset. This is useful when the dataset is small enough to fit in memory.
    """

    def __init__(
        self,
        dataset: Union[Dataset, list],
        format_numpy: bool = True,
        column_names: Optional[list] = None,
    ):
        """
        Args
        ----
            dataset (Dataset): The dataset to preload.
            format_numpy (bool): Whether to convert the dataset to numpy or not.
        """
        if isinstance(dataset, list):
            self.rows = dataset
            self.column_names = column_names
        else:
            if format_numpy:
                dataset.set_format(type="numpy", columns=list(dataset.column_names))
            self.column_names = dataset.column_names
            self.rows = []
            for i in tqdm(range(len(dataset))):
                self.rows.append(dataset[i])
            del dataset

    def train_test_split(self, test_size=0.8, seed=None, shuffle=True):
        """
        Split the dataset into train and test set.

        Args
        ----
            test_size (float): The ratio of the test set.
            seed (int): The seed for the random number generator.
            shuffle (bool): Whether to shuffle the dataset before splitting.
        """
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self.rows)

        split_index = int(len(self.rows) * test_size)
        # NOTE: Thats dirty, but it works
        return {
            "test": PreloadedDataset(
                self.rows[:split_index], column_names=self.column_names
            ),
            "train": PreloadedDataset(
                self.rows[split_index:], column_names=self.column_names
            ),
        }

    def __getitem__(self, idx):
        return self.rows[idx]

    def __len__(self):
        return len(self.rows)


def split_dataset(
    dataset: Union[Dataset, PreloadedDataset], train_split: float, seed: int
):
    """
    Split the dataset into train, validation and test set.

    Args
    ----
        dataset (Dataset): The dataset to split.
        train_split (float): The ratio of the training set.
        seed (int): The seed for the random number generator.
    """
    # TODO: check for bugs

    dataset = dataset.train_test_split(
        test_size=1 - train_split, seed=seed, shuffle=True
    )

    test_val_dataset = dataset["test"].train_test_split(
        test_size=0.5, seed=seed, shuffle=True
    )

    return {
        "train": dataset["train"],
        "validation": test_val_dataset["train"],
        "test": test_val_dataset["test"],
    }


def to_preloaded_dataset(
    dataset: Union[Dataset, DatasetDict], format_numpy: bool = True
):
    """
    Convert a dataset to a preloaded dataset.

    Args
    ----
        dataset (Union[Dataset, DatasetDict]): The dataset to convert.
        format_numpy (bool): Whether to convert the dataset to numpy or not.
    """
    if isinstance(dataset, DatasetDict):
        return {k: to_preloaded_dataset(v, format_numpy) for k, v in dataset.items()}

    return PreloadedDataset(dataset, format_numpy)
