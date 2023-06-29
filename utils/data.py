"""
Utility functions for data handling during a model call.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"


from os.path import join

from torch.utils.data import Dataset as TorchDataset

from datasets import Dataset, load_from_disk, DatasetDict

from tqdm import tqdm

from typing import Optional, List, Union

import copy

from utils import graphormer_data_collator_improved as graphormer_collator_utils
import random

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

    if isinstance(dataset, Dataset) or isinstance(dataset, PreloadedDataset):
        dataset = split_dataset(dataset, train_split, seed)

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
        return copy.deepcopy(self.rows[idx])

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
