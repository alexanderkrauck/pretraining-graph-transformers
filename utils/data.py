"""
Utility functions for data handling during a model call.

Copyright (c) 2023 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
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
from utils import graphormer_data_collator_improved_3d as graphormer_collator_utils_3d
import random

from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import StandardScaler



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

    if not pretraining:
        if dataset_name == "tox21_original":
            dataset = DatasetDict.load_from_disk(
                join(data_dir, "tox21_original/processed/arrow"),
                keep_in_memory=True,
            )

        if dataset_name == "tox21":
            dataset = load_from_disk(
                join(data_dir, "tox21/processed/arrow"),
                keep_in_memory=True,
            )

        if dataset_name == "ZINC":
            dataset = DatasetDict.load_from_disk(
                join(data_dir, "ZINC/processed/arrow"),
                keep_in_memory=True,
            )

        if dataset_name == "qm9":
            dataset = load_from_disk(join(data_dir, "qm9/processed/arrow"))

    else:
        if dataset_name == "pcqm4mv2":
            dataset = load_from_disk(
                join(data_dir, "pcqm4mv2/processed/arrow"),
                keep_in_memory=False,
            )
        if dataset_name == "pcba":
            dataset = load_from_disk(join(data_dir, "pcba/processed/arrow"))
        if dataset_name == "qm9":
            dataset = load_from_disk(join(data_dir, "qm9/processed/arrow"))

    if dataset is None:
        raise ValueError(f"Invalid dataset name for pretraining = {pretraining}.")

    if memory_mode in ["full", "half"]:
        dataset = to_preloaded_dataset(
            dataset, preprocess=True if memory_mode == "full" else False, **kwargs
        )

    if isinstance(dataset, Dataset) or isinstance(dataset, PreloadedDataset):
        dataset = split_dataset(dataset, train_split, seed)

    return dataset


def prepare_cv_dataset_for_training(
    seed: int,
    dataset_name: str,
    data_dir: str,
    memory_mode: str,
    num_folds: int,
    train_split: float,
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
        num_folds (int): Number of folds to use for cross validation.
    """

    if dataset_name == "tox21":
        dataset = load_from_disk(join(data_dir, "tox21/processed/arrow"))

    if dataset_name == "qm9":
        dataset = load_from_disk(join(data_dir, "qm9/processed/arrow"))

    if memory_mode in ["full", "half"]:
        dataset = to_preloaded_dataset(
            dataset, preprocess=True if memory_mode == "full" else False, **kwargs
        )
    else:
        raise ValueError(f"Invalid memory mode {memory_mode}.")

    datasets = dataset.train_test_split(1 - train_split, seed=seed)

    return datasets, datasets["train"].k_fold_index_split(num_folds, seed)


def get_dataset_task(dataset_name: str, **kwargs):
    if dataset_name in ["pcba", "tox21", "tox21_original"]:
        return "classification"
    if dataset_name in ["qm9", "ZINC"]:
        return "regression"


def get_dataset_num_classes(dataset_name: str, **kwargs):
    if dataset_name in ["tox21", "tox21_original"]:
        return 12
    if dataset_name in ["pcba"]:
        return 128
    if dataset_name in ["qm9"]:
        return 19
    if dataset_name in ["ZINC"]:
        return 1
    return None


class PreloadedDataset(TorchDataset):
    """
    A preloaded dataset. This is useful when the dataset is small enough to fit in memory.
    """

    def __init__(
        self,
        dataset: Union[Dataset, list],
        column_names: Optional[list] = None,
        preprocess: bool = True,
        model_type: str = "graphormer",
        **kwargs,
    ):
        """
        Args
        ----
            dataset (Dataset): The dataset to preload.
            preprocess (bool): Whether to preprocess the dataset already.
        """
        if isinstance(dataset, list):
            self.rows = dataset
            self.column_names = column_names
        else:
            self.column_names = dataset.column_names
            self.rows = []
            for i in tqdm(range(len(dataset))):
                row = dataset[i]
                if preprocess:
                    if model_type == "graphormer3d":
                        row = graphormer_collator_utils_3d.preprocess_3D_item(
                            row, **kwargs
                        )
                    else:
                        row = graphormer_collator_utils.preprocess_item(row, **kwargs)
                self.rows.append(row)
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

    def k_fold_index_split(self, n_folds=5, seed=None):
        """
        Split the dataset into K folds.

        Args
        ----
            n_folds (int): The number of folds.
            seed (int): The seed for the random number generator.
        """
        indices = list(range(len(self.rows)))
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = list(kf.split(indices))

        return folds

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
    dataset: Union[Dataset, DatasetDict], preprocess: bool = True, **kwargs
):
    """
    Convert a dataset to a preloaded dataset.

    Args
    ----
        dataset (Union[Dataset, DatasetDict]): The dataset to convert.
        preprocess (bool): Whether to preprocess the dataset already.
    """
    if isinstance(dataset, DatasetDict):
        return {
            k: to_preloaded_dataset(v, preprocess, **kwargs) for k, v in dataset.items()
        }

    return PreloadedDataset(dataset, preprocess, **kwargs)


def is_cross_val_dataset(dataset_name: str, **kwargs):
    """
    Check if the dataset is a cross validation dataset.

    Args
    ----
        dataset_name (str): The name of the dataset.
    """
    return dataset_name.lower() in ["pcba", "tox21", "qm9"]

def get_regression_target_scaler(dataset, num_classes: int):
    targets = []
    for item in dataset:
        targets.append(item["labels"])
    target_scaler = StandardScaler()
    target_array = np.array(targets)
    if num_classes == 1:
        target_array = target_array.reshape(-1, 1)
    target_scaler.fit(target_array)
    return target_scaler