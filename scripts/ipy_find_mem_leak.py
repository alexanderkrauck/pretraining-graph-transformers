#%%
import os
os.chdir("..")

from os.path import join

from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataloader, Subset

from datasets import Dataset, load_from_disk, DatasetDict

from tqdm import tqdm

from typing import Optional, List, Union

import copy

from utils import graphormer_data_collator_improved as graphormer_collator_utils
from utils import graphormer_data_collator_improved_3d as graphormer_collator_utils_3d
import random

from sklearn.model_selection import KFold
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from utils.modeling_graphormer_improved_3d import Graphormer3DConfig, Graphormer3DForGraphClassification
from utils.graphormer_data_collator_improved_3d import Graphormer3DDataCollator
import yaml

from transformers import (
    Trainer,
    TrainingArguments,
)
#%%
dataset_size = 1000
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

        self.last_idx = 0
        if isinstance(dataset, list):
            self.rows = dataset
            self.column_names = column_names
        else:
            self.column_names = dataset.column_names
            self.rows = []
            self.das = dataset

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
        entry = self.das[self.last_idx]
        preprocessed_entry = graphormer_collator_utils_3d.preprocess_3D_item(entry)
        self.last_idx += 1
        return preprocessed_entry

    def __len__(self):
        return 3000000

# %%
data_dir = "data"
dataset = load_from_disk(join(data_dir, "pcqm4mv2/processed/arrow"))

dataset = PreloadedDataset(dataset, preprocess=True, model_type="graphormer3d")
#%%
class DummyModel(torch.nn.Module):
    """
    Implementation of the 3d Graphormer model for graph classification as in the repository of the paper.
    """

    def __init__(self, config):
        super().__init__()

        self.dummy_param = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True)

    def forward(
        self,
        input_nodes,
        pos,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **unused,
    ):
        loss = (self.dummy_param * 0.0).squeeze()
        return (loss, torch.randn(256,50,80))

# %%
with open("configs/dummy_config_3d.yml", "r") as file:
    config = yaml.safe_load(file)

#config["model_args"][""]
model_config = Graphormer3DConfig(**config["model_args"])

collator = Graphormer3DDataCollator(
                model_config=model_config,
                on_the_fly_processing=False,
                collator_mode="pretraining",
                target_scaler=None,
            )

dl = TorchDataloader(dataset,
            batch_size=256,
            sampler=torch.utils.data.RandomSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
model = DummyModel(model_config)

training_args = TrainingArguments(
            output_dir=os.path.join("logs", "checkpoints"),
            logging_dir="test1231231",
            seed=12,
            data_seed=12,
            run_name="gaw",
            report_to="none",
            **config["trainer_args"],
        )
trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=Subset(dataset, list(range(2000))),
            data_collator=collator
        )
# %%
trainer.train()
# %%
