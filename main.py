#!/usr/bin/python
"""This script provides the main entry point for training and evaluating the Graphormer model 
on a given dataset. It handles the preparation of the dataset, the creation and configuration 
of the model, and the setup of the training process, including logging and the application of a 
random seed for reproducibility."""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-05-20"


# Standard library imports
import os
import random
from argparse import ArgumentParser

# Third party imports
import numpy as np
import torch
import yaml
from transformers import (
    GraphormerConfig,
    GraphormerForGraphClassification,
    Trainer,
    TrainingArguments,
)

# Local application imports
from utils import data as data_utils
from utils import graphormer_data_collator_improved as graphormer_collator_utils
from utils import setup as setup_utils


def main(
    name: str = None, logdir: str = "runs", yaml_file: str = "configs/dummy_config.yml"
):
    """
    Entry point for training and evaluating the model.

    Args
    ----
        name:(str)
            Name of the experiment.
        logdir (str): Directories where logs are stored.
        yaml_file (str): The yaml file with the config.
    """
    # TODO: Implement the logic of the main function
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    name = setup_utils.get_experiment_name(config, name)
    logpath = os.path.join(logdir, name)
    logger = setup_utils.setup_logging(logpath, name)
    setup_utils.setup_wandb()

    seed = config["seed"]  # TODO: maybe allow multiple seeds with a loop
    logger.info(f"Set the random seed to : {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TODO: move below to a seperate function maybe
    pretraining = config["pretraining"]
    data_args = config["data_args"]

    dataset = data_utils.prepare_dataset_for_training(pretraining, **data_args)

    # Define your custom model here
    model_config = GraphormerConfig(**config["model_args"])
    model = GraphormerForGraphClassification(model_config)

    # Set up the training arguments
    # TODO: maybe add logic for hyperparameter search

    setup_utils.setup_batch_size(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config["devices"]))

    training_args = TrainingArguments(
        output_dir=os.path.join(logpath, "checkpoints"),
        logging_dir=logpath,
        seed=seed,
        **config["trainer_args"],
    )

    if not pretraining:
        collator = graphormer_collator_utils.GraphormerDataCollator()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=collator,
        )

        trainer.train()
    # TODO: Implement the pretraining logic


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-n", "--name", help="Name of the experiment", default=f"*time*", type=str
    )
    parser.add_argument(
        "-l",
        "--logdir",
        help="Directories where logs are stored",
        default=f"runs",
        type=str,
    )
    parser.add_argument(
        "-y",
        "--yaml_file",
        help="The yaml file with the config",
        default="configs/dummy_config.yml",
        type=str,
    )

    args = parser.parse_args()

    main(**dict(args._get_kwargs()))
