"""
This module provides utility functions for setting up experiments in the project. 

Functions included in this module are:

- load_config: Loads a YAML configuration file and returns a dictionary of the configuration.
- setup_logging: Sets up a logger for the experiment that logs to both the console and a specified log file.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2023-06-03"


import logging
import os
from datetime import datetime
from pathlib import Path
import wandb
import yaml


def get_experiment_name(config, name=None):
    """
    Get the experiment name from the provided name or configuration.

    Args:
    ----
        config (dict): Configuration dictionary.
        name (str, optional): The provided name of the experiment. Defaults to None.

    Returns:
    ----
        str: The experiment name.
    """
    if name is None:
        name = config.get("name", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    elif "name" in config:
        print(
            f"Overwriting name {config['name']} with {name} as command line argument is stronger."
        )

    name = name.replace("*time*", datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    name = name.replace("*seed*", str(config["seed"]))

    return name


def setup_logging(logpath, name, config):
    """
    Set up a logger for experiment.

    Args:
    ----
        logdir (str): Directory where logs are stored.
        name (str, optional): Name of the experiment.
        yaml_file (str): Path to the yaml file with the config.

    Returns:
    ----
        logger: Logger object.
    """

    if os.path.isdir(logpath):
        raise ValueError(
            f"Log directory {logpath} already exists. Consider adding *time* to the name so that the experiment is unique."
        )

    os.makedirs(logpath)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(logpath, "experiment.log"))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Experiment {name} started. Logging to {logpath}.")

    current_commit = get_commit()
    logger.info(
        f"Git commit SHA as of start of experiment: {current_commit}. It is reccomended to only run experiments on a clean commit."
    )
    logger.info(
        f"https://github.com/alexanderkrauck/pretrained-graph-transformer/tree/{current_commit}"
    )

    logger.info(f"Used Config: \n\n{config}\n.")

    with open(os.path.join(logpath, "config.yml"), "w") as file:
        yaml.dump(config, file)

    logger.info(f'Copied the used config to : {os.path.join(logpath, "config.yml")}')

    return logger


def get_commit(repo_path: str = "."):
    """
    Get the current commit of the repository.

    Args
    ----
        repo_path (str): Path to the repository."""

    git_folder = Path(repo_path, ".git")
    head_name = Path(git_folder, "HEAD").read_text().split("\n")[0].split(" ")[-1]
    head_ref = Path(git_folder, head_name)
    commit = head_ref.read_text().replace("\n", "")
    return commit


def setup_wandb(name: str, logdir: str, config: dict):
    """
    Setup the wandb logger.
    """
    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    wandb.init(dir=logdir, name=name, project="pretrained_graph_transformer")

    copied_dict = config.copy()
    del copied_dict["trainer_args"]
    del copied_dict["model_args"]

    wandb.config.update(copied_dict)


def setup_batch_size(config: dict, n_devices: int):
    """
    Setup the batch size for the experiment. The batch size is divided by the number of devices and the gradient accumulation steps.

    Args:
    ----
        config (dict): Configuration dictionary.
        n_devices (int): Number of devices to use."""

    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["trainer_args"]["gradient_accumulation_steps"]
    per_device_batch_size = batch_size / (gradient_accumulation_steps * n_devices)
    if not per_device_batch_size.is_integer():
        raise ValueError(
            f"Batch size {batch_size} is not divisible by the number of devices {n_devices} and the gradient accumulation steps {gradient_accumulation_steps}."
        )
    config["trainer_args"]["per_device_train_batch_size"] = int(per_device_batch_size)
    config["trainer_args"]["per_device_eval_batch_size"] = int(per_device_batch_size)


def log_model_params(model, logger):
    """
    Log the number of parameters of the model.

    Args:
    ----
        model: The model to log the parameters of.
        logger: The logger to log the parameters to."""

    paramsum = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total number of trainable parameters: {paramsum}.")
