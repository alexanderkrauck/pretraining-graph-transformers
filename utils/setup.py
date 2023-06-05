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
import shutil

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
    
    return name

def setup_logging(logpath, name, yaml_file):
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
        raise ValueError(f"Log directory {logpath} already exists. Consider adding *time* to the name so that the experiment is unique.")

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

    logger.info(f"Used Config file: {yaml_file}.")

    logger.info(f'Copied the used config to : {os.path.join(logpath, "config.yml")}')

    shutil.copy(yaml_file, os.path.join(logpath, "config.yml"))

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

def setup_wandb(name:str, logdir: str):
    """
    Setup the wandb logger.
    """
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"]="pretrained_graph_transformer"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"]="true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"]="false"

    os.environ["WANDB_NAME"] = name

    os.environ["WANDB_DIR"] = logdir#os.path.join(logdir, "wandb")

def setup_batch_size(config: dict):
    """
    Setup the batch size for the experiment. The batch size is divided by the number of devices and the gradient accumulation steps.
    
    Args:
    ----
        config (dict): Configuration dictionary."""

    batch_size = config["batch_size"]
    devices = config["devices"]
    gradient_accumulation_steps = config["trainer_args"]["gradient_accumulation_steps"]
    per_device_batch_size = batch_size / (gradient_accumulation_steps * len(devices))
    if not per_device_batch_size.is_integer():
        raise ValueError(
            f"Batch size {batch_size} is not divisible by the number of devices {len(devices)} and the gradient accumulation steps {gradient_accumulation_steps}."
        )
    config["trainer_args"]["per_device_train_batch_size"] = int(per_device_batch_size)
    config["trainer_args"]["per_device_eval_batch_size"] = int(per_device_batch_size)