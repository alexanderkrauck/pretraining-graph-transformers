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
    Trainer,
    TrainingArguments,
)
import wandb


# Local application imports
from utils import data as data_utils
from utils import setup as setup_utils
from utils import evaluate as evaluate_utils


def main(
    name: str = None,
    logdir: str = "runs",
    yaml_file: str = "configs/dummy_config.yml",
    from_pretrained: str = None,
    model_type: str = "graphormer",
    return_trainer_instead=False,
):
    """
    Entry point for training and evaluating the model.

    Args
    ----
        name:(str)
            Name of the experiment.
        logdir (str): Directories where logs are stored.
        yaml_file (str): The yaml file with the config.
        from_pretrained (str): Path to a pretrained model.
        model_type (str): The type of model to use. Either "graphormer" or "graphormer3d".
        return_trainer_instead (bool): If true then the trainer will be returned and not "train" exectued. For debug.
    """

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    seeds = config["seed"]
    if isinstance(seeds, int):
        return main_run(
            config,
            name,
            logdir,
            yaml_file,
            from_pretrained,
            model_type,
            return_trainer_instead,
        )

    for seed in seeds:
        seed_config = config.copy()
        seed_config["seed"] = seed
        main_run(
            seed_config,
            name,
            logdir,
            yaml_file,
            from_pretrained,
            model_type,
            return_trainer_instead,
        )


def main_run(
    config: dict,
    name: str = None,
    logdir: str = "runs",
    yaml_file: str = "configs/dummy_config.yml",
    from_pretrained: str = None,
    model_type: str = "graphormer",
    return_trainer_instead=False,
):
    seed = config["seed"]

    name = setup_utils.get_experiment_name(config, name)
    logpath = os.path.join(logdir, name)
    logger = setup_utils.setup_logging(logpath, name, yaml_file)
    setup_utils.setup_wandb(name, logdir, config)

    logger.info(f"Set the random seed to : {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TODO: move below to a seperate function maybe
    pretraining = config["pretraining"]
    logger.info(f"For this run pretraining is : {pretraining}")

    dataset = data_utils.prepare_dataset_for_training(
        pretraining, seed, model_type = model_type, **config["data_args"], **config["model_args"]
    )
    evaluation_func = evaluate_utils.prepare_evaluation_for_training(
        pretraining, **config["data_args"]
    )

    # Set up the training arguments
    # TODO: maybe add logic for hyperparameter search

    device_count = torch.cuda.device_count()
    logger.info(
        f"Using CUDA devices: {[torch.cuda.get_device_properties(device_index) for device_index in range(device_count)]}"
    )
    setup_utils.setup_batch_size(config, device_count)

    training_args = TrainingArguments(
        output_dir=os.path.join(logpath, "checkpoints"),
        logging_dir=logpath,
        seed=seed,
        data_seed=seed,
        run_name=name,
        report_to=["wandb", "tensorboard"],
        **config["trainer_args"],
    )

    num_classes = data_utils.get_dataset_num_classes(**config["data_args"])
    config["model_args"]["classification_task"] = data_utils.get_dataset_task(
        **config["data_args"]
    )

    model, collator = setup_utils.get_model_and_collator(
        config, model_type, from_pretrained, num_classes
    )

    setup_utils.log_model_params(model, logger)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        compute_metrics=evaluation_func,
        callbacks=[evaluate_utils.CustomEarlyStoppingCallback(**config["evaluation_args"] if "evaluation_args" in config else {})]
    )

    if return_trainer_instead:
        return trainer

    trainer.train()

    test_results = trainer.evaluate(dataset["test"])

    # Prefix all keys in the dictionary with 'test_'
    test_results = {f"test_{k}": v for k, v in test_results.items()}

    # Log the results
    wandb.log(test_results)

    wandb.finish()


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

    parser.add_argument(
        "-p",
        "--from_pretrained",
        help="If provided, the pretrained model to use initially.",
        type=str,
    )

    parser.add_argument(
        "-m",
        "--model_type",
        help="The type of the model to use",
        default="graphormer",
        type=str,
    )

    args = parser.parse_args()

    main(**dict(args._get_kwargs()))
